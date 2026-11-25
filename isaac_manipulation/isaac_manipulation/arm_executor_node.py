"""Robot arm action server — M0609 manipulation.

pick_mineral 3-mode:
  enable_ik=True + use_world_target_ik=True (default, T2 정공법)
     → mineral world XYZ (action goal) + /ground_truth/odom 으로 arm base local
       변환 → DLS-IK 로 APPROACH/DESCEND/LIFT 자세 풀음 → /grasp/command snap.
       arm 이 mineral 까지 동적으로 접근.

  enable_ik=True + use_world_target_ik=False
     → 기존 wrist-cam servo IK (Phase 3b-3). YOLO detection 대기 + IK.
       wrist 마운트 calibration 필요. cheat 졸업 시 활성.

  enable_ik=False
     → simple-ver scripted PICK_DOWN dip + /grasp/command FixedJoint snap.
       mineral 좌표 무시. supervisor APPROACH align 의존. legacy fallback.

place_to_cargo / unload_to_base / deploy_solar_panel — scripted PLACE_BASKET.
"""
from __future__ import annotations

import math
import threading
import time
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from isaac_interfaces.action import ExecuteArmTask
from isaac_interfaces.msg import Detection, DetectionArray
from nav_msgs.msg import Odometry
from rclpy.action import ActionServer, CancelResponse, GoalResponse
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String

from .kinematics import M0609_DH, dls_ik, fk


def _quat_to_yaw(qx: float, qy: float, qz: float, qw: float) -> float:
    siny = 2.0 * (qw * qz + qx * qy)
    cosy = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny, cosy)


def _quat_to_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    n = math.sqrt(x * x + y * y + z * z + w * w) or 1.0
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def _yaw_rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
)


JOINT_NAMES = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
HOME_DEG = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]

# Pick down — joint_1=0 (forward toward mineral), shoulder dip + wrist tilt.
# Mineral is in front of rover (supervisor APPROACH P-control aligns it).
PICK_DOWN_DEG = [0.0, 25.0, 90.0, 0.0, 65.0, 0.0]

# Cargo (back of rover) — HOME 의 joint_1 만 180° 회전. LIFT → HOME 경유 후
# 단순 yaw swing 으로 무게중심 안정성 ↑ (이전 [180,25,90,0,55,0] 큰 반경 dip 시
# rover 뒤로 기울어짐 우려).
CARGO_DEG = [180.0, 0.0, 90.0, 0.0, 90.0, 0.0]

# Legacy fallback when IK is enabled and wrist detection missing.
PICK_TRAJ_DEG: List[List[float]] = [
    [  0.0,  0.0, 90.0, 0.0, 90.0, 0.0],
    [  0.0, 25.0, 90.0, 0.0, 65.0, 0.0],
    [  0.0,  0.0, 90.0, 0.0, 90.0, 0.0],
]


def _wrist_optical_to_link6(
        xyz_optical: np.ndarray,
        mount_xyz_link6: Tuple[float, float, float],
        R_optical_to_link6: np.ndarray) -> np.ndarray:
    """Convert mineral XYZ from wrist cam optical frame to link_6 frame."""
    p_link6 = R_optical_to_link6 @ xyz_optical + np.asarray(mount_xyz_link6)
    return p_link6


def _link6_to_base(p_link6: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Point in link_6 frame → arm base frame using current joints."""
    T = fk(q)
    R = T[:3, :3]
    p = T[:3, 3]
    return R @ p_link6 + p


class _ArmCanceled(Exception):
    """운영자가 동작을 취소함 (E-STOP). _publish_joint_state 가 던진다."""


class ArmExecutorNode(Node):
    SUPPORTED_COMMANDS = {
        "pick_mineral",
        "place_to_cargo",
        "unload_to_base",
        "deploy_solar_panel",
        "home",                 # 취소(E-STOP)로 멈춘 팔을 HOME 으로 되돌린다
    }

    def __init__(self) -> None:
        super().__init__("arm_executor_node")
        self.declare_parameter("publish_hz", 30.0)
        self.declare_parameter("step_duration_sec", 3.0)
        self.declare_parameter("joint_command_topic", "/arm/joint_command")
        self.declare_parameter("joint_state_topic", "/joint_states_raw")
        self.declare_parameter("wrist_detections_topic", "/perception/wrist_detections")
        self.declare_parameter("wrist_detection_wait_sec", 6.0)
        self.declare_parameter("hover_above_mineral_m", 0.05)
        self.declare_parameter("min_confidence", 0.5)
        # 손목 카메라(D455 color)의 link_6 기준 장착 위치·자세. vehicle_v3.usd
        # 실측: 원점 (0.0115, 0.045, 0.05), 광축 = link_6 +z (공구 방향),
        # 영상 오른쪽 = link_6 +x, 영상 아래 = link_6 +y → optical→link_6 회전은 단위행렬.
        self.declare_parameter("wrist_mount_xyz_link6", [0.0115, 0.045, 0.05])
        # Rotation: optical (z fwd, x right, y down) -> link_6 axes.
        # Stored as 9 flat values, row-major 3x3.
        self.declare_parameter("wrist_R_optical_to_link6", [
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ])
        self.declare_parameter("ik_use_orientation", False)
        # 3-mode pick_mineral:
        #   enable_ik=True (default) + world_target=True → T2 DLS-IK world target
        #     (mineral world XYZ → arm base local → IK → APPROACH/DESCEND/LIFT)
        #   enable_ik=True + world_target=False → wrist-cam servo IK (Phase 3b-3)
        #   enable_ik=False → simple-ver scripted PICK_DOWN (legacy fallback)
        self.declare_parameter("enable_ik", True)
        self.declare_parameter("use_world_target_ik", True)
        self.declare_parameter("odom_topic", "/ground_truth/odom")
        # IK phase 높이 (arm base local z offset, mineral world 좌표 기준 상대)
        self.declare_parameter("ik_approach_dz", 0.30)
        self.declare_parameter("ik_descend_dz", 0.05)
        # 부착 방식 — 시뮬레이터의 --grasp 와 맞춰야 한다.
        #   friction (기본): 넓은 집게를 벌린 채 광물 위로 내려 닫고, 손가락
        #                    마찰만으로 든다 (시뮬레이터 --grasp friction --gripper wide).
        #   contact        : RG2 손끝을 광물 위로 가져가 닿을 때까지 천천히 내린다.
        #                    시뮬레이터는 손끝이 광물에 닿은 순간에만 FixedJoint 로
        #                    부착한다 (--grasp contact --gripper rg2).
        #   snap           : 예전 방식 (광물 근처로 팔만 뻗고 순간이동 부착).
        self.declare_parameter("grasp_mode", "friction")
        # 넓은 집게 치수 (isaac_sim/scripts/wide_gripper.py 와 같아야 한다).
        self.declare_parameter("friction_palm_offset_m", 0.097)    # link_6 → palm 아랫면
        self.declare_parameter("friction_finger_length_m", 0.17)   # palm 아랫면 → 손가락 끝
        # palm 아랫면을 광물 윗면보다 이만큼 위에서 멈추고 닫는다.
        self.declare_parameter("friction_palm_clearance_m", 0.03)
        # 손목 카메라가 윗면 높이를 못 쟀을 때 쓰는 값 (광물 중심 → 윗면).
        self.declare_parameter("friction_top_above_center_m", 0.114)
        self.declare_parameter("friction_descend_speed", 0.06)     # m/s
        self.declare_parameter("friction_close_timeout_sec", 4.0)
        # link_6 원점 → 손끝까지 거리 (vehicle_v3.usd 의 RG2-FT inner finger 끝).
        self.declare_parameter("tool_tip_offset_m", 0.194)
        # 손끝의 시작·최저 높이 (인식한 광물 중심 기준). 광물 높이는 약 0.23 m 라
        # 윗면은 중심 +0.10~0.13 m 에 있다. 시작은 그보다 넉넉히 위, 최저는
        # 윗면보다 한참 아래 — 인식 높이가 틀려도 그 사이 어딘가에서 닿는다.
        self.declare_parameter("contact_start_above_m", 0.24)
        self.declare_parameter("contact_end_above_m", 0.02)
        self.declare_parameter("contact_descend_speed", 0.04)   # m/s
        # contact pick 에서 손끝을 광물 위로 가져간 뒤 손목 카메라로 광물 중심을
        # 다시 재서 수평 위치를 고친다. 로버 카메라는 옆에서 보므로 깊이 방향
        # (표면→중심) 이 불확실하지만, 손목 카메라는 위에서 내려다보므로 수평
        # 중심이 bbox 중심으로 바로 나온다.
        self.declare_parameter("use_wrist_correction", True)
        # 손목 카메라가 목표 자리에서 광물을 한 번도 못 보면 집지 않는다 (허공 집기 방지).
        self.declare_parameter("abort_if_wrist_sees_nothing", True)
        self.declare_parameter("wrist_presence_radius_m", 0.5)
        # 손목 카메라가 안정적으로 알아보는 종류에만 적용한다. 초록 가스는 실제로 있어도
        # 손목 카메라에 안 잡히는 일이 잦아서(3 번 중 3 번) 예전처럼 그대로 진행한다.
        self.declare_parameter("wrist_presence_classes", ["blue_mineral", "yellow_mineral"])
        self.declare_parameter("wrist_correction_gate_m", 0.15)   # 이보다 먼 검출은 다른 물체로 봄
        # 손목 카메라의 깊이 영상으로 "땅 위로 솟은 덩어리" 를 직접 잰다 (YOLO 를 거치지
        # 않음). 초록 가스(20 cm 상자)는 위에서 보면 YOLO 가 못 알아보고, 대신 집게
        # 손가락을 가스로 잘못 본다 — 그래서 위치 보정 없이 집다가 손가락이 상자 위에
        # 얹혔다. YOLO 보정이 안 될 때 이 측정으로 중심·윗면·놓인 방향을 얻는다.
        self.declare_parameter("wrist_shape_measure", True)
        self.declare_parameter("wrist_align_yaw", True)           # 집게를 상자 면에 맞춰 돌린다
        self.declare_parameter("wrist_depth_topic", "/camera/wrist/depth")
        self.declare_parameter("wrist_info_topic", "/camera/wrist/camera_info")
        # 집게를 벌렸을 때 안쪽 면 간격 (isaac_sim/scripts/wide_gripper.py 의 GAP_OPEN).
        self.declare_parameter("gripper_open_gap_m", 0.33)
        self.declare_parameter("wrist_settle_sec", 1.0)
        self.declare_parameter("grasp_command_topic", "/grasp/command")
        # Step delay between grasp publish and arm continuing — gives
        # vehicle_v3 ScriptNode time to attach the FixedJoint.
        self.declare_parameter("grasp_publish_delay_sec", 0.3)
        # pickup 직후 시뮬레이터가 실제로 mineral 을 그리퍼에 붙였는지 확인.
        # run_vehicle_v3 의 GraspBridge 가 std_msgs/String 으로 발행 (data =
        # 잡은 mineral 이름, 없으면 ""). 이 시간 안에 부착 확인이 안 되면 pick 을
        # 실패로 끝내 mission_manager 가 수집으로 세지 않게 한다. 0 = 확인 안 함.
        # status 토픽이 한 번도 안 들어온 경우(구버전 sim)는 기존처럼 성공 처리.
        self.declare_parameter("grasp_status_topic", "/grasp/status")
        self.declare_parameter("grasp_confirm_timeout_sec", 3.0)
        # CARGO_SWING 완료 후 release 까지 대기 — Isaac articulation 이
        # ROS joint_command 를 따라가는 visual lag 흡수. ROS 입장 swing 끝나도
        # mineral 이 시각적으로 cargo (back basket) 에 도달하기 전이라면
        # release 시점에 너무 일찍 사라지는 느낌을 줌 (2026-05-27 사용자 보고).
        # 2.5s — CARGO_SWING 의 실 회전 시간 (~1.3s, joint_1 0→180°) 후 articulation
        # 의 visual lag 흡수. 1.5s 시 mineral hide 가 cargo 도달 전에 일어나는
        # 사례 관찰 (2026-05-27 rover_2 시연). 2.5s 로 안정.
        self.declare_parameter("cargo_settle_sec", 2.5)

        self.joint_pub = self.create_publisher(
            JointState, str(self.get_parameter("joint_command_topic").value), 10)
        # /grasp/command — published as geometry_msgs/Twist (OmniGraph
        # constraint: no generic-string subscriber in isaacsim.ros2.bridge).
        # Encoding:
        #   pickup x y z target_id   → linear=(x,y,z) angular.x=+1.0
        #   release                  → linear=(0,0,0) angular.x=-1.0
        # vehicle_v3 ScriptNode decodes the sign of angular.x.
        self.grasp_pub = self.create_publisher(
            Twist, str(self.get_parameter("grasp_command_topic").value), 10)

        self.create_subscription(
            DetectionArray, str(self.get_parameter("wrist_detections_topic").value),
            self._on_wrist_detections, SENSOR_QOS)

        self.create_subscription(
            JointState, str(self.get_parameter("joint_state_topic").value),
            self._on_joint_states, SENSOR_QOS)

        # rover (= arm base via /ground_truth/odom chassisFrameId="base_link")
        # world pose — world target IK 에서 mineral world → arm base local 변환에 필요.
        self.create_subscription(
            Odometry, str(self.get_parameter("odom_topic").value),
            self._on_odom, SENSOR_QOS)

        self._grasp_status_lock = threading.Lock()
        self._grasp_status_held: str = ""
        self._grasp_status_mono: float = 0.0   # 마지막 수신 시각 (monotonic)
        self._grasp_status_count: int = 0
        self.create_subscription(
            String, str(self.get_parameter("grasp_status_topic").value),
            self._on_grasp_status, 10)

        self._wrist_lock = threading.Lock()
        self._wrist_dets: List[Detection] = []
        self._wrist_stamp_ns: int = 0
        self._wrist_depth: Optional[Image] = None
        self._wrist_depth_ns: int = 0
        self._wrist_K: Optional[np.ndarray] = None
        self.create_subscription(
            Image, str(self.get_parameter("wrist_depth_topic").value),
            self._on_wrist_depth, SENSOR_QOS)
        self.create_subscription(
            CameraInfo, str(self.get_parameter("wrist_info_topic").value),
            self._on_wrist_info, SENSOR_QOS)
        # Current arm joint positions (rad) from /joint_states_raw. None until
        # first message arrives — we fall back to HOME_DEG for IK base then.
        self._current_arm_q_rad: Optional[np.ndarray] = None
        self._joint_state_logged_once = False

        self._odom_lock = threading.Lock()
        self._rover_xyz: Optional[np.ndarray] = None
        self._rover_yaw: Optional[float] = None
        self._rover_R: Optional[np.ndarray] = None
        self._odom_hist: List[Tuple[float, np.ndarray]] = []   # (monotonic, xyz)

        # 상대 토픽 "execute_arm_task" 사용 — 단일 시연에선 root namespace 라
        # "/execute_arm_task" 로 광고되고 (기존 동작 유지), multi 시연에선
        # namespace=ns 가 적용되어 "/<ns>/execute_arm_task" 로 자동 광고된다.
        # ROS2 action 의 sub-topic (_action/status 등) 도 따라서 namespace 안.
        self.action_server = ActionServer(
            self,
            ExecuteArmTask,
            "execute_arm_task",
            execute_callback=self._execute_callback,
            goal_callback=self._goal_callback,
            cancel_callback=self._cancel_callback,
        )

        # 취소(E-STOP) 처리: 실행 중인 goal 과, 멈춘 자리의 관절각.
        self._active_goal = None
        self._last_cmd_deg: Optional[List[float]] = None
        self._frozen_deg: Optional[List[float]] = None

        self._initial_home_sent = False
        self.create_timer(1.0, self._send_initial_home_once)

        self.get_logger().info(
            "arm_executor_node ready: /execute_arm_task (wrist-cam visual servo + DLS-IK)")

    def _send_initial_home_once(self) -> None:
        if self._initial_home_sent:
            return
        self._publish_joint_state(HOME_DEG)
        self._initial_home_sent = True

    def _on_wrist_detections(self, msg: DetectionArray) -> None:
        with self._wrist_lock:
            self._wrist_dets = list(msg.detections)
            self._wrist_stamp_ns = self.get_clock().now().nanoseconds

    def _on_wrist_depth(self, msg: Image) -> None:
        with self._wrist_lock:
            self._wrist_depth = msg
            self._wrist_depth_ns = self.get_clock().now().nanoseconds

    def _on_wrist_info(self, msg: CameraInfo) -> None:
        if self._wrist_K is None and msg.k[0] > 0.0:
            self._wrist_K = np.array(msg.k, dtype=np.float64).reshape(3, 3)

    def _on_joint_states(self, msg: JointState) -> None:
        """Extract M0609 6 arm joints from rover's full joint state."""
        name_to_pos = dict(zip(msg.name, msg.position))
        try:
            q = np.array([name_to_pos[n] for n in JOINT_NAMES], dtype=np.float64)
        except KeyError as e:
            if not self._joint_state_logged_once:
                self.get_logger().warn(
                    f"joint_states_raw missing arm joint {e} — IK will use HOME as base. "
                    f"Available joints: {list(msg.name)}")
                self._joint_state_logged_once = True
            return
        self._current_arm_q_rad = q
        if not self._joint_state_logged_once:
            self.get_logger().info(
                f"joint_states_raw arm joints OK: "
                f"q_deg = [{','.join(f'{math.degrees(v):.1f}' for v in q)}]")
            self._joint_state_logged_once = True

    def _on_odom(self, msg: Odometry) -> None:
        """Rover (articulation root = arm base) world pose 추출.
        chassisFrameId='base_link' = vehicle_v3 의 m0609/base_link → arm base.
        """
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        with self._odom_lock:
            self._rover_xyz = np.array([p.x, p.y, p.z], dtype=np.float64)
            self._rover_yaw = _quat_to_yaw(o.x, o.y, o.z, o.w)
            self._rover_R = _quat_to_matrix(o.x, o.y, o.z, o.w)
            now = time.monotonic()
            self._odom_hist.append((now, self._rover_xyz))
            while self._odom_hist and now - self._odom_hist[0][0] > 1.0:
                self._odom_hist.pop(0)

    def _best_wrist_detection(self,
                              wait_sec: float,
                              min_conf: float) -> Optional[Detection]:
        deadline_ns = self.get_clock().now().nanoseconds + int(wait_sec * 1e9)
        while self.get_clock().now().nanoseconds < deadline_ns:
            with self._wrist_lock:
                dets = [d for d in self._wrist_dets if d.confidence >= min_conf]
                stamp = self._wrist_stamp_ns
            if dets:
                # freshness check: must be within last 0.5s
                now_ns = self.get_clock().now().nanoseconds
                if now_ns - stamp < int(0.5e9):
                    return max(dets, key=lambda d: d.confidence)
            time.sleep(0.05)
        return None

    def _goal_callback(self, goal_request: ExecuteArmTask.Goal) -> GoalResponse:
        command = goal_request.command.strip()
        if command not in self.SUPPORTED_COMMANDS:
            self.get_logger().warning(f"Rejecting unsupported arm command: {command}")
            return GoalResponse.REJECT
        self.get_logger().info(
            f"Accepted arm goal command={command} target_id={goal_request.target_id} "
            f"xyz=({goal_request.target_x:.2f},{goal_request.target_y:.2f},"
            f"{goal_request.target_z:.2f})")
        return GoalResponse.ACCEPT

    def _cancel_callback(self, goal_handle) -> CancelResponse:
        self.get_logger().info("Arm goal cancel requested")
        return CancelResponse.ACCEPT

    def _execute_callback(self, goal_handle):
        """goal 하나를 실행한다. 취소 요청이 오면 팔을 그 자리에 세운다.

        모든 동작은 _publish_joint_state 로 관절 목표를 조금씩 보내서 움직인다. 취소가
        요청되면 그 함수가 _ArmCanceled 를 던져, 어느 단계에 있든 다음 목표를 보내지
        않고 여기로 빠져나온다 → 팔은 마지막으로 받은 목표 자세에 그대로 서 있다
        (쥐고 있던 광물도 쥔 채). 다시 움직이려면 "home" 명령을 보낸다.
        """
        self._active_goal = goal_handle
        try:
            return self._run_command(goal_handle)
        except _ArmCanceled:
            self._frozen_deg = list(self._last_cmd_deg) if self._last_cmd_deg else None
            where = ("[" + ",".join(f"{v:.0f}" for v in self._frozen_deg) + "]°"
                     if self._frozen_deg else "?")
            self.get_logger().warn(f"arm goal canceled — 팔을 그 자리에 세움 {where}")
            goal_handle.canceled()
            return ExecuteArmTask.Result(success=False,
                                         message="canceled — arm stopped in place")
        finally:
            self._active_goal = None

    def _execute_home(self, goal_handle):
        """멈춰 있던 팔을 HOME 으로 되돌리고 집게를 연다 (E-STOP 해제 뒤)."""
        start = self._frozen_deg
        if start is None and self._current_arm_q_rad is not None:
            start = [math.degrees(float(v)) for v in self._current_arm_q_rad[:6]]
        if start is None:
            start = list(HOME_DEG)
        self._goto(start, HOME_DEG, goal_handle, "HOME_RECOVER", 0.0, 0.9, duration=3.0)
        self._publish_joint_state(HOME_DEG)
        self._frozen_deg = None
        # 쥐고 있던 것이 있으면 놓는다 (HOME 은 적재함 밖이라 땅에 떨어진다 — 다시 집으면 된다).
        self._publish_grasp("release")
        goal_handle.succeed()
        self.get_logger().info("arm home done")
        return ExecuteArmTask.Result(success=True, message="home completed")

    def _run_command(self, goal_handle):
        command = goal_handle.request.command.strip()
        self.get_logger().info(f"Executing arm command: {command}")

        if command == "home":
            return self._execute_home(goal_handle)
        if self._frozen_deg is not None:
            # 취소로 멈춘 자리에서 바로 다음 동작을 시작하면 목표가 튄다 — 먼저 HOME 으로.
            start = self._frozen_deg
            self._goto(start, HOME_DEG, goal_handle, "HOME_RECOVER", 0.0, 0.05, duration=3.0)
            self._frozen_deg = None
            self._publish_grasp("release")
        if command == "pick_mineral":
            if bool(self.get_parameter("enable_ik").value):
                if bool(self.get_parameter("use_world_target_ik").value):
                    if str(self.get_parameter("grasp_mode").value) in (
                            "contact", "friction"):
                        return self._execute_pick_contact(goal_handle)
                    return self._execute_pick_world_target(goal_handle)
                return self._execute_pick_ik(goal_handle)
            return self._execute_pick_scripted(goal_handle)
        # Other commands (place_to_cargo, unload_to_base, deploy_solar_panel)
        # — fall back to scripted PLACE_BASKET sequence as visual demo.
        return self._execute_scripted(goal_handle, PLACE_BASKET_TRAJ_DEG, command)

    def _publish_grasp(self, cmd: str, x: float = 0.0, y: float = 0.0,
                       z: float = 0.0, target_id: str = "") -> None:
        """Publish grasp command as Twist (hijacked for OmniGraph compatibility)."""
        msg = Twist()
        if cmd == "pickup":
            msg.linear.x = float(x)
            msg.linear.y = float(y)
            msg.linear.z = float(z)
            msg.angular.x = 1.0    # pickup mode marker
        else:
            msg.angular.x = -1.0   # release mode marker
        # Publish multiple times so OmniGraph subscriber is sure to capture it.
        # 2026-05-27 시연에서 rover_1 의 release burst (3×0.05s) 가 Isaac
        # ros2_bridge 의 subscription frame 과 race 로 drop 되어 mineral 이
        # attach 된 채 남는 사례 관찰. 6×0.1s 로 burst window 0.6s 확장해
        # bridge 가 최소 한 frame 은 받도록 보장.
        for _ in range(6):
            self.grasp_pub.publish(msg)
            time.sleep(0.1)
        delay = float(self.get_parameter("grasp_publish_delay_sec").value)
        if delay > 0.0:
            time.sleep(delay)
        self.get_logger().info(
            f"grasp/command -> {cmd} xyz=({x:.3f},{y:.3f},{z:.3f}) "
            f"target_id={target_id}")

    def _on_grasp_status(self, msg: String) -> None:
        with self._grasp_status_lock:
            self._grasp_status_held = msg.data
            self._grasp_status_mono = time.monotonic()
            self._grasp_status_count += 1

    def _confirm_grasp(self, since_mono: float) -> bool:
        """pickup 발행(since_mono) 이후 시뮬레이터가 부착을 보고했는지 확인."""
        timeout = float(self.get_parameter("grasp_confirm_timeout_sec").value)
        if timeout <= 0.0:
            return True
        with self._grasp_status_lock:
            seen_any = self._grasp_status_count > 0
        if not seen_any:
            self.get_logger().warn(
                "grasp status 토픽 수신 없음 — 부착 확인 생략 (성공으로 간주)")
            return True
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._grasp_status_lock:
                held = self._grasp_status_held
                fresh = self._grasp_status_mono >= since_mono
            if fresh and held:
                self.get_logger().info(f"grasp confirmed — holding {held}")
                return True
            time.sleep(0.05)
        return False

    def _abort_pick_not_attached(self, goal_handle, cur_deg: List[float]):
        """부착 실패 — 팔을 HOME 으로 되돌리고 action 을 실패로 종료."""
        self.get_logger().warn(
            "grasp NOT attached — 팔이 닿는 범위에 mineral 없음. pick 실패 처리")
        self._goto(cur_deg, HOME_DEG, goal_handle, "HOME_ABORT", 0.5, 1.0)
        self._publish_joint_state(HOME_DEG)
        goal_handle.abort()
        return ExecuteArmTask.Result(
            success=False, message="grasp not attached (no mineral in reach)")

    def _execute_pick_scripted(self, goal_handle):
        """Simple-ver pick — single forward dip + single cargo swing:
            HOME -> PICK_DOWN (forward, joint_1=0) -> GRASP publish -> HOME
                 -> CARGO (joint_1=180) -> RELEASE publish -> HOME
        """
        req = goal_handle.request
        tx, ty, tz = float(req.target_x), float(req.target_y), float(req.target_z)

        cur_q_rad = (self._current_arm_q_rad
                     if self._current_arm_q_rad is not None
                     else np.radians(HOME_DEG))
        cur_deg = list(np.degrees(cur_q_rad))

        # 1) HOME
        self._goto(cur_deg, HOME_DEG, goal_handle, "HOME_PRE", 0.0, 0.1)
        cur_deg = list(HOME_DEG)

        # 2) PICK_DOWN — forward dip (joint_1=0, mineral in front of rover)
        self._goto(cur_deg, PICK_DOWN_DEG, goal_handle, "PICK_DOWN", 0.1, 0.3)
        cur_deg = list(PICK_DOWN_DEG)

        # 3) GRASP — vehicle_v3 attaches nearest mineral to gripper
        t_pickup = time.monotonic()
        self._publish_grasp("pickup", x=tx, y=ty, z=tz, target_id=req.target_id)
        if not self._confirm_grasp(t_pickup):
            return self._abort_pick_not_attached(goal_handle, cur_deg)
        fb = ExecuteArmTask.Feedback()
        fb.state = "GRASP_CLOSE"
        fb.progress = 0.35
        fb.message = "pickup published"
        goal_handle.publish_feedback(fb)

        # 4) LIFT — back to HOME (mineral comes along via FixedJoint)
        self._goto(cur_deg, HOME_DEG, goal_handle, "LIFT", 0.35, 0.55)
        cur_deg = list(HOME_DEG)

        # 5) CARGO — single swing to back of rover
        self._goto(cur_deg, CARGO_DEG, goal_handle, "CARGO_SWING", 0.55, 0.8)
        cur_deg = list(CARGO_DEG)

        # 5.5) Visual settle — articulation lag 흡수 (mineral 이 cargo 도착 후 hide).
        settle = float(self.get_parameter("cargo_settle_sec").value)
        if settle > 0.0:
            time.sleep(settle)

        # 6) RELEASE — vehicle_v3 detaches + hides the mineral (cargo stowed)
        self._publish_grasp("release")
        fb = ExecuteArmTask.Feedback()
        fb.state = "RELEASE"
        fb.progress = 0.85
        fb.message = "release published"
        goal_handle.publish_feedback(fb)

        # 7) HOME guarantee
        self._goto(cur_deg, HOME_DEG, goal_handle, "HOME_POST", 0.85, 1.0)
        self._publish_joint_state(HOME_DEG)
        goal_handle.succeed()
        self.get_logger().info("pick_mineral (scripted) done")
        return ExecuteArmTask.Result(success=True, message="pick_mineral completed")

    def _execute_pick_world_target(self, goal_handle):
        """T2 DLS-IK + /grasp/command FixedJoint snap (정공법 + grip cheat).

        Action goal 의 target_xyz 를 mineral world XYZ 로 받아:
          1. /ground_truth/odom 도착 대기 (arm base world pose)
          2. mineral world → arm base local 변환 (yaw 보정)
          3. APPROACH / DESCEND / LIFT 의 target 을 mineral 기준 z offset 으로 IK
          4. State machine: HOME → APPROACH → DESCEND → grasp pickup → LIFT
             → CARGO → grasp release → HOME
        """
        req = goal_handle.request
        mineral_world = np.array(
            [req.target_x, req.target_y, req.target_z], dtype=np.float64)

        # 1) /ground_truth/odom 도착 대기 (최대 2초)
        deadline_ns = self.get_clock().now().nanoseconds + int(2e9)
        while self.get_clock().now().nanoseconds < deadline_ns:
            with self._odom_lock:
                if self._rover_xyz is not None:
                    break
            time.sleep(0.05)
        with self._odom_lock:
            arm_base_xyz = (None if self._rover_xyz is None
                            else self._rover_xyz.copy())
            yaw = self._rover_yaw
        if arm_base_xyz is None or yaw is None:
            self.get_logger().error(
                "world_target_ik: no /ground_truth/odom — abort")
            goal_handle.abort()
            return ExecuteArmTask.Result(success=False, message="no odom")

        # 2) mineral world → arm base local (yaw 보정)
        R = _yaw_rot_z(yaw)
        target_local = R.T @ (mineral_world - arm_base_xyz)
        self.get_logger().info(
            f"mineral world=({mineral_world[0]:.2f},{mineral_world[1]:.2f},"
            f"{mineral_world[2]:.2f}) arm_base world=({arm_base_xyz[0]:.2f},"
            f"{arm_base_xyz[1]:.2f},{arm_base_xyz[2]:.2f}) yaw={math.degrees(yaw):.1f}° "
            f"→ target_local=({target_local[0]:.3f},{target_local[1]:.3f},"
            f"{target_local[2]:.3f})")

        # 3) IK target — APPROACH (위), DESCEND (mineral 옆)
        dz_a = float(self.get_parameter("ik_approach_dz").value)
        dz_d = float(self.get_parameter("ik_descend_dz").value)
        t_approach = target_local + np.array([0.0, 0.0, dz_a])
        t_descend = target_local + np.array([0.0, 0.0, dz_d])

        q_home = np.array([math.radians(v) for v in HOME_DEG])
        q_approach, ok_a, err_a = dls_ik(t_approach, None, q_home,
                                          position_only=True)
        q_descend, ok_d, err_d = dls_ik(t_descend, None, q_approach,
                                         position_only=True)
        if not (ok_a and ok_d):
            self.get_logger().error(
                f"IK 미수렴: APPROACH={ok_a}(err={err_a:.4f}) "
                f"DESCEND={ok_d}(err={err_d:.4f})")
            goal_handle.abort()
            return ExecuteArmTask.Result(
                success=False,
                message=f"IK fail: a={ok_a} d={ok_d}")

        deg_approach = list(np.degrees(q_approach))
        deg_descend = list(np.degrees(q_descend))
        self.get_logger().info(
            f"IK solved: APPROACH q_deg={[round(v,1) for v in deg_approach]} "
            f"DESCEND q_deg={[round(v,1) for v in deg_descend]}")

        # 4) State machine — LIFT 제거 (DESCEND → HOME_MID 직접 보간으로 통합):
        #   HOME → APPROACH → DESCEND → pickup → HOME_MID → CARGO → release → HOME
        cur_deg = list(HOME_DEG)
        self._goto(cur_deg, HOME_DEG, goal_handle, "HOME_PRE", 0.0, 0.05)
        cur_deg = list(HOME_DEG)

        self._goto(cur_deg, deg_approach, goal_handle, "APPROACH", 0.05, 0.25)
        cur_deg = list(deg_approach)

        self._goto(cur_deg, deg_descend, goal_handle, "DESCEND", 0.25, 0.45)
        cur_deg = list(deg_descend)

        # 5) Grasp pickup — FixedJoint snap (mineral → gripper link origin)
        t_pickup = time.monotonic()
        self._publish_grasp("pickup",
                            x=float(mineral_world[0]),
                            y=float(mineral_world[1]),
                            z=float(mineral_world[2]),
                            target_id=req.target_id)
        if not self._confirm_grasp(t_pickup):
            return self._abort_pick_not_attached(goal_handle, cur_deg)
        fb = ExecuteArmTask.Feedback()
        fb.state = "GRASP_PICKUP"
        fb.progress = 0.5
        fb.message = "pickup published"
        goal_handle.publish_feedback(fb)

        # 6) DESCEND → HOME 직접 보간 (LIFT 통합). mineral 은 collision off 라
        # rover body 통과해도 충돌 영향 X. arm 자세 dip → 직립 자연 펴짐.
        self._goto(cur_deg, HOME_DEG, goal_handle, "HOME_MID", 0.5, 0.7)
        cur_deg = list(HOME_DEG)

        # 7) CARGO yaw swing (HOME 의 joint_1 만 180°)
        self._goto(cur_deg, CARGO_DEG, goal_handle, "CARGO_SWING", 0.7, 0.85)
        cur_deg = list(CARGO_DEG)

        # 7.5) Visual settle — arm 이 시각적으로 cargo 도착까지 대기.
        settle = float(self.get_parameter("cargo_settle_sec").value)
        if settle > 0.0:
            time.sleep(settle)

        # 8) Release — detach + MakeInvisible
        self._publish_grasp("release")
        fb = ExecuteArmTask.Feedback()
        fb.state = "RELEASE"
        fb.progress = 0.9
        fb.message = "release published"
        goal_handle.publish_feedback(fb)

        # 9) HOME
        self._goto(cur_deg, HOME_DEG, goal_handle, "HOME_POST", 0.9, 1.0)
        self._publish_joint_state(HOME_DEG)
        goal_handle.succeed()
        self.get_logger().info("pick_mineral (T2 DLS-IK world target) done")
        return ExecuteArmTask.Result(
            success=True, message="pick_mineral completed (IK)")

    def _held_now(self, since_mono: float) -> str:
        """since_mono 이후에 받은 grasp status 가 알려준 광물 이름 (없으면 "")."""
        with self._grasp_status_lock:
            if self._grasp_status_mono >= since_mono:
                return self._grasp_status_held
        return ""

    def _wait_base_still(self, timeout: float = 4.0) -> float:
        """받침대가 멈출 때까지 기다린다 (최근 0.4 s 이동 < 3 mm). 기다린 시간 반환.

        mission_manager 는 정지 명령과 동시에 팔 goal 을 보내는데, 로버는 감속
        한계 때문에 그 뒤로도 4~7 cm 더 굴러간다 (실측). 그 전에 목표를 받침대
        좌표로 바꾸면 그만큼 어긋난다.
        """
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            with self._odom_lock:
                hist = list(self._odom_hist)
            if hist and hist[-1][0] - hist[0][0] >= 0.4:
                recent = [p for t, p in hist if hist[-1][0] - t <= 0.4]
                if float(np.linalg.norm(recent[-1] - recent[0])) < 0.003:
                    break
            time.sleep(0.05)
        return time.monotonic() - t0

    def _wrist_measure(self, expected: np.ndarray):
        """손목 카메라 검출로 광물 중심과 윗면을 팔 받침대 좌표로 잰다.

        expected (받침대 좌표) 에서 수평으로 gate 안에 있는 검출만 쓰고, 서로
        다른 영상 5 장의 중앙값을 돌려준다. 3 장도 못 모으면 None.

        return: (중심, 표면) — 표면은 카메라가 실제로 본 점 (위에서 내려다보므로
        윗면). 인식 노드는 표면 점을 물체 반지름만큼 시선 방향으로 밀어 중심을
        만들고 그 반지름을 bbox_size_m.z (= 지름) 로 같이 보내므로 되돌릴 수 있다.
        """
        gate = float(self.get_parameter("wrist_correction_gate_m").value)
        min_conf = float(self.get_parameter("min_confidence").value)
        mount = np.array([float(v) for v in
                          self.get_parameter("wrist_mount_xyz_link6").value])
        R_oc = np.array(list(self.get_parameter("wrist_R_optical_to_link6").value),
                        dtype=np.float64).reshape(3, 3)
        time.sleep(max(0.0, float(self.get_parameter("wrist_settle_sec").value)))
        samples: List[np.ndarray] = []
        rejected: List[str] = []
        self._wrist_nearest_m = None     # 이번 측정에서 본 광물 중 예상 위치에 가장 가까운 거리
        last_stamp = 0
        t_end = time.monotonic() + 2.5
        while time.monotonic() < t_end and len(samples) < 5:
            with self._wrist_lock:
                dets = list(self._wrist_dets)
                stamp = self._wrist_stamp_ns
            q = self._current_arm_q_rad
            fresh = self.get_clock().now().nanoseconds - stamp < int(0.5e9)
            if stamp != last_stamp and fresh and q is not None:
                last_stamp = stamp
                T = fk(q)
                best = None
                for d in dets:
                    xyz = np.array([d.world_position.x, d.world_position.y,
                                    d.world_position.z])
                    if d.confidence < min_conf or not xyz.any():
                        rejected.append(f"{d.class_name} conf={d.confidence:.2f} "
                                        f"{'깊이 없음' if not xyz.any() else '신뢰도 낮음'}")
                        continue
                    p_base = T[:3, :3] @ (R_oc @ xyz + mount) + T[:3, 3]
                    off = float(np.linalg.norm((p_base - expected)[:2]))
                    # 카메라가 본 표면 점이 집게 자신이면 버린다 (어두운 손가락을
                    # 초록 가스로 보는 일이 있다 — 2026-11-25 실측, conf 0.5~0.7).
                    dist0 = float(np.linalg.norm(xyz))
                    r0 = min(0.5 * float(d.bbox_size_m.z), 0.9 * dist0)
                    if self._is_own_gripper(R_oc @ (xyz * (1.0 - r0 / dist0)) + mount):
                        rejected.append(f"{d.class_name} conf={d.confidence:.2f} "
                                        f"집게 손가락을 광물로 봄")
                        continue
                    if self._wrist_nearest_m is None or off < self._wrist_nearest_m:
                        self._wrist_nearest_m = off
                    if off < gate and (best is None or off < best[0]):
                        dist = float(np.linalg.norm(xyz))
                        r_obj = min(0.5 * float(d.bbox_size_m.z), 0.9 * dist)
                        surf = xyz * (1.0 - r_obj / dist)
                        s_base = T[:3, :3] @ (R_oc @ surf + mount) + T[:3, 3]
                        best = (off, p_base, s_base)
                    elif off >= gate:
                        dv = (p_base - expected) * 100.0
                        rejected.append(f"{d.class_name} conf={d.confidence:.2f} "
                                        f"예상 위치에서 {off * 100:.0f} cm "
                                        f"(받침대 기준 {dv[0]:+.0f},{dv[1]:+.0f},{dv[2]:+.0f} cm, "
                                        f"카메라에서 {float(np.linalg.norm(xyz)) * 100:.0f} cm, "
                                        f"지름 추정 {float(d.bbox_size_m.z) * 100:.0f} cm)")
                if best is not None:
                    samples.append(np.concatenate([best[1], best[2]]))
            time.sleep(0.05)
        if len(samples) < 3:
            self.get_logger().info(
                f"wrist measure: 쓸 수 있는 검출 {len(samples)}장 — 버린 것: "
                f"{rejected[-2:] if rejected else '검출 자체가 없음'}")
            return None
        med = np.median(np.array(samples), axis=0)
        return med[:3], med[3:]

    def _is_own_gripper(self, p_link6: np.ndarray) -> bool:
        """link_6 좌표의 점이 넓은 집게(손가락·palm) 자리인가."""
        palm = float(self.get_parameter("friction_palm_offset_m").value)
        tip = palm + float(self.get_parameter("friction_finger_length_m").value)
        half_gap = 0.5 * float(self.get_parameter("gripper_open_gap_m").value)
        if p_link6[2] < palm + 0.01:
            return True
        return bool(p_link6[2] < tip + 0.03 and abs(p_link6[0]) > half_gap - 0.03)

    def _wrist_measure_shape(self, expected: np.ndarray):
        """손목 카메라 깊이 영상에서 목표 자리의 '땅 위로 솟은 덩어리' 를 잰다.

        물체 종류를 가리지 않는다 (YOLO 결과를 쓰지 않음). 받침대 좌표로
          center (x, y) — 위에서 본 바닥 모양에 가장 꼭 맞는 직사각형의 중심
          top_z         — 윗면 높이
          size (a, b)   — 직사각형 두 변의 길이
          angle         — 변 a 의 방향 (rad, 받침대 좌표)
        를 dict 로 돌려준다. 못 재면 None.
        """
        log = self.get_logger()
        with self._wrist_lock:
            msg, stamp = self._wrist_depth, self._wrist_depth_ns
        K, q = self._wrist_K, self._current_arm_q_rad
        if msg is None or K is None or q is None:
            log.info("wrist shape: 깊이 영상·camera_info·관절각 중 없는 것이 있음")
            return None
        if self.get_clock().now().nanoseconds - stamp > int(1.0e9):
            log.info("wrist shape: 깊이 영상이 1 초 넘게 오래됨")
            return None
        if msg.encoding == "32FC1":
            depth = np.frombuffer(bytes(msg.data), dtype=np.float32)
        elif msg.encoding == "16UC1":
            depth = np.frombuffer(bytes(msg.data), dtype=np.uint16).astype(np.float32) * 1e-3
        else:
            log.info(f"wrist shape: 모르는 깊이 형식 {msg.encoding}")
            return None
        depth = depth.reshape(msg.height, msg.width)[::2, ::2]
        v, u = np.mgrid[0:msg.height:2, 0:msg.width:2]
        ok = np.isfinite(depth) & (depth > 0.05) & (depth < 1.2)
        z = depth[ok].astype(np.float64)
        pts = np.stack([(u[ok] - K[0, 2]) * z / K[0, 0], (v[ok] - K[1, 2]) * z / K[1, 1], z], axis=1)
        mount = np.array([float(x) for x in self.get_parameter("wrist_mount_xyz_link6").value])
        R_oc = np.array(list(self.get_parameter("wrist_R_optical_to_link6").value),
                        dtype=np.float64).reshape(3, 3)
        p6 = pts @ R_oc.T + mount
        palm = float(self.get_parameter("friction_palm_offset_m").value)
        tip = palm + float(self.get_parameter("friction_finger_length_m").value)
        half_gap = 0.5 * float(self.get_parameter("gripper_open_gap_m").value)
        own = (p6[:, 2] < palm + 0.01) | ((p6[:, 2] < tip + 0.03)
                                           & (np.abs(p6[:, 0]) > half_gap - 0.03))
        T = fk(q)
        P = p6[~own] @ T[:3, :3].T + T[:3, 3]
        d = P[:, :2] - expected[:2]
        r = np.hypot(d[:, 0], d[:, 1])
        # 땅: 목표에서 22~40 cm 떨어진 고리의 점들에 평면을 맞춘다 (비탈·차체 기울기 포함).
        ring = (r > 0.22) & (r < 0.40)
        if int(ring.sum()) < 80:
            log.info(f"wrist shape: 둘레의 땅이 거의 안 보임 ({int(ring.sum())}점)")
            return None
        A = np.c_[d[ring], np.ones(int(ring.sum()))]
        coef = np.linalg.lstsq(A, P[ring, 2], rcond=None)[0]
        for _ in range(2):                       # 돌·다른 광물은 빼고 다시 맞춘다
            res = P[ring, 2] - A @ coef
            keep = np.abs(res - np.median(res)) < 0.03
            if int(keep.sum()) < 40:
                break
            coef = np.linalg.lstsq(A[keep], P[ring, 2][keep], rcond=None)[0]
        h = P[:, 2] - (d @ coef[:2] + coef[2])
        obj = (h > 0.05) & (r < 0.30)
        if int(obj.sum()) < 150:
            log.info(f"wrist shape: 땅 위로 5 cm 넘게 솟은 점이 {int(obj.sum())}개뿐 — 물체 없음")
            return None
        # 1 cm 격자에서 이어진 덩어리 중 목표에 가장 가까운 큰 것 하나만.
        cell = 0.01
        n = int(round(0.60 / cell))
        ij = np.clip(((d[obj] + 0.30) / cell).astype(int), 0, n - 1)
        grid = np.zeros((n, n), dtype=bool)
        grid[ij[:, 0], ij[:, 1]] = True
        from scipy import ndimage
        grid = ndimage.binary_closing(grid, iterations=2)
        lab, n_lab = ndimage.label(grid)
        if n_lab == 0:
            return None
        sizes = ndimage.sum(grid, lab, index=range(1, n_lab + 1))
        cms = ndimage.center_of_mass(grid, lab, index=range(1, n_lab + 1))
        best, best_score = 0, None
        for k in range(n_lab):
            if sizes[k] < 60:                    # 60 cm² 미만은 잡음
                continue
            dist = math.hypot(cms[k][0] * cell - 0.30, cms[k][1] * cell - 0.30)
            if best_score is None or dist < best_score:
                best, best_score = k + 1, dist
        if best == 0:
            return None
        sel = lab[ij[:, 0], ij[:, 1]] == best
        xy = d[obj][sel]
        top_z = float(np.percentile(P[obj][sel][:, 2], 95))
        # 가장 꼭 맞는 직사각형: 1° 씩 돌려 가며 넓이가 가장 작은 방향.
        best_fit = None
        for deg in range(0, 90, 1):
            a = math.radians(deg)
            ca, sa = math.cos(a), math.sin(a)
            pu = xy[:, 0] * ca + xy[:, 1] * sa
            pv = -xy[:, 0] * sa + xy[:, 1] * ca
            u0, u1 = np.percentile(pu, [1, 99])
            v0, v1 = np.percentile(pv, [1, 99])
            area = (u1 - u0) * (v1 - v0)
            if best_fit is None or area < best_fit[0]:
                best_fit = (area, a, u0, u1, v0, v1)
        _, a, u0, u1, v0, v1 = best_fit
        cu, cv = (u0 + u1) / 2, (v0 + v1) / 2
        centre = expected[:2] + np.array([cu * math.cos(a) - cv * math.sin(a),
                                          cu * math.sin(a) + cv * math.cos(a)])
        size = (float(u1 - u0), float(v1 - v0))
        if not (0.06 <= min(size) and max(size) <= 0.40):
            log.info(f"wrist shape: 덩어리 크기 {size[0] * 100:.0f}x{size[1] * 100:.0f} cm — "
                     f"광물로 보기 어려움")
            return None
        return {"center": centre, "top_z": top_z, "size": size, "angle": a,
                "points": int(sel.sum())}

    def _execute_pick_contact(self, goal_handle):
        """광물 바로 위에서 수직으로 내려가 집는 pick. grasp_mode 에 따라 둘로 갈린다.

        contact (RG2 + 접촉 조건부 부착)
          HOME → ABOVE (손끝이 광물 중심 위, 공구 축 수직 아래)
               → 천천히 하강하면서 pickup 을 계속 보냄
               → 시뮬레이터가 "손끝이 광물에 닿음" 을 status 로 알리면 정지
               → ABOVE → HOME → CARGO → release → HOME
          최저 높이까지 내려가도 닿지 않으면 실패로 끝낸다 (수집으로 세지 않음).

        friction (넓은 집게 + 마찰 파지)
          HOME → ABOVE → 손목 카메라로 수평 위치와 윗면 높이를 잰다
               → 손가락을 벌린 채 palm 이 윗면 바로 위에 올 때까지 하강
               → 닫기 → 시뮬레이터가 "손가락 사이에 광물이 끼어 멈춤" 을 알리면
               → 들어올린 뒤 아직 쥐고 있는지 확인 → HOME → CARGO → 열기 → HOME
          닫았는데 아무것도 안 끼었거나, 들거나 옮기다 떨어뜨리면 실패.
        """
        req = goal_handle.request
        mineral_world = np.array(
            [req.target_x, req.target_y, req.target_z], dtype=np.float64)

        deadline_ns = self.get_clock().now().nanoseconds + int(2e9)
        while self.get_clock().now().nanoseconds < deadline_ns:
            with self._odom_lock:
                if self._rover_xyz is not None:
                    break
            time.sleep(0.05)
        waited = self._wait_base_still()
        with self._odom_lock:
            base_xyz = None if self._rover_xyz is None else self._rover_xyz.copy()
            base_R = None if self._rover_R is None else self._rover_R.copy()
        if base_xyz is None or base_R is None:
            self.get_logger().error("contact pick: no odom — abort")
            goal_handle.abort()
            return ExecuteArmTask.Result(success=False, message="no odom")

        # 광물 중심을 팔 받침대 좌표로 (roll/pitch 포함 — 경사에서 yaw 만 쓰면
        # 팔 길이 0.5 m 에 기울기 10° 면 9 cm 가 어긋난다).
        center = base_R.T @ (mineral_world - base_xyz)
        friction = str(self.get_parameter("grasp_mode").value) == "friction"
        if friction:
            # 여기서 tip 은 link_6 → palm 아랫면, above 는 "광물 중심에서 palm
            # 아랫면까지의 높이" 다. 시작은 손가락 끝이 윗면보다 5 cm 위.
            tip = float(self.get_parameter("friction_palm_offset_m").value)
            finger_len = float(self.get_parameter("friction_finger_length_m").value)
            clearance = float(self.get_parameter("friction_palm_clearance_m").value)
            top_above = float(self.get_parameter("friction_top_above_center_m").value)
            z_start = top_above + 0.05 + finger_len
            z_end = top_above + clearance
            speed = max(0.005, float(self.get_parameter("friction_descend_speed").value))
        else:
            tip = float(self.get_parameter("tool_tip_offset_m").value)
            z_start = float(self.get_parameter("contact_start_above_m").value)
            z_end = float(self.get_parameter("contact_end_above_m").value)
            speed = max(0.005, float(self.get_parameter("contact_descend_speed").value))

        # 공구 축(link_6 z)이 받침대 기준 수직 아래, 손가락 벌림 축은 광물 방향에
        # 수직 — HOME 자세를 광물 방위각만큼 돌린 자세.
        q_home = np.radians(HOME_DEG)
        R_home = fk(q_home)[:3, :3]
        tool_yaw = math.atan2(center[1], center[0])
        R_tool = _yaw_rot_z(tool_yaw) @ R_home

        def solve(above: float, seed: np.ndarray):
            target = center + np.array([0.0, 0.0, above + tip])
            return dls_ik(target, R_tool, seed, max_iters=200)

        q_above, ok_a, err_a = solve(z_start, q_home)
        q_low, ok_l, err_l = solve(z_end, q_above)
        self.get_logger().info(
            f"contact pick: mineral world=({mineral_world[0]:.2f},"
            f"{mineral_world[1]:.2f},{mineral_world[2]:.2f}) → 받침대 기준 "
            f"({center[0]:.3f},{center[1]:.3f},{center[2]:.3f}) "
            f"IK above={ok_a}({err_a:.4f}) low={ok_l}({err_l:.4f}) "
            f"[정지 대기 {waited:.1f} s]")
        if not (ok_a and ok_l):
            goal_handle.abort()
            return ExecuteArmTask.Result(
                success=False,
                message=f"out of reach (above={ok_a} low={ok_l}, "
                        f"x={center[0]:.2f} y={center[1]:.2f} z={center[2]:.2f})")

        cur_deg = list(HOME_DEG)
        self._goto(cur_deg, HOME_DEG, goal_handle, "HOME_PRE", 0.0, 0.05)
        deg_above = list(np.degrees(q_above))
        self._goto(cur_deg, deg_above, goal_handle, "ABOVE", 0.05, 0.2)
        cur_deg = deg_above

        # ── 손목 카메라로 수평 위치 보정 (최대 2 회: 고친 뒤 한 번 더 본다) ──
        def base_drift() -> str:
            """goal 을 받은 뒤 받침대가 얼마나 움직였는지 (진단용)."""
            with self._odom_lock:
                p_now, R_now = self._rover_xyz.copy(), self._rover_R.copy()
            d = base_R.T @ (p_now - base_xyz)
            ang = math.degrees(math.acos(float(np.clip(
                (np.trace(base_R.T @ R_now) - 1.0) * 0.5, -1.0, 1.0))))
            moved = base_R.T @ (mineral_world - base_xyz) - R_now.T @ (mineral_world - p_now)
            return (f"받침대 이동 ({d[0] * 100:+.1f},{d[1] * 100:+.1f},{d[2] * 100:+.1f}) cm "
                    f"회전 {ang:.1f}° → 목표가 받침대 기준으로 "
                    f"({-moved[0] * 100:+.1f},{-moved[1] * 100:+.1f},{-moved[2] * 100:+.1f}) cm 이동")

        self.get_logger().info(f"contact pick: ABOVE 도착 — {base_drift()}")

        def retarget(new_center: np.ndarray, tag: str) -> bool:
            """목표 중심을 바꾸고 ABOVE 자세를 다시 풀어 옮긴다. 안 닿으면 취소."""
            nonlocal center, q_above, q_low, deg_above, cur_deg
            saved = center
            center = new_center
            q_try, ok1, _ = solve(z_start, q_above)
            q_try_low, ok2, _ = solve(z_end, q_try)
            if not (ok1 and ok2):
                center = saved
                self.get_logger().warn(f"contact pick: {tag} 위치는 팔이 닿지 않음 — 취소")
                return False
            q_above, q_low = q_try, q_try_low
            deg_above = list(np.degrees(q_above))
            self._goto(cur_deg, deg_above, goal_handle, "ALIGN", 0.2, 0.25, duration=1.0)
            cur_deg = deg_above
            return True

        # 팔을 뻗는 동안 받침대가 밀린 만큼(실측 2~6 cm) 목표를 다시 계산한다.
        with self._odom_lock:
            base_xyz, base_R = self._rover_xyz.copy(), self._rover_R.copy()
        fresh = base_R.T @ (mineral_world - base_xyz)
        if float(np.linalg.norm(fresh - center)) > 0.005:
            retarget(fresh, "odom 재계산")
        top_seen = None          # 손목 카메라가 잰 윗면 높이 (받침대 z)
        use_shape = friction and bool(self.get_parameter("wrist_shape_measure").value)
        gap = float(self.get_parameter("gripper_open_gap_m").value)
        if bool(self.get_parameter("use_wrist_correction").value):
            shape_mode = False       # YOLO 대신 깊이 영상으로 재는 중
            for attempt in range(3):
                if attempt == 2 and not shape_mode:
                    break                                  # YOLO 보정은 두 번까지
                measured = None if shape_mode else self._wrist_measure(center)
                shape = None
                if measured is None and use_shape:
                    # YOLO 가 위에서 본 모습을 못 알아봄 → 깊이 영상으로 덩어리를 직접 잰다.
                    shape = self._wrist_measure_shape(center)
                    if shape is not None and float(np.linalg.norm(
                            shape["center"] - center[:2])) > 2.0 * float(
                            self.get_parameter("wrist_correction_gate_m").value):
                        self.get_logger().info(
                            "wrist shape: 덩어리가 예상 위치에서 "
                            f"{np.linalg.norm(shape['center'] - center[:2]) * 100:.0f} cm — 버림")
                        shape = None
                if shape is not None:
                    shape_mode = True
                    top_seen = float(shape["top_z"])
                    shift = shape["center"] - center[:2]
                    # 집게를 닫는 방향(link_6 x)을 직사각형의 가까운 변 방향에 맞춘다.
                    close_dir = math.atan2(R_tool[1, 0], R_tool[0, 0])
                    turn = (shape["angle"] - close_dir + math.pi / 4) % (math.pi / 2) - math.pi / 4
                    a_len, b_len = shape["size"]
                    k = int(round((close_dir + turn - shape["angle"]) / (math.pi / 2))) % 2
                    width = a_len if k == 0 else b_len       # 닫는 방향으로 본 폭
                    other = b_len if k == 0 else a_len
                    if width > gap - 0.03 and other < width - 0.02:
                        turn += math.pi / 2 if turn < 0 else -math.pi / 2
                        width = other
                    if not bool(self.get_parameter("wrist_align_yaw").value):
                        turn = 0.0
                    self.get_logger().info(
                        f"contact pick: 깊이 영상 측정 #{attempt + 1} — 덩어리 "
                        f"{a_len * 100:.0f}x{b_len * 100:.0f} cm ({shape['points']}점), "
                        f"dx={shift[0] * 100:+.1f} cm dy={shift[1] * 100:+.1f} cm, "
                        f"윗면은 인식 중심 +{(top_seen - center[2]) * 100:.1f} cm, "
                        f"집게 {math.degrees(turn):+.0f}° 돌림 (닫는 방향 폭 "
                        f"{width * 100:.0f} cm / 벌림 {gap * 100:.0f} cm)")
                    # 방향 추정은 ±6° 쯤 흔들린다 (실측) — 그 안이면 다시 돌리지 않는다.
                    if abs(turn) < math.radians(8.0):
                        turn = 0.0
                    if float(np.linalg.norm(shift)) < 0.005 and turn == 0.0:
                        break
                    saved_R, saved_yaw = R_tool, tool_yaw
                    tool_yaw += turn
                    R_tool = _yaw_rot_z(tool_yaw) @ R_home
                    trial = center.copy()
                    trial[:2] = shape["center"]
                    if not retarget(trial, "깊이 영상 보정"):
                        R_tool, tool_yaw = saved_R, saved_yaw     # 돌린 자세는 안 닿음
                        if not retarget(trial, "깊이 영상 보정(돌리지 않고)"):
                            break
                    continue
                if measured is None:
                    # 보정 범위(gate) 밖이라도 근처에 광물이 보였으면 있는 것이다
                    # (가스처럼 큰 것은 중심 추정이 15 cm 넘게 어긋난다) → 그대로 진행.
                    near = getattr(self, "_wrist_nearest_m", None)
                    nothing = near is None or near > float(
                        self.get_parameter("wrist_presence_radius_m").value)
                    cls = str(goal_handle.request.metadata or "")
                    trusted = cls in list(self.get_parameter("wrist_presence_classes").value)
                    if attempt == 0 and nothing and trusted and bool(
                            self.get_parameter("abort_if_wrist_sees_nothing").value):
                        self.get_logger().warn(
                            "contact pick: 손목 카메라에 광물이 안 보임 — 그 자리에 광물이 "
                            "없는 것으로 보고 집지 않음")
                        self._goto(cur_deg, HOME_DEG, goal_handle, "HOME_ABORT", 0.6, 1.0)
                        self._publish_joint_state(HOME_DEG)
                        goal_handle.abort()
                        return ExecuteArmTask.Result(
                            success=False, message="no mineral at the target (wrist camera)")
                    self.get_logger().warn(
                        "contact pick: 손목 카메라에 광물이 안 잡힘 — 로버 카메라 "
                        "추정값 그대로 진행")
                    break
                seen, surface = measured
                top_seen = float(surface[2])
                shift = seen[:2] - center[:2]
                self.get_logger().info(
                    f"contact pick: 손목 카메라 보정 #{attempt + 1} "
                    f"dx={shift[0] * 100:+.1f} cm dy={shift[1] * 100:+.1f} cm "
                    f"(중심 높이 차 {(seen[2] - center[2]) * 100:+.1f} cm 는 참고만, "
                    f"윗면은 인식 중심 +{(top_seen - center[2]) * 100:.1f} cm)")
                if float(np.linalg.norm(shift)) < 0.005:
                    break
                trial = center.copy()
                trial[:2] = seen[:2]
                if not retarget(trial, "손목 카메라 보정"):
                    break

        if friction:
            return self._finish_pick_friction(
                goal_handle, solve, center, mineral_world, q_above, deg_above,
                z_start, top_seen)

        # ── 하강: 직선(Cartesian)으로 조금씩, 매 스텝 pickup 발행 + status 확인 ──
        publish_hz = max(5.0, float(self.get_parameter("publish_hz").value))
        dt = 1.0 / publish_hz
        n_steps = max(2, int(round((z_start - z_end) / speed * publish_hz)))
        pickup = Twist()
        pickup.linear.x, pickup.linear.y, pickup.linear.z = (
            float(mineral_world[0]), float(mineral_world[1]), float(mineral_world[2]))
        pickup.angular.x = 1.0
        t_start = time.monotonic()
        q = q_above
        held = ""
        reached = z_start
        for k in range(1, n_steps + 1):
            above = z_start + (z_end - z_start) * k / n_steps
            q_next, ok, _ = solve(above, q)
            if not ok:
                self.get_logger().warn(f"contact pick: IK 실패 (손끝 +{above:.3f} m) — 하강 중단")
                break
            q = q_next
            reached = above
            cur_deg = list(np.degrees(q))
            self._publish_joint_state(cur_deg)
            self.grasp_pub.publish(pickup)
            fb = ExecuteArmTask.Feedback()
            fb.state = "DESCEND_CONTACT"
            fb.progress = float(0.25 + 0.2 * k / n_steps)
            fb.message = f"tip +{above:.3f} m"
            goal_handle.publish_feedback(fb)
            time.sleep(dt)
            held = self._held_now(t_start)
            if held:
                break
        if not held:
            # 팔이 명령을 따라오는 지연만큼 조금 더 기다려 본다.
            t_wait = time.monotonic() + 1.0
            while time.monotonic() < t_wait and not held:
                self.grasp_pub.publish(pickup)
                time.sleep(dt)
                held = self._held_now(t_start)
        if not held:
            self.get_logger().warn(
                f"contact pick: 손끝을 +{reached:.3f} m 까지 내렸지만 접촉 없음 — 실패")
            self._goto(cur_deg, deg_above, goal_handle, "RETREAT", 0.45, 0.6)
            self._goto(deg_above, HOME_DEG, goal_handle, "HOME_ABORT", 0.6, 1.0)
            self._publish_joint_state(HOME_DEG)
            goal_handle.abort()
            return ExecuteArmTask.Result(
                success=False, message="no contact with mineral")
        self.get_logger().info(
            f"contact pick: 접촉 — {held} (손끝 높이 +{reached:.3f} m, "
            f"인식 중심 기준)")

        self._goto(cur_deg, deg_above, goal_handle, "LIFT", 0.45, 0.6)
        self._goto(deg_above, HOME_DEG, goal_handle, "HOME_MID", 0.6, 0.7)
        cur_deg = list(HOME_DEG)
        self._goto(cur_deg, CARGO_DEG, goal_handle, "CARGO_SWING", 0.7, 0.85)
        cur_deg = list(CARGO_DEG)
        settle = float(self.get_parameter("cargo_settle_sec").value)
        if settle > 0.0:
            time.sleep(settle)
        self._publish_grasp("release")
        fb = ExecuteArmTask.Feedback()
        fb.state = "RELEASE"
        fb.progress = 0.9
        fb.message = "release published"
        goal_handle.publish_feedback(fb)
        self._goto(cur_deg, HOME_DEG, goal_handle, "HOME_POST", 0.9, 1.0)
        self._publish_joint_state(HOME_DEG)
        goal_handle.succeed()
        self.get_logger().info("pick_mineral (contact) done")
        return ExecuteArmTask.Result(
            success=True, message=f"pick_mineral completed (contact, {held})")

    def _finish_pick_friction(self, goal_handle, solve, center, mineral_world,
                              q_above, deg_above, z_start, top_seen):
        """ABOVE 에 정렬된 뒤의 마찰 파지 절차 (하강 → 닫기 → 들기 → 적재)."""
        log = self.get_logger()
        clearance = float(self.get_parameter("friction_palm_clearance_m").value)
        finger_len = float(self.get_parameter("friction_finger_length_m").value)
        top_above = float(self.get_parameter("friction_top_above_center_m").value)
        if top_seen is not None:
            # 윗면을 직접 쟀으면 그 값을 쓴다 (인식 중심의 높이 오차와 무관해짐).
            top_above = float(np.clip(top_seen - center[2], 0.02, 0.20))
        z_end = top_above + clearance
        speed = max(0.005, float(self.get_parameter("friction_descend_speed").value))
        publish_hz = max(5.0, float(self.get_parameter("publish_hz").value))
        dt = 1.0 / publish_hz
        log.info(
            f"friction pick: 윗면 = 인식 중심 +{top_above * 100:.1f} cm "
            f"({'손목 카메라 실측' if top_seen is not None else '기본값'}) → palm 을 "
            f"+{z_end * 100:.1f} cm 까지 내림 (손가락 끝은 중심 "
            f"{(z_end - finger_len) * 100:+.1f} cm)")

        pickup = Twist()
        pickup.linear.x, pickup.linear.y, pickup.linear.z = (
            float(mineral_world[0]), float(mineral_world[1]), float(mineral_world[2]))
        pickup.angular.x = 1.0
        release = Twist()
        release.angular.x = -1.0

        def give_up(cur, message: str):
            """집게를 열고 HOME 으로 돌아가 실패로 끝낸다."""
            for _ in range(6):
                self.grasp_pub.publish(release)
                time.sleep(0.1)
            if cur is not deg_above:
                self._goto(cur, deg_above, goal_handle, "RETREAT", 0.45, 0.6, duration=2.0)
            self._goto(deg_above, HOME_DEG, goal_handle, "HOME_ABORT", 0.6, 1.0)
            self._publish_joint_state(HOME_DEG)
            goal_handle.abort()
            return ExecuteArmTask.Result(success=False, message=message)

        def still_held(timeout: float = 2.0) -> str:
            """지금도 쥐고 있는지. 팔이 막 멈춘 직후에는 손가락 간격이 잠깐
            흔들려 '쥠' 이 한두 번 빠질 수 있으므로 timeout 동안 기다려 본다."""
            t0 = time.monotonic()
            while time.monotonic() - t0 < timeout:
                time.sleep(0.1)
                name = self._held_now(t0)
                if name:
                    return name
            return ""

        # ── 하강 (직선, 손가락 벌린 채) ──
        n_steps = max(2, int(round((z_start - z_end) / speed * publish_hz)))
        q = q_above
        cur_deg = deg_above
        for k in range(1, n_steps + 1):
            above = z_start + (z_end - z_start) * k / n_steps
            q_next, ok, _ = solve(above, q)
            if not ok:
                log.warn(f"friction pick: IK 실패 (palm +{above:.3f} m) — 실패")
                return give_up(cur_deg, "out of reach during descent")
            q = q_next
            cur_deg = list(np.degrees(q))
            self._publish_joint_state(cur_deg)
            fb = ExecuteArmTask.Feedback()
            fb.state = "DESCEND_OPEN"
            fb.progress = float(0.25 + 0.15 * k / n_steps)
            fb.message = f"palm +{above:.3f} m"
            goal_handle.publish_feedback(fb)
            time.sleep(dt)
        time.sleep(0.8)          # 팔이 명령을 따라잡을 시간 (실측 지연 0.3~0.4 s)

        # ── 닫기: 손가락이 광물에 막혀 멈출 때까지 ──
        t_close = time.monotonic()
        deadline = t_close + float(self.get_parameter("friction_close_timeout_sec").value)
        held = ""
        while time.monotonic() < deadline and not held:
            self.grasp_pub.publish(pickup)
            time.sleep(0.1)
            held = self._held_now(t_close)
        if not held:
            log.warn("friction pick: 집게를 닫았지만 손가락 사이에 광물이 없음 — 실패")
            return give_up(cur_deg, "nothing between the fingers")
        log.info(f"friction pick: 쥠 — {held} ({time.monotonic() - t_close:.1f} s)")
        fb = ExecuteArmTask.Feedback()
        fb.state = "GRASP_CLOSE"
        fb.progress = 0.45
        fb.message = held
        goal_handle.publish_feedback(fb)
        time.sleep(0.5)

        # ── 들기: 마찰만으로 버티는지 확인 ──
        self._goto(cur_deg, deg_above, goal_handle, "LIFT", 0.45, 0.6, duration=2.5)
        cur_deg = deg_above
        if not still_held():
            log.warn("friction pick: 들어올리다 놓침 — 실패")
            return give_up(cur_deg, "mineral slipped during lift")
        log.info("friction pick: 들어올림 — 마찰로 유지 중")

        self._goto(cur_deg, HOME_DEG, goal_handle, "HOME_MID", 0.6, 0.7)
        cur_deg = list(HOME_DEG)
        self._goto(cur_deg, CARGO_DEG, goal_handle, "CARGO_SWING", 0.7, 0.85)
        cur_deg = list(CARGO_DEG)
        settle = float(self.get_parameter("cargo_settle_sec").value)
        if settle > 0.0:
            time.sleep(settle)
        carried = still_held()
        # 열기 = 시뮬레이터가 그 순간 손가락 사이에 있는 광물을 적재 처리.
        self._publish_grasp("release")
        fb = ExecuteArmTask.Feedback()
        fb.state = "RELEASE"
        fb.progress = 0.9
        fb.message = "release published"
        goal_handle.publish_feedback(fb)
        self._goto(cur_deg, HOME_DEG, goal_handle, "HOME_POST", 0.9, 1.0)
        self._publish_joint_state(HOME_DEG)
        if not carried:
            log.warn("friction pick: 적재함으로 옮기다 떨어뜨림 — 실패")
            goal_handle.abort()
            return ExecuteArmTask.Result(
                success=False, message="mineral dropped during transport")
        goal_handle.succeed()
        log.info("pick_mineral (friction) done")
        return ExecuteArmTask.Result(
            success=True, message=f"pick_mineral completed (friction, {carried})")

    def _execute_pick_ik(self, goal_handle):
        """pick_mineral with IK + wrist-cam servo (Phase 3b-3, off by default).

        Tune wrist_mount_xyz_link6 / wrist_R_optical_to_link6 before enabling.
        """
        # 1) Ensure HOME first (uses current joint state if available)
        cur_q_rad = (self._current_arm_q_rad
                     if self._current_arm_q_rad is not None
                     else np.radians(HOME_DEG))
        cur_deg = list(np.degrees(cur_q_rad))
        self._goto(cur_deg, HOME_DEG, goal_handle, "HOME_PRE", 0.0, 0.1)
        cur_deg = list(HOME_DEG)

        # 2) Wait for wrist detection
        wait_s = float(self.get_parameter("wrist_detection_wait_sec").value)
        min_conf = float(self.get_parameter("min_confidence").value)
        det = self._best_wrist_detection(wait_s, min_conf)
        if det is None:
            self.get_logger().warn(
                f"No wrist detection within {wait_s}s — falling back to scripted PICK_TRAJ")
            return self._execute_scripted(goal_handle, PICK_TRAJ_DEG, "pick_mineral_fallback")

        # 3) Convert wrist optical XYZ -> link_6 -> arm base; subtract hover
        xyz_optical = np.array([det.world_position.x, det.world_position.y,
                                det.world_position.z], dtype=np.float64)
        mount_xyz = tuple(float(v) for v in self.get_parameter("wrist_mount_xyz_link6").value)
        R_flat = list(self.get_parameter("wrist_R_optical_to_link6").value)
        R_oc = np.array(R_flat, dtype=np.float64).reshape(3, 3)
        p_link6 = _wrist_optical_to_link6(xyz_optical, mount_xyz, R_oc)
        # Use actual current arm joints (post-HOME) for FK transform base.
        q_cur_rad = (self._current_arm_q_rad
                     if self._current_arm_q_rad is not None
                     else np.radians(HOME_DEG))
        p_base_mineral = _link6_to_base(p_link6, q_cur_rad)

        hover = float(self.get_parameter("hover_above_mineral_m").value)
        target_pos = p_base_mineral + np.array([0.0, 0.0, hover])
        self.get_logger().info(
            f"wrist det class={det.class_name} optical=({xyz_optical[0]:.3f},"
            f"{xyz_optical[1]:.3f},{xyz_optical[2]:.3f}) -> base_mineral="
            f"({p_base_mineral[0]:.3f},{p_base_mineral[1]:.3f},{p_base_mineral[2]:.3f}) "
            f"-> target_hover=({target_pos[0]:.3f},{target_pos[1]:.3f},{target_pos[2]:.3f})")

        # 4) DLS-IK
        use_orient = bool(self.get_parameter("ik_use_orientation").value)
        target_R = fk(q_cur_rad)[:3, :3] if use_orient else None
        q_sol, ok, err = dls_ik(target_pos, target_R, q_cur_rad,
                                position_only=not use_orient)
        if not ok:
            self.get_logger().warn(
                f"DLS-IK did not converge (err={err:.4f}) — falling back to scripted PICK_TRAJ")
            return self._execute_scripted(goal_handle, PICK_TRAJ_DEG, "pick_mineral_fallback")

        target_deg = list(np.degrees(q_sol))
        self.get_logger().info(
            f"IK ok err={err:.5f} target_joints_deg = "
            f"[{','.join(f'{v:.1f}' for v in target_deg)}]")

        # 5) Interpolate to IK target (APPROACH_DESCEND)
        self._goto(cur_deg, target_deg, goal_handle, "APPROACH_DESCEND", 0.1, 0.5)
        cur_deg = target_deg

        # 6) GRASP (no real gripper yet — placeholder)
        fb = ExecuteArmTask.Feedback()
        fb.state = "GRASP_CLOSE"
        fb.progress = 0.55
        fb.message = "gripper close (placeholder)"
        goal_handle.publish_feedback(fb)
        time.sleep(0.5)

        # 7) LIFT (return to HOME briefly)
        self._goto(cur_deg, HOME_DEG, goal_handle, "LIFT", 0.6, 0.75)
        cur_deg = list(HOME_DEG)

        # 8) PLACE_BASKET scripted
        for i in range(len(PLACE_BASKET_TRAJ_DEG) - 1):
            self._goto(cur_deg, PLACE_BASKET_TRAJ_DEG[i + 1], goal_handle,
                       f"PLACE_BASKET_{i + 1}",
                       0.75 + i * 0.05, 0.75 + (i + 1) * 0.05)
            cur_deg = list(PLACE_BASKET_TRAJ_DEG[i + 1])

        # 9) HOME guarantee
        self._publish_joint_state(HOME_DEG)
        goal_handle.succeed()
        self.get_logger().info("pick_mineral done")
        return ExecuteArmTask.Result(success=True, message="pick_mineral completed")

    def _execute_scripted(self, goal_handle, trajectory: List[List[float]], tag: str):
        publish_hz = max(5.0, float(self.get_parameter("publish_hz").value))
        step_dur = max(0.2, float(self.get_parameter("step_duration_sec").value))
        n_interp = max(2, int(round(publish_hz * step_dur)))
        dt = 1.0 / publish_hz
        total_steps = max(1, len(trajectory) - 1)

        cur = list(trajectory[0])
        for seg_idx in range(total_steps):
            target = trajectory[seg_idx + 1]
            for k in range(1, n_interp + 1):
                t = k / n_interp
                interp = [cur[i] + (target[i] - cur[i]) * t for i in range(6)]
                self._publish_joint_state(interp)
                fb = ExecuteArmTask.Feedback()
                fb.state = tag
                fb.progress = float((seg_idx + t) / total_steps)
                fb.message = f"seg {seg_idx + 1}/{total_steps}"
                goal_handle.publish_feedback(fb)
                time.sleep(dt)
            cur = list(target)

        self._publish_joint_state(HOME_DEG)
        goal_handle.succeed()
        self.get_logger().info(f"Arm command {tag} done")
        return ExecuteArmTask.Result(success=True, message=f"{tag} completed")

    def _goto(self, cur_deg: List[float], target_deg: List[float], goal_handle,
              tag: str, progress_start: float, progress_end: float,
              duration: Optional[float] = None) -> None:
        publish_hz = max(5.0, float(self.get_parameter("publish_hz").value))
        step_dur = max(0.2, float(self.get_parameter("step_duration_sec").value)
                       if duration is None else float(duration))
        n_interp = max(2, int(round(publish_hz * step_dur)))
        dt = 1.0 / publish_hz
        for k in range(1, n_interp + 1):
            t = k / n_interp
            interp = [cur_deg[i] + (target_deg[i] - cur_deg[i]) * t for i in range(6)]
            self._publish_joint_state(interp)
            fb = ExecuteArmTask.Feedback()
            fb.state = tag
            fb.progress = float(progress_start + (progress_end - progress_start) * t)
            fb.message = tag
            goal_handle.publish_feedback(fb)
            time.sleep(dt)

    def _publish_joint_state(self, positions_deg: List[float]) -> None:
        goal = self._active_goal
        if goal is not None and goal.is_cancel_requested:
            raise _ArmCanceled()             # 다음 목표를 보내지 않는다 = 그 자리에 선다
        self._last_cmd_deg = [float(v) for v in positions_deg]
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = [math.radians(v) for v in positions_deg]
        self.joint_pub.publish(msg)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ArmExecutorNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
