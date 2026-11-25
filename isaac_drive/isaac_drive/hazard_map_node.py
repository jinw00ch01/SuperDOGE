"""hazard_map_node — 로버 깊이 카메라로 주행 중에 위험 지도를 만든다.

미리 만든 obstacle_grid.npy 를 대신한다. 계산은 navigation/hazard_map.py 에 있고
이 노드는 ROS 입·출력만 맡는다.

  입력
    · <depth_topic>   sensor_msgs/Image (32FC1 z-depth m, 또는 16UC1 mm)
    · <info_topic>    sensor_msgs/CameraInfo
    · <odom_topic>    nav_msgs/Odometry — 팔 받침대(base_link)의 world 자세
    · <joint_topic>   sensor_msgs/JointState — 팔이 HOME 이 아닐 때는 영상을 쓰지
                      않는다 (팔과 집게가 카메라 앞에 들어와 장애물로 찍힘)
    · <detections_topic> isaac_interfaces/DetectionArray — 광물은 장애물이 아니라
                      집을 대상이므로 그 주변은 위험에서 뺀다
  출력
    · <map_topic>     nav_msgs/OccupancyGrid — -1 안 봄 / 0~49 안전 / 100 위험
                      (부풀리지 않은 값. 로버 반경만큼 부풀리는 것은 받는 쪽 몫)
                      안전 칸의 값은 그 칸의 경사(도, 49 에서 자름). 위험 판정은 받는 쪽이
                      50 초과로 보므로 예전 소비자는 그대로 동작한다. 경사 비용 계획에 쓴다.

지도의 범위(크기)는 운용 구역 크기로, terrain meta.json 의 size_m 에서만 읽는다.
"""
from __future__ import annotations

import array
import json
import math
import os
from collections import deque

import numpy as np
import rclpy
from isaac_interfaces.msg import DetectionArray
from nav_msgs.msg import OccupancyGrid, Odometry
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy)
from sensor_msgs.msg import CameraInfo, Image, JointState

from isaac_drive.navigation.hazard_map import HazardMap

SENSOR_QOS = QoSProfile(reliability=ReliabilityPolicy.BEST_EFFORT,
                        durability=DurabilityPolicy.VOLATILE,
                        history=HistoryPolicy.KEEP_LAST, depth=5)
MAP_QOS = QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                     durability=DurabilityPolicy.TRANSIENT_LOCAL,
                     history=HistoryPolicy.KEEP_LAST, depth=1)

ARM_JOINTS = ["joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6"]
ARM_HOME_RAD = np.radians([0.0, 0.0, 90.0, 0.0, 90.0, 0.0])


def _stamp_sec(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class HazardMapNode(Node):
    def __init__(self) -> None:
        super().__init__("hazard_map_node")
        self.declare_parameter("terrain_dir", "")
        self.declare_parameter("map_size_m", [50.0, 50.0])      # terrain_dir 없을 때
        self.declare_parameter("cell_size", 0.1)
        self.declare_parameter("depth_topic", "/camera/rover/depth")
        self.declare_parameter("info_topic", "/camera/rover/camera_info")
        self.declare_parameter("odom_topic", "/ground_truth/odom")
        self.declare_parameter("joint_topic", "/joint_states_raw")
        self.declare_parameter("detections_topic", "/perception/detections")
        self.declare_parameter("map_topic", "/hazard_map")
        self.declare_parameter("frame_id", "map")
        self.declare_parameter("camera_offset_xyz", [0.147, 0.0, -0.312])
        # 깊이 영상 stamp 가 실제 찍힌 시각보다 늦는 정도. 녹화 자료에서 정답
        # 지도와의 일치가 가장 좋았던 값 (0.02~0.03 s 에서 정밀도 99.8 %).
        self.declare_parameter("image_latency_sec", 0.03)
        self.declare_parameter("integrate_hz", 5.0)
        self.declare_parameter("publish_hz", 2.0)
        self.declare_parameter("pixel_stride", 2)
        self.declare_parameter("min_range_m", 0.5)
        self.declare_parameter("max_range_m", 8.0)
        self.declare_parameter("slope_deg", 25.0)
        self.declare_parameter("step_m", 0.12)
        self.declare_parameter("rough_m", 0.04)
        self.declare_parameter("arm_home_tolerance_deg", 5.0)
        self.declare_parameter("mineral_clear_radius_m", 0.4)
        # 막힘 감지: 전진(후진) 명령을 stall_sec 동안 받았는데 stall_move_m 도 못 갔으면
        # 카메라가 못 본 무언가에 막힌 것으로 보고, 진행 방향 앞에 위험 원을 찍는다.
        # 접촉 센서가 없는 로버의 "범퍼" 대용 (바퀴 명령과 실제 이동의 비교).
        self.declare_parameter("cmd_topic", "/cmd_vel")
        # 운영자가 지도를 비울 때 (HUD 의 MAP CLR). 잘못 찍힌 위험 칸이 길을 막을 때 쓴다.
        self.declare_parameter("reset_topic", "/hazard_map/reset")
        self.declare_parameter("clear_marks_topic", "/hazard_map/clear_marks")
        self.declare_parameter("mark_topic", "/hazard_map/mark")
        self.declare_parameter("stall_sec", 4.0)
        self.declare_parameter("stall_cmd_mps", 0.15)    # 구간 평균 명령이 이보다 작으면 판단 안 함
        self.declare_parameter("stall_move_m", 0.15)
        self.declare_parameter("stall_mark_ahead_m", 0.9)
        self.declare_parameter("stall_mark_radius_m", 0.45)
        self.declare_parameter("mineral_min_confidence", 0.5)
        # 먼 검출은 위치가 부정확해서 엉뚱한 곳(바위)을 지울 수 있다.
        self.declare_parameter("mineral_max_range_m", 4.0)

        size = [float(v) for v in self.get_parameter("map_size_m").value]
        terrain_dir = str(self.get_parameter("terrain_dir").value)
        meta_path = os.path.join(terrain_dir, "meta.json") if terrain_dir else ""
        if meta_path and os.path.isfile(meta_path):
            with open(meta_path) as f:
                size = [float(v) for v in json.load(f)["size_m"]]
        self.hm = HazardMap(
            map_size=size,
            cell_size=float(self.get_parameter("cell_size").value),
            cam_offset=[float(v) for v in self.get_parameter("camera_offset_xyz").value],
            min_range=float(self.get_parameter("min_range_m").value),
            max_range=float(self.get_parameter("max_range_m").value),
            slope_deg=float(self.get_parameter("slope_deg").value),
            step_m=float(self.get_parameter("step_m").value),
            rough_m=float(self.get_parameter("rough_m").value))
        self.latency = float(self.get_parameter("image_latency_sec").value)
        self.stride = max(1, int(self.get_parameter("pixel_stride").value))
        self.min_period = 1.0 / max(0.5, float(self.get_parameter("integrate_hz").value))
        self.arm_tol = math.radians(float(self.get_parameter("arm_home_tolerance_deg").value))
        self.clear_r = float(self.get_parameter("mineral_clear_radius_m").value)
        self.min_conf = float(self.get_parameter("mineral_min_confidence").value)
        self.mineral_range = float(self.get_parameter("mineral_max_range_m").value)

        self.K = None
        self._odom = deque(maxlen=600)          # (t, p, q)
        self._arm_dev = None                    # HOME 에서 가장 많이 벗어난 관절 (rad)
        self._last_t = -1e9
        self._minerals: list = []               # [(x, y)]
        self.n_used = self.n_skip_arm = self.n_skip_pose = 0

        self.create_subscription(CameraInfo, str(self.get_parameter("info_topic").value),
                                 self._on_info, SENSOR_QOS)
        self.create_subscription(Odometry, str(self.get_parameter("odom_topic").value),
                                 self._on_odom, SENSOR_QOS)
        self.create_subscription(JointState, str(self.get_parameter("joint_topic").value),
                                 self._on_joints, SENSOR_QOS)
        self.create_subscription(Image, str(self.get_parameter("depth_topic").value),
                                 self._on_depth, SENSOR_QOS)
        self.create_subscription(DetectionArray,
                                 str(self.get_parameter("detections_topic").value),
                                 self._on_detections, 10)
        self.pub = self.create_publisher(
            OccupancyGrid, str(self.get_parameter("map_topic").value), MAP_QOS)
        self.create_timer(1.0 / max(0.2, float(self.get_parameter("publish_hz").value)),
                          self._publish)
        self.create_timer(10.0, self._status)
        from geometry_msgs.msg import Twist
        self._bumps: list = []                  # [(x, y)] 막혀서 찍은 위험 원
        self._cmd_lin = 0.0
        self._stall_ref = None                  # (시각, 위치, 방향 부호)
        self.create_subscription(Twist, str(self.get_parameter("cmd_topic").value),
                                 self._on_cmd, 10)
        from std_msgs.msg import Empty
        self.create_subscription(Empty, str(self.get_parameter("reset_topic").value),
                                 self._on_reset, 10)
        from geometry_msgs.msg import PointStamped
        self.create_subscription(Empty, str(self.get_parameter("clear_marks_topic").value),
                                 self._on_clear_marks, 10)
        self.create_subscription(PointStamped, str(self.get_parameter("mark_topic").value),
                                 self._on_mark, 10)
        self.get_logger().info(
            f"hazard_map_node ready — {size[0]:.0f}×{size[1]:.0f} m @ {self.hm.cell:.2f} m, "
            f"거리 {self.hm.min_range}~{self.hm.max_range} m, 경사 한계 "
            f"{float(self.get_parameter('slope_deg').value):.0f}°, 단차 {self.hm.step_m} m "
            f"— 미리 만든 지도 없음")

    # ── 입력 ────────────────────────────────────────────────────────────
    def _on_info(self, msg: CameraInfo) -> None:
        if self.K is None and msg.k[0] > 0:
            self.K = (float(msg.k[0]), float(msg.k[4]), float(msg.k[2]), float(msg.k[5]))

    def _on_cmd(self, msg) -> None:
        self._cmd_lin = float(msg.linear.x)

    def _on_reset(self, _msg) -> None:
        """지금까지 쌓은 판정을 버린다. 카메라가 다시 보는 대로 새로 채워진다."""
        cells = int(self.hm.obstacle_mask().sum())
        self.hm.evidence[:] = 0
        self.hm.seen[:] = 0
        self.hm.grade[:] = np.nan
        bumps = len(self._bumps)
        self._bumps.clear()
        self._stall_ref = None
        self.get_logger().warn(
            f"operator reset: 위험 칸 {cells}개, 막힘 표시 {bumps}곳을 지움 — 지도를 다시 만든다")
        self._publish(force=True)

    def _on_clear_marks(self, _msg) -> None:
        """몸으로 찍은 표시(막힘·기울기 반사)만 지운다. 카메라로 본 위험 칸은 그대로 둔다.

        탐색 노드가 "갈 수 있는 곳이 하나도 없다" 고 할 때 부른다 — 표시는 한 번의
        실패로 찍히고 다시 확인되지 않으므로, 로버를 가둘 바에는 지우고 다시 겪는 편이 낫다.
        """
        n = len(self._bumps)
        self._bumps.clear()
        self._stall_ref = None
        self.get_logger().warn(f"막힘·기울기 표시 {n}곳을 지움 (탐색 노드 요청: 갈 곳이 없음)")
        self._publish(force=True)

    def _on_mark(self, msg) -> None:
        """다른 노드가 알려 준 위험 지점 (mission_manager 의 기울기 반사: 앞이 들린 자리의 앞).

        막힘 표시와 똑같이 다룬다 — 카메라가 못 본 낮은 턱·짧은 급경사를 몸으로 알아낸 것.
        """
        bx, by = float(msg.point.x), float(msg.point.y)
        if any(math.hypot(bx - x0, by - y0) < 0.3 for (x0, y0) in self._bumps):
            return
        self._bumps.append((bx, by))
        self.get_logger().warn(
            f"기울기 반사가 알려 준 자리 ({bx:.2f},{by:.2f}) 을 위험으로 표시 "
            f"(누적 {len(self._bumps)}곳)")
        self._publish(force=True)

    def _check_stall(self, t: float, pos: np.ndarray, quat: np.ndarray) -> None:
        """명령한 거리와 실제로 간 거리를 stall_sec 구간마다 비교한다.

        벽에 막힌 경우뿐 아니라 비탈에서 바퀴만 헛도는 경우도 잡는다 (27° 비탈에서
        바퀴는 0.2~0.3 m/s 로 도는데 차체는 0.03 m/s — 마찰 한계). 명령이 잠깐
        작아져도 구간이 끊기지 않도록 속도 문턱이 아니라 누적 거리로 본다.
        """
        cmd = self._cmd_lin
        sign = 1.0 if cmd > 0.05 else (-1.0 if cmd < -0.05 else 0.0)
        x, y, z, w = quat
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        ref = self._stall_ref
        if sign == 0.0:
            self._stall_ref = None
            return
        if ref is None or ref[2] != sign or t < ref[3]:
            self._stall_ref = [t, pos.copy(), sign, t, 0.0, yaw, 0.0]
            return
        # 상위 노드가 3 m/s 를 보내도 실제로 낼 수 있는 속도는 1 m/s 안쪽이다.
        ref[4] += min(abs(cmd), 0.8) * min(t - ref[3], 0.2)
        ref[3] = t
        ref[6] = max(ref[6], abs((yaw - ref[5] + math.pi) % (2.0 * math.pi) - math.pi))
        if t - ref[0] < float(self.get_parameter("stall_sec").value):
            return
        moved = float(np.hypot(*(pos[:2] - ref[1][:2])))
        asked = ref[4]
        self._stall_ref = [t, pos.copy(), sign, t, 0.0, yaw, 0.0]     # 다음 구간
        if ref[6] > math.radians(40.0):          # 방향을 크게 트는 중 — 제자리여도 막힌 게 아니다
            return
        if asked < float(self.get_parameter("stall_cmd_mps").value) * (t - ref[0]) \
                or moved > max(float(self.get_parameter("stall_move_m").value),
                               0.15 * asked):
            return
        d = sign * float(self.get_parameter("stall_mark_ahead_m").value)
        bx, by = float(pos[0] + d * math.cos(yaw)), float(pos[1] + d * math.sin(yaw))
        if any(math.hypot(bx - x0, by - y0) < 0.3 for (x0, y0) in self._bumps):
            return                              # 이미 찍은 자리 — 계획기가 돌아갈 때까지 기다림
        self._bumps.append((bx, by))
        self.get_logger().warn(
            f"막힘 감지: {t - ref[0]:.1f} s 동안 {'전진' if sign > 0 else '후진'} 명령 "
            f"{asked:.2f} m 에 실제 {moved:.2f} m → ({bx:.2f},{by:.2f}) 을 위험으로 표시 "
            f"(누적 {len(self._bumps)}곳)")
        self._publish()

    def _on_odom(self, msg: Odometry) -> None:
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        self._check_stall(_stamp_sec(msg.header), np.array([p.x, p.y, p.z]),
                          np.array([q.x, q.y, q.z, q.w]))
        self._odom.append((_stamp_sec(msg.header),
                           np.array([p.x, p.y, p.z]),
                           np.array([q.x, q.y, q.z, q.w])))

    def _on_joints(self, msg: JointState) -> None:
        pos = dict(zip(msg.name, msg.position))
        try:
            q = np.array([pos[n] for n in ARM_JOINTS])
        except KeyError:
            return
        dev = float(np.abs(q - ARM_HOME_RAD).max())
        # 팔이 차체 뒤쪽 반(joint_1 이 ±90° 밖)을 향하고 있으면 앞을 보는 깊이 카메라를
        # 가리지 않는다 (균형추로 뒤로 뻗은 경우). 그때는 영상을 그대로 쓴다.
        heading = abs((float(q[0]) + math.pi) % (2.0 * math.pi) - math.pi)
        self._arm_dev = 0.0 if heading >= math.radians(90.0) else dev

    def _on_detections(self, msg: DetectionArray) -> None:
        if not self._odom:
            return
        rover = self._odom[-1][1]
        for d in msg.detections:
            x, y = d.world_position.x, d.world_position.y
            if d.confidence < self.min_conf or (x == 0.0 and y == 0.0):
                continue
            if math.hypot(x - rover[0], y - rover[1]) > self.mineral_range:
                continue
            for k, (mx, my) in enumerate(self._minerals):
                if (mx - x) ** 2 + (my - y) ** 2 < 0.5 ** 2:
                    self._minerals[k] = (0.8 * mx + 0.2 * x, 0.8 * my + 0.2 * y)
                    break
            else:
                self._minerals.append((float(x), float(y)))

    def _pose_at(self, t: float):
        buf = self._odom
        if len(buf) < 2 or abs(t - buf[-1][0]) > 1.0:
            return None
        i = len(buf) - 1
        while i > 1 and buf[i - 1][0] > t:
            i -= 1
        t0, p0, q0 = buf[i - 1]
        t1, p1, q1 = buf[i]
        a = 0.0 if t1 <= t0 else float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0))
        if float(np.dot(q0, q1)) < 0.0:
            q1 = -q1
        return (1 - a) * p0 + a * p1, _quat_to_matrix((1 - a) * q0 + a * q1)

    def _on_depth(self, msg: Image) -> None:
        t = _stamp_sec(msg.header)
        if t - self._last_t < self.min_period or self.K is None:
            return
        if self._arm_dev is not None and self._arm_dev > self.arm_tol:
            self.n_skip_arm += 1
            self._last_t = t
            return
        pose = self._pose_at(t - self.latency)
        if pose is None:
            self.n_skip_pose += 1
            return
        if msg.encoding == "32FC1":
            depth = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width)
        elif msg.encoding == "16UC1":
            depth = np.frombuffer(msg.data, np.uint16).reshape(
                msg.height, msg.width).astype(np.float32) * 1e-3
        else:
            self.get_logger().warn(f"지원하지 않는 depth encoding: {msg.encoding}",
                                   throttle_duration_sec=10.0)
            return
        self._last_t = t
        fx, fy, cx, cy = self.K
        s = self.stride
        if self.hm.integrate(depth[::s, ::s], fx, fy, cx, cy, pose[0], pose[1], stride=s):
            self.n_used += 1

    # ── 출력 ────────────────────────────────────────────────────────────
    def _publish(self, force: bool = False) -> None:
        if not force and self.hm.frames == 0 and not self._bumps:
            return
        for (x, y) in self._minerals:
            self.hm.clear_disk(x, y, self.clear_r)
        cells = np.full((self.hm.rows, self.hm.cols), -1, dtype=np.int8)
        obs = self.hm.observed_mask()
        g = np.nan_to_num(self.hm.grade, nan=0.0)
        cells[obs] = np.clip(np.rint(g[obs]), 0, 49).astype(np.int8)
        cells[self.hm.obstacle_mask()] = 100
        if self._bumps:
            r = float(self.get_parameter("stall_mark_radius_m").value) / self.hm.cell
            ii, jj = np.mgrid[0:self.hm.rows, 0:self.hm.cols]
            for (x, y) in self._bumps:
                ci, cj = self.hm.world_to_cell(x, y)
                cells[(ii - int(ci)) ** 2 + (jj - int(cj)) ** 2 <= r * r] = 100
        msg = OccupancyGrid()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = str(self.get_parameter("frame_id").value)
        msg.info.resolution = float(self.hm.cell)
        msg.info.width = int(self.hm.cols)
        msg.info.height = int(self.hm.rows)
        msg.info.origin.position.x = -self.hm.map_w / 2.0
        msg.info.origin.position.y = -self.hm.map_h / 2.0
        msg.info.origin.orientation.w = 1.0
        msg.data = array.array("b", cells.tobytes())
        self.pub.publish(msg)

    def _status(self) -> None:
        obs, haz = self.hm.observed_mask(), self.hm.obstacle_mask()
        self.get_logger().info(
            f"hazard map: 영상 {self.n_used}장 반영 (팔 사용 중 {self.n_skip_arm}, "
            f"자세 없음 {self.n_skip_pose}), 본 면적 {obs.mean() * 100:.1f}% "
            f"({obs.sum() * self.hm.cell ** 2:.0f} m²), 위험 칸 {int(haz.sum())}, "
            f"광물 {len(self._minerals)}개 제외")


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = HazardMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
