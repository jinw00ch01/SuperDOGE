"""Mission manager — supervisor level FSM.

Pattern: cmd_vel mux that never touches coverage_node source.

Modes (orthogonal to phase, controlled via /mission/mode):
- AUTO    : phase-driven autonomy (default)
- MANUAL  : pass-through from /teleop/cmd_vel directly to /cmd_vel
            (phase still tracked for telemetry, but ignored for actuation)

Phases (AUTO mode):
- EXPLORE          : pass-through from /coverage/cmd_vel_raw
- APPROACH         : A* way-point pursuit toward best target
- PICK_READY       : stop within stop_distance, arm action client fires
                     (hysteresis: APPROACH->PICK_READY at stop_dist,
                      PICK_READY->APPROACH at stop_dist + resume_buffer)
- RETURN_TO_BASE   : A* path to (0, 0) basecamp — triggered when
                     collected_count >= collection_goal OR battery critical
- MISSION_COMPLETE : stopped at basecamp, mission ended
- RECOVERY         : 넘어져서 self_right_node 가 일으키는 중 (주행 정지)
- ESTOP            : 운영자가 세움. RESUME 명령이 올 때까지 주행 정지

Operator commands (HUD 의 ACTIONS 버튼 → <ns>/mission/command, std_msgs/String):
  ESTOP       주행 정지 + 진행 중인 채취 동작 취소 (팔이 그 자리에 선다)
  RESUME      ESTOP 해제. 세워 둔 팔을 HOME 으로 되돌린 뒤, 복귀/완료 중이었으면
              그대로, 아니면 EXPLORE 부터
  RTB         지금 기지로 복귀 (목표 수를 못 채웠어도)
  RTB_CANCEL  운영자가 시킨 복귀를 취소하고 탐색 재개
  RESET       채집 수를 0 으로. 임무가 끝났거나 복귀 중이면 탐색부터 다시 시작
  REPLAN      지금 경로를 버리고 다시 계획 (coverage 에도 요청)
  SKIP        접근 중인 목표를 failed_pick_ignore_sec 동안 건너뜀
명령과 phase 전환은 <ns>/mission/event (String) 로도 나간다 — HUD 의 EVENT LOG.

Coverage_node must be launched with topic remap:
  ros2 run isaac_drive coverage_node --ros-args -r /cmd_vel:=/coverage/cmd_vel_raw

On shutdown the node publishes zero twist a few times so the rover does
not keep the last angular velocity.
"""
from __future__ import annotations

import math
import os
from typing import List, Optional, Tuple

import rclpy
from geometry_msgs.msg import PointStamped, PoseStamped, Twist
from nav_msgs.msg import Odometry, Path
from rclpy.action import ActionClient
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from std_msgs.msg import Empty, String

from isaac_interfaces.action import ExecuteArmTask
from isaac_interfaces.msg import BatteryState, Detection, DetectionArray, MissionState
from isaac_drive.navigation.path_planner import astar, simplify_path
from isaac_drive.navigation.online_map import (OnlineObstacleLayer,
                                               basecamp_keepout, blank_terrain)
from isaac_drive.navigation.terrain_loader import load_terrain


def _default_terrain_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.normpath(os.path.join(here, "..", "..", "..", "..", "src",
                                      "a2_isaac", "isaac_sim", "assets",
                                      "generated_terrains", "terrain_00004")),
        os.path.expanduser("~/dev_ws/rover_ws/src/a2_isaac/isaac_sim/assets/"
                           "generated_terrains/terrain_00004"),
    ]
    for p in candidates:
        if os.path.isdir(p):
            return p
    return candidates[-1]


SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
)


VALUE_PRIORITY = {"yellow_mineral": 50.0, "green_gas": 25.0, "blue_mineral": 10.0}

# Per-class stop distance — balanced for wrist cam visibility (max depth ~1.1m)
# and rover P-control overshoot. T2 standalone uses tighter values (0.25 blue,
# 0.75 yellow/green) but assumes precise rover positioning; we widen here.
#
# 2026-11-28: 0.55 / 0.70 → 0.47 / 0.60. 예전 값은 "멈추라는 명령 뒤에 0.1~0.16 m 더 굴러간다"
# 를 전제로 한 것이었다 (실제로 선 자리: 파랑 0.40~0.45, 초록·노랑 0.51~0.59). 명령 지연을
# 없애고 나니 명령한 자리에 그대로 서서 (0.61~0.77), 팔이 닿는 한계(받침대에서 0.72 m 쯤,
# logs/min2_fix3 "out of reach x=0.77") 에 걸렸다. 서는 자리를 예전에 실제로 서던 곳으로 맞춘다.
CLASS_STOP_DIST_M = {
    "blue_mineral":   0.47,
    "yellow_mineral": 0.60,
    "green_gas":      0.60,
}


def _score(det: Detection) -> float:
    return float(det.value_score) + 0.1 * float(det.confidence)


def _stop_dist_for(class_name: str, default_m: float) -> float:
    return float(CLASS_STOP_DIST_M.get(class_name, default_m))


def turn_rate_cap(heading_err: float, decel: float = 0.4, spin_gain: float = 1.9) -> float:
    """남은 각도 안에서 멈출 수 있는 회전 명령의 상한 (rad/s).

    방향 오차가 클 때 로버는 거의 제자리에서 돈다. 시뮬레이터가 각가속도를
    제한하고(화성 0.99 rad/s²) 명령의 약 1.9 배로 돌기 때문에, 오차에 비례한
    명령만 주면 목표 방향을 지나쳤다 되돌아오기를 반복한다 (좌우로 흔들림).
    Navigator 의 제자리 회전과 같은 식.
    """
    return math.sqrt(2.0 * decel * abs(heading_err)) / spin_gain


def confirm_hits(seen: list, items, now_ns: int, need: int, radius_m: float,
                 window_ns: int) -> list:
    """같은 자리에서 여러 번 본 검출만 통과시킨다.

    seen: [[class, x, y, 횟수, 마지막으로 본 시각]] — 호출할 때마다 갱신된다.
    items: [(class, x, y)]. return: items 와 같은 길이의 bool 목록.
    위치 추정은 가끔 한 번씩 몇 m 튄다 (깊이 화소가 광물 뒤 배경을 잡을 때).
    그런 값이 그대로 목표가 되면 로버는 엉뚱한 곳으로 가고, 목표 고정(lock)이
    그 값에 걸려 진짜 위치의 검출을 "다른 광물" 로 버린다.
    """
    seen[:] = [s for s in seen if now_ns - s[4] < window_ns]
    r2 = radius_m * radius_m
    out = []
    for cls, x, y in items:
        best, bd = None, r2
        for s in seen:
            if s[0] != cls:
                continue
            dd = (s[1] - x) ** 2 + (s[2] - y) ** 2
            if dd < bd:
                best, bd = s, dd
        if best is None:
            best = [cls, x, y, 0, now_ns]
            seen.append(best)
        best[1] = 0.5 * best[1] + 0.5 * x
        best[2] = 0.5 * best[2] + 0.5 * y
        best[3] += 1
        best[4] = now_ns
        out.append(best[3] >= need)
    return out


class _CmdTap:
    """cmd_vel publisher 를 감싸 마지막으로 보낸 진행 방향을 기억한다 (기울기 반사가 쓴다)."""

    def __init__(self, pub) -> None:
        self._pub = pub
        self.last_dir = 1.0

    def publish(self, msg: Twist) -> None:
        if abs(msg.linear.x) > 0.05:
            self.last_dir = 1.0 if msg.linear.x > 0.0 else -1.0
        self._pub.publish(msg)


def nose_up_and_roll_deg(q) -> Tuple[float, float]:
    """자세 quaternion → (앞이 들린 각, 옆으로 기운 각) [deg].

    차체 x 축(앞)과 y 축(왼쪽)이 수평면에서 얼마나 올라가 있는지다. 평지 0,
    앞바퀴가 턱에 올라타면 앞이 들린 각이 +.
    """
    x, y, z, w = q.x, q.y, q.z, q.w
    nose = math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (x * z - w * y)))))
    roll = math.degrees(math.asin(max(-1.0, min(1.0, 2.0 * (y * z + w * x)))))
    return nose, roll


class MissionManagerNode(Node):
    def __init__(self) -> None:
        super().__init__("mission_manager_node")

        self.declare_parameter("approach_engage_dist_m", 8.0)
        self.declare_parameter("approach_stop_dist_m", 0.75)   # default fallback
        self.declare_parameter("approach_resume_buffer_m", 0.3)
        self.declare_parameter("coverage_cmd_stale_sec", 1.0)
        self.declare_parameter("post_pick_cooldown_sec", 8.0)
        self.declare_parameter("post_pick_skip_radius_m", 1.0)
        # pick 이 실패한 지점(오검출 등) 을 이 시간 동안 후보에서 제외 — 같은
        # 허상 target 으로 무한 재시도하는 것 방지. 반경은 post_pick_skip_radius_m.
        self.declare_parameter("failed_pick_ignore_sec", 60.0)
        # 지도 없는 주행(map_source=online)에서는 광물도 위험 칸으로 잡힌다. 그 광물을
        # 집으러 갈 때는 목표 주변 이 반경의 위험 칸을 없는 것으로 치고 계획한다.
        self.declare_parameter("approach_target_clear_m", 0.5)
        # 같은 자리(반경 안)에서 이 횟수만큼 본 검출만 목표 후보로 삼는다.
        # 한 번 튄 위치 추정이 목표가 되는 것을 막는다. 1 이면 예전 동작.
        self.declare_parameter("target_confirm_hits", 3)
        self.declare_parameter("target_confirm_radius_m", 1.0)
        self.declare_parameter("target_confirm_window_sec", 3.0)
        # 계획이 목표에 못 미친 곳에서 끝났을 때 (목표 주변이 막힘) 다시 계획해 보는 주기.
        self.declare_parameter("approach_short_plan_retry_sec", 1.0)
        self.declare_parameter("arm_action_name", "/execute_arm_task")
        self.declare_parameter("enable_arm_action", True)
        self.declare_parameter("approach_lin_speed", 0.6)
        self.declare_parameter("approach_creep_speed", 0.25)
        # 멈출 자리를 지나치지 않게 속도를 줄이는 기준 (_approach_speed).
        #   approach_brake_decel_m_s2 : 줄일 때 쓰는 감속도. 구동부 한계(화성 0.72)보다 낮게.
        #   approach_brake_delay_sec  : 명령이 바퀴에 닿기까지의 지연 — 그동안 가는 거리를 뺀다.
        #   approach_creep_lead_m     : 멈출 자리의 이만큼 앞에서 이미 느린 속도가 되게.
        self.declare_parameter("approach_brake_decel_m_s2", 0.4)
        self.declare_parameter("approach_brake_delay_sec", 0.3)
        self.declare_parameter("approach_creep_lead_m", 0.35)
        # 멈춘 뒤 광물이 받침대에서 이보다 가까우면(지나쳤으면) 집기 전에 물러난다.
        # 반대로 멈춘 자리가 stop 거리 + pick_reach_slack_m 보다 멀면 (비탈에서 뒤로 밀림)
        # 천천히 다가가서 맞춘다.
        self.declare_parameter("pick_min_dist_m", 0.30)
        self.declare_parameter("pick_backoff_speed", 0.15)
        self.declare_parameter("pick_reach_slack_m", 0.06)
        self.declare_parameter("steer_gain", 1.5)
        self.declare_parameter("detection_stale_sec", 1.5)
        # 접근을 포기하는 기준. 예전에는 "마지막으로 본 지 8 초" 였는데, 장애물을 돌아가는
        # 동안에는 카메라가 광물을 등지므로 접근의 절반이 그 8 초에 걸려 끊겼다 (광물은
        # 움직이지 않으니 안 보인다고 버릴 이유가 없다). 지금은 "가까워지고 있는가" 로 본다:
        # 남은 경로 길이가 approach_stall_sec 동안 approach_stall_progress_m 만큼도 줄지
        # 않으면 포기하고, 전체로는 approach_budget_base_sec + 처음 경로 길이 ×
        # approach_budget_sec_per_m 를 넘기지 않는다.
        # 본 광물을 기억해 두는 목록 (_remember_minerals). 지금 안 보여도 가까운 것부터 찾아간다.
        #   mineral_memory_range_m : 이보다 멀리서 본 것은 적지 않는다. 처음에는 12 m 로 두었는데
        #       13 m·23 m 밖에서 또렷이 보이는 광물이 지도에도 안 뜨고 가지도 않았다
        #       (logs/min2_fix5 t=128). 멀리서 잰 거리는 틀릴 수 있지만 가까이서 다시 잰 값으로
        #       고치면 된다 (_remember_minerals) — 접근을 시작하는 거리(30 m)까지 적는다.
        #   target_switch_margin_m : 접근 중에 이만큼 더 가까운 광물이 보이면 그쪽으로 바꾼다
        #   mineral_memory_max_fails : 접근·집기에 이만큼 실패한 자리는 목록에서 지운다
        # 지도 노드가 "광물이 아님 (건물 벽)" 이라고 알려 오는 자리 — 그 둘레의 검출은 무시한다.
        self.declare_parameter("mineral_veto_topic", "/hazard_map/not_mineral")
        self.declare_parameter("mineral_veto_radius_m", 1.0)
        # 기억한 광물 목록을 HUD 지도에 그리도록 내보내는 토픽 (JSON, 1 Hz).
        self.declare_parameter("known_minerals_topic", "/mission/known_minerals")
        self.declare_parameter("mineral_memory_range_m", 30.0)
        # 기억에 적는 기준은 목표로 고정하는 기준(0.6, 3 번)보다 낮다. 지나가며 0.4 s 쯤만
        # 보인 광물(검출 3 번, 신뢰도 0.53~0.69)이 화면에는 상자로 뜨는데 지도에 안 남고
        # 가지도 않았다 (logs/min2_fix7 t=19). 잘못 적은 것은 가서 걸러 낸다:
        #   mineral_look_* : 기억한 자리에 look_max_m 안으로 다가와 그쪽을 look_sec 동안
        #       보고 있는데 검출이 한 번도 없으면 "없는 것" 으로 지운다 (팔을 뻗기 전에).
        self.declare_parameter("memory_min_confidence", 0.5)
        self.declare_parameter("memory_confirm_hits", 2)
        #       접근 속도가 0.5~0.9 m/s 라 이 구간은 2~3 초면 지나간다 — look_sec 를 4 초로
        #       두었더니 한 번도 걸리지 않고 빈 자리에 팔을 뻗었다 (logs/min2_turn1 t=130).
        self.declare_parameter("mineral_look_max_m", 3.5)
        self.declare_parameter("mineral_look_min_m", 1.0)
        self.declare_parameter("mineral_look_cone_deg", 20.0)
        self.declare_parameter("mineral_look_sec", 1.5)
        # 멈춰서 보기: 인식 노드는 빨리 도는 중(0.25 rad/s 초과)의 검출에 좌표를 주지 않는다
        # (그때의 좌표는 90 % 오차가 0.24 → 1.86 m 로 커진다 — logs/min2_turn1). 그래서
        # 탐색 중 제자리에서 돌다가 본 광물은 화면에 상자만 뜨고 지도에 남지 않았다.
        # 좌표 없는 검출이 look_stop_hits 번 이어지면 look_stop_sec 동안 멈춰서 좌표를 얻는다.
        self.declare_parameter("look_stop_hits", 2)
        self.declare_parameter("look_stop_sec", 1.2)
        self.declare_parameter("look_stop_cooldown_sec", 6.0)
        self.declare_parameter("look_stop_min_yaw_rate", 0.2)
        self.declare_parameter("target_switch_margin_m", 1.0)
        self.declare_parameter("mineral_memory_max_fails", 3)
        self.declare_parameter("approach_stall_sec", 25.0)
        self.declare_parameter("approach_stall_progress_m", 0.3)
        self.declare_parameter("approach_budget_base_sec", 40.0)
        self.declare_parameter("approach_budget_sec_per_m", 8.0)
        self.declare_parameter("explore_resume_delay_sec", 2.0)
        self.declare_parameter("phase_log_period_sec", 1.0)
        self.declare_parameter("coverage_cmd_topic", "/coverage/cmd_vel_raw")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("teleop_cmd_topic", "/teleop/cmd_vel")
        self.declare_parameter("detections_topic", "/perception/detections")
        self.declare_parameter("odom_topic", "/ground_truth/odom")
        self.declare_parameter("phase_topic", "/mission/phase")
        self.declare_parameter("state_topic", "/mission/state")
        self.declare_parameter("mode_topic", "/mission/mode")
        self.declare_parameter("estop_topic", "/mission/estop")
        # 운영자 명령 (HUD 의 ACTIONS 버튼). std_msgs/String:
        #   ESTOP | RESUME | RTB | RTB_CANCEL | RESET | REPLAN | SKIP
        self.declare_parameter("command_topic", "/mission/command")
        # 운영자 명령과 phase 전환을 사람이 읽는 한 줄로 내보낸다 (HUD 의 EVENT LOG).
        self.declare_parameter("event_topic", "/mission/event")
        # 기울기 반사 — 앞이 들리거나 옆으로 크게 기울면 밀어붙이지 않고 물러난다.
        # 급경사 지형에서 넘어지는 경우의 거의 전부가 "짧고 가파른 턱을 최대 명령으로
        # 밀다가 차체는 서 있는 채 앞이 40~55° 까지 들려 뒤로 넘어가는" 것이었다.
        # (terrain_00015, 16 대 × 15 분: 반사 없이 8 회 → 30°/32° 반사로 2 회)
        self.declare_parameter("tilt_guard", True)
        self.declare_parameter("tilt_guard_pitch_deg", 30.0)    # 앞이 들린 각
        self.declare_parameter("tilt_guard_roll_deg", 32.0)     # 옆으로 기운 각
        self.declare_parameter("tilt_guard_back_mps", 0.5)
        # 물러나기는 거리로 끝낸다: 구동 제한(화성 0.72 m/s²) 때문에 0.9 m/s 로 가던 중이면
        # 후진 명령을 내도 1.3 초·0.5 m 는 앞으로 더 간다 (시간으로 끊으면 물러나지 못한다).
        self.declare_parameter("tilt_guard_back_m", 0.5)        # 가장 멀리 나간 자리에서 이만큼
        self.declare_parameter("tilt_guard_back_sec", 4.0)      # 그래도 이 시간이면 끝 (시뮬레이션 시각)
        self.declare_parameter("tilt_guard_mark_ahead_m", 0.9)  # 그 자리 앞을 위험으로 표시
        # 물러났는데도 기울기가 그대로면 곧바로 또 발동한다 (가파른 비탈을 다 내려올 때까지
        # 물러나는 것 — 의도한 동작). 다만 끝없이 후진만 하지 않게 연달아 이 횟수까지만.
        # 그 뒤에는 기울기가 문턱 − 3° 아래로 내려가거나 지금보다 5° 더 기울 때까지 쉰다.
        self.declare_parameter("tilt_guard_chain_max", 3)
        self.declare_parameter("tilt_guard_skip_after", 3)      # 같은 목표로 가다 이만큼 걸리면 건너뜀
        self.declare_parameter("hazard_mark_topic", "/hazard_map/mark")
        # self_right_node 가 넘어진 로버를 세우는 동안 True 를 보낸다 → 바퀴 정지.
        self.declare_parameter("righting_active_topic", "righting/active")
        self.declare_parameter("righting_status_topic", "righting/status")
        self.declare_parameter("battery_topic", "/battery_state")
        self.declare_parameter("min_confidence", 0.6)
        self.declare_parameter("terrain_dir", _default_terrain_dir())
        # 장애물 지도의 출처. online (기본): hazard_map_node 가 깊이 카메라로
        # 주행 중에 만드는 지도. prebuilt: obstacle_grid.npy (원본, 정답 지도).
        self.declare_parameter("map_source", "online")
        self.declare_parameter("hazard_map_topic", "/hazard_map")
        self.declare_parameter("path_cell_size", 0.2)
        # rover 실제 차폭 (m0609 팔 + 후방 바스켓 포함) ≈ 1.5m.
        # 이력: 0.7m 는 가장자리 충돌 잦아 1.0m 로 올렸으나, 1.0m 는 좁은 통로
        # (≤2.5m) 통과 못 해 path 가 비효율적으로 빙 둘러감 (2026-05-26 시연
        # 검증). 절충값 0.8m — obstacle 중심에서 0.8m 마진, rover 차체 끝
        # 0.05m 여유. coverage 의 robot_radius 0.7m 와도 가까워 두 grid 간
        # 시각 차이 최소.
        self.declare_parameter("path_robot_radius", 0.8)
        # target 이 이 거리 이내로 흔들리면 replan skip — perception 의
        # mineral 위치 jitter 가 0.4m 안으로 흔들리며 1초당 6+ 회 replan
        # 폭주를 만들어 path 진동 보임. 0.5m 로 올려 perception covariance
        # 일반 범위 안의 jitter 무시.
        self.declare_parameter("path_replan_target_delta_m", 0.5)
        self.declare_parameter("waypoint_reach_dist_m", 0.4)
        # Mission termination
        self.declare_parameter("collection_goal", 5)
        self.declare_parameter("cargo_capacity", 5)
        self.declare_parameter("basecamp_x", 0.0)
        self.declare_parameter("basecamp_y", 0.0)
        # Basecamp 자체는 obstacle 로 마킹된 영역 (terrain meta: keepout 반경 6m,
        # visual_footprint 8x8m). path_robot_radius=1.0m inflate 후 obstacle
        # 외곽 ≈ 7m. fringe 는 그보다 살짝 바깥 (7.5m), arrive 는 fringe 가깝게
        # 둬서 fringe 도달이 곧 mission_complete.
        self.declare_parameter("basecamp_fringe_radius_m", 7.5)
        self.declare_parameter("basecamp_arrive_radius_m", 8.0)
        self.declare_parameter("rtb_lin_speed", 0.9)
        self.declare_parameter("teleop_stale_sec", 0.5)
        self.declare_parameter("state_publish_hz", 2.0)

        # ── Multi-rover 협조 (T2 패턴, 838d5ed) ────────────────────────
        # enable_mineral_claim: 둘이 같은 mineral 안 노리도록 /mineral_claims
        # 토픽 (TTL 5s, radius 1.5m) — 다른 rover claim 근처 mineral 은 후보 제외.
        # enable_rover_avoid: A* 의 obstacle_grid 에 다른 rover 위치를
        # rover_avoid_radius_m 만큼 dynamic inflate. 0.8m 이상 움직이면 replan.
        # 둘 다 default False — 단일 rover 시연엔 부담 없음. mvp_multi.launch.py
        # 가 True 로 override.
        self.declare_parameter("enable_mineral_claim", False)
        self.declare_parameter("claim_topic", "/mineral_claims")
        self.declare_parameter("claim_skip_radius_m", 1.5)
        self.declare_parameter("claim_ttl_sec", 5.0)
        self.declare_parameter("claim_publish_period_sec", 0.5)
        self.declare_parameter("claim_rover_id", "")  # frame_id 로 사용 — 빈값 fallback
        self.declare_parameter("enable_rover_avoid", False)
        self.declare_parameter("rover_positions_topic", "/rover_positions")
        self.declare_parameter("rover_position_publish_period_sec", 0.2)
        self.declare_parameter("rover_position_ttl_sec", 2.0)
        self.declare_parameter("rover_avoid_radius_m", 1.2)
        self.declare_parameter("rover_replan_trigger_m", 0.8)

        self.phase = "EXPLORE"
        self.mode = "AUTO"
        self.estop = False
        self._phase_before_estop = "EXPLORE"
        self.operator_rtb = False           # 운영자가 누른 RTB (목표 달성 전)
        self._arm_goal_handle = None
        self._arm_stopped = False           # ESTOP 으로 팔을 세웠다 → 풀리면 HOME 으로 보낸다
        self.righting_active = False
        self._righting_stamp_ns = 0
        self.righting_status = ""           # self_right_node 가 보내는 진행 상황 (HUD 의 TASK)
        self._guard_until = None            # 물러나기가 끝나는 odom 시각 (s). None = 평소
        self._guard_start = 0.0
        self._guard_p0 = (0.0, 0.0)         # 발동한 자리
        self._guard_fwd = (1.0, 0.0)        # 가려던 방향 (수평 단위 벡터)
        self._guard_far = 0.0               # 발동 뒤 가려던 쪽으로 가장 멀리 나간 거리
        self._guard_dir = -1.0              # 물러나는 방향 (-1 = 후진)
        self._guard_why = ""
        self._guard_count = 0               # 누적 횟수
        self._guard_chain = 0               # 기울기가 풀리지 않은 채 연달아 발동한 횟수
        self._guard_peak = 0.0              # 그 동안 가장 크게 넘은 정도 (문턱 대비, 도)
        self._guard_target = None           # (x, y, 횟수) 같은 목표로 가다 걸린 횟수
        self._phase_before_recovery = "EXPLORE"
        # Mineral collection / battery
        self.collected_count: int = 0
        self.cargo_count: int = 0
        self.collected_by_class: dict[str, int] = {
            "blue_mineral": 0, "yellow_mineral": 0, "green_gas": 0}
        self.battery_percent: float = 100.0
        self.battery_low: bool = False
        self.battery_critical: bool = False
        # Teleop
        self.last_teleop_cmd: Optional[Twist] = None
        self.last_teleop_ns: int = 0
        # Last error string for MissionState
        self.last_error: str = ""
        self.target: Optional[Detection] = None
        self.last_det_stamp_ns: int = 0
        self.last_coverage_cmd: Optional[Twist] = None
        self.last_coverage_cmd_ns: int = 0
        self.last_odom: Optional[Odometry] = None
        # A* planning state
        self.ogrid = None
        self.waypoints: List[Tuple[float, float]] = []
        self.wp_idx: int = 0
        self.last_plan_target: Optional[Tuple[float, float]] = None
        # Multi-rover 협조 state
        # other_claims[rid] = (x, y, ts_ns)
        self.other_claims: dict = {}
        # other_rovers[rid] = (x, y, ts_ns)
        self.other_rovers: dict = {}
        # 마지막 plan 시 다른 rover 위치 (replan trigger 판단용)
        self._last_plan_other_rovers: dict = {}
        # 빈 문자열이면 namespace 자동 추출 fallback (get_namespace())
        rid_param = str(self.get_parameter("claim_rover_id").value).strip()
        self._my_rover_id = rid_param or self.get_namespace().strip("/") or "rover"
        # Arm action state
        self.arm_in_flight: bool = False
        self.arm_client: Optional[ActionClient] = None
        # Post-pick cooldown
        self.last_pick_world: Optional[Tuple[float, float]] = None
        # (x, y, expire_ns) — pick 실패 지점 임시 제외 목록
        self.failed_pick_points: list = []
        self.plan_goal_gap = 0.0        # 마지막 계획의 끝점 ↔ 목표 거리 (m)
        self._seen_dets: list = []      # confirm_hits 의 기억
        self._seen_mem: list = []       # 같은 것, 기억용 (기준이 낮다)
        self._appr_look_ns: int = 0     # 기억한 자리를 가까이서 보기 시작한 시각 (0 = 아님)
        self.yaw_rate: float = 0.0      # odom 자세를 차분한 회전 속도 (rad/s)
        self._yaw_prev = None           # (stamp s, yaw)
        self._blind_hits: list = []     # 좌표 없는 검출을 본 시각들 (ns)
        self._look_stop_until_ns: int = 0
        self._look_stop_next_ns: int = 0
        self._defer_since_ns = 0
        self.last_plan_ns = 0
        self.last_pick_done_ns: int = 0
        # Target lock-on (kept while APPROACH, even if detection briefly lost)
        self.lock_target: Optional[Detection] = None
        self.lock_started_ns: int = 0
        # 접근 진척 감시 (_approach_progress): 남은 경로 길이의 최소값과 그것이 줄어든 시각.
        self._appr_best: float = float("inf")
        self._appr_progress_ns: int = 0
        self._appr_started_ns: int = 0
        self._appr_budget_ns: int = 0
        self._appr_unseen_logged: bool = False
        # 본 광물 목록: {"cls", "x", "y", "range"(잰 거리), "det"(Detection), "fails"}
        self.known: list = []
        self.vetoed: list = []                  # [(x, y)] 광물이 아니라고 판정된 자리
        self._chosen_from: int = 0
        # 실제 속력 (odom 위치 차분, m/s) — 멈췄는지·멈추는 데 얼마나 갈지 보는 데 쓴다.
        self.speed: float = 0.0
        self._speed_prev = None                 # (stamp s, x, y)
        # PICK_READY 에 들어와 아직 팔에 명령을 보내지 않은 상태 (멈추기·물러나기 중).
        self._pick_pending: bool = False
        self._pick_adjust: str = ""             # "" | "back" (지나쳐서 물러남) | "fwd" (멀어서 다가감)
        # EXPLORE resume cooldown — force stop briefly after any APPROACH/PICK
        # exit so coverage_node can replan from the rover's new position.
        self.explore_entry_ns: int = 0

        terrain_dir = str(self.get_parameter("terrain_dir").value)
        self.map_layer = None
        self.map_replans = 0
        map_source = str(self.get_parameter("map_source").value).strip().lower()
        try:
            loader = blank_terrain if map_source == "online" else load_terrain
            _meta, self.ogrid, _fog = loader(
                terrain_dir,
                cell_size=float(self.get_parameter("path_cell_size").value),
                robot_radius=float(self.get_parameter("path_robot_radius").value),
            )
            if map_source == "online":
                self.map_layer = OnlineObstacleLayer(self.ogrid, basecamp_keepout(_meta))
                from nav_msgs.msg import OccupancyGrid
                from rclpy.qos import (DurabilityPolicy, HistoryPolicy,
                                       QoSProfile, ReliabilityPolicy)
                self.create_subscription(
                    OccupancyGrid, str(self.get_parameter("hazard_map_topic").value),
                    self._on_hazard_map,
                    QoSProfile(reliability=ReliabilityPolicy.RELIABLE,
                               durability=DurabilityPolicy.TRANSIENT_LOCAL,
                               history=HistoryPolicy.KEEP_LAST, depth=1))
                self.get_logger().info(
                    f"장애물 지도: 주행 중 작성 (깊이 카메라) — 빈 격자 "
                    f"{self.ogrid.rows}x{self.ogrid.cols} @ {self.ogrid.cell_size:.2f}m 로 시작")
            else:
                self.get_logger().info(
                    f"obstacle_grid loaded from {terrain_dir} "
                    f"({self.ogrid.rows}x{self.ogrid.cols} @ {self.ogrid.cell_size:.2f}m)")
        except Exception as e:
            self.get_logger().error(
                f"terrain load FAILED ({e}) — APPROACH/RTB will fall back to straight-line P-control")

        if bool(self.get_parameter("enable_arm_action").value):
            self.arm_client = ActionClient(
                self, ExecuteArmTask, str(self.get_parameter("arm_action_name").value))

        self.cmd_pub = _CmdTap(self.create_publisher(
            Twist, str(self.get_parameter("cmd_vel_topic").value), 10))
        self._guard_pub = self.cmd_pub._pub
        self.hazard_mark_pub = self.create_publisher(
            PointStamped, str(self.get_parameter("hazard_mark_topic").value), 10)
        self.create_subscription(
            PointStamped, str(self.get_parameter("mineral_veto_topic").value),
            self._on_mineral_veto, 10)
        self.known_pub = self.create_publisher(
            String, str(self.get_parameter("known_minerals_topic").value), 5)
        self.create_timer(1.0, self._publish_known)
        self.phase_pub = self.create_publisher(
            String, str(self.get_parameter("phase_topic").value), 10)
        self.state_pub = self.create_publisher(
            MissionState, str(self.get_parameter("state_topic").value), 10)
        self.event_pub = self.create_publisher(
            String, str(self.get_parameter("event_topic").value), 20)
        # Tell coverage_node to drop its stale DRIVE path and replan from
        # the rover's new post-APPROACH/PICK position.
        self.replan_pub = self.create_publisher(
            Empty, "/coverage/replan_request", 10)
        # Supervisor 가 APPROACH/RTB 시 자체적으로 그린 A* path / target.
        # coverage_node 의 /mission/path / /mission/markers 와 별개 토픽이라
        # 두 source 가 시간상 분리돼서 UI 에 표시됨 (EXPLORE 때는 coverage,
        # 그 외엔 supervisor).
        self.supervisor_path_pub = self.create_publisher(
            Path, "/supervisor/path", 1)
        self.supervisor_target_pub = self.create_publisher(
            PointStamped, "/supervisor/target", 1)

        self.create_subscription(
            DetectionArray, str(self.get_parameter("detections_topic").value),
            self._on_detections, SENSOR_QOS)
        self.create_subscription(
            Twist, str(self.get_parameter("coverage_cmd_topic").value),
            self._on_coverage_cmd, 10)
        self.create_subscription(
            Odometry, str(self.get_parameter("odom_topic").value),
            self._on_odom, SENSOR_QOS)
        self.create_subscription(
            Twist, str(self.get_parameter("teleop_cmd_topic").value),
            self._on_teleop_cmd, 10)
        self.create_subscription(
            String, str(self.get_parameter("mode_topic").value),
            self._on_mode, 10)
        from std_msgs.msg import Bool as _Bool
        self.create_subscription(
            String, str(self.get_parameter("righting_status_topic").value),
            lambda m: setattr(self, "righting_status", m.data), 10)
        self.create_subscription(
            _Bool, str(self.get_parameter("righting_active_topic").value),
            self._on_righting, 10)
        self.create_subscription(
            Empty, str(self.get_parameter("estop_topic").value),
            self._on_estop, 10)
        self.create_subscription(
            String, str(self.get_parameter("command_topic").value),
            self._on_command, 10)
        self.create_subscription(
            BatteryState, str(self.get_parameter("battery_topic").value),
            self._on_battery, 10)

        # ── Multi-rover 협조 pub/sub + timer ──────────────────────────
        # /mineral_claims, /rover_positions 는 namespace 없는 공유 토픽 (모든
        # rover 가 동일 채널 sub/pub). frame_id 가 rover_id 라 자기 자신 무시.
        if bool(self.get_parameter("enable_rover_avoid").value):
            pos_topic = str(self.get_parameter("rover_positions_topic").value)
            self.rover_pos_pub = self.create_publisher(
                PointStamped, pos_topic, 10)
            self.create_subscription(
                PointStamped, pos_topic, self._on_rover_position, 10)
            self.create_timer(
                float(self.get_parameter(
                    "rover_position_publish_period_sec").value),
                self._publish_rover_position)
            self.get_logger().info(
                f"rover avoidance 활성 — id={self._my_rover_id} "
                f"topic={pos_topic} inflate="
                f"{float(self.get_parameter('rover_avoid_radius_m').value):.2f}m")
        else:
            self.rover_pos_pub = None

        if bool(self.get_parameter("enable_mineral_claim").value):
            claim_topic = str(self.get_parameter("claim_topic").value)
            self.claim_pub = self.create_publisher(
                PointStamped, claim_topic, 10)
            self.create_subscription(
                PointStamped, claim_topic, self._on_claim, 10)
            self.create_timer(
                float(self.get_parameter("claim_publish_period_sec").value),
                self._publish_claim)
            self.get_logger().info(
                f"mineral claim 협조 활성 — id={self._my_rover_id} "
                f"topic={claim_topic} radius="
                f"{float(self.get_parameter('claim_skip_radius_m').value):.2f}m "
                f"ttl={float(self.get_parameter('claim_ttl_sec').value):.1f}s")
        else:
            self.claim_pub = None

        self.create_timer(0.05, self._tick)
        self.create_timer(
            float(self.get_parameter("phase_log_period_sec").value), self._log_phase)
        state_hz = max(float(self.get_parameter("state_publish_hz").value), 0.5)
        self.create_timer(1.0 / state_hz, self._publish_state)
        self.get_logger().info(
            f"mission_manager_node ready. mode=AUTO phase=EXPLORE goal="
            f"{int(self.get_parameter('collection_goal').value)} minerals.")

    def _on_detections(self, msg: DetectionArray) -> None:
        # While the arm is executing a pick, ignore new detections so the
        # current target/phase does not get swapped mid-action.
        if self.arm_in_flight:
            return
        min_conf = float(self.get_parameter("min_confidence").value)
        cooldown_ns = int(float(self.get_parameter("post_pick_cooldown_sec").value) * 1e9)
        skip_r = float(self.get_parameter("post_pick_skip_radius_m").value)
        now_ns = self.get_clock().now().nanoseconds

        def in_cooldown(d: Detection) -> bool:
            if self.last_pick_world is None:
                return False
            if now_ns - self.last_pick_done_ns >= cooldown_ns:
                return False
            dx = d.world_position.x - self.last_pick_world[0]
            dy = d.world_position.y - self.last_pick_world[1]
            return (dx * dx + dy * dy) <= (skip_r * skip_r)

        self.failed_pick_points = [
            f for f in self.failed_pick_points if f[2] > now_ns]

        def near_failed_pick(d: Detection) -> bool:
            for fx, fy, _ in self.failed_pick_points:
                dx = d.world_position.x - fx
                dy = d.world_position.y - fy
                if (dx * dx + dy * dy) <= (skip_r * skip_r):
                    return True
            return False

        mem_conf = min(min_conf, float(self.get_parameter("memory_min_confidence").value))
        # 돌다가 본 광물 (좌표 없음) → 잠깐 멈춰서 좌표를 얻는다.
        if (self.phase == "EXPLORE" and self.lock_target is None and self.mode == "AUTO"
                and self.yaw_rate > float(self.get_parameter("look_stop_min_yaw_rate").value)
                and now_ns >= self._look_stop_next_ns
                and any(d.confidence >= mem_conf and d.world_position.x == 0.0
                        and d.world_position.y == 0.0 and d.world_position.z == 0.0
                        for d in msg.detections)):
            self._blind_hits = [h for h in self._blind_hits if now_ns - h < int(1e9)] + [now_ns]
            if len(self._blind_hits) >= int(self.get_parameter("look_stop_hits").value):
                self._blind_hits = []
                self._look_stop_until_ns = now_ns + int(
                    float(self.get_parameter("look_stop_sec").value) * 1e9)
                self._look_stop_next_ns = now_ns + int(
                    float(self.get_parameter("look_stop_cooldown_sec").value) * 1e9)
                self.get_logger().info(
                    f"도는 중({self.yaw_rate:.2f} rad/s)에 광물이 보임 — 잠깐 멈춰서 위치를 잰다")
                self._event("도는 중에 광물이 보임 — 멈춰서 위치 확인")
        loose = [
            d for d in msg.detections
            if d.confidence >= mem_conf
            and not (d.world_position.x == 0.0
                     and d.world_position.y == 0.0
                     and d.world_position.z == 0.0)
            and not in_cooldown(d)
            and not near_failed_pick(d)
            and not self._is_vetoed(d.world_position.x, d.world_position.y)
            # Multi-rover: 다른 rover 가 claim 한 mineral 은 후보 제외
            and not self._is_claimed_by_other(
                d.world_position.x, d.world_position.y)
        ]
        # 기억: 낮은 기준으로 확인된 것을 모두 적는다 (목표 고정과는 별개).
        seen_ok = []
        if loose:
            ok = confirm_hits(
                self._seen_mem,
                [(d.class_name, d.world_position.x, d.world_position.y) for d in loose],
                now_ns, int(self.get_parameter("memory_confirm_hits").value),
                float(self.get_parameter("target_confirm_radius_m").value),
                int(float(self.get_parameter("target_confirm_window_sec").value) * 1e9))
            seen_ok = [d for d, keep in zip(loose, ok) if keep]
            self._remember_minerals(seen_ok)
        candidates = [d for d in loose if d.confidence >= min_conf]
        if candidates:
            ok = confirm_hits(
                self._seen_dets,
                [(d.class_name, d.world_position.x, d.world_position.y) for d in candidates],
                now_ns, int(self.get_parameter("target_confirm_hits").value),
                float(self.get_parameter("target_confirm_radius_m").value),
                int(float(self.get_parameter("target_confirm_window_sec").value) * 1e9))
            pending = [d for d, keep in zip(candidates, ok) if not keep]
            candidates = [d for d, keep in zip(candidates, ok) if keep]
            # 더 값진 광물이 보이는데 아직 확인 횟수를 못 채웠다면, 덜 값진 것에
            # 먼저 고정되지 않도록 잠깐(최대 1.5 s) 기다린다.
            if (self.lock_target is None and candidates and pending
                    and max(map(_score, pending)) > max(map(_score, candidates)) + 1.0):
                if self._defer_since_ns == 0:
                    self._defer_since_ns = now_ns
                if now_ns - self._defer_since_ns < int(1.5e9):
                    return
            else:
                self._defer_since_ns = 0
        if self.lock_target is None and not candidates:
            return

        if self.lock_target is not None:
            # Lock 중 — 같은 mineral 의 detection 만 받아 좌표 + lock timer 갱신.
            # 다른 mineral 은 무시 (lock 핵심). 같은 mineral 판정 =
            # class_name 일치 + 좌표 2m 이내.
            lx = self.lock_target.world_position.x
            ly = self.lock_target.world_position.y
            match_r2 = 2.0 * 2.0
            # 고정한 목표의 갱신은 기억용 기준으로 본다 — 신뢰도가 낮게 나오는 광물도
            # 다가가며 위치가 고쳐지고 "보인다" 로 친다.
            same = [
                d for d in seen_ok
                if d.class_name == self.lock_target.class_name
                and (d.world_position.x - lx) ** 2
                    + (d.world_position.y - ly) ** 2 < match_r2
            ]
            if same:
                # Pick the candidate closest to the locked coordinate, not the
                # most confident one — prevents lock swap when a different
                # mineral of the same class happens to land within the match
                # radius with higher confidence.
                best = min(
                    same,
                    key=lambda d: (d.world_position.x - lx) ** 2
                                  + (d.world_position.y - ly) ** 2,
                )
                self.target = best
                self.last_det_stamp_ns = now_ns
                self.lock_target = best
                self.lock_started_ns = now_ns  # 갱신: 시야 안 들어오면 timeout reset
            elif self.last_odom is not None:
                # 같은 방향인데 더 가까운 같은 종류의 검출 = 같은 광물이고, 고정해 둔
                # 위치의 거리가 길게 잘못 잡혔던 것 (먼 광물의 깊이 오차). 옮겨 잡는다.
                p = self.last_odom.pose.pose.position
                lb = math.atan2(ly - p.y, lx - p.x)
                ld = math.hypot(lx - p.x, ly - p.y)
                best = None
                for d in candidates:
                    if d.class_name != self.lock_target.class_name:
                        continue
                    dx, dy = d.world_position.x - p.x, d.world_position.y - p.y
                    diff = abs((math.atan2(dy, dx) - lb + math.pi) % (2 * math.pi) - math.pi)
                    dist = math.hypot(dx, dy)
                    if diff < math.radians(6.0) and dist < ld and \
                            (best is None or dist < best[0]):
                        best = (dist, d)
                if best is not None:
                    d = best[1]
                    self.get_logger().info(
                        f"고정한 목표 위치 수정: ({lx:.2f},{ly:.2f}) → "
                        f"({d.world_position.x:.2f},{d.world_position.y:.2f}) "
                        f"(같은 방향, {ld - best[0]:.1f} m 더 가까움)")
                    self.target = d
                    self.last_det_stamp_ns = now_ns
                    self.lock_target = d
                    self.lock_started_ns = now_ns
            # 지금 보이는 다른 광물이 잡아 둔 목표보다 뚜렷이 가까우면 그쪽부터 집는다.
            # 원래 목표는 목록(self.known)에 남아 있어 나중에 다시 간다.
            if self.phase == "APPROACH" and self.last_odom is not None \
                    and self.lock_target is not None:
                p = self.last_odom.pose.pose.position
                lx = self.lock_target.world_position.x
                ly = self.lock_target.world_position.y
                d_lock = math.hypot(lx - p.x, ly - p.y)
                others = [d for d in candidates
                          if (d.world_position.x - lx) ** 2
                          + (d.world_position.y - ly) ** 2 >= match_r2]
                if others:
                    near = min(others, key=lambda d: math.hypot(
                        d.world_position.x - p.x, d.world_position.y - p.y))
                    d_near = math.hypot(near.world_position.x - p.x,
                                        near.world_position.y - p.y)
                    if d_near < d_lock - float(
                            self.get_parameter("target_switch_margin_m").value):
                        self.get_logger().info(
                            f"더 가까운 광물이 보임 — 목표를 {self.lock_target.class_name} "
                            f"({lx:.2f},{ly:.2f}) {d_lock:.1f} m → {near.class_name} "
                            f"({near.world_position.x:.2f},{near.world_position.y:.2f}) "
                            f"{d_near:.1f} m 로 바꿈 (원래 목표는 기억해 둠)")
                        self._event(f"더 가까운 {near.class_name} ({d_near:.1f} m) 부터 집음")
                        self.target = near
                        self.lock_target = near
                        self.last_det_stamp_ns = now_ns
                        self.lock_started_ns = now_ns
                        self._appr_best = float("inf")
                        self._appr_progress_ns = now_ns
                        self._appr_started_ns = now_ns
                        self._appr_budget_ns = 0
                        self._appr_unseen_logged = False
            return
        # Lock 없음 — 목표는 _tick 이 기억한 목록에서 가까운 것으로 고른다 (_choose_target).

    # ── 광물이 아닌 것 (건물 벽) ────────────────────────────────────────
    def _is_vetoed(self, x: float, y: float) -> bool:
        r = float(self.get_parameter("mineral_veto_radius_m").value)
        return any(math.hypot(x - vx, y - vy) <= r for vx, vy in self.vetoed)

    def _on_mineral_veto(self, msg: PointStamped) -> None:
        """지도 노드의 판정: 이 자리의 "광물" 은 건물 벽이다. 기억에서 지우고, 가던 중이면 그만둔다."""
        x, y = float(msg.point.x), float(msg.point.y)
        if any(math.hypot(x - vx, y - vy) < 0.3 for vx, vy in self.vetoed):
            return
        self.vetoed.append((x, y))
        r = float(self.get_parameter("mineral_veto_radius_m").value)
        self.known = [k for k in self.known if math.hypot(k["x"] - x, k["y"] - y) > r]
        tgt = self.lock_target or self.target
        if tgt is not None and not self.arm_in_flight and math.hypot(
                tgt.world_position.x - x, tgt.world_position.y - y) <= r:
            self.get_logger().info(
                f"목표 {tgt.class_name} ({tgt.world_position.x:.2f},{tgt.world_position.y:.2f}) "
                f"은 광물이 아님 (건물 벽) — 버리고 EXPLORE 로")
            self._event(f"{msg.header.frame_id or '광물'} 로 본 것이 건물 벽 — 목표에서 뺌")
            self.target = None
            self.lock_target = None
            self.last_det_stamp_ns = 0

    # ── 본 광물 기억 ────────────────────────────────────────────────────
    def _remember_minerals(self, dets) -> None:
        """확인된 검출을 목록에 적는다. 같은 광물이면 더 가까이서 잰 위치로 고친다.

        로버는 광물을 지나가며 잠깐씩만 본다. 예전에는 "지금 보이는 것 중 값진 것" 만
        목표가 될 수 있어서, 돌아가느라 시야에서 벗어난 광물과 다른 것을 집는 동안 본 광물은
        잊혔다. 광물은 움직이지 않으므로 본 자리를 적어 두면 된다.
        (실제 로버라면 위치 추정 오차만큼 적어 둔 좌표가 틀어진다 — 그래서 접근하며 다시
        보일 때 좌표를 고치고, 집을 때는 손목 카메라로 다시 잰다. 그 구조는 여기도 같다.)
        """
        if self.last_odom is None:
            return
        p = self.last_odom.pose.pose.position
        max_r = float(self.get_parameter("mineral_memory_range_m").value)
        for d in dets:
            x, y = float(d.world_position.x), float(d.world_position.y)
            rng = math.hypot(x - p.x, y - p.y)
            if rng > max_r:
                continue
            bearing = math.atan2(y - p.y, x - p.x)
            for k in self.known:
                if k["cls"] != d.class_name:
                    continue
                # 멀리서 잰 것일수록 거리 오차가 크다 → 같은 광물로 보는 반경도 넓게.
                tol = max(1.0, 0.15 * max(rng, k["range"]))
                same = math.hypot(k["x"] - x, k["y"] - y) < tol
                if not same and k["range"] > 6.0:
                    # 멀리서 적어 둔 것과 같은 방향인데 더 가깝게 잡힘 = 같은 광물이고 적어 둔
                    # 거리가 길게 틀렸던 것 (먼 광물은 깊이 화소가 뒤쪽 배경을 읽기도 한다).
                    kb = math.atan2(k["y"] - p.y, k["x"] - p.x)
                    diff = abs((bearing - kb + math.pi) % (2 * math.pi) - math.pi)
                    same = diff < math.radians(6.0) and rng < math.hypot(k["x"] - p.x, k["y"] - p.y)
                if same:
                    if rng <= k["range"] * 1.2:
                        k.update(x=x, y=y, det=d, range=min(rng, k["range"]))
                    break
            else:
                self.known.append({"cls": d.class_name, "x": x, "y": y, "range": rng,
                                   "det": d, "fails": 0})

    def _forget_mineral(self, x: float, y: float, radius: float, failed: bool = False) -> None:
        """(x, y) 근처의 기억을 지운다. failed=True 면 실패 횟수만 올리고 한도를 넘을 때 지운다."""
        limit = int(self.get_parameter("mineral_memory_max_fails").value)
        keep = []
        for k in self.known:
            if math.hypot(k["x"] - x, k["y"] - y) <= radius:
                if not failed:
                    continue
                k["fails"] += 1
                if k["fails"] >= limit:
                    self.get_logger().info(
                        f"광물 기억 삭제: {k['cls']} ({k['x']:.2f},{k['y']:.2f}) — "
                        f"{k['fails']}번 실패")
                    continue
            keep.append(k)
        self.known = keep

    def _publish_known(self) -> None:
        """기억한 (아직 못 모은) 광물을 가까운 순으로 내보낸다 — HUD 지도가 번호와 함께 그린다."""
        import json
        if self.last_odom is None:
            return
        p = self.last_odom.pose.pose.position
        tgt = (self.lock_target or self.target) if self.phase in ("APPROACH", "PICK_READY") \
            else None
        items = sorted(
            ({"c": k["cls"], "x": round(k["x"], 2), "y": round(k["y"], 2),
              "d": round(math.hypot(k["x"] - p.x, k["y"] - p.y), 1), "f": int(k["fails"]),
              # 지금 가고 있는 광물 — HUD 가 고리로 표시한다
              "t": int(tgt is not None and tgt.class_name == k["cls"] and math.hypot(
                  tgt.world_position.x - k["x"], tgt.world_position.y - k["y"]) < 1.0)}
             for k in self.known), key=lambda it: it["d"])
        msg = String()
        msg.data = json.dumps({"items": items}, separators=(",", ":"))
        self.known_pub.publish(msg)

    def _choose_target(self, now_ns: int) -> None:
        """잡아 둔 목표가 없을 때: 기억한 광물 중 가장 가까운 것을 목표로 삼는다."""
        if self.last_odom is None or not self.known:
            return
        p = self.last_odom.pose.pose.position
        skip_r = float(self.get_parameter("post_pick_skip_radius_m").value)
        cooldown_ns = int(float(self.get_parameter("post_pick_cooldown_sec").value) * 1e9)
        self.failed_pick_points = [f for f in self.failed_pick_points if f[2] > now_ns]
        best = None
        for k in self.known:
            if any(math.hypot(k["x"] - fx, k["y"] - fy) <= skip_r
                   for fx, fy, _ in self.failed_pick_points):
                continue
            if self.last_pick_world is not None and now_ns - self.last_pick_done_ns < cooldown_ns \
                    and math.hypot(k["x"] - self.last_pick_world[0],
                                   k["y"] - self.last_pick_world[1]) <= skip_r:
                continue
            if self._is_claimed_by_other(k["x"], k["y"]):
                continue
            d = math.hypot(k["x"] - p.x, k["y"] - p.y)
            if best is None or d < best[0]:
                best = (d, k)
        if best is not None:
            self.target = best[1]["det"]
            self.last_det_stamp_ns = now_ns
            self._chosen_from = len(self.known)     # phase 기록에 "N 개 중" 을 남기려고

    def _on_coverage_cmd(self, msg: Twist) -> None:
        self.last_coverage_cmd = msg
        self.last_coverage_cmd_ns = self.get_clock().now().nanoseconds

    def _on_odom(self, msg: Odometry) -> None:
        self.last_odom = msg
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        p = msg.pose.pose.position
        prev = self._speed_prev
        if prev is None or t < prev[0] or t - prev[0] > 1.0:
            self._speed_prev = (t, p.x, p.y)
        elif t - prev[0] >= 0.1:
            self.speed = math.hypot(p.x - prev[1], p.y - prev[2]) / (t - prev[0])
            self._speed_prev = (t, p.x, p.y)
        q = msg.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        yp = self._yaw_prev
        if yp is None or t < yp[0] or t - yp[0] > 1.0:
            self._yaw_prev = (t, yaw)
        elif t - yp[0] >= 0.1:
            self.yaw_rate = abs((yaw - yp[1] + math.pi) % (2 * math.pi) - math.pi) / (t - yp[0])
            self._yaw_prev = (t, yaw)

    def _on_teleop_cmd(self, msg: Twist) -> None:
        self.last_teleop_cmd = msg
        self.last_teleop_ns = self.get_clock().now().nanoseconds

    def _on_mode(self, msg: String) -> None:
        new_mode = (msg.data or "").strip().upper()
        if new_mode not in ("AUTO", "MANUAL"):
            self.get_logger().warn(
                f"ignored unknown mission mode: {msg.data!r} (expected AUTO/MANUAL)")
            return
        if new_mode == self.mode:
            return
        self.get_logger().info(f"mode: {self.mode} -> {new_mode}")
        self._event(f"mode {self.mode} -> {new_mode}")
        self.mode = new_mode
        # Reset autonomy lock/target when switching to MANUAL so re-engaging
        # AUTO starts from a clean phase.
        if self.mode == "MANUAL":
            self.target = None
            self.lock_target = None
            self.waypoints = []
            self.wp_idx = 0
            self.last_plan_target = None

    def _on_righting(self, msg) -> None:
        if msg.data and not self.righting_active:
            self.get_logger().warn("넘어짐 복구 중 — 주행 정지")
            if self.phase != "RECOVERY":
                self._phase_before_recovery = self.phase
            self._enter_phase("RECOVERY", reason="rover flipped")
        elif not msg.data and self.righting_active:
            self.get_logger().info("넘어짐 복구 끝 — 주행 재개")
            self._leave_recovery()
        self.righting_active = bool(msg.data)
        self._righting_stamp_ns = self.get_clock().now().nanoseconds

    def _leave_recovery(self) -> None:
        """RECOVERY → 하던 일로. 채취 도중이었으면 목표를 버리고 탐색부터 다시."""
        self.righting_status = ""
        if self.phase != "RECOVERY":
            return
        back = self._phase_before_recovery
        if self.estop:
            back = "ESTOP"
        elif back not in ("RETURN_TO_BASE", "MISSION_COMPLETE"):
            back = "EXPLORE"
            self.target = None
            self.lock_target = None
        self.waypoints = []                     # 자세가 바뀌었으니 다시 계획
        self._enter_phase(back, reason="righted")

    def _on_estop(self, _msg: Empty) -> None:
        self._engage_estop()

    def _engage_estop(self) -> str:
        if self.estop:
            return "ESTOP 이미 걸려 있음"
        self.get_logger().warn("ESTOP received — mission halted.")
        self.estop = True
        self.last_error = "ESTOP triggered"
        if self.phase != "ESTOP":
            self._phase_before_estop = self.phase
        # 팔이 움직이는 중이면 그것도 멈춘다 (arm_executor 가 취소를 받아 준다).
        handle = self._arm_goal_handle
        arm_note = ""
        if self.arm_in_flight and handle is not None:
            try:
                handle.cancel_goal_async()
                self._arm_stopped = True
                arm_note = ", 팔도 그 자리에 세움"
            except Exception as e:
                self.get_logger().warn(f"arm goal cancel failed: {e}")
        self.cmd_pub.publish(Twist())
        self._enter_phase("ESTOP", reason="operator ESTOP")
        return "ESTOP 걸림 — 주행 정지" + arm_note

    def _release_estop(self) -> str:
        if not self.estop:
            return "RESUME 무시 — ESTOP 상태가 아님"
        self.estop = False
        self.last_error = ""
        if self.phase == "RECOVERY":         # 넘어짐 복구가 끝나면 _leave_recovery 가 이어 간다
            if self._phase_before_recovery == "ESTOP":
                self._phase_before_recovery = self._phase_before_estop
            return "ESTOP 해제 — 넘어짐 복구가 끝나면 재개"
        back = self._phase_before_estop
        if back not in ("RETURN_TO_BASE", "MISSION_COMPLETE"):
            back = "EXPLORE"                 # 접근·채취 도중이었으면 탐색부터 다시
            self.target = None
            self.lock_target = None
        self.explore_entry_ns = self.get_clock().now().nanoseconds
        self._enter_phase(back, reason="operator RESUME")
        if back == "EXPLORE":
            self.replan_pub.publish(Empty())
        arm_note = ""
        if self._arm_stopped:
            arm_note = " (먼저 팔을 HOME 으로)"
            if not self.arm_in_flight:       # 취소가 이미 끝났으면 지금, 아니면 끝나는 대로
                self._send_arm_home()
        return f"ESTOP 해제 — {back} 로 재개{arm_note}"

    def _send_arm_home(self) -> None:
        """ESTOP 으로 세워 둔 팔을 HOME 으로 되돌린다. 끝날 때까지 로버는 서 있는다."""
        if self.arm_client is None or self.arm_in_flight:
            return
        if not self.arm_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn("arm action server not available — 팔을 HOME 으로 못 보냄")
            self._arm_stopped = False
            return
        goal = ExecuteArmTask.Goal()
        goal.command = "home"
        self.arm_in_flight = True
        self.get_logger().info("arm action sent: home (ESTOP 해제 뒤 복귀)")

        def accepted(future) -> None:
            try:
                handle = future.result()
            except Exception as e:
                self.get_logger().error(f"arm home goal failed: {e}")
                self.arm_in_flight = False
                return
            if not handle.accepted:
                self.arm_in_flight = False
                return
            self._arm_goal_handle = handle
            handle.get_result_async().add_done_callback(done)

        def done(_future) -> None:
            self.arm_in_flight = False
            self._arm_goal_handle = None
            if self.estop:                   # 복귀 도중 다시 세웠다 — 풀리면 또 보낸다
                return
            self._arm_stopped = False
            self.explore_entry_ns = self.get_clock().now().nanoseconds
            self.get_logger().info("arm home done — 주행 재개")
            self._event("팔 HOME 복귀 끝 — 주행 재개")

        self.arm_client.send_goal_async(goal).add_done_callback(accepted)

    # ── 운영자 명령 (HUD) ───────────────────────────────────────────────
    def _event(self, text: str) -> None:
        m = String()
        m.data = text
        self.event_pub.publish(m)

    def _on_command(self, msg: String) -> None:
        cmd = (msg.data or "").strip().upper()
        handlers = {
            "ESTOP": self._engage_estop, "RESUME": self._release_estop,
            "RTB": self._cmd_rtb, "RTB_CANCEL": self._cmd_rtb_cancel,
            "RESET": self._cmd_reset, "REPLAN": self._cmd_replan, "SKIP": self._cmd_skip,
        }
        if cmd not in handlers:
            self.get_logger().warn(f"operator command 무시 (모르는 명령): {msg.data!r}")
            self._event(f"command {msg.data!r} 무시 — 모르는 명령")
            return
        result = handlers[cmd]()
        self.get_logger().info(f"operator command {cmd}: {result}")
        self._event(f"command {cmd}: {result}")
        self._publish_state()

    def _cmd_rtb(self) -> str:
        if self.estop:
            return "거부 — ESTOP 상태 (먼저 RESUME)"
        if self.phase == "MISSION_COMPLETE":
            return "거부 — 이미 임무 완료"
        if self.phase == "RETURN_TO_BASE":
            return "이미 복귀 중"
        if self.arm_in_flight:
            return "거부 — 채취 동작 중 (끝난 뒤 다시)"
        self.operator_rtb = True
        self.target = None
        self.lock_target = None
        self._enter_phase("RETURN_TO_BASE", reason="operator RTB")
        return f"기지 복귀 시작 (채집 {self.collected_count}개)"

    def _cmd_rtb_cancel(self) -> str:
        if self.phase != "RETURN_TO_BASE":
            return "무시 — 복귀 중이 아님"
        goal = int(self.get_parameter("collection_goal").value)
        if (goal >= 0 and self.collected_count >= goal) or self.battery_critical:
            return "거부 — 목표 달성/배터리 때문에 하는 복귀는 취소할 수 없음"
        self.operator_rtb = False
        self.explore_entry_ns = self.get_clock().now().nanoseconds
        self._enter_phase("EXPLORE", reason="operator RTB cancel")
        self.replan_pub.publish(Empty())
        return "복귀 취소 — 탐색 재개"

    def _cmd_reset(self) -> str:
        if self.arm_in_flight:
            return "거부 — 채취 동작 중"
        before = self.collected_count
        self.collected_count = 0
        self.cargo_count = 0
        for k in self.collected_by_class:
            self.collected_by_class[k] = 0
        self.failed_pick_points = []
        self.known = []
        self.vetoed = []
        self.last_error = "" if not self.estop else self.last_error
        self.operator_rtb = False
        restarted = ""
        if not self.estop and self.phase in ("MISSION_COMPLETE", "RETURN_TO_BASE"):
            self.target = None
            self.lock_target = None
            self.explore_entry_ns = self.get_clock().now().nanoseconds
            self._enter_phase("EXPLORE", reason="operator RESET")
            self.replan_pub.publish(Empty())
            restarted = ", 임무 다시 시작"
        return f"채집 수 {before} → 0{restarted}"

    def _cmd_replan(self) -> str:
        self.waypoints = []
        self.wp_idx = 0
        self.last_plan_target = None
        self.last_plan_ns = 0
        self.replan_pub.publish(Empty())
        return f"경로 다시 계획 ({self.phase})"

    def _cmd_skip(self) -> str:
        if self.arm_in_flight:
            return "거부 — 채취 동작 중"
        tgt = self.lock_target or self.target
        if tgt is None or self.phase not in ("APPROACH", "PICK_READY"):
            return "무시 — 접근 중인 목표가 없음"
        now_ns = self.get_clock().now().nanoseconds
        ignore_sec = float(self.get_parameter("failed_pick_ignore_sec").value)
        x, y = float(tgt.world_position.x), float(tgt.world_position.y)
        self.failed_pick_points.append((x, y, now_ns + int(ignore_sec * 1e9)))
        self.target = None
        self.lock_target = None
        self.lock_started_ns = 0
        self.last_det_stamp_ns = 0
        return f"목표 {tgt.class_name} ({x:.1f}, {y:.1f}) 를 {ignore_sec:.0f} s 동안 건너뜀"

    # ── 기울기 반사 ─────────────────────────────────────────────────────
    def _tilt_guard(self) -> bool:
        """앞이 들리거나 옆으로 크게 기울면 잠깐 물러난다. True = 이번 tick 의 명령을 냈다.

        시간은 odom 의 시각(시뮬레이션 시각)으로 잰다 — 시뮬레이터가 실시간보다 느려도
        물러나는 거리가 같다.
        """
        odom = self.last_odom
        if odom is None or not bool(self.get_parameter("tilt_guard").value):
            self._guard_until = None
            return False
        t = odom.header.stamp.sec + odom.header.stamp.nanosec * 1e-9
        if self._guard_until is not None:
            p = odom.pose.pose.position
            along = ((p.x - self._guard_p0[0]) * self._guard_fwd[0]
                     + (p.y - self._guard_p0[1]) * self._guard_fwd[1])     # 가려던 쪽이 +
            self._guard_far = max(self._guard_far, along)
            retreated = self._guard_far - along
            if self._guard_start <= t < self._guard_until \
                    and retreated < float(self.get_parameter("tilt_guard_back_m").value):
                back = Twist()
                back.linear.x = self._guard_dir * float(
                    self.get_parameter("tilt_guard_back_mps").value)
                self._guard_pub.publish(back)        # last_dir 를 바꾸지 않게 안쪽 publisher 로
                return True
            self._guard_until = None                 # 다 물러났다 (또는 시간 초과·시각 되감김)
            self._after_guard(f"앞으로 {self._guard_far:.2f} m 더 간 뒤 {retreated:.2f} m 물러남, "
                              f"{t - self._guard_start:.1f} s")
            self.cmd_pub.publish(Twist())
            return True
        if self.phase == "PICK_READY":               # 서서 팔을 쓰는 중
            return False
        nose, roll = nose_up_and_roll_deg(odom.pose.pose.orientation)
        lim_p = float(self.get_parameter("tilt_guard_pitch_deg").value)
        lim_r = float(self.get_parameter("tilt_guard_roll_deg").value)
        over = max(nose - lim_p, abs(roll) - lim_r)          # 문턱을 넘은 정도 (도)
        if over < -3.0:
            self._guard_chain, self._guard_peak = 0, 0.0     # 기울기가 풀렸다 → 다시 장전
        if over > 0.0 and self._guard_chain >= int(self.get_parameter("tilt_guard_chain_max").value):
            if over < self._guard_peak + 5.0:
                return False            # 이미 연달아 물러났다 — 더 나빠지지 않는 한 계획기에 맡긴다
        if nose > lim_p:
            # 앞이 들렸다 = 내리막은 뒤쪽. 어느 쪽으로 가던 중이든 뒤로 내려온다.
            self._guard_dir, self._guard_why = -1.0, f"앞이 {nose:.0f}° 들림"
        elif abs(roll) > lim_r:
            # 옆으로 기울었다 → 온 길로 되돌아간다.
            self._guard_dir = -self.cmd_pub.last_dir
            self._guard_why = f"옆으로 {abs(roll):.0f}° 기움"
        else:
            return False
        self._guard_start = t
        self._guard_until = t + float(self.get_parameter("tilt_guard_back_sec").value)
        self._guard_count += 1
        self._guard_chain += 1
        self._guard_peak = max(self._guard_peak, over)
        if self._guard_chain == int(self.get_parameter("tilt_guard_chain_max").value):
            self.get_logger().warn(
                f"tilt guard: 연달아 {self._guard_chain}번 물러남 — 기울기가 풀리거나 "
                "5° 더 기울 때까지 반사를 쉬고 계획기에 맡긴다")
        # 가려던 쪽(물러나는 방향의 반대)의 바로 앞을 위험으로 표시 → 계획기가 돌아간다.
        q = odom.pose.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self._guard_p0 = (float(odom.pose.pose.position.x), float(odom.pose.pose.position.y))
        self._guard_fwd = (-self._guard_dir * math.cos(yaw), -self._guard_dir * math.sin(yaw))
        self._guard_far = 0.0
        d = -self._guard_dir * float(self.get_parameter("tilt_guard_mark_ahead_m").value)
        mark = PointStamped()
        mark.header.stamp = odom.header.stamp
        mark.header.frame_id = "world"
        mark.point.x = float(odom.pose.pose.position.x + d * math.cos(yaw))
        mark.point.y = float(odom.pose.pose.position.y + d * math.sin(yaw))
        self.hazard_mark_pub.publish(mark)
        self._guard_mark = (mark.point.x, mark.point.y)
        self.get_logger().warn(
            f"tilt guard #{self._guard_count}: {self._guard_why} "
            f"(nose {nose:+.0f}°, roll {roll:+.0f}°, phase {self.phase}) at "
            f"({odom.pose.pose.position.x:.2f},{odom.pose.pose.position.y:.2f}) → "
            f"{'후진' if self._guard_dir < 0 else '전진'}으로 물러남, "
            f"({mark.point.x:.2f},{mark.point.y:.2f}) 위험 표시")
        self._event(f"기울기 반사: {self._guard_why} → 물러나서 다른 길로")
        back = Twist()
        back.linear.x = self._guard_dir * float(self.get_parameter("tilt_guard_back_mps").value)
        self._guard_pub.publish(back)
        return True

    def _after_guard(self, moved: str = "") -> None:
        """다 물러난 뒤: 경로를 버리고 다시 계획. 같은 목표에서 거듭 걸리면 그 목표를 건너뛴다."""
        note = self._cmd_replan()
        now_ns = self.get_clock().now().nanoseconds
        if self.phase == "EXPLORE":
            # coverage 가 새 경로를 낼 때까지 예전 명령을 따라가지 않는다.
            self.last_coverage_cmd = None
            self.last_coverage_cmd_ns = 0
            self.explore_entry_ns = now_ns
        tgt = self.lock_target or self.target
        if tgt is not None and self.phase == "APPROACH":
            x, y = float(tgt.world_position.x), float(tgt.world_position.y)
            prev = self._guard_target
            n = prev[2] + 1 if prev and math.hypot(prev[0] - x, prev[1] - y) < 1.0 else 1
            self._guard_target = (x, y, n)
            if n >= int(self.get_parameter("tilt_guard_skip_after").value):
                self._guard_target = None
                note = self._cmd_skip() + f" (기울기 반사 {n}회)"
                self._event(note)
        self.get_logger().info(f"tilt guard: 물러남 끝 ({moved}) — {note}")

    def _on_battery(self, msg: BatteryState) -> None:
        self.battery_percent = float(msg.percentage)
        self.battery_low = bool(msg.is_low)
        was_critical = self.battery_critical
        self.battery_critical = bool(msg.is_critical)
        if self.battery_critical and not was_critical:
            self.get_logger().warn(
                f"battery critical ({self.battery_percent:.1f}%) — RTB will engage.")

    def _tick(self) -> None:
        # 1. Hard overrides (highest priority): ESTOP / MANUAL / MISSION_COMPLETE.
        if self.estop or self.righting_active or self.mode == "MANUAL" or self.arm_in_flight:
            self._guard_until = None        # 반사 도중이었으면 버린다 (나중에 이어서 물러나지 않게)
        if self.estop:
            self.cmd_pub.publish(Twist())
            return

        if self.righting_active:
            # 복구 노드가 죽어 신호가 끊긴 경우(3 s)에는 풀어 준다.
            if self.get_clock().now().nanoseconds - self._righting_stamp_ns > int(3e9):
                self.righting_active = False
                self._leave_recovery()
            else:
                self.cmd_pub.publish(Twist())
                return

        if self.mode == "MANUAL":
            # Phase stays where it was (for telemetry) but we don't transition.
            # Forward fresh teleop cmd; stale -> zero twist.
            stale_sec = float(self.get_parameter("teleop_stale_sec").value)
            age = (self.get_clock().now().nanoseconds - self.last_teleop_ns) / 1e9 \
                if self.last_teleop_ns else float("inf")
            if self.last_teleop_cmd is not None and age < stale_sec:
                self.cmd_pub.publish(self.last_teleop_cmd)
            else:
                self.cmd_pub.publish(Twist())
            return

        if self.phase == "MISSION_COMPLETE":
            self.cmd_pub.publish(Twist())
            return

        # During arm action, hold rover stopped and freeze FSM transitions.
        if self.arm_in_flight:
            self.cmd_pub.publish(Twist())
            return

        # 기울기 반사: 물러나는 동안에는 계획·접근 명령을 내지 않는다.
        if self._tilt_guard():
            return

        # 2. Mission termination gates (AUTO only): collection goal / battery.
        # goal=0 은 "spawn 후 즉시 RTB" 모드 (빠른 베이스캠프 도달 시연).
        # 음수 goal 은 "infinite EXPLORE" (RTB by goal 비활성).
        goal = int(self.get_parameter("collection_goal").value)
        if self.phase != "RETURN_TO_BASE" and (
            (goal >= 0 and self.collected_count >= goal)
            or self.battery_critical
        ):
            why = (f"goal reached ({self.collected_count}/{goal})"
                   if self.collected_count >= goal
                   else f"battery critical ({self.battery_percent:.1f}%)")
            self._enter_phase("RETURN_TO_BASE", reason=why)

        # 3. RTB drives toward basecamp; arrival -> MISSION_COMPLETE.
        if self.phase == "RETURN_TO_BASE":
            self._drive_to_basecamp()
            return

        now_ns = self.get_clock().now().nanoseconds
        stale_ns = int(float(self.get_parameter("detection_stale_sec").value) * 1e9)
        if self.lock_target is None and self.phase == "EXPLORE":
            self._choose_target(now_ns)
        det_fresh = (self.target is not None) and (now_ns - self.last_det_stamp_ns < stale_ns)

        # 고정한 목표는 안 보여도 기억한 위치로 계속 간다 (광물은 움직이지 않는다).
        # 포기는 시간이 아니라 진척으로 정한다 — _approach_progress.
        if not det_fresh and self.lock_target is not None:
            self.target = self.lock_target
            det_fresh = True
            if (not self._appr_unseen_logged and self.phase == "APPROACH"
                    and now_ns - self.lock_started_ns > int(8e9)):
                self._appr_unseen_logged = True
                self.get_logger().info(
                    f"목표 {self.target.class_name} 이 8 s 째 안 보임 — "
                    f"기억한 위치 ({self.target.world_position.x:.2f},"
                    f"{self.target.world_position.y:.2f}) 로 계속 접근")
                self._event("목표가 안 보임 — 기억한 위치로 계속 접근")

        engage_d = float(self.get_parameter("approach_engage_dist_m").value)
        default_stop_d = float(self.get_parameter("approach_stop_dist_m").value)
        target_cls = self.target.class_name if self.target is not None else ""
        stop_d = _stop_dist_for(target_cls, default_stop_d)
        resume_d = stop_d + float(self.get_parameter("approach_resume_buffer_m").value)

        dist = self._distance_to_target() if det_fresh else float("inf")

        prev_phase = self.phase
        if not det_fresh:
            self.phase = "EXPLORE"
        elif prev_phase == "PICK_READY":
            # Hysteresis: stay PICK_READY until rover drifts past resume_d
            self.phase = "PICK_READY" if dist <= resume_d else "APPROACH"
        else:
            if dist <= stop_d:
                self.phase = "PICK_READY"
            elif dist <= engage_d or (prev_phase == "APPROACH"
                                      and self.lock_target is not None):
                self.phase = "APPROACH"         # 돌아가느라 멀어져도 잡은 목표는 놓지 않는다
            else:
                self.phase = "EXPLORE"

        if self.phase != prev_phase:
            self.get_logger().info(
                f"phase: {prev_phase} -> {self.phase}"
                + (f" (target {self.target.class_name} world=("
                   f"{self.target.world_position.x:.2f},"
                   f"{self.target.world_position.y:.2f}), dist={dist:.2f}m, "
                   f"stop_d={stop_d:.2f}m)"
                   if det_fresh else ""))
            self._event(f"phase {prev_phase} -> {self.phase}"
                        + (f" ({self.target.class_name} {dist:.1f} m)" if det_fresh else ""))
            if self.phase != "APPROACH":
                self.waypoints = []
                self.wp_idx = 0
                self.last_plan_target = None
                self.plan_goal_gap = 0.0
            if self.phase == "APPROACH" and prev_phase != "APPROACH":
                # Lock onto this target's coordinates for the duration of APPROACH
                self.lock_target = self.target
                self.lock_started_ns = now_ns
                self._appr_best = float("inf")
                self._appr_progress_ns = now_ns
                self._appr_started_ns = now_ns
                self._appr_budget_ns = 0
                self._appr_unseen_logged = False
                if self._chosen_from >= 2:
                    self.get_logger().info(
                        f"기억한 광물 {self._chosen_from}개 중 가장 가까운 "
                        f"{self.target.class_name} ({dist:.1f} m) 부터 감")
                    self._event(f"기억한 광물 {self._chosen_from}개 중 가장 가까운 것부터")
            if self.phase != "APPROACH" and self.phase != "PICK_READY":
                self.lock_target = None
                self.lock_started_ns = 0
            if self.phase == "PICK_READY" and prev_phase != "PICK_READY":
                self._pick_pending = True       # 멈춘 것을 확인하고 보낸다 (아래 PICK_READY)
                self._pick_adjust = ""
            elif self.phase != "PICK_READY":
                self._pick_pending = False
            if self.phase == "EXPLORE" and prev_phase != "EXPLORE":
                # Drop stale coverage cmd, ask coverage to replan from the new
                # rover position, and start a brief cooldown so the replan can
                # complete before we start passing through its cmd_vel.
                self.last_coverage_cmd = None
                self.last_coverage_cmd_ns = 0
                self.explore_entry_ns = now_ns
                self.replan_pub.publish(Empty())

        if self.phase == "EXPLORE":
            # Hold zero twist briefly so coverage can publish a fresh path
            # for the post-APPROACH/PICK rover position.
            resume_delay_ns = int(
                float(self.get_parameter("explore_resume_delay_sec").value) * 1e9)
            in_resume_cooldown = (
                self.explore_entry_ns > 0
                and (now_ns - self.explore_entry_ns) < resume_delay_ns)
            stale_sec = float(self.get_parameter("coverage_cmd_stale_sec").value)
            cmd_age = (now_ns - self.last_coverage_cmd_ns) / 1e9 \
                if self.last_coverage_cmd_ns else float("inf")
            if in_resume_cooldown or now_ns < self._look_stop_until_ns:
                self.cmd_pub.publish(Twist())
            elif self.last_coverage_cmd is not None and cmd_age < stale_sec:
                self.cmd_pub.publish(self.last_coverage_cmd)
            else:
                self.cmd_pub.publish(Twist())
            # EXPLORE: supervisor 는 매 tick 발행 안 함 — coverage 의 /mission/
            # path · /mission/markers 가 dashboard 의 path/target source.
            # 빈 path 를 20Hz 로 쏘면 coverage 의 3Hz 발행을 덮어쓰는 race
            # condition 발생. EXPLORE 진입 시점에 _enter_phase 가 한 번만
            # 청소용 빈 path 발행 → 그 후엔 coverage 가 takeover.
            self.waypoints = []
        elif self.phase == "APPROACH":
            twist = self._approach_twist(dist)
            if self.target is not None and not self._approach_progress(now_ns):
                twist = Twist()
            self.cmd_pub.publish(twist)
            # _maybe_replan 후 waypoints + target 발행.
            tgt = (self.target.world_position.x, self.target.world_position.y) \
                  if self.target is not None else None
            self._publish_supervisor_viz(target_xy=tgt)
        else:  # PICK_READY
            self.cmd_pub.publish(self._pick_ready_twist(dist))
            tgt = (self.target.world_position.x, self.target.world_position.y) \
                  if self.target is not None else None
            self._publish_supervisor_viz(target_xy=tgt)

    def _distance_to_target(self) -> float:
        if self.target is None or self.last_odom is None:
            return float("inf")
        rx = self.last_odom.pose.pose.position.x
        ry = self.last_odom.pose.pose.position.y
        tx = self.target.world_position.x
        ty = self.target.world_position.y
        return math.hypot(tx - rx, ty - ry)

    def _approach_twist(self, dist: float) -> Twist:
        t = Twist()
        if self.target is None or self.last_odom is None:
            return t

        rx = self.last_odom.pose.pose.position.x
        ry = self.last_odom.pose.pose.position.y
        tx = self.target.world_position.x
        ty = self.target.world_position.y

        self._maybe_replan(rx, ry, tx, ty)

        if self.waypoints:
            wp_reach = float(self.get_parameter("waypoint_reach_dist_m").value)
            while self.wp_idx < len(self.waypoints) - 1:
                wx, wy = self.waypoints[self.wp_idx]
                if math.hypot(wx - rx, wy - ry) < wp_reach:
                    self.wp_idx += 1
                else:
                    break
            goal_x, goal_y = self.waypoints[self.wp_idx]
            stop_d = _stop_dist_for(
                self.target.class_name,
                float(self.get_parameter("approach_stop_dist_m").value))
            if (self.wp_idx == len(self.waypoints) - 1
                    and math.hypot(goal_x - rx, goal_y - ry) < wp_reach
                    and dist > stop_d + 0.3):
                # 계획의 끝에 왔는데 광물은 아직 멀다 = 계획이 목표에 못 미친 곳에서
                # 끝났다 (목표 주변이 실제로 막혀 있음). 발밑의 점을 향해 조향하면
                # 제자리에서 좌우로 맴돌기만 하므로, 이 광물은 한동안 포기한다.
                ignore_ns = int(float(
                    self.get_parameter("failed_pick_ignore_sec").value) * 1e9)
                self.failed_pick_points.append(
                    (tx, ty, self.get_clock().now().nanoseconds + ignore_ns))
                self._forget_mineral(tx, ty, 1.0, failed=True)
                self.get_logger().warn(
                    f"APPROACH: 광물 ({tx:.2f},{ty:.2f}) 까지 {dist:.2f} m 남기고 "
                    f"길이 막힘 — 이 광물은 잠시 제외하고 EXPLORE 로")
                self.target = None
                self.lock_target = None
                self.last_det_stamp_ns = 0
                return Twist()
        else:
            # A* 실패 — 이 mineral 까지 obstacle 회피 path 없음. 직진하면
            # obstacle 통과 위험. 정지 + target/lock 폐기로 다음 tick 에
            # det_fresh=False → 자동 EXPLORE 복귀. 같은 mineral 이 다시
            # 감지되면 rover 위치가 바뀐 상태에서 재시도 가능.
            self.get_logger().warn(
                f"APPROACH A* failed for target ({tx:.2f},{ty:.2f}) — "
                f"dropping target, returning to EXPLORE",
                throttle_duration_sec=2.0)
            self.target = None
            self.lock_target = None
            self.last_det_stamp_ns = 0
            return Twist()

        q = self.last_odom.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        bearing = math.atan2(goal_y - ry, goal_x - rx)
        heading_err = math.atan2(math.sin(bearing - yaw), math.cos(bearing - yaw))

        lin = self._approach_speed(dist)
        if abs(heading_err) > 0.4:
            lin *= 0.3
        t.linear.x = lin
        t.angular.z = float(self.get_parameter("steer_gain").value) * heading_err
        max_ang = 1.2
        if abs(heading_err) > 0.4:
            max_ang = min(max_ang, turn_rate_cap(heading_err))
        if t.angular.z > max_ang:
            t.angular.z = max_ang
        elif t.angular.z < -max_ang:
            t.angular.z = -max_ang
        return t

    def _approach_speed(self, dist: float) -> float:
        """목표까지 dist 남았을 때의 전진 속도 — 멈출 자리를 지나치지 않는 한에서 빠르게.

        예전에는 "1.5 m 안이면 0.25 m/s, 밖이면 최고 속도" 의 두 단계였다. 화성 구동부는
        0.72 m/s² 로만 줄일 수 있고 명령이 바퀴에 닿기까지도 시간이 걸려서, 최고 속도로
        들어오면 느린 구간(0.8 m)이 끝나기 전에 속도가 다 줄지 않았다. 내리막이면 더 길어져
        멈출 자리를 0.5 m 지나쳐 광물을 깔고 섰다 (logs/min2_on t=409, blue_mineral_0024).

        지금은 제동 곡선이다: 멈출 자리의 creep_lead 앞에서 느린 속도가 되도록
        v = sqrt(v_creep² + 2·a·남은 거리), 남은 거리에서는 명령 지연 동안 갈 거리를 뺀다.
        """
        stop_d = _stop_dist_for(self.target.class_name if self.target is not None else "",
                                float(self.get_parameter("approach_stop_dist_m").value))
        v_max = float(self.get_parameter("approach_lin_speed").value)
        v_creep = float(self.get_parameter("approach_creep_speed").value)
        a = float(self.get_parameter("approach_brake_decel_m_s2").value)
        room = (dist - stop_d - float(self.get_parameter("approach_creep_lead_m").value)
                - self.speed * float(self.get_parameter("approach_brake_delay_sec").value))
        if room <= 0.0:
            return min(v_creep, v_max)
        return min(v_max, math.sqrt(v_creep * v_creep + 2.0 * a * room))

    def _pick_ready_twist(self, dist: float) -> Twist:
        """PICK_READY: 완전히 멈추고, 팔이 닿는 거리인지 확인한 뒤 팔에 명령을 보낸다.

        멈추라는 명령을 낸 뒤에도 로버는 조금 움직인다 (내리막이면 더 가고, 오르막이면 뒤로
        밀린다). 광물이 받침대 바로 앞·아래로 들어오면 손목 카메라에 보이지 않아 "그 자리에
        광물이 없음" 으로, 0.72 m 보다 멀면 "팔이 닿지 않음" 으로 실패한다. 그래서 선 뒤에
        거리를 다시 보고, 너무 가까우면 물러나고 멀면 다가가서 맞춘다.
        """
        t = Twist()
        if not self._pick_pending or self.arm_in_flight:
            return t
        stop_d = _stop_dist_for(self.target.class_name if self.target is not None else "",
                                float(self.get_parameter("approach_stop_dist_m").value))
        v_adj = float(self.get_parameter("pick_backoff_speed").value)
        if self._pick_adjust == "back" and dist >= stop_d - 0.1:
            self._pick_adjust = ""
        elif self._pick_adjust == "fwd" and dist <= stop_d:
            self._pick_adjust = ""
        if not self._pick_adjust:
            if dist < float(self.get_parameter("pick_min_dist_m").value):
                self._pick_adjust = "back"
                self.get_logger().warn(
                    f"PICK_READY: 광물을 지나쳐 {dist:.2f} m 까지 붙음 — "
                    f"{stop_d - 0.1:.2f} m 가 될 때까지 물러남")
                self._event(f"광물에 너무 붙음 ({dist:.2f} m) — 물러나서 집기")
            elif self.speed > 0.03:
                return t                                # 아직 구르는 중 — 설 때까지 기다림
            elif dist > stop_d + float(self.get_parameter("pick_reach_slack_m").value):
                self._pick_adjust = "fwd"
                self.get_logger().info(
                    f"PICK_READY: 멈춘 자리가 {dist:.2f} m 로 멀다 — {stop_d:.2f} m 까지 다가감")
        if self._pick_adjust:
            t.linear.x = -v_adj if self._pick_adjust == "back" else v_adj
            return t
        self._send_arm_goal()
        self._pick_pending = not self.arm_in_flight     # 팔 서버가 없으면 다음 tick 에 다시
        return t

    def _approach_remaining(self) -> float:
        """로버에서 목표까지 남은 길 (계획한 경로를 따라, m). 경로가 없으면 직선 거리."""
        if self.last_odom is None or self.target is None:
            return float("inf")
        p = self.last_odom.pose.pose.position
        x, y = float(p.x), float(p.y)
        total = 0.0
        for wx, wy in self.waypoints[self.wp_idx:]:
            total += math.hypot(wx - x, wy - y)
            x, y = wx, wy
        return total + math.hypot(self.target.world_position.x - x,
                                  self.target.world_position.y - y)

    def _looked_and_absent(self, now_ns: int) -> bool:
        """기억한 자리를 가까이서 정면으로 보고 있는데 광물이 검출되지 않는가."""
        if self.target is None or self.last_odom is None:
            self._appr_look_ns = 0
            return False
        pose = self.last_odom.pose.pose
        q = pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        dx = self.target.world_position.x - pose.position.x
        dy = self.target.world_position.y - pose.position.y
        dist = math.hypot(dx, dy)
        off = abs((math.atan2(dy, dx) - yaw + math.pi) % (2 * math.pi) - math.pi)
        looking = (float(self.get_parameter("mineral_look_min_m").value) <= dist
                   <= float(self.get_parameter("mineral_look_max_m").value)
                   and off <= math.radians(float(self.get_parameter("mineral_look_cone_deg").value)))
        if not looking or self.last_det_stamp_ns >= self._appr_look_ns > 0:
            self._appr_look_ns = now_ns if looking else 0
            return False
        if self._appr_look_ns == 0:
            self._appr_look_ns = now_ns
            return False
        return now_ns - self._appr_look_ns > int(
            float(self.get_parameter("mineral_look_sec").value) * 1e9)

    def _approach_progress(self, now_ns: int) -> bool:
        """접근이 진척되고 있는가. 아니면 목표를 한동안 제외하고 False (→ EXPLORE)."""
        rem = self._approach_remaining()
        if not math.isfinite(rem):
            return True
        if self._appr_budget_ns == 0:
            self._appr_budget_ns = int(1e9 * (
                float(self.get_parameter("approach_budget_base_sec").value)
                + rem * float(self.get_parameter("approach_budget_sec_per_m").value)))
        step = float(self.get_parameter("approach_stall_progress_m").value)
        if rem < self._appr_best - step:
            self._appr_best = rem
            self._appr_progress_ns = now_ns
        elif rem > self._appr_best + 1.5:
            self._appr_best = rem          # 다시 계획해서 길이 길어짐 — 새 길 기준으로 본다
        if self._looked_and_absent(now_ns):
            tx, ty = self.target.world_position.x, self.target.world_position.y
            self._forget_mineral(tx, ty, 1.0)
            self.get_logger().warn(
                f"기억한 자리에 광물이 없음: {self.target.class_name} ({tx:.2f},{ty:.2f}) — "
                f"가까이서 그쪽을 봐도 검출이 없다. 기억에서 지우고 EXPLORE 로")
            self._event("기억한 자리에 광물이 안 보임 — 지움")
            self.target = None
            self.lock_target = None
            self.last_det_stamp_ns = 0
            return False
        stall_ns = int(float(self.get_parameter("approach_stall_sec").value) * 1e9)
        why = ""
        if now_ns - self._appr_progress_ns > stall_ns:
            why = f"{stall_ns / 1e9:.0f} s 동안 가까워지지 못함"
        elif now_ns - self._appr_started_ns > self._appr_budget_ns:
            why = f"접근 시간 {self._appr_budget_ns / 1e9:.0f} s 초과"
        if not why:
            return True
        tx, ty = self.target.world_position.x, self.target.world_position.y
        ignore_ns = int(float(self.get_parameter("failed_pick_ignore_sec").value) * 1e9)
        self.failed_pick_points.append((tx, ty, now_ns + ignore_ns))
        self._forget_mineral(tx, ty, 1.0, failed=True)
        self.get_logger().warn(
            f"APPROACH 포기: {self.target.class_name} ({tx:.2f},{ty:.2f}) — {why} "
            f"(남은 길 {rem:.1f} m). 이 광물은 잠시 제외하고 EXPLORE 로")
        self._event(f"접근 포기 — {why}")
        self.target = None
        self.lock_target = None
        self.last_det_stamp_ns = 0
        return False

    # ── Multi-rover 협조 메서드 (T2 패턴, 838d5ed) ─────────────────────
    def _on_claim(self, msg: PointStamped) -> None:
        """다른 rover 의 mineral claim 수신. frame_id == rover_id."""
        rid = (msg.header.frame_id or "").strip()
        if not rid or rid == self._my_rover_id:
            return
        now_ns = self.get_clock().now().nanoseconds
        self.other_claims[rid] = (
            float(msg.point.x), float(msg.point.y), now_ns)

    def _publish_claim(self) -> None:
        """phase 가 APPROACH/PICK_READY 일 때 자신 target XY publish."""
        if self.claim_pub is None:
            return
        if self.phase not in ("APPROACH", "PICK_READY"):
            return
        if self.target is None:
            return
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._my_rover_id
        msg.point.x = float(self.target.world_position.x)
        msg.point.y = float(self.target.world_position.y)
        msg.point.z = 0.0
        self.claim_pub.publish(msg)

    def _on_rover_position(self, msg: PointStamped) -> None:
        """다른 rover 의 odom 위치 수신. frame_id == rover_id."""
        rid = (msg.header.frame_id or "").strip()
        if not rid or rid == self._my_rover_id:
            return
        now_ns = self.get_clock().now().nanoseconds
        self.other_rovers[rid] = (
            float(msg.point.x), float(msg.point.y), now_ns)

    def _publish_rover_position(self) -> None:
        """자신 odom XY 를 /rover_positions 에 publish (frame_id=self id)."""
        if self.rover_pos_pub is None or self.last_odom is None:
            return
        p = self.last_odom.pose.pose.position
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._my_rover_id
        msg.point.x = float(p.x)
        msg.point.y = float(p.y)
        msg.point.z = 0.0
        self.rover_pos_pub.publish(msg)

    def _other_rovers_alive(self) -> dict:
        """TTL 안 살아있는 다른 rover 만 반환 {rid: (x, y)}."""
        ttl_ns = int(float(
            self.get_parameter("rover_position_ttl_sec").value) * 1e9)
        now_ns = self.get_clock().now().nanoseconds
        alive = {}
        for rid, (rx, ry, ts) in list(self.other_rovers.items()):
            if now_ns - ts > ttl_ns:
                self.other_rovers.pop(rid, None)
                continue
            alive[rid] = (rx, ry)
        return alive

    def _build_dynamic_grid(self, base=None):
        """static obstacle_grid + 다른 rover 위치를 inflate 한 snapshot grid.

        enable_rover_avoid=False 거나 다른 rover 없으면 static grid 그대로.
        base: 기준 격자 (없으면 self.ogrid.grid).
        """
        if self.ogrid is None:
            return None
        if base is None:
            base = self.ogrid.grid
        if self.rover_pos_pub is None:
            return base
        others = self._other_rovers_alive()
        if not others:
            return base
        import numpy as np
        g = base.copy()
        r_m = float(self.get_parameter("rover_avoid_radius_m").value)
        cells_r = max(1, int(math.ceil(r_m / self.ogrid.cell_size)))
        cells_r2 = cells_r * cells_r
        for rid, (rx, ry) in others.items():
            cell = self.ogrid.world_to_cell(rx, ry, clip=True)
            if cell is None:
                continue
            i0, j0 = cell
            for di in range(-cells_r, cells_r + 1):
                for dj in range(-cells_r, cells_r + 1):
                    if di * di + dj * dj > cells_r2:
                        continue
                    ni, nj = i0 + di, j0 + dj
                    if self.ogrid.in_bounds(ni, nj):
                        g[ni, nj] = 1
        return g

    def _other_rovers_moved(self) -> bool:
        """다른 rover 가 마지막 plan 후 trigger_m 이상 움직였으면 True."""
        if self.rover_pos_pub is None:
            return False
        trigger = float(self.get_parameter("rover_replan_trigger_m").value)
        trigger2 = trigger * trigger
        cur = self._other_rovers_alive()
        prev = self._last_plan_other_rovers
        if set(cur.keys()) != set(prev.keys()):
            return True
        for rid, (cx, cy) in cur.items():
            px, py = prev.get(rid, (cx, cy))
            if (cx - px) ** 2 + (cy - py) ** 2 > trigger2:
                return True
        return False

    def _is_claimed_by_other(self, det_x: float, det_y: float) -> bool:
        """다른 rover 가 ttl 안에 claim 한 mineral 좌표 근처면 True."""
        if not self.other_claims:
            return False
        skip_r = float(self.get_parameter("claim_skip_radius_m").value)
        ttl_ns = int(float(self.get_parameter("claim_ttl_sec").value) * 1e9)
        now_ns = self.get_clock().now().nanoseconds
        skip_r2 = skip_r * skip_r
        for rid, (cx, cy, ts) in list(self.other_claims.items()):
            if now_ns - ts > ttl_ns:
                self.other_claims.pop(rid, None)
                continue
            dx = det_x - cx
            dy = det_y - cy
            if dx * dx + dy * dy < skip_r2:
                return True
        return False

    def _on_hazard_map(self, msg) -> None:
        """새 위험 지도 → 계획용 격자 갱신. 가던 경로가 막혔으면 버리고 다시 짠다."""
        self.map_layer.update_from_msg(msg)
        if not self.waypoints or self.last_odom is None:
            return
        p = self.last_odom.pose.pose.position
        rest = [(p.x, p.y)] + list(self.waypoints[self.wp_idx:])
        ignore = None
        if self.phase == "APPROACH" and self.target is not None:
            # 다가가는 광물 주변은 "막혔다" 로 보지 않는다 (계획도 그렇게 짰다).
            ignore = (self.target.world_position.x, self.target.world_position.y,
                      float(self.get_parameter("approach_target_clear_m").value)
                      + self.ogrid.robot_radius + self.ogrid.cell_size)
        if self.map_layer.path_blocked(rest, ignore=ignore):
            self.waypoints = []          # 다음 _maybe_replan 에서 새로 계획
            self.map_replans += 1
            self.get_logger().info(
                f"새로 본 장애물이 경로를 막음 → 다시 계획 (누적 {self.map_replans}회)",
                throttle_duration_sec=2.0)

    def _maybe_replan(self, rx: float, ry: float, tx: float, ty: float) -> None:
        if self.ogrid is None:
            return
        need_replan = (
            not self.waypoints
            or self.last_plan_target is None
            or math.hypot(tx - self.last_plan_target[0],
                          ty - self.last_plan_target[1])
                > float(self.get_parameter("path_replan_target_delta_m").value)
            # Multi-rover: 다른 rover 가 trigger_m 이상 움직였으면 replan.
            or self._other_rovers_moved()
        )
        now_ns = self.get_clock().now().nanoseconds
        if not need_replan and self.plan_goal_gap > 0.3:
            # 지난 계획은 목표에 못 미친 곳에서 끝났다 (목표 주변이 막혀 있었음).
            # 지도가 바뀌어 열렸을 수 있으니 주기적으로 다시 짜 본다.
            retry_ns = int(float(
                self.get_parameter("approach_short_plan_retry_sec").value) * 1e9)
            need_replan = (now_ns - self.last_plan_ns) > retry_ns
        if not need_replan:
            return
        self.last_plan_ns = now_ns

        # static grid 또는 dynamic grid (다른 rover 위치 inflate 후) 사용.
        # multi-rover off 시 static grid 그대로 반환되어 비용 없음.
        base = None
        if self.map_layer is not None:
            # 온라인 지도에서는 목표 광물 자체가 위험 칸이다 → 목표 주변만 빼고 계획.
            base = self.map_layer.grid_excluding(
                tx, ty, float(self.get_parameter("approach_target_clear_m").value))
        active_grid = self._build_dynamic_grid(base)
        if active_grid is None:
            active_grid = self.ogrid.grid

        start = self.ogrid.world_to_cell(rx, ry, clip=True)
        goal = self.ogrid.world_to_cell(tx, ty, clip=True)
        if start and active_grid[start] == 1:
            start = self._nearest_free(start, active_grid)
        if goal is not None and active_grid[goal] == 1:
            goal = self._nearest_free(goal, active_grid)

        # 1차 시도 — 표준 grid 로.
        cost = getattr(self.ogrid, "cost", None)       # 기울기 비용 (지형 스캐너가 있을 때)
        path = astar(active_grid, start, goal, cost=cost) if start and goal else None

        # 2차 시도 — 실패 시 시작 cell 주변 5×5 일시 free 처리 후 재시도.
        if not path and start and goal:
            relaxed = active_grid.copy()
            si, sj = start
            for di in range(-2, 3):
                for dj in range(-2, 3):
                    ii, jj = si + di, sj + dj
                    if self.ogrid.in_bounds(ii, jj):
                        relaxed[ii, jj] = 0
            path = astar(relaxed, start, goal, cost=cost)
            if path:
                self.get_logger().warn(
                    f"A* recovered with relaxed start cell "
                    f"(rover was in inflated obstacle area)")
        # Multi-rover: replan 시점의 다른 rover 위치 캐시 (trigger 판단용).
        self._last_plan_other_rovers = self._other_rovers_alive()

        if not path:
            self.get_logger().warn(
                f"A* no path: start={start} goal={goal} target=({tx:.2f},{ty:.2f})",
                throttle_duration_sec=2.0)
            self.waypoints = []
            self.last_plan_target = (tx, ty)
            return

        path = simplify_path(active_grid, path, cost=cost)
        self.waypoints = [self.ogrid.cell_to_world(i, j) for (i, j) in path]
        self.wp_idx = 1 if len(self.waypoints) > 1 else 0
        self.last_plan_target = (tx, ty)
        ex, ey = self.waypoints[-1]
        gap = math.hypot(ex - tx, ey - ty)
        changed = (gap > 0.3) != (self.plan_goal_gap > 0.3)
        self.plan_goal_gap = gap
        if gap > 0.3:
            if changed:
                self.get_logger().warn(
                    f"A* replanned: 목표 ({tx:.2f},{ty:.2f}) 주변이 막혀 "
                    f"{gap:.2f} m 떨어진 곳까지만 계획함")
        else:
            self.get_logger().info(
                f"A* replanned: {len(self.waypoints)} waypoints "
                f"to ({tx:.2f},{ty:.2f})")

    def _send_arm_goal(self) -> None:
        if self.arm_client is None or self.target is None or self.arm_in_flight:
            return
        if not self.arm_client.wait_for_server(timeout_sec=1.0):
            self.get_logger().warn(
                "arm action server not available — skipping pick. Will retry on next PICK_READY.",
                throttle_duration_sec=5.0)
            return
        goal = ExecuteArmTask.Goal()
        goal.command = "pick_mineral"
        goal.target_id = str(self.target.mineral_id)
        goal.target_x = float(self.target.world_position.x)
        goal.target_y = float(self.target.world_position.y)
        goal.target_z = float(self.target.world_position.z)
        goal.metadata = self.target.class_name
        self.arm_in_flight = True
        self._pending_pick_world = (goal.target_x, goal.target_y)
        self._pending_pick_class = self.target.class_name
        future = self.arm_client.send_goal_async(goal)
        future.add_done_callback(self._on_arm_goal_accepted)
        self.get_logger().info(
            f"arm action sent: pick_mineral target_id={goal.target_id} "
            f"class={goal.metadata} xyz=({goal.target_x:.2f},{goal.target_y:.2f},"
            f"{goal.target_z:.2f})")

    def _on_arm_goal_accepted(self, future) -> None:
        try:
            handle = future.result()
        except Exception as e:
            self.get_logger().error(f"arm goal future failed: {e}")
            self.arm_in_flight = False
            return
        if not handle.accepted:
            self.get_logger().warn("arm goal rejected by server")
            self.arm_in_flight = False
            return
        self._arm_goal_handle = handle
        handle.get_result_async().add_done_callback(self._on_arm_result)

    def _on_arm_result(self, future) -> None:
        try:
            result = future.result().result
            ok = bool(result.success)
            msg_text = result.message
        except Exception as e:
            ok, msg_text = False, f"exception: {e}"
        self.arm_in_flight = False
        self._arm_goal_handle = None
        self.last_pick_done_ns = self.get_clock().now().nanoseconds
        self.last_pick_world = getattr(self, "_pending_pick_world", None)
        if not ok and msg_text.startswith("canceled"):
            # 운영자가 세운 것 — 채취 실패로 치지 않는다 (그 자리를 피하지도 않는다).
            self.target = None
            self.lock_target = None      # 잡아 둔 목표도 같이 놓는다
            self.last_det_stamp_ns = 0
            self.get_logger().info("arm action canceled by operator — 팔이 그 자리에 섬")
            self._event("채취 동작 취소됨 — 팔이 그 자리에 섬")
            if not self.estop and self._arm_stopped:
                self._send_arm_home()        # 취소가 끝나기 전에 이미 RESUME 을 눌렀다
            return
        if self.last_pick_world is not None:
            # 집었으면 그 자리의 기억을 지운다. 손목 카메라가 "그 자리에 없음" 이라고 한 것도
            # 지운다 (잘못 적힌 자리). 다른 실패는 횟수만 올려 나중에 다시 가 본다.
            gone = ok or "no mineral" in msg_text
            self._forget_mineral(self.last_pick_world[0], self.last_pick_world[1],
                                 float(self.get_parameter("post_pick_skip_radius_m").value),
                                 failed=not gone)
        if ok:
            self.collected_count += 1
            cap = int(self.get_parameter("cargo_capacity").value)
            self.cargo_count = min(self.collected_count, cap) if cap > 0 else self.collected_count
            # 종류별 카운트 — arm action 의 metadata 에 class_name 이 들어있음.
            picked_class = (self.target.class_name if self.target is not None
                            else getattr(self, "_pending_pick_class", ""))
            if picked_class in self.collected_by_class:
                self.collected_by_class[picked_class] += 1
            # 직전 실패 메시지는 깨끗하게 — 한 번 실패가 영원히 화면에 박혀
            # 있는 현상 방지. 다음 실패가 발생하면 그때 다시 채워짐.
            if not self.estop:
                self.last_error = ""
        else:
            if not self.estop:
                self.last_error = f"arm failed: {msg_text}"
            if self.last_pick_world is not None:
                ignore_ns = int(float(
                    self.get_parameter("failed_pick_ignore_sec").value) * 1e9)
                self.failed_pick_points.append(
                    (self.last_pick_world[0], self.last_pick_world[1],
                     self.last_pick_done_ns + ignore_ns))
        # Drop the current target so PICK_READY exits and EXPLORE resumes.
        self.target = None
        self.lock_target = None      # 잡아 둔 목표도 같이 놓는다
        self.last_det_stamp_ns = 0
        goal = int(self.get_parameter("collection_goal").value)
        self.get_logger().info(
            f"arm action done success={ok} message={msg_text!r} — "
            f"collected={self.collected_count}/{goal} — cooldown engaged")

    def _enter_phase(self, new_phase: str, *, reason: str = "") -> None:
        if self.phase == new_phase:
            return
        prev = self.phase
        self.phase = new_phase
        # Stale path 폐기 — EXPLORE 로 복귀할 때처럼 RTB 진입 시에도 이전 phase
        # (APPROACH/PICK_READY) 의 waypoint 가 남아있으면 rover 의 새 위치
        # 기준 obstacle 회피 path 가 안 그려져서 일직선으로 가는 사고가 남.
        # 다음 _drive_to_basecamp tick 에서 fresh A* 가 자동 실행됨.
        if new_phase in ("MISSION_COMPLETE", "EXPLORE", "RETURN_TO_BASE"):
            self.waypoints = []
            self.wp_idx = 0
            self.last_plan_target = None
        # EXPLORE / MISSION_COMPLETE / RTB 진입 시 supervisor source 의
        # path/target 을 한 번 청소 — 직전 phase 의 mineral 별·점선이
        # dashboard 에 남아있다가 새 viz tick 전까지 stale 로 보이는 현상
        # 차단 (2026-05-27 rover_1 RTB 시연에서 옛 mineral target 잔재 관찰).
        # RTB 의 경우 다음 _drive_to_basecamp tick 에서 새 basecamp target 으로
        # 즉시 덮어쓰지만, 그 한 frame 사이 stale frame 을 None 으로 막음.
        if new_phase in ("EXPLORE", "MISSION_COMPLETE", "RETURN_TO_BASE"):
            try:
                self._publish_supervisor_viz(target_xy=None)
            except Exception:
                pass
        # phase 전환 즉시 MissionState 도 발행 — _publish_state 의 timer 다음
        # tick (수십 ms) 까지 기다리면 backend 가 race 로 옛 explore slot
        # fallback 잔재를 한 frame 그림 (2026-05-27 MISSION_COMPLETE 깜박임
        # 디버깅). synchronous 발행으로 새 phase 가 path/target 메시지보다
        # 먼저 backend 에 도달하게 한다.
        try:
            self._publish_state()
        except Exception:
            pass
        suffix = f" ({reason})" if reason else ""
        self.get_logger().info(f"phase: {prev} -> {new_phase}{suffix}")
        self._event(f"phase {prev} -> {new_phase}{suffix}")
        if new_phase != "RETURN_TO_BASE":
            self.operator_rtb = False

    def _drive_to_basecamp(self) -> None:
        if self.last_odom is None:
            self.cmd_pub.publish(Twist())
            return
        bx = float(self.get_parameter("basecamp_x").value)
        by = float(self.get_parameter("basecamp_y").value)
        rx = self.last_odom.pose.pose.position.x
        ry = self.last_odom.pose.pose.position.y
        dist_to_center = math.hypot(bx - rx, by - ry)

        arrive_r = float(self.get_parameter("basecamp_arrive_radius_m").value)
        if dist_to_center <= arrive_r:
            self._enter_phase("MISSION_COMPLETE",
                              reason=f"basecamp reached at ({rx:.2f},{ry:.2f})")
            self.cmd_pub.publish(Twist())
            return

        # Basecamp 중심 (0,0) 자체가 obstacle 영역 → goal nudge 실패해 A* fail.
        # rover 에서 basecamp 향한 방향의 fringe 점을 target 으로 사용.
        fringe_r = float(self.get_parameter("basecamp_fringe_radius_m").value)
        if dist_to_center > fringe_r:
            ux = (bx - rx) / dist_to_center
            uy = (by - ry) / dist_to_center
            tx = bx - ux * fringe_r
            ty = by - uy * fringe_r
        else:
            tx, ty = bx, by

        # APPROACH 와 동일하게 A* 한 번 만들어두면 다음부터 같은 target 이라
        # replan 안 함. RTB 진입 시 _enter_phase 에서 waypoints/last_plan_target
        # 을 비웠으므로 첫 호출에서 fresh plan 생성.
        self._maybe_replan(rx, ry, tx, ty)

        if not self.waypoints:
            # A* 실패 — basecamp 까지 obstacle 회피 path 없음. 직진하면
            # obstacle 통과 위험. 자리에서 MISSION_COMPLETE 로 종료.
            self._enter_phase(
                "MISSION_COMPLETE",
                reason=f"RTB A* failed at ({rx:.2f},{ry:.2f}) — stopping in place")
            self.last_error = "RTB unreachable"
            self.cmd_pub.publish(Twist())
            return

        wp_reach = float(self.get_parameter("waypoint_reach_dist_m").value)
        while self.wp_idx < len(self.waypoints) - 1:
            wx, wy = self.waypoints[self.wp_idx]
            if math.hypot(wx - rx, wy - ry) < wp_reach:
                self.wp_idx += 1
            else:
                break
        gx, gy = self.waypoints[self.wp_idx]
        # UI 표시 — fringe target 을 분홍 별로.
        self._publish_supervisor_viz(target_xy=(tx, ty))

        q = self.last_odom.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        bearing = math.atan2(gy - ry, gx - rx)
        heading_err = math.atan2(math.sin(bearing - yaw), math.cos(bearing - yaw))

        t = Twist()
        lin = float(self.get_parameter("rtb_lin_speed").value)
        if abs(heading_err) > 0.4:
            lin *= 0.3
        t.linear.x = lin
        max_ang = 1.2
        if abs(heading_err) > 0.4:
            max_ang = min(max_ang, turn_rate_cap(heading_err))
        t.angular.z = max(-max_ang, min(max_ang,
            float(self.get_parameter("steer_gain").value) * heading_err))
        self.cmd_pub.publish(t)

    def _publish_supervisor_viz(self, target_xy=None) -> None:
        """Supervisor 의 현재 path/target 을 UI 용 토픽으로 발행.
        target_xy: (tx, ty) 또는 None. None 이면 target marker 안 그림."""
        stamp = self.get_clock().now().to_msg()
        # Path
        msg = Path()
        msg.header.stamp = stamp
        msg.header.frame_id = "map"
        for wx, wy in (self.waypoints or []):
            ps = PoseStamped()
            ps.header = msg.header
            ps.pose.position.x = float(wx)
            ps.pose.position.y = float(wy)
            ps.pose.orientation.w = 1.0
            msg.poses.append(ps)
        self.supervisor_path_pub.publish(msg)
        # Target — None 이면 NaN 으로 보내서 dashboard 가 hide.
        tgt = PointStamped()
        tgt.header.stamp = stamp
        tgt.header.frame_id = "map"
        if target_xy is None:
            tgt.point.x = float("nan")
            tgt.point.y = float("nan")
        else:
            tgt.point.x = float(target_xy[0])
            tgt.point.y = float(target_xy[1])
        self.supervisor_target_pub.publish(tgt)

    def _publish_state(self) -> None:
        s = MissionState()
        # ESTOP 이 걸려 있는 동안에는 안에서 phase 가 어떻게 바뀌든(넘어짐 복구 등)
        # 밖으로는 항상 ESTOP 으로 알린다 — HUD 의 PHASE·TASK 가 정지 상태임을 보여야 한다.
        phase = "ESTOP" if self.estop else self.phase
        s.state = phase if self.mode == "AUTO" else f"MANUAL/{phase}"
        s.previous_state = ""
        s.battery_percent = float(self.battery_percent)
        s.low_battery = bool(self.battery_low)
        s.critical_battery = bool(self.battery_critical)
        s.cargo_count = int(self.cargo_count)
        s.cargo_capacity = int(self.get_parameter("cargo_capacity").value)
        s.collected_count = int(self.collected_count)
        s.collection_goal = int(self.get_parameter("collection_goal").value)
        s.collected_blue   = int(self.collected_by_class.get("blue_mineral", 0))
        s.collected_yellow = int(self.collected_by_class.get("yellow_mineral", 0))
        s.collected_green  = int(self.collected_by_class.get("green_gas", 0))
        s.active_task = ("E-STOP" + (" · 팔 정지" if self._arm_stopped else "")
                         + (" · 넘어짐 복구 중" if self.phase == "RECOVERY" else "")
                         if self.estop
                         else (self.righting_status or "self-righting") if self.phase == "RECOVERY"
                         else "arm → home" if self.arm_in_flight and self._arm_stopped
                         else "picking" if self.arm_in_flight
                         else f"기울기 반사 — {self._guard_why}" if self._guard_until is not None
                         else "manual" if self.mode == "MANUAL"
                         else self.phase.lower())
        s.last_error = self.last_error
        self.state_pub.publish(s)

    def _nearest_free(self, cell: Tuple[int, int], grid=None) -> Tuple[int, int]:
        # search radius 25 cell (= 5m at cell_size=0.2). 이전 12 cell (2.4m) 은
        # basecamp 같은 큰 obstacle 영역 옆에서 free 못 찾는 케이스가 있었음.
        if grid is None:
            grid = self.ogrid.grid
        i0, j0 = cell
        for r in range(1, 25):
            for di in range(-r, r + 1):
                for dj in range(-r, r + 1):
                    if max(abs(di), abs(dj)) != r:
                        continue
                    ni, nj = i0 + di, j0 + dj
                    if self.ogrid.in_bounds(ni, nj) and grid[ni, nj] == 0:
                        return (ni, nj)
        return cell

    def _log_phase(self) -> None:
        s = String()
        s.data = self.phase
        self.phase_pub.publish(s)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MissionManagerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # Publish zero twist a few times so the rover does not keep moving
        # after shutdown. Wrap in try/except: under SIGINT the publisher's
        # context may already be invalid by the time we get here.
        for _ in range(5):
            try:
                node.cmd_pub.publish(Twist())
            except Exception:
                break
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
