"""다중 rover 시연용 통합 런치 (Phase 3 of multi-rover MVP).

run_vehicle_v3.py --rovers rover_1 rover_2 ... 와 짝지어 사용. 각 rover 마다
별도 namespace 의 5 노드 묶음을 띄움. 토픽은 /<namespace>/... prefix.

기존 mvp.launch.py 는 변경 없이 보존 (단일 rover 시연용).

사용법 (T1 Isaac Sim 띄운 후):
  ros2 launch isaac_bringup mvp_multi.launch.py
    → default 로 rover_1, rover_2 두 대 노드 묶음 띄움
  ros2 launch isaac_bringup mvp_multi.launch.py rovers:='rover_1 rover_2 rover_3'
    → 임의 N대

각 rover 의 토픽 격리:
  /rover_1/ground_truth/odom, /rover_1/cmd_vel, /rover_1/arm/joint_command, ...
  /rover_2/ground_truth/odom, /rover_2/cmd_vel, /rover_2/arm/joint_command, ...

mineral claim 공유 토픽 (모든 rover 공통, namespace 없음):
  /mineral_claims  (Phase 4 에서 추가)
"""

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _default_terrain_root() -> str:
    env = os.environ.get("ISAAC_TERRAIN_ROOT")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    candidate = os.path.normpath(
        os.path.join(here, "..", "..", "..", "isaac_sim", "assets", "generated_terrains")
    )
    if os.path.isdir(candidate):
        return candidate
    return os.path.expanduser(
        "~/dev_ws/rover_ws/src/a2_isaac/isaac_sim/assets/generated_terrains"
    )


def _rover_group(ns: str, terrain_dir: str, collection_goal: int,
                 arrive_r: float, fringe_r: float, reveal_radius: float,
                 map_source: str = "online", governor: str = "",
                 self_right: bool = False, slope_deg: float = 25.0,
                 tilt_guard: bool = True, terrain_scan: bool = True):
    """단일 rover 의 5 노드 묶음 — 토픽 모두 /<ns>/... prefix.

    terrain_dir 는 T1 Isaac Sim 의 --terrain 인자와 동일한 terrain 폴더 경로.
    coverage/mission_manager 가 같은 obstacle_grid 를 사용해야 anchor 도달 가능.
    collection_goal 은 각 rover 가 N개 채집 시 RETURN_TO_BASE 전환.
    """
    nsp = "/" + ns
    mapping = []
    if governor:
        # 학습한 구동 조절기: supervisor 의 명령을 받아 깎아서 바퀴로 보낸다.
        mapping.append(Node(
            package="isaac_drive",
            # governor == "balance": 물러서기 + 팔 균형추까지 하는 균형 정책 (balance_node).
            executable="balance_node" if governor == "balance" else "drive_governor_node",
            name="balance_node" if governor == "balance" else "drive_governor_node",
            namespace=ns,
            output="screen",
            parameters=[{
                "cmd_in_topic":  f"{nsp}/cmd_vel_raw",
                "cmd_out_topic": f"{nsp}/cmd_vel",
                "odom_topic":    f"{nsp}/ground_truth/odom",
                "joint_topic":   f"{nsp}/joint_states_raw",
                **({"joint_command_topic": f"{nsp}/arm/joint_command",
                    "phase_topic":         f"{nsp}/mission/phase"}
                   if governor == "balance" else {}),
            }],
        ))
    if self_right:
        # 넘어지면 팔로 다시 세운다 (학습한 정책). 세우는 동안 supervisor 는 바퀴 정지.
        mapping.append(Node(
            package="isaac_drive",
            executable="self_right_node",
            name="self_right_node",
            namespace=ns,
            output="screen",
            parameters=[{
                "odom_topic":          f"{nsp}/ground_truth/odom",
                "joint_topic":         f"{nsp}/joint_states_raw",
                "joint_command_topic": f"{nsp}/arm/joint_command",
                "active_topic":        f"{nsp}/righting/active",
                "status_topic":        f"{nsp}/righting/status",
            }],
        ))
    if map_source == "online":
        # 깊이 카메라 → 위험 지도 (미리 만든 obstacle_grid 대신).
        mapping.append(Node(
            package="isaac_drive",
            executable="hazard_map_node",
            name="hazard_map_node",
            namespace=ns,
            output="screen",
            parameters=[{
                "terrain_dir":      terrain_dir,
                "slope_deg":        float(slope_deg),
                "depth_topic":      f"{nsp}/camera/rover/depth",
                "info_topic":       f"{nsp}/camera/rover/camera_info",
                "odom_topic":       f"{nsp}/ground_truth/odom",
                "joint_topic":      f"{nsp}/joint_states_raw",
                "detections_topic": f"{nsp}/perception/detections",
                "map_topic":        f"{nsp}/hazard_map",
                "reset_topic":      f"{nsp}/hazard_map/reset",
                "clear_marks_topic": f"{nsp}/hazard_map/clear_marks",
                "scan_topic":       f"{nsp}/terrain_scan" if terrain_scan else "",
                "elevation_topic":  f"{nsp}/elevation/summary",
                "veto_topic":       f"{nsp}/hazard_map/not_mineral",
                "mark_topic":       f"{nsp}/hazard_map/mark",
                "cmd_topic":        f"{nsp}/cmd_vel",
            }],
        ))
    return mapping + [
        # T5 cheat 어댑터 — /<ns>/ground_truth/odom → /<ns>/rover/estimated_pose
        Node(
            package="isaac_drive",
            executable="odom_to_estimated_pose",
            name="odom_to_estimated_pose",
            namespace=ns,
            output="screen",
            parameters=[{
                "odom_topic": f"{nsp}/ground_truth/odom",
                "pose_topic": f"{nsp}/rover/estimated_pose",
            }],
        ),
        # T2 perception — nav + wrist YOLO + depth backproject
        Node(
            package="isaac_perception",
            executable="yolo_perception_node",
            name="yolo_perception_node",
            namespace=ns,
            output="screen",
            parameters=[{
                "nav_rgb_topic":     f"{nsp}/camera/rover/image_raw",
                "nav_depth_topic":   f"{nsp}/camera/rover/depth",
                "nav_info_topic":    f"{nsp}/camera/rover/camera_info",
                "odom_topic":        f"{nsp}/ground_truth/odom",
                "nav_det_topic":     f"{nsp}/perception/detections",
                "nav_ann_topic":     f"{nsp}/perception/image_annotated",
                "wrist_rgb_topic":   f"{nsp}/camera/wrist/image_raw",
                "wrist_depth_topic": f"{nsp}/camera/wrist/depth",
                "wrist_info_topic":  f"{nsp}/camera/wrist/camera_info",
                "wrist_det_topic":   f"{nsp}/perception/wrist_detections",
                "wrist_ann_topic":   f"{nsp}/perception/wrist_image_annotated",
            }],
        ),
        # T3 coverage — BCD anchor sweep + A* obstacle 회피
        # /cmd_vel → /coverage/cmd_vel_raw remap: supervisor 가 mux.
        # minimap_publisher 와 mission_state 가 absolute "/mission/..." 로
        # 박혀 있어 두 rover 가 충돌 → remap 으로 namespace 강제 격리.
        # matplotlib viewer: 두 rover 모두 활성. minimap_rover_id 가 StateWriter
        # 에 전달되어 각자 unique npz (예: starcraft_map_state_rover_1.npz)
        # 와 별도 viewer subprocess 가 떠 두 matplotlib 창이 독립 동작.
        Node(
            package="isaac_drive",
            executable="coverage_node",
            name="coverage_node",
            namespace=ns,
            output="screen",
            parameters=[{
                "pose_topic":    f"{nsp}/rover/estimated_pose",
                "cmd_vel_topic": f"{nsp}/coverage/cmd_vel_raw",
                "terrain_dir": terrain_dir,
                "map_source": map_source,
                "hazard_map_topic": f"{nsp}/hazard_map",
                "clear_marks_topic": f"{nsp}/hazard_map/clear_marks",
                # matplotlib Starcraft Map viewer 비활성 — Web HUD 의
                # COVERAGE MAP canvas 가 두 rover 추적 가능해진 시점부터
                # 더이상 필요 없음. enable_minimap_topics 만 True 로
                # /<ns>/mission/{minimap,path,markers} 발행 유지 → Web HUD
                # 가 그 토픽 받아 그림.
                "enable_minimap": False,
                "enable_minimap_topics": True,
                "minimap_rover_id": ns,
                "reveal_radius": reveal_radius,
            }],
            remappings=[
                # minimap_publisher 의 absolute 토픽 → namespace 강제
                ("/mission/minimap",  f"{nsp}/mission/minimap"),
                ("/mission/path",     f"{nsp}/mission/path"),
                ("/mission/markers",  f"{nsp}/mission/markers"),
                # coverage_node 의 MissionState publisher (absolute /mission_state)
                ("/mission_state",    f"{nsp}/mission_state"),
                # mission_manager 의 replan publish 를 같은 ns 의 coverage 가 받게
                ("/coverage/replan_request", f"{nsp}/coverage/replan_request"),
            ],
        ),
        # Supervisor — EXPLORE→APPROACH→PICK phase + cmd_vel mux + arm action
        Node(
            package="isaac_supervisor",
            executable="mission_manager_node",
            name="mission_manager_node",
            namespace=ns,
            output="screen",
            parameters=[{
                "approach_engage_dist_m": 30.0,
                "mineral_veto_topic": f"{nsp}/hazard_map/not_mineral",
                "known_minerals_topic": f"{nsp}/mission/known_minerals",
                "approach_lin_speed": 1.2,
                "coverage_cmd_topic": f"{nsp}/coverage/cmd_vel_raw",
                "righting_active_topic": f"{nsp}/righting/active",
                "righting_status_topic": f"{nsp}/righting/status",
                "cmd_vel_topic":      f"{nsp}/cmd_vel_raw" if governor else f"{nsp}/cmd_vel",
                "detections_topic":   f"{nsp}/perception/detections",
                "odom_topic":         f"{nsp}/ground_truth/odom",
                "phase_topic":        f"{nsp}/mission/phase",
                # 상대 토픽 — namespace=ns 가 자동 prefix → /<ns>/execute_arm_task
                "arm_action_name":    "execute_arm_task",
                "collection_goal":    collection_goal,
                # ── Multi-rover 협조 활성 (단일 시연은 default False) ──
                "enable_mineral_claim":  True,
                "enable_rover_avoid":    True,
                "claim_rover_id":        ns,
                "basecamp_arrive_radius_m": arrive_r,
                "basecamp_fringe_radius_m": fringe_r,
                "state_topic":        f"{nsp}/mission/state",
                "mode_topic":         f"{nsp}/mission/mode",
                "estop_topic":        f"{nsp}/mission/estop",
                # HUD 의 ACTIONS 버튼 · 수동 주행 (mission_web_node 가 보낸다)
                "command_topic":      f"{nsp}/mission/command",
                "event_topic":        f"{nsp}/mission/event",
                "hazard_mark_topic":  f"{nsp}/hazard_map/mark",
                "tilt_guard":         tilt_guard,
                "teleop_cmd_topic":   f"{nsp}/teleop/cmd_vel",
                "terrain_dir":        terrain_dir,
                "map_source":         map_source,
                "hazard_map_topic":   f"{nsp}/hazard_map",
                # 협조 param (enable_mineral_claim/enable_rover_avoid 등) 은
                # main 의 mission_manager_node 에 declare 안 됨 — 단순 namespace
                # 복사 단계 (옵션 X) 라 미설정. 협조 단계 (옵션 Y) 진입 시
                # mission_manager 코드 + param declare 추가 후 활성.
            }],
            remappings=[
                # mission_manager_node 의 absolute publisher → namespace 강제
                ("/coverage/replan_request", f"{nsp}/coverage/replan_request"),
                ("/supervisor/path",         f"{nsp}/supervisor/path"),
                ("/supervisor/target",       f"{nsp}/supervisor/target"),
            ],
        ),
        # T2 manipulation — DLS-IK 상태머신 + /grasp/command FixedJoint snap
        Node(
            package="isaac_manipulation",
            executable="arm_executor_node",
            name="arm_executor_node",
            namespace=ns,
            output="screen",
            parameters=[{
                "grasp_mode": LaunchConfiguration("grasp_mode"),
                "joint_command_topic":    f"{nsp}/arm/joint_command",
                "joint_state_topic":      f"{nsp}/joint_states_raw",
                "wrist_detections_topic": f"{nsp}/perception/wrist_detections",
                "wrist_depth_topic":      f"{nsp}/camera/wrist/depth",
                "wrist_info_topic":       f"{nsp}/camera/wrist/camera_info",
                "odom_topic":             f"{nsp}/ground_truth/odom",
                "grasp_command_topic":    f"{nsp}/grasp/command",
                "grasp_status_topic":     f"{nsp}/grasp/status",
            }],
        ),
    ]


def _launch_setup(context, *args, **kwargs):
    rovers_str = LaunchConfiguration("rovers").perform(context).strip()
    rovers = [r for r in rovers_str.split() if r]
    if not rovers:
        rovers = ["rover_1", "rover_2"]
    terrain_id = LaunchConfiguration("terrain_id").perform(context)
    terrain_root = LaunchConfiguration("terrain_root").perform(context)
    terrain_dir = os.path.join(terrain_root, terrain_id)
    collection_goal = int(LaunchConfiguration("collection_goal").perform(context))
    arrive_r = float(LaunchConfiguration("basecamp_arrive_radius_m").perform(context))
    fringe_r = float(LaunchConfiguration("basecamp_fringe_radius_m").perform(context))
    enable_dashboard = LaunchConfiguration("enable_dashboard").perform(context)
    enable_web_video = LaunchConfiguration("enable_web_video").perform(context)
    web_video_port = LaunchConfiguration("web_video_port").perform(context)
    web_port = LaunchConfiguration("web_port").perform(context)
    hud_key = LaunchConfiguration("hud_key").perform(context).strip()
    hud_actions = LaunchConfiguration("hud_actions").perform(context).strip()
    reveal_radius = float(LaunchConfiguration("reveal_radius").perform(context))
    map_source = LaunchConfiguration("map_source").perform(context).strip().lower()
    slope_deg = float(LaunchConfiguration("hazard_slope_deg").perform(context))
    governor = LaunchConfiguration("drive_governor").perform(context).strip().lower()
    governor = ("balance" if governor == "balance"
                else "governor" if governor in ("true", "1", "yes") else "")
    tilt_guard = LaunchConfiguration("tilt_guard").perform(context).strip().lower() in (
        "1", "true", "yes", "on")
    self_right = LaunchConfiguration("self_right").perform(context).strip().lower() in (
        "true", "1", "yes")
    terrain_scan = LaunchConfiguration("terrain_scan").perform(context).strip().lower() in (
        "1", "true", "yes", "on")
    print(f"[mvp_multi] terrain_dir={terrain_dir}  rovers={rovers}  "
          f"collection_goal={collection_goal} (per rover)  map_source={map_source}  "
          f"drive_governor={governor}")
    nodes = []
    for ns in rovers:
        nodes.extend(_rover_group(ns, terrain_dir, collection_goal,
                                  arrive_r, fringe_r, reveal_radius, map_source,
                                  governor, self_right, slope_deg, tilt_guard,
                                  terrain_scan))

    # Web HUD — 단일 인스턴스가 모든 rover namespace 추적 (옵션 X 단계 2).
    # mission_web_node 의 rover_namespaces param 으로 ns 리스트 전달.
    if str(enable_dashboard).lower() in ("true", "1", "yes"):
        nodes.append(Node(
            package="isaac_supervisor",
            executable="mission_web_node",
            name="mission_web_node",
            output="screen",
            additional_env={
                "WEB_VIDEO_PORT": str(web_video_port),
                "WEB_PORT": str(web_port),
                "WEB_TOKEN": hud_key,
                "WEB_ACTIONS": hud_actions,
                "WEB_PUBLIC": LaunchConfiguration("hud_public").perform(context).strip(),
            },
            parameters=[{
                "rover_namespaces": rovers,
                "terrain_preview_path": os.path.join(terrain_dir, "preview.png"),
            }],
        ))
    if str(enable_web_video).lower() in ("true", "1", "yes"):
        nodes.append(Node(
            package="web_video_server",
            executable="web_video_server",
            name="web_video_server",
            output="screen",
            parameters=[{
                "port": int(web_video_port),
                # 영상은 HUD 서버(web_port)가 /stream 으로 중계한다 — 이 포트는 밖에 열지 않는다.
                "address": "127.0.0.1",
            }],
        ))
    return nodes


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription([
        DeclareLaunchArgument(
            "rovers", default_value="rover_1 rover_2",
            description="공백 구분 namespace 리스트. 예: 'rover_1 rover_2 rover_3'"),
        DeclareLaunchArgument(
            "terrain_id", default_value="terrain_00026",
            description="T1 Isaac Sim 의 --terrain 인자와 동일해야 obstacle_grid 정합."),
        DeclareLaunchArgument(
            "terrain_root", default_value=_default_terrain_root(),
            description="terrain_<id> 디렉토리들이 모여있는 루트."),
        DeclareLaunchArgument(
            "collection_goal", default_value="2",
            description="각 rover 가 N개 채집 시 RETURN_TO_BASE. 빠른 시연 1."),
        DeclareLaunchArgument(
            "basecamp_arrive_radius_m", default_value="8.0",
            description="베이스캠프 중심에서 이 거리 안에 들어오면 MISSION_COMPLETE."),
        DeclareLaunchArgument(
            "basecamp_fringe_radius_m", default_value="7.5",
            description="RTB 시 A* target = basecamp 중심에서 이 거리 fringe."),
        DeclareLaunchArgument(
            "enable_dashboard", default_value="true",
            description="mission_web_node (멀티 rover 추적 Web HUD) 활성."),
        DeclareLaunchArgument(
            "enable_web_video", default_value="true",
            description="web_video_server MJPEG 스트리머 활성."),
        DeclareLaunchArgument(
            "web_video_port", default_value="8090",
            description="web_video_server HTTP 포트 (127.0.0.1 전용). HUD 서버가 /stream 으로 중계한다."),
        DeclareLaunchArgument(
            "web_port", default_value="8088",
            description="Web HUD 포트. 화면·상태·카메라 영상이 모두 이 포트 하나로 나간다."),
        DeclareLaunchArgument(
            "hud_key", default_value="",
            description="HUD 접속 키. 주면 http://<주소>:<web_port>/?key=<키> 로 들어온 브라우저만 "
                        "볼 수 있다. 외부 주소(터널)로 열 때는 반드시 줄 것."),
        DeclareLaunchArgument(
            "hud_actions", default_value="true",
            description="false 면 HUD 의 ACTIONS 버튼·수동 주행을 끈다 (보기 전용)."),
        DeclareLaunchArgument(
            "terrain_scan", default_value="true",
            description="지형 스캐너(로버 위 360° 거리 센서, /<ns>/terrain_scan)의 점으로 높이 "
                        "지도를 만들어 급경사·턱을 위험 지도에 합치고 HUD 에 주변 높낮이를 보낸다. "
                        "시뮬레이터를 --terrain-scanner off 로 띄웠으면 아무 일도 하지 않는다."),
        DeclareLaunchArgument(
            "tilt_guard", default_value="true",
            description="기울기 반사 — 앞이 30° 넘게 들리거나 옆으로 32° 넘게 기울면 밀어붙이지 "
                        "않고 0.5 m 물러난 뒤 그 앞을 위험으로 표시하고 다른 길을 찾는다."),
        DeclareLaunchArgument(
            "hud_public", default_value="false",
            description="true 면 HUD 를 터널로 내보내 인터넷 주소(https://…)를 만든다. 주소는 "
                        "시작 로그와 HUD 위쪽에 나온다. hud_key 를 안 줬으면 접속 키를 만들어 "
                        "붙인다. cloudflared 가 깔려 있어야 한다 (PATH 또는 ~/.local/bin). "
                        "영상·상태가 Cloudflare 의 서버를 거쳐 나간다."),
        DeclareLaunchArgument(
            "reveal_radius", default_value="5.0",
            description="coverage_node 의 fog reveal 반경 (m). 멀티 rover 시연 "
                        "가시성 위해 default 5.0. 더 키우려면 reveal_radius:=7.0."),
        DeclareLaunchArgument(
            "grasp_mode", default_value="friction",
            description="arm_executor 의 파지 방식. friction: 넓은 집게로 쥐어 "
                        "마찰로 든다 (시뮬레이터 --grasp friction 과 짝). contact: "
                        "손끝이 닿을 때까지 내려 부착 (--grasp contact --gripper rg2). "
                        "snap: 예전 방식 (--grasp magic --gripper rg2)."),
        DeclareLaunchArgument(
            "map_source", default_value="online",
            description="장애물 지도의 출처. online: 로버 깊이 카메라로 주행 중에 작성 "
                        "(hazard_map_node). prebuilt: 지형과 함께 만들어 둔 "
                        "obstacle_grid.npy (원본 방식, 정답 지도)."),
        DeclareLaunchArgument(
            "drive_governor", default_value="false",
            description="true: 학습한 구동 조절기(drive_governor_node)가 주행 명령을 "
                        "차체 상태에 맞춰 깎는다. 시뮬레이터를 --drive-limits off 로 "
                        "띄워야 한다 (안 그러면 고정 제한이 먼저 깎는다). "
                        "balance: 균형 정책(balance_node — 물러서기 + 팔 균형추, "
                        "policies/balance.npz 필요). "
                        "false: 시뮬레이터의 고정 제한만 사용."),
        DeclareLaunchArgument(
            "hazard_slope_deg", default_value="25.0",
            description="온라인 위험 지도가 '못 가는 비탈' 로 치는 경사 (도). 레골리스 마찰 "
                        "0.58/0.52 에서는 25° 부터 바퀴가 80~90% 헛돈다."),
        DeclareLaunchArgument(
            "self_right", default_value="false",
            description="true: 로버가 넘어지면 팔로 다시 세운다 (self_right_node, "
                        "학습한 정책 policies/self_right.npz 필요)."),
        OpaqueFunction(function=_launch_setup),
    ])
