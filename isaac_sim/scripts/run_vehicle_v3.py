"""vehicle_v3.usd (액션그래프 내장 로버) 를 terrain 에 올려 구동.

v3 는 ROS2 센서 그래프가 USD 에 내장돼 있다 — 이 런처는 그래프를 짜지 않는다.
terrain 로드 + v3 reference + play 만 한다. 팀 누구든 이 패턴(또는 이 스크립트)
으로 v3 를 띄워 자기 노드를 개발하면 된다 — 실물 로봇처럼.

실행: <isaac-python> isaac_sim/scripts/run_vehicle_v3.py [--terrain terrain_00004] [--headless]
"""
import argparse
import json
import math
import os
import shutil
import sys
import tempfile

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

_p = argparse.ArgumentParser(description="vehicle_v3 (graph-embedded rover) 런처")
_p.add_argument("--terrain", default="terrain_00004")
_p.add_argument("--headless", action="store_true")
_p.add_argument("--no-chase", action="store_true",
                help="Chase cam(/camera/chase) 비활성화 — GPU 부담 측정/비교용. "
                     "비활성 시 Web HUD chase 슬롯은 offline 표시.")
_p.add_argument("--no-overview", action="store_true",
                help="Overview cam(/camera/overview) 비활성화 — 같은 의도.")
_p.add_argument("--rovers", nargs="*", default=[],
                help="다중 rover 모드 — namespace 리스트 (예: rover_1 rover_2). "
                     "비우면 단일 rover (기존 동작, /World/Rover + T5 sun camera + "
                     "chase/overview cam). 다중 모드는 chase/overview/sun_camera 비활성.")
_p.add_argument("--spawn-spacing", type=float, default=0.0,
                help="다중 rover 시 사이 간격 (m). >0 이면 meta spawn_locations "
                     "무시하고 spawn[0] 기준 X 방향으로 N 미터씩 배치 (A* 회피 검증용).")
_p.add_argument("--render-every", type=int, default=1,
                help="N step 마다 1회만 렌더 (기본 1 = 매 step). 렌더가 병목인 "
                     "GPU 에서 2 로 주면 물리는 그대로, 카메라/토픽 발행만 "
                     "sim-time 기준 절반 주기 → 실시간 배율(RTF) 상승.")
_p.add_argument("--max-rtf", type=float, default=None,
                help="실시간 배율 상한 (sim-time/wall-time). 0 = 무제한. 기본: "
                     "--render-every 1 이면 무제한(기존 동작), 2 이상이면 1.0 — "
                     "ROS 노드가 wall-clock 기준이라 sim 이 실시간보다 빨라지면 "
                     "제어 주기가 어긋남.")
_p.add_argument("--no-viewport", action="store_true",
                help="GUI 메인 뷰포트 렌더 중단 — 창은 뜨지만 3D 화면은 갱신 안 됨. "
                     "카메라 토픽은 정상. Web HUD 로만 볼 때 GPU 절약.")
_p.add_argument("--render-preset", choices=["default", "performance"],
                default="default",
                help="performance = RTX 반사/간접광/AO/반투명 끄고 DLSS Performance. "
                     "화면 톤이 달라지므로 YOLO 검출률 확인 필요.")
_p.add_argument("--debug-grasp", action="store_true",
                help="grasp FixedJoint 가 있는 동안 finger / mineral world 좌표와 "
                     "간격을 주기적으로 출력 (magic grasp 진단용).")
_p.add_argument("--physics", choices=["mars", "earth"], default="mars",
                help="물리 환경. mars(기본): 중력 3.72 m/s² + 레골리스·바위·바퀴 "
                     "접촉 재질 + terrain meta 의 구역별 마찰. earth: 손대지 "
                     "않음 (9.81, PhysX 기본 마찰 — 예전 동작).")
_p.add_argument("--grasp", choices=["contact", "magic"], default="contact",
                help="광물 부착 방식. contact(기본): 그리퍼 손끝이 광물에 닿았을 "
                     "때만 그 자리에서 부착. magic: 예전 방식 — 팔 받침대 2.5 m "
                     "안의 가장 가까운 광물을 손가락으로 순간이동시켜 부착.")
_p.add_argument("--drive-limits", default="auto",
                help="구동부 속도·가속 제한. auto(기본): --physics 의 중력에 맞춘 값 "
                     "(earth 는 제한 없음 = 원래 동작). off: 제한 없음. "
                     "또는 'max_lin=0.8,acc_lin=0.5' 처럼 항목별 override "
                     "(항목: max_lin, max_ang, acc_lin, acc_ang).")
_p.add_argument("--kit-arg", action="append", default=[],
                help="Kit 에 그대로 넘길 인자 (반복 가능). 예: "
                     "--kit-arg=--/rtx/reflections/enabled=false")
_a, _ = _p.parse_known_args()

from isaacsim import SimulationApp

_PERF_PRESET_ARGS = [
    "--/rtx/translucency/enabled=false",
    "--/rtx/reflections/enabled=false",
    "--/rtx/indirectDiffuse/enabled=false",
    "--/rtx/ambientOcclusion/enabled=false",
    "--/rtx/directLighting/sampledLighting/enabled=false",
    "--/rtx/raytracing/cached/enabled=false",
    "--/rtx-transient/dlssg/enabled=false",
    "--/rtx-transient/dldenoiser/enabled=false",
    "--/rtx/post/dlss/execMode=0",
]
_kit_args = list(_a.kit_arg)
if _a.render_preset == "performance":
    _kit_args = _PERF_PRESET_ARGS + _kit_args

app = SimulationApp({
    "headless": _a.headless,
    "extra_args": _kit_args,
    # SimulationApp 은 headless 일 때만 이 플래그를 적용. GUI 는 main() 에서 처리.
    "disable_viewport_updates": bool(_a.no_viewport and _a.headless),
})

from isaacsim.core.utils.extensions import enable_extension

enable_extension("isaacsim.ros2.bridge")
app.update()

import omni.usd
from isaacsim.core.api import World
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdPhysics, UsdShade

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
from planet_physics import apply_planet_physics  # noqa: E402
from contact_grasp import ContactGrasp  # noqa: E402
from drive_limits import apply_drive_limits, limits_for_gravity, parse_limits  # noqa: E402
ISAAC_SIM = os.path.dirname(HERE)
WORLD = f"{ISAAC_SIM}/worlds/{_a.terrain}.usd"
V3 = f"{ISAAC_SIM}/assets/vehicle/vehicle_v3.usd"
TERRAIN_DIR = f"{ISAAC_SIM}/assets/generated_terrains/{_a.terrain}"
ROVER_PRIM = "/World/Rover"
# rover 의 실시간 transform 은 articulation rigid body 인 Body prim 에서만
# 갱신됨. ROVER_PRIM (reference xform root) 은 spawn 시점 그대로 고정.
# build_vehicle_v3.ROVER_CAM 의 경로 패턴 차용: /Root/Vehicle/rover/Body.
ROVER_BODY_PRIM = ROVER_PRIM + "/Vehicle/rover/Body"

# T5 localization (PR #11): vehicle_v3 USD 에 baked 된 좁은 24mm sun camera 를
# runtime 에 wide lens 로 재설정 + 시각용 sun disk/light 를 World 에 추가해
# sun_yaw bearing 추출이 안정적으로 동작하도록 한다.
SUN_CAMERA_PRIM = f"{ROVER_PRIM}/Vehicle/rover/Body/SunCamera"
WORLD_SUN_YAW = 0.929
WORLD_SUN_ELEVATION = math.radians(55.0)
VISUAL_SUN_PRIM = "/World/VisualSun"
VISUAL_SUN_LIGHT = "/World/VisualSunLight"

# World-fixed overview camera — terrain 전체 부감 (50m 아레나 한눈에). vehicle_v3
# 와 무관하게 World prim 에 별도 author. UI 메인 화면 overview 슬롯이 구독.
OVERVIEW_CAM = "/World/OverviewCam"
OVERVIEW_GRAPH = "/World/OverviewGraph"
# 위치: 맵 중심 (0,0) 남쪽 -y 에서 비스듬히 부감. stage 가 Z-up 이라 카메라
# default forward(-z, 즉 아래)에 X 축 +α° 회전을 적용하면 시선이
# (0, sin α, -cos α). 카메라 (0, -45, 55) 에서 (0, 0, 0) 향하려면
# 벡터 = (0, 45, -55), 정규화 = (0, 0.633, -0.774) → α = asin(0.633) ≈ 39.3°.
# 광각 14mm 로 50m 아레나 전체 프레임에 들어옴.
OVERVIEW_XYZ = (0.0, -45.0, 55.0)
OVERVIEW_PITCH_DEG = 39.3
# 16:9 해상도 — Web HUD overview slot 의 가로:세로 비율에 맞춰 letterbox 제거.
OVERVIEW_W = 960
OVERVIEW_H = 540
OVERVIEW_TOPIC = "/camera/overview/image_raw"
OVERVIEW_INFO_TOPIC = "/camera/overview/camera_info"

# Chase camera — rover 뒤쪽 위에서 3인칭 TPS 풍 관망. /camera/chase/image_raw.
# rover transform 을 매 step Python 으로 폴링해서 cam 위치/시선 업데이트.
# vehicle_v3 reference 와 무관한 World prim 으로 author.
CHASE_CAM = "/World/ChaseCam"
CHASE_GRAPH = "/World/ChaseGraph"
CHASE_W = 960
CHASE_H = 540
CHASE_TOPIC = "/camera/chase/image_raw"
CHASE_INFO_TOPIC = "/camera/chase/camera_info"
# rover local frame 기준 chase cam offset (x=뒤, z=위, y=좌우 평행). rover
# 의 forward 가 local +x (Ackermann 주행 + spawn yaw 1.576 검증). 뒤쪽 6m,
# 위쪽 3m, 옆 0 (정후방). target 은 rover origin 보다 약간 위(+1.0) 로 잡아
# 부감 효과.
CHASE_OFFSET_LOCAL = (-6.0, 0.0, 3.0)
CHASE_TARGET_Z_BIAS = 1.0


def _add_overview_camera(stage) -> str:
    """World prim 에 부감 카메라 prim 추가. visibility=invisible 로 viewport
    오버레이 흔적 제거 — ROS 캡쳐와는 무관."""
    cam = UsdGeom.Camera.Define(stage, OVERVIEW_CAM)
    xf = UsdGeom.Xformable(cam)
    xf.AddTranslateOp().Set(Gf.Vec3d(*OVERVIEW_XYZ))
    xf.AddRotateXOp().Set(float(OVERVIEW_PITCH_DEG))
    cam.GetFocalLengthAttr().Set(14.0)
    # 16:9 aperture — 가로 21mm × 세로 11.8mm 비율 유지로 영상 stretching 방지.
    cam.GetHorizontalApertureAttr().Set(20.955)
    cam.GetVerticalApertureAttr().Set(20.955 * (OVERVIEW_H / OVERVIEW_W))
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 100000.0))
    UsdGeom.Imageable(cam.GetPrim()).MakeInvisible()
    print(f"[run_v3] overview camera 생성: {OVERVIEW_CAM} "
          f"@ {OVERVIEW_XYZ} pitch={OVERVIEW_PITCH_DEG}°")
    return OVERVIEW_CAM


def _create_overview_graph() -> None:
    """별도 OmniGraph — RP + ROS2CameraHelper + CameraInfoHelper.
    vehicle_v3 내장 그래프와 분리해 reference 충돌 회피.
    """
    import omni.graph.core as og
    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": OVERVIEW_GRAPH, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnPlaybackTick"),
                ("RPOverview", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("CamOverviewRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("CamOverviewInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
            ],
            keys.SET_VALUES: [
                ("RPOverview.inputs:width", OVERVIEW_W),
                ("RPOverview.inputs:height", OVERVIEW_H),
                ("CamOverviewRgb.inputs:topicName", OVERVIEW_TOPIC),
                ("CamOverviewRgb.inputs:type", "rgb"),
                ("CamOverviewRgb.inputs:frameId", "overview_camera"),
                ("CamOverviewInfo.inputs:topicName", OVERVIEW_INFO_TOPIC),
                ("CamOverviewInfo.inputs:frameId", "overview_camera"),
            ],
            keys.CONNECT: [
                ("OnTick.outputs:tick", "RPOverview.inputs:execIn"),
                ("RPOverview.outputs:execOut", "CamOverviewRgb.inputs:execIn"),
                ("RPOverview.outputs:renderProductPath",
                 "CamOverviewRgb.inputs:renderProductPath"),
                ("RPOverview.outputs:execOut", "CamOverviewInfo.inputs:execIn"),
                ("RPOverview.outputs:renderProductPath",
                 "CamOverviewInfo.inputs:renderProductPath"),
            ],
        },
    )

    # cameraPrim relationship 연결 — build_vehicle_v3 의 _set_targets 패턴 차용.
    stage = omni.usd.get_context().get_stage()
    try:
        from isaacsim.core.utils.prims import set_targets
        set_targets(
            prim=stage.GetPrimAtPath(f"{OVERVIEW_GRAPH}/RPOverview"),
            attribute="inputs:cameraPrim",
            target_prim_paths=[OVERVIEW_CAM])
    except Exception as e:
        print(f"[run_v3] ⚠ overview cameraPrim 연결 실패: {e}")
        return
    print(f"[run_v3] overview graph 구성 완료 → {OVERVIEW_TOPIC}")


def _add_chase_camera(stage, cam_path: str = CHASE_CAM):
    """rover 뒤에서 따라가는 chase 카메라 prim. 매 step _update_chase_cam 이
    transform op 의 4x4 매트릭스를 갱신해 rover 를 따라간다.
    multi 모드에선 cam_path 로 per-rover 격리 (예: /World/ChaseCam_rover_1)."""
    cam = UsdGeom.Camera.Define(stage, cam_path)
    xf = UsdGeom.Xformable(cam)
    xf.ClearXformOpOrder()
    transform_op = xf.AddTransformOp()
    cam.GetFocalLengthAttr().Set(20.0)
    cam.GetHorizontalApertureAttr().Set(20.955)
    cam.GetVerticalApertureAttr().Set(20.955 * (CHASE_H / CHASE_W))
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.1, 100000.0))
    UsdGeom.Imageable(cam.GetPrim()).MakeInvisible()
    print(f"[run_v3] chase camera 생성: {cam_path} offset_local={CHASE_OFFSET_LOCAL}")
    return transform_op


def _create_chase_graph(graph_path: str = CHASE_GRAPH,
                        cam_path: str = CHASE_CAM,
                        topic: str = CHASE_TOPIC,
                        info_topic: str = CHASE_INFO_TOPIC,
                        frame_id: str = "chase_camera") -> None:
    """vehicle_v3 / overview 와 분리된 별도 OmniGraph (RP + CamHelper).
    multi 모드에선 graph_path/topic 을 ns 별로 분리 (per-rover chase stream).
    """
    import omni.graph.core as og
    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnTick", "omni.graph.action.OnPlaybackTick"),
                ("RPChase", "isaacsim.core.nodes.IsaacCreateRenderProduct"),
                ("CamChaseRgb", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("CamChaseInfo", "isaacsim.ros2.bridge.ROS2CameraInfoHelper"),
            ],
            keys.SET_VALUES: [
                ("RPChase.inputs:width", CHASE_W),
                ("RPChase.inputs:height", CHASE_H),
                ("CamChaseRgb.inputs:topicName", topic),
                ("CamChaseRgb.inputs:type", "rgb"),
                ("CamChaseRgb.inputs:frameId", frame_id),
                ("CamChaseInfo.inputs:topicName", info_topic),
                ("CamChaseInfo.inputs:frameId", frame_id),
            ],
            keys.CONNECT: [
                ("OnTick.outputs:tick", "RPChase.inputs:execIn"),
                ("RPChase.outputs:execOut", "CamChaseRgb.inputs:execIn"),
                ("RPChase.outputs:renderProductPath",
                 "CamChaseRgb.inputs:renderProductPath"),
                ("RPChase.outputs:execOut", "CamChaseInfo.inputs:execIn"),
                ("RPChase.outputs:renderProductPath",
                 "CamChaseInfo.inputs:renderProductPath"),
            ],
        },
    )
    stage = omni.usd.get_context().get_stage()
    try:
        from isaacsim.core.utils.prims import set_targets
        set_targets(
            prim=stage.GetPrimAtPath(f"{graph_path}/RPChase"),
            attribute="inputs:cameraPrim",
            target_prim_paths=[cam_path])
    except Exception as e:
        print(f"[run_v3] ⚠ chase cameraPrim 연결 실패: {e}")
        return
    print(f"[run_v3] chase graph 구성 완료 → {topic}")


def _find_articulation_root(stage, rover_root: str = ""):
    """PhysicsArticulationRootAPI 가 있는 첫 prim.

    rover_root 가 주어지면 그 prim 의 자손 안에서만 검색 (multi-rover 격리).
    없으면 stage 전체 traverse (단일 모드).
    """
    if rover_root:
        root = stage.GetPrimAtPath(rover_root)
        if not root.IsValid():
            return None
        for prim in Usd.PrimRange(root):
            if prim.HasAPI("PhysicsArticulationRootAPI"):
                return prim
        return None
    for prim in stage.Traverse():
        if prim.HasAPI("PhysicsArticulationRootAPI"):
            return prim
    for prim in stage.Traverse():
        if prim.GetName() == "base_link" and "m0609" in str(prim.GetPath()):
            return prim
    return None


def _update_chase_cam(stage, transform_op, state: dict) -> None:
    """rover 의 world transform 을 읽어 chase cam 위치/시선을 매 step 갱신.

    rover local +x = forward (Ackermann 주행 + spawn yaw=π/2 검증). chase 는
    rover 뒤(-x), 위(+z) 에서 rover 머리 위쪽(+target_z_bias) 을 바라봄.
    USD Camera convention: -Z forward, +Y up, +X right (camera local).

    GT_SCRIPT 와 동일 패턴: articulation root 를 traverse 로 찾고 매 호출마다
    새 XformCache 로 stale 방지. ROVER_PRIM (reference xform root) 은 spawn
    시점 고정이라 사용 불가.
    """
    artic = state.get("artic")
    if artic is None or not artic.IsValid():
        # state 에 "rover_root" 가 있으면 그 안에서만 검색 (multi-rover).
        rover_root = state.get("rover_root", "")
        artic = _find_articulation_root(stage, rover_root)
        state["artic"] = artic
        if artic is None:
            return
    cache = UsdGeom.XformCache()
    m = cache.GetLocalToWorldTransform(artic)
    # ExtractTranslation / row vectors 로 rover 의 world frame 축들.
    rover_pos = m.ExtractTranslation()
    # row 0 = rover local +x in world, row 1 = +y, row 2 = +z.
    rx = Gf.Vec3d(m[0][0], m[0][1], m[0][2])
    ry = Gf.Vec3d(m[1][0], m[1][1], m[1][2])
    rz = Gf.Vec3d(m[2][0], m[2][1], m[2][2])
    ox, oy, oz = CHASE_OFFSET_LOCAL
    eye = Gf.Vec3d(rover_pos) + rx * ox + ry * oy + rz * oz
    target = Gf.Vec3d(rover_pos) + rz * CHASE_TARGET_Z_BIAS

    # look-at 매트릭스 — camera local 의 -z(forward) 가 (target - eye) 향함.
    fwd = (target - eye)
    fwd_len = fwd.GetLength()
    if fwd_len < 1e-6:
        return
    fwd = fwd / fwd_len
    world_up = Gf.Vec3d(0, 0, 1)
    right = Gf.Cross(fwd, world_up)
    right_len = right.GetLength()
    if right_len < 1e-6:
        right = Gf.Vec3d(1, 0, 0)
    else:
        right = right / right_len
    up = Gf.Cross(right, fwd)

    # USD Matrix4d (row-major). camera->world transform:
    #   col 0 = right, col 1 = up, col 2 = -forward, col 3 = eye.
    cam_xform = Gf.Matrix4d(
        right[0], right[1], right[2], 0.0,
        up[0],    up[1],    up[2],    0.0,
        -fwd[0],  -fwd[1],  -fwd[2],  0.0,
        eye[0],   eye[1],   eye[2],   1.0,
    )
    transform_op.Set(cam_xform)


def configure_sun_camera(stage) -> None:
    """Widen the baked v3 sun camera at runtime.

    vehicle_v3.usd was built with a narrow 24 mm lens. The visible sun is a
    bearing target, so it needs the wide view used in the proven feature/local
    setup; otherwise the detector may lock onto bright terrain instead.
    """
    prim = stage.GetPrimAtPath(SUN_CAMERA_PRIM)
    if not prim or not prim.IsValid():
        print(f"[run_v3] ⚠ sun camera 없음: {SUN_CAMERA_PRIM}")
        return
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0.10, 0.0, 2.50))
    # Match the proven v2 sun-camera mount: optical axis points upward from
    # the rover body, with image axes stable for bearing extraction.
    xform.AddRotateXYZOp().Set(Gf.Vec3f(180.0, 0.0, -90.0))
    cam = UsdGeom.Camera(prim)
    cam.GetFocalLengthAttr().Set(6.0)
    cam.GetHorizontalApertureAttr().Set(20.955)
    cam.GetVerticalApertureAttr().Set(15.2908)
    cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 10000.0))
    print(f"[run_v3] sun camera wide lens applied: {SUN_CAMERA_PRIM}")


def create_visual_sun(stage) -> None:
    """Create a visible sun disk/sphere for the upward sun camera.

    Isaac DistantLight illuminates the scene but is not a visible object in the
    camera image. T5 sun_yaw needs an actual bright blob, so we add one far
    above the map at the same azimuth as the localization world_sun_yaw.
    """
    distance = 2000.0
    radius = 100.0

    horizontal_distance = distance * math.cos(WORLD_SUN_ELEVATION)
    x = math.cos(WORLD_SUN_YAW) * horizontal_distance
    y = math.sin(WORLD_SUN_YAW) * horizontal_distance
    z = distance * math.sin(WORLD_SUN_ELEVATION)

    sun = UsdGeom.Sphere.Define(stage, Sdf.Path(VISUAL_SUN_PRIM))
    sun.GetRadiusAttr().Set(radius)
    sun.GetDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.86, 0.22)])
    UsdGeom.XformCommonAPI(sun.GetPrim()).SetTranslate(Gf.Vec3d(x, y, z))

    mat = UsdShade.Material.Define(stage, Sdf.Path("/World/VisualSunMaterial"))
    shader = UsdShade.Shader.Define(
        stage,
        Sdf.Path("/World/VisualSunMaterial/PreviewSurface"),
    )
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(1.0, 0.82, 0.12)
    )
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(1.0, 0.82, 0.12)
    )
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.0)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(sun.GetPrim()).Bind(mat)

    light = UsdLux.SphereLight.Define(stage, Sdf.Path(VISUAL_SUN_LIGHT))
    light.CreateRadiusAttr(radius)
    light.CreateIntensityAttr(250000.0)
    light.CreateExposureAttr(0.0)
    light.CreateColorAttr(Gf.Vec3f(1.0, 0.86, 0.18))
    UsdGeom.XformCommonAPI(light.GetPrim()).SetTranslate(Gf.Vec3d(x, y, z))

    print(
        f"[run_v3] visual sun: {VISUAL_SUN_PRIM} "
        f"pos=({x:.1f}, {y:.1f}, {z:.1f}), "
        f"yaw={WORLD_SUN_YAW:.3f} rad, "
        f"elev={math.degrees(WORLD_SUN_ELEVATION):.1f} deg, "
        f"radius={radius:.1f}"
    )



# ─── 단일 rover (기존 동작 호환) ──────────────────────────────────────
DEFAULT_ROVER_PRIM = "/World/Rover"


# ─── ScriptNode 소스 — 다중 rover 용 ─────────────────────────────────
# build_vehicle_v3.py 의 GT_SCRIPT / GRASP_SCRIPT 와 동일 동작이지만 stage
# 전역 traverse 대신 rover root prim 아래만 검색 → 두 rover 가 같은 stage
# 에 있어도 서로 간섭하지 않음. rover root 는 스크립트에 박지 않고 노드
# 자신의 prim path 에서 구한다 (ScriptNode 는 전역 이름공간을 공유하므로
# rover 별로 다른 상수를 박으면 마지막 rover 값으로 덮어써짐).

GT_SCRIPT_NSAWARE_TPL = '''
from pxr import UsdGeom, Usd
import omni.usd
import omni.graph.core as og


def setup(db):
    pass


def _find_rover_root(node_path):
    """ScriptNode prim path 에서 rover root 추출.
    예: /World/Rover_1/ActionGraph/ReadGtPose → /World/Rover_1
    """
    parts = node_path.split("/")
    if len(parts) < 3:
        return None
    return "/" + parts[1] + "/" + parts[2]


def compute(db):
    stage = omni.usd.get_context().get_stage()
    # 자신 prim path 에서 rover root 동적 추출 — script content 가 공유돼도 인스턴스별 다른 path
    my_path = ""
    try:
        my_path = db.node.get_prim_path()
    except Exception:
        try:
            my_path = str(db.node.get_prim())
        except Exception:
            return False
    rover_root = _find_rover_root(my_path)
    if not rover_root:
        return False
    root_prim = stage.GetPrimAtPath(rover_root)
    if not root_prim or not root_prim.IsValid():
        return False
    artic = None
    for prim in Usd.PrimRange(root_prim):
        if prim.HasAPI("PhysicsArticulationRootAPI"):
            artic = prim
            break
    if artic is None:
        for prim in Usd.PrimRange(root_prim):
            if prim.GetName() == "base_link" and "m0609" in str(prim.GetPath()):
                artic = prim
                break
    if artic is None:
        return False
    cache = UsdGeom.XformCache()
    M = cache.GetLocalToWorldTransform(artic)
    t = M.ExtractTranslation()
    q = M.ExtractRotationQuat()
    db.outputs.position = [float(t[0]), float(t[1]), float(t[2])]
    qi = q.GetImaginary()
    db.outputs.orientation = [float(qi[0]), float(qi[1]), float(qi[2]),
                              float(q.GetReal())]
    db.outputs.execOut = og.ExecutionAttributeState.ENABLED
    return True
'''


GRASP_SCRIPT_NSAWARE_TPL = '''
from pxr import UsdPhysics, UsdGeom, Gf, Sdf, Usd
import omni.usd
import omni.graph.core as og
import math

# ── 중요: 이 스크립트는 모든 rover 에 "같은 내용" 으로 들어가야 한다 ──────────
# omni.graph.scriptnode 는 스크립트의 전역 이름(상수·함수·dict)을 ScriptNode
# 구현 모듈의 globals 하나에 주입한다 (OgnScriptNode.compute 의
# compute_fn.__globals__.update). 즉 stage 안의 모든 ScriptNode 가 전역
# 이름공간을 공유하고, 마지막에 초기화된 노드의 값이 전체를 덮어쓴다.
# 예전처럼 rover 별로 ROVER_ROOT / GRASP_JOINT_PATH 상수를 다르게 박으면
# 나중에 초기화된 rover 의 값으로 모든 rover 가 동작한다 → rover_2 의 pickup
# 명령이 rover_1 의 그리퍼·주변 광물에 적용되는 cross-talk (2026-11-23 재현).
# 그래서 rover root 는 매 compute 마다 자기 노드 prim path 에서 구하고,
# 상태는 rover root 별 dict 에 둔다. 이름은 다른 ScriptNode 스크립트와
# 겹치지 않게 전부 _grasp_ 접두어.
_GRASP_JOINT_BY_ROOT = __GRASP_JOINT_TABLE__
_GRASP_GRIPPER_LINK_NAME = "right_inner_finger"
_GRASP_SEARCH_RADIUS = 2.5  # m  WRIST_T_LINK6 best-effort 누적 IK 오차 흡수용 cheat (2026-05-27)

try:
    _GRASP_STATES  # 다른 rover 노드가 먼저 만들었으면 재사용 (attach 상태 보존)
except NameError:
    _GRASP_STATES = {}


def _grasp_component(v, i):
    try:
        return float(v[i])
    except Exception:
        return 0.0


def _grasp_rover_root(db):
    # ScriptNode prim path 에서 rover root 추출.
    # 예: /World/Rover_1/ActionGraph/GraspScript -> /World/Rover_1
    node_path = ""
    try:
        node_path = db.node.get_prim_path()
    except Exception:
        try:
            node_path = str(db.node.get_prim())
        except Exception:
            return None
    parts = str(node_path).split("/")
    if len(parts) < 3:
        return None
    return "/" + parts[1] + "/" + parts[2]


def _grasp_state(rover_root):
    st = _GRASP_STATES.get(rover_root)
    if st is None:
        st = {"attached_obj_path": None, "gripper_link_path": None,
              "arm_base_path": None}
        _GRASP_STATES[rover_root] = st
    return st


def _grasp_find_gripper_link(stage, rover_root):
    root = stage.GetPrimAtPath(rover_root)
    if not root.IsValid():
        return None
    for prim in Usd.PrimRange(root):
        if (prim.GetName() == _GRASP_GRIPPER_LINK_NAME
                and "onrobot" in str(prim.GetPath())):
            return str(prim.GetPath())
    return None


def _grasp_find_arm_base(stage, rover_root):
    # 로봇팔 articulation root (m0609 base_link) — snap distance 기준점.
    # IK 자세 폭주로 finger 가 텔레포트해도 base_link 는 안정.
    root = stage.GetPrimAtPath(rover_root)
    if not root.IsValid():
        return None
    for prim in Usd.PrimRange(root):
        if prim.HasAPI("PhysicsArticulationRootAPI"):
            return str(prim.GetPath())
    for prim in Usd.PrimRange(root):
        if prim.GetName() == "base_link" and "m0609" in str(prim.GetPath()):
            return str(prim.GetPath())
    return None


def _grasp_held_by_others(stage, joint_path):
    # 다른 rover 의 grasp FixedJoint 가 잡고 있는 mineral path 집합.
    held = set()
    for other in _GRASP_JOINT_BY_ROOT.values():
        if other == joint_path:
            continue
        jp = stage.GetPrimAtPath(other)
        if not jp or not jp.IsValid():
            continue
        for t in UsdPhysics.FixedJoint(jp).GetBody1Rel().GetTargets():
            held.add(str(t))
    return held


def _grasp_find_nearest_mineral(stage, tx, ty, exclude):
    cache = UsdGeom.XformCache()
    best_path = None
    best_d2 = _GRASP_SEARCH_RADIUS * _GRASP_SEARCH_RADIUS
    for prim in stage.Traverse():
        parent = prim.GetParent()
        if not parent or parent.GetName() != "Minerals":
            continue
        if not prim.IsValid():
            continue
        if str(prim.GetPath()) in exclude:
            continue
        imageable = UsdGeom.Imageable(prim)
        try:
            if imageable.ComputeVisibility() == UsdGeom.Tokens.invisible:
                continue
        except Exception:
            pass
        M = cache.GetLocalToWorldTransform(prim)
        p = M.ExtractTranslation()
        dx = p[0] - tx
        dy = p[1] - ty
        d2 = dx * dx + dy * dy
        if d2 < best_d2:
            best_d2 = d2
            best_path = str(prim.GetPath())
    return best_path, math.sqrt(best_d2) if best_path else float("inf")


def _grasp_attach(stage, joint_path, link_path, obj_path):
    if stage.GetPrimAtPath(joint_path).IsValid():
        stage.RemovePrim(joint_path)
    link_prim = stage.GetPrimAtPath(link_path)
    obj_prim = stage.GetPrimAtPath(obj_path)
    if not link_prim.IsValid() or not obj_prim.IsValid():
        return False
    joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
    joint.CreateBody0Rel().SetTargets([Sdf.Path(link_path)])
    joint.CreateBody1Rel().SetTargets([Sdf.Path(obj_path)])
    # Body0 = right_inner_finger frame origin (knuckle 회전축) 에 mineral attach.
    # 시연용으로 lever arm = 0 유지 → cargo basket 으로 swing 시 mineral inertia
    # 가 회전축 위에 있어 안정.
    # 졸업 과제 (list_to_fix): 우리 main 의 vehicle_v3.usd 의 right_inner_finger
    # frame 축 정의 검사 + finger tip 까지 정확한 offset 측정해 시각 개선.
    # T2 제안값 (0.06, 0, 0.14) 시도 → 좌표계 mismatch 의심 + cargo 회전 시
    # lever arm 효과로 oscillation 관찰 (2026-05-26 시연 검증). 원복.
    joint.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    return True


def _grasp_set_mineral_collision(stage, obj_path, enabled):
    if not obj_path:
        return
    prim = stage.GetPrimAtPath(obj_path)
    if not prim or not prim.IsValid():
        return
    for child in Usd.PrimRange(prim):
        if child.HasAPI(UsdPhysics.CollisionAPI):
            ca = UsdPhysics.CollisionAPI(child)
            enabled_attr = ca.GetCollisionEnabledAttr()
            if not enabled_attr:
                enabled_attr = ca.CreateCollisionEnabledAttr()
            enabled_attr.Set(bool(enabled))


def _grasp_hide(stage, obj_path):
    prim = stage.GetPrimAtPath(obj_path)
    if prim and prim.IsValid():
        imageable = UsdGeom.Imageable(prim)
        imageable.MakeInvisible()


def setup(db):
    pass


def compute(db):
    lin = db.inputs.linearVelocity
    ang = db.inputs.angularVelocity
    mode = _grasp_component(ang, 0)  # +1 pickup, -1 release
    if abs(mode) <= 0.5:
        return True
    rover_root = _grasp_rover_root(db)
    joint_path = _GRASP_JOINT_BY_ROOT.get(rover_root or "")
    if not joint_path:
        print(f"[grasp] rover root 판별 실패 — node 가 속한 root={rover_root!r} 가 "
              f"joint 테이블에 없음: {sorted(_GRASP_JOINT_BY_ROOT)}")
        return False
    st = _grasp_state(rover_root)
    stage = omni.usd.get_context().get_stage()
    if mode > 0.5:
        gripper_path = (st.get("gripper_link_path")
                        or _grasp_find_gripper_link(stage, rover_root))
        if not gripper_path:
            print(f"[grasp] {rover_root} gripper link 못 찾음")
            return False
        st["gripper_link_path"] = gripper_path
        # arm_base (articulation root) — snap 거리 측정 기준. finger 와 달리
        # IK 자세 폭주에 영향 안 받음 (rover body 와 함께 stable).
        arm_base_path = (st.get("arm_base_path")
                         or _grasp_find_arm_base(stage, rover_root))
        if not arm_base_path:
            print(f"[grasp] {rover_root} arm_base 못 찾음 — m0609 articulation root 없음")
            return False
        st["arm_base_path"] = arm_base_path
        # supervisor 가 보낸 추정 mineral 좌표 (참고용 log).
        req_x = _grasp_component(lin, 0)
        req_y = _grasp_component(lin, 1)
        # pickup 명령은 burst (6회) 로 온다. 이미 이 rover 가 광물을 잡고 있으면
        # joint 를 지웠다 다시 만들지 않는다 (매번 재생성하면 그 사이 프레임에
        # collision 꺼진 광물이 자유낙하).
        held = st.get("attached_obj_path")
        if held and stage.GetPrimAtPath(joint_path).IsValid():
            return True
        # ── snap 판정: arm_base GT 기준 nearest mineral ──────────────────
        # finger world transform 은 PhysX articulation 폭주 시 텔레포트 가능
        # (2026-05-27 rover_1 시연에서 gripper GT (19.78,20.95) 폭주 관찰).
        # arm_base 는 articulation root 라 stable. SEARCH_RADIUS 2.5m 안에
        # mineral 이 들어오면 nearest 를 그리퍼 finger 로 snap (attach 대상은
        # 그리퍼 그대로 — 시각적으로 그리퍼가 광물 들고 있는 모습 유지).
        arm_prim = stage.GetPrimAtPath(arm_base_path)
        bx, by = req_x, req_y
        if arm_prim and arm_prim.IsValid():
            cache = UsdGeom.XformCache()
            M = cache.GetLocalToWorldTransform(arm_prim)
            apos = M.ExtractTranslation()
            bx = float(apos[0])
            by = float(apos[1])
        mineral_path, dist = _grasp_find_nearest_mineral(
            stage, bx, by, _grasp_held_by_others(stage, joint_path))
        if not mineral_path:
            print(f"[grasp] {rover_root} pickup ignored — no mineral near arm_base "
                  f"({bx:.2f},{by:.2f}) within {_GRASP_SEARCH_RADIUS}m "
                  f"(requested=({req_x:.2f},{req_y:.2f}))")
            return True
        if _grasp_attach(stage, joint_path, gripper_path, mineral_path):
            st["attached_obj_path"] = mineral_path
            _grasp_set_mineral_collision(stage, mineral_path, False)
            print(f"[grasp] pickup OK — attached {mineral_path} to {gripper_path} "
                  f"(arm_base GT=({bx:.2f},{by:.2f}), "
                  f"requested=({req_x:.2f},{req_y:.2f}), "
                  f"target dist {dist:.2f}m, snapped, collision off)")
        return True
    # mode < -0.5 : release
    obj = st.get("attached_obj_path")
    if stage.GetPrimAtPath(joint_path).IsValid():
        stage.RemovePrim(joint_path)
    if obj:
        _grasp_hide(stage, obj)
        print(f"[grasp] {rover_root} release + hide {obj}")
    st["attached_obj_path"] = None
    return True
'''


# --grasp contact 에서는 부착·해제를 전부 main loop (ContactGrasp + GraspBridge)
# 가 한다. ScriptNode 가 같이 움직이면 joint 를 먼저 지워 버려 광물 숨김이
# 빠지므로 아무것도 하지 않는 스크립트로 바꾼다.
GRASP_SCRIPT_NOOP = '''
def setup(db):
    pass


def compute(db):
    return True
'''


# ─── helpers ─────────────────────────────────────────────────────────

def _ns_to_prim_name(ns: str) -> str:
    """네임스페이스 string 을 prim 이름으로 정규화 (예: 'rover_1' → 'Rover_1')."""
    s = ns.strip("/").strip()
    if not s:
        return "Rover"
    # Sdf path component 는 영문/숫자/_ 만 안전
    safe = "".join(c if (c.isalnum() or c == "_") else "_" for c in s)
    return safe[:1].upper() + safe[1:]


# ─── Standalone rclpy bridge — OmniGraph SubscribeTwist 우회 ───────────
# release msg 가 OmniGraph 측 race 로 한쪽 rover 의 SubscribeTwist 에 한 frame
# 도 안 들어오는 사례 (2026-05-27 rover_1/2 release fail 관찰). standard
# rclpy subscriber 가 동일 토픽을 별도로 받아 main thread 에서 stage 직접
# hide. ScriptNode 의 release 분기는 그대로 두므로 dual mechanism — 둘 중
# 하나만 잡혀도 mineral 사라짐 보장.
def _force_release_for_ns(stage, ns: str) -> bool:
    """ns 의 rover 의 grasp FixedJoint 와 attached mineral 강제 hide.
    ScriptNode 의 release 분기와 동일 효과. 이미 detach 됐으면 no-op.
    return: True = 실제 hide 수행, False = no-op."""
    ns_norm = (ns or "").strip("/").strip()
    # 단일 rover 모드(ns 없음)는 GraspBridge.publish_status 와 같은 경로를 쓴다.
    joint_path = (f"/World/grip_fixed_joint_{ns_norm}" if ns_norm
                  else "/World/grip_fixed_joint")
    ns_norm = ns_norm or "rover"
    joint_prim = stage.GetPrimAtPath(joint_path)
    if not joint_prim or not joint_prim.IsValid():
        return False
    # FixedJoint 의 body1 (mineral) path
    mineral_path = None
    try:
        fj = UsdPhysics.FixedJoint(joint_prim)
        targets = fj.GetBody1Rel().GetTargets()
        if targets:
            mineral_path = str(targets[0])
    except Exception:
        mineral_path = None
    # detach
    stage.RemovePrim(joint_path)
    if not mineral_path:
        print(f"[grasp-bridge] {ns_norm} FORCE-detach (no body1 target)")
        return True
    mineral_prim = stage.GetPrimAtPath(mineral_path)
    if not mineral_prim or not mineral_prim.IsValid():
        print(f"[grasp-bridge] {ns_norm} FORCE-detach (mineral {mineral_path} not found)")
        return True
    # collision 복원
    if mineral_prim.HasAPI(UsdPhysics.CollisionAPI):
        attr = UsdPhysics.CollisionAPI(mineral_prim).GetCollisionEnabledAttr()
        if attr:
            attr.Set(True)
    # invisible
    try:
        UsdGeom.Imageable(mineral_prim).MakeInvisible()
    except Exception:
        pass
    print(f"[grasp-bridge] {ns_norm} FORCE-release + hide {mineral_path}")
    return True


def _debug_grasp_trace(stage, namespaces, step: int) -> None:
    """grasp FixedJoint 가 존재하는 rover 마다 finger/mineral world 좌표 출력."""
    cache = UsdGeom.XformCache()
    for ns in namespaces:
        ns_norm = (ns or "").strip("/").strip() or "rover"
        jp = stage.GetPrimAtPath(f"/World/grip_fixed_joint_{ns_norm}")
        if not jp or not jp.IsValid():
            continue
        fj = UsdPhysics.FixedJoint(jp)
        b0 = fj.GetBody0Rel().GetTargets()
        b1 = fj.GetBody1Rel().GetTargets()
        if not b0 or not b1:
            print(f"[grasp-dbg] step={step} {ns_norm} joint 에 body target 없음")
            continue
        f_prim = stage.GetPrimAtPath(b0[0])
        m_prim = stage.GetPrimAtPath(b1[0])
        fp = cache.GetLocalToWorldTransform(f_prim).ExtractTranslation()
        mp = cache.GetLocalToWorldTransform(m_prim).ExtractTranslation()
        gap = (Gf.Vec3d(fp) - Gf.Vec3d(mp)).GetLength()
        vis = UsdGeom.Imageable(m_prim).ComputeVisibility()
        en = jp.GetAttribute("physics:jointEnabled").Get()
        print(f"[grasp-dbg] step={step} {ns_norm} "
              f"finger=({fp[0]:.2f},{fp[1]:.2f},{fp[2]:.2f}) "
              f"mineral=({mp[0]:.2f},{mp[1]:.2f},{mp[2]:.2f}) gap={gap:.3f}m "
              f"vis={vis} jointEnabled={en} {m_prim.GetName()}")


class GraspBridge:
    """별도 rclpy node — 모든 rover 의 /grasp/command 를 standard subscriber
    로 듣고 release 신호를 main thread 의 pending queue 에 push. main loop 가
    매 step 마다 drain → stage 직접 hide. OmniGraph subscriber 의 single-
    thread frame race 영향 받지 않음."""

    def __init__(self, namespaces):
        import rclpy
        from rclpy.node import Node as _RclpyNode
        from geometry_msgs.msg import Twist
        from functools import partial as _partial

        if not rclpy.ok():
            rclpy.init()
        from std_msgs.msg import String as _String
        self._String = _String
        node = _RclpyNode("grasp_bridge_listener")
        self._pending = []
        self._pickups = []
        # ns → (status publisher, grasp FixedJoint path). arm_executor 가 pickup
        # 직후 이 토픽으로 실제 부착 여부를 확인한다 (data = 잡은 mineral prim
        # 이름, 안 잡았으면 빈 문자열).
        self._status = {}
        for ns in namespaces:
            ns_norm = (ns or "").strip("/").strip()
            topic = (f"/{ns_norm}/grasp/command"
                     if ns_norm else "/grasp/command")
            node.create_subscription(
                Twist, topic,
                _partial(self._on_grasp, ns_norm), 100)
            node.get_logger().info(
                f"[grasp-bridge] subscribed: {topic} (depth=100)")
            status_topic = (f"/{ns_norm}/grasp/status"
                            if ns_norm else "/grasp/status")
            # 단일 rover 모드는 USD 내장 GraspScript 의 joint 경로를 씀.
            joint_path = (f"/World/grip_fixed_joint_{ns_norm}"
                          if ns_norm else "/World/grip_fixed_joint")
            self._status[ns_norm] = (
                node.create_publisher(_String, status_topic, 10), joint_path)
        self._node = node

    def publish_status(self, stage) -> None:
        """각 rover 의 grasp FixedJoint 존재 여부를 /<ns>/grasp/status 로 발행."""
        for pub, joint_path in self._status.values():
            held = ""
            jp = stage.GetPrimAtPath(joint_path)
            if jp and jp.IsValid():
                targets = UsdPhysics.FixedJoint(jp).GetBody1Rel().GetTargets()
                held = targets[0].name if targets else "unknown"
            msg = self._String()
            msg.data = held
            pub.publish(msg)

    def _on_grasp(self, ns, msg):
        if msg.angular.x < -0.5:
            self._pending.append(ns)
        elif msg.angular.x > 0.5:
            self._pickups.append(ns)

    def drain_pickups(self):
        out = self._pickups[:]
        self._pickups.clear()
        return out

    @property
    def node(self):
        return self._node

    def drain_releases(self):
        out = self._pending[:]
        self._pending.clear()
        return out


def _spawn_for(spots, idx, fallback=(0.0, 0.0, 1.0)):
    if not spots or idx >= len(spots):
        return fallback
    s = spots[idx]
    return (float(s["x"]), float(s["y"]), float(s["z"]) + 0.3)


def _close_spawn_for(spots, idx, spacing):
    """검증용 — spawn_locations[0] 기준 X 방향 spacing m 씩 떨어진 자리.

    spawn_locations[0] 의 z 사용 (해당 지점 terrain 높이) 으로 wrong-z 묻힘 방지.
    """
    base = (0.0, 0.0, 1.0)
    if spots:
        s0 = spots[0]
        base = (float(s0["x"]), float(s0["y"]), float(s0["z"]) + 0.3)
    return (base[0] + idx * spacing, base[1], base[2])


def _load_rover(stage, world, prim_path: str, spawn,
                usd_source: str = V3) -> None:
    """vehicle_v3 USD 를 prim_path 에 reference + 위치 설정.

    usd_source 가 V3 가 아니면 (다중 rover 시 per-rover copy) 별도 파일이라
    USD prototype 공유가 일어나지 않아 OmniGraph 데이터 분리 자연 보장.
    """
    add_reference_to_stage(usd_path=usd_source, prim_path=prim_path)
    prim = stage.GetPrimAtPath(prim_path)
    # 안전망 — 혹시라도 instance 화 됐으면 강제 해제
    try:
        prim.SetInstanceable(False)
    except Exception:
        pass
    xf = UsdGeom.Xformable(prim)
    top = None
    for op in xf.GetOrderedXformOps():
        if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
            top = op
            break
    if top is None:
        top = xf.AddTranslateOp()
    top.Set(Gf.Vec3d(*spawn))
    print(f"[run_v3] reference {os.path.basename(usd_source)} @ {prim_path}  spawn={spawn}")


def _per_rover_usd_copies(namespaces) -> dict:
    """rover 별 vehicle_v3 USD 복사본 경로 dict 반환 {ns: usd_path}.

    같은 파일을 여러 prim path 에 reference 하면 USD/OmniGraph 가 prototype
    공유로 데이터가 섞이는 문제 차단 — 각 rover 가 자기 전용 USD 파일을 봄.
    """
    out_dir = tempfile.mkdtemp(prefix="a2_isaac_rovers_")
    copies = {}
    for ns in namespaces:
        dst = os.path.join(out_dir, f"vehicle_v3_{ns}.usd")
        shutil.copyfile(V3, dst)
        copies[ns] = dst
        fixed = _localize_asset_paths(dst)
        print(f"[run_v3] per-rover USD copy: {V3.split('/')[-1]} → {dst}"
              + (f"  (asset 경로 {fixed}개 현지화)" if fixed else ""))
    return copies


def _localize_asset_paths(usd_path: str) -> int:
    """다른 PC 의 절대경로로 저장된 asset 경로를 이 checkout 기준으로 치환.

    vehicle_v3.usd 의 MDL 재질이 저장한 PC 의 절대경로
    (.../a2_isaac/isaac_sim/assets/...) 를 들고 있어, checkout 위치가 다르면
    재질 로드가 실패함. 존재하지 않는 절대경로 중 '/isaac_sim/assets/' 를
    포함하고 이 checkout 에 같은 파일이 있는 것만 바꿈. 치환 개수 반환.
    """
    from pxr import UsdUtils

    layer = Sdf.Layer.FindOrOpen(usd_path)
    if layer is None:
        return 0
    marker = "/isaac_sim/assets/"
    count = 0

    def _fix(path: str) -> str:
        nonlocal count
        raw = path[5:] if path.startswith("file:") else path
        if not raw.startswith("/") or os.path.exists(raw):
            return path
        i = raw.find(marker)
        if i < 0:
            return path
        local = ISAAC_SIM + raw[i + len("/isaac_sim"):]
        if not os.path.exists(local):
            return path
        count += 1
        return local

    UsdUtils.ModifyAssetPaths(layer, _fix)
    if count:
        layer.Save()
    return count


def _patch_topic_names(stage, rover_root: str, ns: str) -> int:
    """rover_root 아래 모든 OmniGraph 노드의 *:topicName 속성에 /{ns} prefix 추가.

    이미 prefix 가 박혀 있으면 (재실행 등) skip. ns 가 비어 있으면 아무것도 안 함.
    return: 변경한 attribute 개수
    """
    if not ns:
        return 0
    ns_norm = ns.strip("/").strip()
    if not ns_norm:
        return 0
    prefix = f"/{ns_norm}"
    root = stage.GetPrimAtPath(rover_root)
    if not root.IsValid():
        return 0
    patched = 0
    for prim in Usd.PrimRange(root):
        for attr in prim.GetAttributes():
            name = attr.GetName()
            if not (name.endswith(":topicName") or name == "topicName"):
                continue
            cur = attr.Get()
            if not isinstance(cur, str) or not cur:
                continue
            # 이미 prefix 적용돼 있으면 skip
            if cur.startswith(prefix + "/") or cur == prefix:
                continue
            # leading slash 보장
            if not cur.startswith("/"):
                cur = "/" + cur
            new_topic = prefix + cur
            attr.Set(new_topic)
            patched += 1
    return patched


def _grasp_joint_table() -> dict:
    """{rover root prim path: grasp FixedJoint path} — 전 rover 공통 테이블.
    joint 경로 규칙은 _force_release_for_ns 와 같아야 한다."""
    table = {}
    for ns in _a.rovers:
        ns_norm = (ns.strip("/").strip() or "rover")
        table[f"/World/{_ns_to_prim_name(ns)}"] = f"/World/grip_fixed_joint_{ns_norm}"
    return table


def _patch_script_nodes(stage, rover_root: str, ns: str) -> int:
    """rover_root 아래 GT/GRASP ScriptNode 의 inputs:script 를 rover-scoped 버전으로 교체.

    return: 패치한 ScriptNode 개수
    """
    ns_norm = (ns.strip("/").strip() or "rover")
    joint_path = f"/World/grip_fixed_joint_{ns_norm}"
    root = stage.GetPrimAtPath(rover_root)
    if not root.IsValid():
        return 0
    patched = 0
    for prim in Usd.PrimRange(root):
        if prim.GetTypeName() != "OmniGraphNode":
            continue
        name = prim.GetName()
        script_attr = prim.GetAttribute("inputs:script")
        if not script_attr or not script_attr.IsValid():
            continue
        if name == "ReadGtPose":
            # GT_SCRIPT 는 self-introspection (placeholder 없음). 모든 rover 동일 content,
            # 런타임에 자기 prim path 로 rover root 동적 결정.
            new_src = GT_SCRIPT_NSAWARE_TPL
            script_attr.Set(new_src)
            check = script_attr.Get()
            ok = (isinstance(check, str) and "_find_rover_root" in check)
            print(f"[run_v3]     {script_attr.GetPath()} ← ReadGtPose ({len(new_src)} chars, self-introspect={'OK' if ok else 'FAIL'})")
            patched += 1
        elif name == "GraspScript" and _a.grasp == "contact":
            script_attr.Set(GRASP_SCRIPT_NOOP)
            print(f"[run_v3]     {script_attr.GetPath()} ← 비움 (--grasp contact)")
            patched += 1
        elif name == "GraspScript":
            # 모든 rover 에 동일한 스크립트 (전 rover 의 root→joint 테이블 포함).
            # rover 별로 내용이 다르면 ScriptNode 전역 공유 때문에 cross-talk.
            new_src = GRASP_SCRIPT_NSAWARE_TPL.replace(
                "__GRASP_JOINT_TABLE__", repr(_grasp_joint_table()))
            script_attr.Set(new_src)
            check = script_attr.Get()
            ok = (isinstance(check, str) and rover_root in check and joint_path in check)
            print(f"[run_v3]     {script_attr.GetPath()} ← GraspScript ({len(new_src)} chars, joint table={'OK' if ok else 'FAIL'})")
            patched += 1
    return patched


def main() -> None:
    for f in (WORLD, V3):
        if not os.path.isfile(f):
            print(f"[run_v3] ✗ 파일 없음: {f}")
            app.close()
            sys.exit(1)

    world = World(stage_units_in_meters=1.0)
    add_reference_to_stage(usd_path=WORLD, prim_path="/World/MarsScene")
    print(f"[run_v3] 씬 로드: {WORLD}")

    # spawn 좌표 (terrain meta.json) — 단일/다중 모드 공통
    spots = []
    terrain_meta: dict = {}
    meta = os.path.join(TERRAIN_DIR, "meta.json")
    if os.path.isfile(meta):
        with open(meta) as f:
            terrain_meta = json.load(f)
        spots = terrain_meta.get("spawn_locations") or []

    stage = omni.usd.get_context().get_stage()
    chase_xform_op = None
    # multi 모드의 per-rover chase cam — main loop 에서 모두 갱신.
    # 각 항목: {"op": Sdf TransformOp, "state": dict(rover_root=...)}.
    multi_chase_cams: list = []

    if _a.rovers:
        # ── 다중 rover (T2 패턴) ──
        # sun_camera/visual_sun 은 단일 검증만 마쳤어 multi 미가동.
        # chase cam 은 per-rover 로 생성 (Web HUD active rover 카메라 슬롯용).
        usd_copies = _per_rover_usd_copies(_a.rovers)
        spacing = float(_a.spawn_spacing)
        if spacing > 0.0:
            print(f"[run_v3] close-spawn 모드: 간격 {spacing:.2f}m (A* 회피 검증)")
        for i, ns in enumerate(_a.rovers):
            prim_path = f"/World/{_ns_to_prim_name(ns)}"
            if spacing > 0.0:
                spawn_xyz = _close_spawn_for(spots, i, spacing)
            else:
                spawn_xyz = _spawn_for(spots, i, fallback=(i * 3.0, 0.0, 1.0))
            _load_rover(stage, world, prim_path, spawn_xyz,
                        usd_source=usd_copies[ns])
            for _ in range(5):
                app.update()
            n_topics = _patch_topic_names(stage, prim_path, ns)
            n_scripts = _patch_script_nodes(stage, prim_path, ns)
            print(f"[run_v3]   patched {n_topics} topicName attrs, "
                  f"{n_scripts} ScriptNode(s) → namespace /{ns}")
            # per-rover chase cam + OmniGraph (--no-chase 시 skip)
            if not _a.no_chase:
                ns_norm = ns.strip("/").strip()
                cam_path = f"/World/ChaseCam_{ns_norm}"
                graph_path = f"/World/ChaseGraph_{ns_norm}"
                topic = f"/{ns_norm}/camera/chase/image_raw"
                info_topic = f"/{ns_norm}/camera/chase/camera_info"
                op = _add_chase_camera(stage, cam_path=cam_path)
                _create_chase_graph(graph_path=graph_path, cam_path=cam_path,
                                    topic=topic, info_topic=info_topic,
                                    frame_id=f"{ns_norm}_chase_camera")
                multi_chase_cams.append({
                    "op": op,
                    "state": {"rover_root": prim_path},
                })
    else:
        # ── 단일 rover (기존 동작) ──
        spawn = (0.0, 0.0, 1.0)
        spawn_yaw = 0.0
        if spots:
            s = spots[0]
            spawn = (float(s["x"]), float(s["y"]), float(s["z"]) + 0.3)
            spawn_yaw = float(s.get("yaw", 0.0))

        # v3 reference — 그래프가 USD 에 내장돼 있어 그대로 따라온다.
        add_reference_to_stage(usd_path=V3, prim_path=ROVER_PRIM)
        # T5 localization (PR #11): sun camera lens 와이드 + 시각 sun disk 추가.
        configure_sun_camera(stage)
        create_visual_sun(stage)
        xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROVER_PRIM))
        top = None
        rot = None
        for op in xf.GetOrderedXformOps():
            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                top = op
            elif op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                rot = op
        if top is None:
            top = xf.AddTranslateOp()
        top.Set(Gf.Vec3d(*spawn))
        if abs(spawn_yaw) > 1e-9:
            if rot is None:
                rot = xf.AddRotateXYZOp()
            rot.Set(Gf.Vec3f(0.0, 0.0, math.degrees(spawn_yaw)))
        print(f"[run_v3] vehicle_v3 spawn: {spawn}, "
              f"yaw={math.degrees(spawn_yaw):.1f} deg")

        # World-fixed overview camera + 자체 OmniGraph 추가.
        if not _a.no_overview:
            _add_overview_camera(stage)
            _create_overview_graph()
        else:
            print("[run_v3] overview cam 비활성 (--no-overview)")
        # rover 뒤에서 따라가는 chase 카메라
        if not _a.no_chase:
            chase_xform_op = _add_chase_camera(stage)
            _create_chase_graph()
        else:
            print("[run_v3] chase cam 비활성 (--no-chase)")

    for _ in range(20):
        app.update()
    # 중력·접촉 재질 — 로버가 stage 에 올라온 뒤, physics 파싱(reset) 전에.
    rover_roots = ([f"/World/{_ns_to_prim_name(ns)}" for ns in _a.rovers]
                   if _a.rovers else [ROVER_PRIM])
    physics_info = apply_planet_physics(
        stage, world, _a.physics, meta=terrain_meta,
        scene_root="/World/MarsScene", rover_roots=rover_roots)
    drive_limits = None
    if _a.drive_limits != "off":
        base = (limits_for_gravity(physics_info["gravity"])
                if _a.physics != "earth" else None)
        if _a.drive_limits != "auto":
            # 항목별 override — earth 에서는 지구 기준값 위에 덮는다.
            drive_limits = parse_limits(_a.drive_limits,
                                        base or limits_for_gravity(9.81))
        else:
            drive_limits = base
    apply_drive_limits(stage, rover_roots, drive_limits)
    contact_grasp = None
    if _a.grasp == "contact":
        if _a.rovers:
            table = _grasp_joint_table()
            grasp_rovers = {ns.strip("/").strip(): (
                f"/World/{_ns_to_prim_name(ns)}",
                table[f"/World/{_ns_to_prim_name(ns)}"]) for ns in _a.rovers}
        else:
            # 단일 모드는 USD 내장 GraspScript 를 그대로 쓰므로 여기서 비운다.
            grasp_rovers = {"": (ROVER_PRIM, "/World/grip_fixed_joint")}
            for prim in Usd.PrimRange(stage.GetPrimAtPath(ROVER_PRIM)):
                if (prim.GetTypeName() == "OmniGraphNode"
                        and prim.GetName() == "GraspScript"):
                    prim.GetAttribute("inputs:script").Set(GRASP_SCRIPT_NOOP)
        contact_grasp = ContactGrasp(stage, grasp_rovers)
    world.reset()
    world.play()
    render_every = max(1, int(_a.render_every))
    if _a.no_viewport and not _a.headless:
        try:
            from omni.kit.viewport.utility import get_active_viewport
            get_active_viewport().updates_enabled = False
            print("[run_v3] 메인 뷰포트 렌더 중단 (--no-viewport)")
        except Exception as exc:
            print(f"[run_v3] ⚠ --no-viewport 적용 실패: {exc}")
    max_rtf = _a.max_rtf
    if max_rtf is None:
        max_rtf = 1.0 if render_every > 1 else 0.0
    physics_dt = float(world.get_physics_dt())
    print(f"[run_v3] render 설정: every={render_every} preset={_a.render_preset} "
          f"headless={_a.headless} no_viewport={_a.no_viewport} "
          f"max_rtf={max_rtf:g} physics_dt={physics_dt:.5f}")
    if _a.rovers:
        nss = ", ".join(f"/{n}" for n in _a.rovers)
        print(f"[run_v3] ready — {len(_a.rovers)}대 vehicle namespaces: {nss}")
    else:
        print("[run_v3] ready — v3 내장 Action Graph 가 센서 토픽 발행 중 "
              "(/imu/data /joint_states_raw /camera/* /camera/overview/* /camera/chase/*)")

    # ── Standalone rclpy GraspBridge — OmniGraph SubscribeTwist 우회 ──
    # release msg drop race 대비 dual mechanism. init 실패해도 main 진행.
    grasp_bridge = None
    grasp_executor = None
    bridge_namespaces = _a.rovers if _a.rovers else [""]
    try:
        grasp_bridge = GraspBridge(bridge_namespaces)
        from rclpy.executors import SingleThreadedExecutor
        grasp_executor = SingleThreadedExecutor()
        grasp_executor.add_node(grasp_bridge.node)
        print(f"[run_v3] GraspBridge ready — backup release listener "
              f"for {bridge_namespaces}")
    except Exception as exc:
        print(f"[run_v3] GraspBridge init FAILED ({exc}) — "
              f"기본 ScriptNode release 만으로 동작")
        grasp_bridge = None
        grasp_executor = None

    step = 0
    chase_state: dict = {}
    # GPU/RTF 부담 측정용 — 600 step 사이 wall-clock 도 함께 출력.
    import time
    last_log_wall = time.time()
    next_step_wall = time.perf_counter()
    try:
        while app.is_running():
            world.step(render=(step % render_every == 0))
            if max_rtf > 0.0:
                # 실시간 배율 상한 — 다음 step 예정 시각까지 대기. 밀렸으면
                # 빚을 쌓지 않고 현재 시각 기준으로 다시 잡음.
                next_step_wall += physics_dt / max_rtf
                wait = next_step_wall - time.perf_counter()
                if wait > 0.0:
                    time.sleep(wait)
                elif wait < -0.1:
                    next_step_wall = time.perf_counter()
            # rclpy bridge poll + pending release 처리. main thread 에서
            # 직접 stage 조작 (안전).
            if grasp_executor is not None and grasp_bridge is not None:
                try:
                    # 한 step 에 밀린 명령을 모두 처리 (pickup 은 팔이 내려오는
                    # 동안 연속으로 온다).
                    for _ in range(8):
                        grasp_executor.spin_once(timeout_sec=0.0)
                    sim_now = float(world.current_time)
                    for ns in grasp_bridge.drain_releases():
                        if contact_grasp is not None:
                            contact_grasp.disarm(ns)
                        _force_release_for_ns(stage, ns)
                    if contact_grasp is not None:
                        for ns in grasp_bridge.drain_pickups():
                            contact_grasp.arm(ns, sim_now)
                        contact_grasp.step(sim_now)
                    if step % 6 == 0:
                        grasp_bridge.publish_status(stage)
                except Exception as exc:
                    print(f"[run_v3] grasp bridge tick error: {exc}")
            if _a.debug_grasp and step % 12 == 0:
                _debug_grasp_trace(stage, bridge_namespaces, step)
            # 단일 모드 chase cam.
            if chase_xform_op is not None:
                _update_chase_cam(stage, chase_xform_op, chase_state)
            # multi 모드 per-rover chase cams.
            for entry in multi_chase_cams:
                _update_chase_cam(stage, entry["op"], entry["state"])
            if step % 600 == 0:
                now = time.time()
                dt = now - last_log_wall
                last_log_wall = now
                # 600 step / dt = 실시간 step rate. 60Hz sim 이면 ~60 step/s
                # = 600 step / 10s 가 정상. 그보다 길면 RTF<1 = lag.
                rate = (600.0 / dt) if dt > 0 else 0.0
                print(f"[run_v3] running... step {step}  "
                      f"({rate:.1f} step/s, 600-step wall={dt:.2f}s, "
                      f"RTF={rate * physics_dt:.2f})")
            step += 1
    except KeyboardInterrupt:
        pass
    finally:
        # rclpy bridge cleanup (init 실패 시는 None 이라 skip).
        if grasp_executor is not None:
            try:
                grasp_executor.shutdown()
            except Exception:
                pass
        if grasp_bridge is not None:
            try:
                grasp_bridge.node.destroy_node()
            except Exception:
                pass
        try:
            import rclpy as _rclpy
            if _rclpy.ok():
                _rclpy.shutdown()
        except Exception:
            pass
        app.close()


if __name__ == "__main__":
    main()
