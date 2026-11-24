"""HUD 초상화(isaac_supervisor/web/static/img/rover_portrait.png)를 지금 로버 모델로 다시 찍는다.

    tools/isaac-pypi isaac_sim/scripts/render_portrait.py

시뮬레이터가 실제로 쓰는 것과 같은 조립을 거친 로버를 찍는다 — vehicle_v3.usd 에
넓은 집게(wide_gripper.install_wide_gripper)를 달고, 화성 중력에서 지면에 내려앉힌
뒤, 팔을 --pose 자세로 든 모습. 그림을 손으로 그리거나 합성하지 않으므로 로버
모델이나 집게가 바뀌면 이 스크립트를 다시 돌리면 된다.

옵션
  --terrain terrain_00026   배경 지형
  --at X,Y | auto           로버를 세울 곳 (world m). auto = 평평하고 주변에 바위·구조물이 없는 곳
  --yaw DEG                 로버가 향하는 방향
  --pose j1,...,j6          팔 관절각 (rad)
  --cam X,Y,Z               카메라 위치 (로버 기준: x 앞, y 왼쪽, z 위)
  --look X,Y,Z              카메라가 보는 점 (로버 기준)
  --focal MM                초점거리 (가로 필름 20.955 mm 기준)
  --size N                  정사각 한 변 픽셀
  --out PATH                저장 위치
"""
import argparse
import json
import math
import os
import sys

_p = argparse.ArgumentParser()
_p.add_argument("--terrain", default="terrain_00026")
_p.add_argument("--at", default="auto")
_p.add_argument("--yaw", type=float, default=150.0)
# 팔을 들어 넓은 집게(가로 막대 + 손가락 둘)가 카메라를 향하게 한 자세
_p.add_argument("--pose", default="1.2,-0.1,0.9,0.0,2.34,1.2")
# 기준점은 팔 받침대(m0609/base_link) — 로버 앞 왼쪽 위에서 내려다본다
_p.add_argument("--cam", default="1.75,1.4,1.3")
_p.add_argument("--look", default="-0.1,0.0,0.06")
_p.add_argument("--focal", type=float, default=30.0)
_p.add_argument("--size", type=int, default=768)
_p.add_argument("--subframes", type=int, default=64)
_p.add_argument("--out", default="")
args, _ = _p.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True, "width": args.size, "height": args.size})

import numpy as np  # noqa: E402
import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import World  # noqa: E402
from isaacsim.core.prims import Articulation  # noqa: E402
from isaacsim.core.utils.stage import add_reference_to_stage  # noqa: E402
from pxr import Gf, Usd, UsdGeom, UsdPhysics  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from asset_paths import localized_copy  # noqa: E402
from planet_physics import apply_planet_physics  # noqa: E402
from wheel_colliders import lift_link_speed_cap, round_wheel_colliders  # noqa: E402
from wide_gripper import install_wide_gripper  # noqa: E402

ISAAC_SIM = os.path.dirname(HERE)
REPO = os.path.dirname(ISAAC_SIM)
OUT = args.out or os.path.join(REPO, "isaac_supervisor", "web", "static", "img",
                               "rover_portrait.png")
ARM_JOINTS = [f"joint_{k}" for k in range(1, 7)]


def vec(text):
    return np.array([float(v) for v in text.split(",")])


def clear_flat_spot(hm: np.ndarray, obstacles: np.ndarray, res: float):
    """평평하고(반경 2 m 안의 높이 차가 가장 작은) 장애물에서 5 m 넘게 떨어진 곳."""
    from scipy import ndimage

    k = int(round(4.0 / res))
    relief = ndimage.maximum_filter(hm, k) - ndimage.minimum_filter(hm, k)
    zoom = hm.shape[0] / obstacles.shape[0]
    far = ndimage.distance_transform_edt(obstacles == 0) * (50.0 / obstacles.shape[0])
    far = ndimage.zoom(far, zoom, order=1)[:hm.shape[0], :hm.shape[1]]
    ii, jj = np.mgrid[0:hm.shape[0], 0:hm.shape[1]]
    x, y = (jj + 0.5) * res - 25.0, (ii + 0.5) * res - 25.0
    ok = (far > 5.0) & (np.hypot(x, y) > 9.0) & (np.abs(x) < 19.0) & (np.abs(y) < 19.0)
    if not ok.any():
        raise SystemExit("평평하고 빈 곳을 못 찾았다 — --at X,Y 로 지정할 것")
    i, j = np.unravel_index(np.argmin(np.where(ok, relief, np.inf)), hm.shape)
    return float(x[i, j]), float(y[i, j]), float(relief[i, j])


yaw = math.radians(args.yaw)
tdir = f"{ISAAC_SIM}/assets/generated_terrains/{args.terrain}"
with open(f"{tdir}/meta.json") as f:
    meta = json.load(f)
hm = np.load(f"{tdir}/heightmap.npy")
res = 50.0 / hm.shape[0]
if args.at == "auto":
    x0, y0, relief = clear_flat_spot(hm, np.load(f"{tdir}/obstacle_grid.npy"), res)
    print(f"[portrait] 자리: ({x0:.1f}, {y0:.1f}) — 반경 2 m 높이 차 {relief:.2f} m")
else:
    x0, y0 = vec(args.at)
ground = float(hm[int((y0 + 25.0) / res), int((x0 + 25.0) / res)])

world = World(stage_units_in_meters=1.0, physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0)
add_reference_to_stage(usd_path=f"{ISAAC_SIM}/worlds/{args.terrain}.usd",
                       prim_path="/World/MarsScene")
ROOT = "/World/Rover"
# 재질 경로가 다른 PC 의 절대경로로 저장돼 있다 — 고친 복사본을 쓴다 (안 그러면 빨갛게 나온다).
rover_usd, fixed = localized_copy(f"{ISAAC_SIM}/assets/vehicle/vehicle_v3.usd")
print(f"[portrait] 로버 모델 복사본 {rover_usd} (재질 경로 {fixed}개 고침)")
add_reference_to_stage(usd_path=rover_usd, prim_path=ROOT)
stage = omni.usd.get_context().get_stage()
xf = UsdGeom.Xformable(stage.GetPrimAtPath(ROOT))
xf.ClearXformOpOrder()
xf.AddTranslateOp().Set(Gf.Vec3d(float(x0), float(y0), ground + 0.45))
xf.AddRotateZOp().Set(float(args.yaw))
for _ in range(10):
    app.update()
for prim in Usd.PrimRange(stage.GetPrimAtPath(ROOT)):      # ROS 그래프는 필요 없다
    if prim.GetTypeName() == "OmniGraph":
        prim.SetActive(False)

round_wheel_colliders(stage, [ROOT])
lift_link_speed_cap(stage, [ROOT])
apply_planet_physics(stage, world, "mars", meta=meta, scene_root="/World/MarsScene",
                     rover_roots=[ROOT])
grip = install_wide_gripper(stage, ROOT)
if grip is None:
    raise SystemExit("넓은 집게를 달지 못했다 — 로버 모델 구조가 바뀌었는지 확인")
for _ in range(5):
    app.update()

art_root = next(str(p.GetPath()) for p in Usd.PrimRange(stage.GetPrimAtPath(ROOT))
                if p.HasAPI(UsdPhysics.ArticulationRootAPI))
art = Articulation(prim_paths_expr=art_root, name="rover")
world.scene.add(art)
world.reset()
arm_idx = [art.get_dof_index(n) for n in ARM_JOINTS]
pose = vec(args.pose)[None, :]
art.set_joint_positions(pose, joint_indices=arm_idx)
art.set_joint_position_targets(pose, joint_indices=arm_idx)
for k in range(180):                                   # 3 s — 바퀴가 땅에 앉고 팔이 멈출 때까지
    art.set_joint_position_targets(pose, joint_indices=arm_idx)
    world.step(render=False)

pos, quat = art.get_world_poses()
p = np.asarray(pos[0], float)
w, qx, qy, qz = [float(v) for v in quat[0]]
R = np.array([[1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * w), 2 * (qx * qz + qy * w)],
              [2 * (qx * qy + qz * w), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * w)],
              [2 * (qx * qz - qy * w), 2 * (qy * qz + qx * w), 1 - 2 * (qx * qx + qy * qy)]])
tilt = math.degrees(math.acos(max(-1.0, min(1.0, R[2, 2]))))
q_now = art.get_joint_positions()[0][arm_idx]
print(f"[portrait] 로버 위치 {np.round(p, 2).tolist()}, 기울기 {tilt:.1f}°, "
      f"팔 {np.round(q_now, 2).tolist()} (목표 {np.round(pose[0], 2).tolist()})")

eye = p + R @ vec(args.cam)
target = p + R @ vec(args.look)
cam = UsdGeom.Camera.Define(stage, "/World/PortraitCam")
cam.CreateFocalLengthAttr(float(args.focal))
cam.CreateHorizontalApertureAttr(20.955)
cam.CreateVerticalApertureAttr(20.955)
cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 500.0))
view = Gf.Matrix4d().SetLookAt(Gf.Vec3d(*eye.tolist()), Gf.Vec3d(*target.tolist()),
                               Gf.Vec3d(0.0, 0.0, 1.0))
cxf = UsdGeom.Xformable(cam.GetPrim())
cxf.ClearXformOpOrder()
cxf.AddTransformOp().Set(view.GetInverse())

product = rep.create.render_product("/World/PortraitCam", (args.size, args.size))
rgb = rep.AnnotatorRegistry.get_annotator("rgb")
rgb.attach([product])
for _ in range(3):                                     # 첫 프레임들은 조명·텍스처가 덜 올라와 있다
    rep.orchestrator.step(rt_subframes=args.subframes)
img = np.asarray(rgb.get_data())[..., :3]
if img.size == 0 or img.max() == 0:
    raise SystemExit("렌더 결과가 비어 있다")

from PIL import Image  # noqa: E402

os.makedirs(os.path.dirname(OUT), exist_ok=True)
Image.fromarray(img).save(OUT, optimize=True)
print(f"[portrait] {OUT} ({img.shape[1]}×{img.shape[0]}, 평균 밝기 {img.mean():.0f})")
app.close()
