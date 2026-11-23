"""planet_physics 가 실제 시뮬레이션에 먹혔는지 측정으로 확인.

  tools/isaac-pypi isaac_sim/scripts/check_planet_physics.py --terrain terrain_00026 --physics mars

측정 항목
  · 자유낙하 가속도 — 상자를 떨어뜨려 z(t) 를 2차식으로 맞춘다.
  · 구역별 동마찰 계수 — 바퀴와 같은 재질의 납작한 상자를 평평한 곳에서
    밀어 감속도를 잰다 (mu = a / g). 경사 영향을 지우려고 +x, -x 두 방향 평균.
로버·ROS 는 띄우지 않는다.
"""
import argparse
import json
import math
import os
import sys

try:
    sys.stdout.reconfigure(line_buffering=True)   # app.close() 가 버퍼를 버리므로
except Exception:
    pass

_p = argparse.ArgumentParser()
_p.add_argument("--terrain", default="terrain_00026")
_p.add_argument("--physics", choices=["mars", "earth"], default="mars")
_a, _ = _p.parse_known_args()

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.core.utils.stage import add_reference_to_stage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import planet_physics as pp  # noqa: E402

ISAAC_SIM = os.path.dirname(HERE)
WORLD = f"{ISAAC_SIM}/worlds/{_a.terrain}.usd"
TDIR = f"{ISAAC_SIM}/assets/generated_terrains/{_a.terrain}"
V0 = 1.5          # 미는 속도 (m/s)
DT = 1.0 / 60.0


def find_flat_patches(meta, hm, obstacle):
    """구역마다 가장 평평하고 장애물 없는 3x3 m 자리의 중심 (x, y, z, 경사°)."""
    res = float(meta["resolution_m"])
    ox, oy = meta["origin"]["x"], meta["origin"]["y"]
    gy, gx = np.gradient(hm, res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    zones = meta.get("physics_zones") or []
    names = ["regolith"] + [str(z["type"]) for z in zones]
    best = {}
    half = int(1.5 / res)
    step = int(0.5 / res)
    n = hm.shape[0]
    bc = meta["basecamp"]["center"]
    for i in range(half + step, n - half - step, step):
        for j in range(half + step, n - half - step, step):
            x, y = ox + j * res, oy + i * res
            if math.hypot(x - bc["x"], y - bc["y"]) < 8.0:
                continue                      # 베이스캠프 구조물 근처 제외
            if obstacle[i - half:i + half, j - half:j + half].any():
                continue
            # 3x3 m 자리 전체가 한 구역 안에 있어야 함
            corners = {pp.zone_at(x + sx * 1.5, y + sy * 1.5, zones)
                       for sx in (-1, 1) for sy in (-1, 1)}
            if len(corners) != 1:
                continue
            name = corners.pop()
            s = float(slope[i - half:i + half, j - half:j + half].max())
            if name in names and (name not in best or s < best[name][3]):
                best[name] = (x, y, float(hm[i, j]), s)
    return best


def main():
    meta = json.load(open(os.path.join(TDIR, "meta.json")))
    hm = np.load(os.path.join(TDIR, "heightmap.npy"))
    obstacle = np.load(os.path.join(TDIR, "obstacle_grid.npy"))
    patches = find_flat_patches(meta, hm, obstacle)

    world = World(stage_units_in_meters=1.0, physics_dt=DT, rendering_dt=DT)
    add_reference_to_stage(usd_path=WORLD, prim_path="/World/MarsScene")
    stage = omni.usd.get_context().get_stage()
    for _ in range(10):
        app.update()
    info = pp.apply_planet_physics(stage, world, _a.physics, meta=meta,
                                   scene_root="/World/MarsScene", rover_roots=())
    wheel_mat = None
    if _a.physics != "earth":
        from pxr import UsdShade
        wheel_mat = UsdShade.Material(
            stage.GetPrimAtPath(f"{pp.MATERIAL_ROOT}/WheelGrouser"))

    sliders = {}
    for name, (x, y, z, s) in patches.items():
        obj = world.scene.add(DynamicCuboid(
            prim_path=f"/World/Test/slider_{name}", name=f"slider_{name}",
            position=np.array([x, y, z + 0.12]), scale=np.array([0.5, 0.5, 0.1]),
            mass=5.0, color=np.array([0.9, 0.2, 0.2])))
        if wheel_mat is not None:
            pp.bind_physics(obj.prim, wheel_mat)
        sliders[name] = obj
    x0, y0, z0, _ = patches["regolith"]
    faller = world.scene.add(DynamicCuboid(
        prim_path="/World/Test/faller", name="faller",
        position=np.array([x0 + 1.2, y0 + 1.2, z0 + 8.0]),
        scale=np.array([0.2, 0.2, 0.2]), mass=1.0))

    world.reset()
    # ── 자유낙하 ────────────────────────────────────────────────────────
    zs = []
    for _ in range(36):
        world.step(render=False)
        zs.append(float(faller.get_world_pose()[0][2]))
    t = np.arange(1, len(zs) + 1) * DT
    g_meas = -2.0 * float(np.polyfit(t, zs, 2)[0])

    # ── 미끄럼 ──────────────────────────────────────────────────────────
    for _ in range(120):                     # 안착
        world.step(render=False)
    rest = {n: o.get_world_pose() for n, o in sliders.items()}
    decel = {n: [] for n in sliders}
    for sign in (+1.0, -1.0):
        for n, o in sliders.items():
            o.set_world_pose(*rest[n])
            o.set_linear_velocity(np.zeros(3))
            o.set_angular_velocity(np.zeros(3))
        for _ in range(30):
            world.step(render=False)
        for o in sliders.values():
            o.set_linear_velocity(np.array([sign * V0, 0.0, 0.0]))
        prev = {n: o.get_world_pose()[0] for n, o in sliders.items()}
        speeds = {n: [] for n in sliders}
        for _ in range(360):
            world.step(render=False)
            for n, o in sliders.items():
                p = o.get_world_pose()[0]
                speeds[n].append(float(np.hypot(*(p - prev[n])[:2])) / DT)
                prev[n] = p
        for n, v in speeds.items():
            v = np.array(v)
            tt = np.arange(len(v)) * DT
            sel = (v < 0.9 * V0) & (v > 0.2 * V0)
            # 정지 이후 표본 제외: 처음으로 0.2*V0 아래로 내려간 시점까지만
            below = np.nonzero(v <= 0.2 * V0)[0]
            if len(below):
                sel[below[0]:] = False
            if sel.sum() >= 4:
                decel[n].append(-float(np.polyfit(tt[sel], v[sel], 1)[0]))

    result = {"physics": _a.physics, "terrain": _a.terrain,
              "gravity_measured": round(g_meas, 3),
              "gravity_configured": info.get("gravity"), "zones": {}}
    print(f"\n[check] physics={_a.physics} terrain={_a.terrain}")
    print(f"[check] 자유낙하 측정 g = {g_meas:.3f} m/s²  (설정 {info.get('gravity')})")
    for n, (x, y, z, s) in patches.items():
        a = decel[n]
        mu = float(np.mean(a)) / g_meas if a else float("nan")
        want = (info.get("materials") or {}).get(n, {}).get("dynamic_friction")
        result["zones"][n] = {"patch_xy": [round(x, 2), round(y, 2)],
                              "max_slope_deg": round(s, 2),
                              "mu_dynamic_measured": round(mu, 3),
                              "mu_dynamic_configured": want,
                              "runs": len(a)}
        print(f"[check] {n:9s} 자리=({x:6.1f},{y:6.1f}) 최대경사 {s:4.1f}°  "
              f"측정 동마찰 {mu:.3f}  (설정 {want})  표본 {len(a)}방향")
    print("[check] RESULT " + json.dumps(result, ensure_ascii=False))


try:
    main()
finally:
    app.close()
