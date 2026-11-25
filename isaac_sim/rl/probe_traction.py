"""접지력 시험 — 평지에서 바퀴를 일정하게 헛돌리며 차체가 실제로 받는 힘을 잰다 (측정).

    tools/isaac-pypi isaac_sim/rl/probe_traction.py --wheels capsule

바퀴 속도를 "차체 속도 + slip" 으로 유지하면 접촉은 계속 미끄럼 상태이고, Coulomb
마찰이라면 차체 가속도는 slip 과 무관하게 (동마찰 계수 × 중력) 이어야 한다.
    유효 마찰계수 = 차체 가속도 / 중력
이 값이 설정값(레골리스 0.52)보다 작으면 시뮬레이션의 접촉이 힘을 덜 전달하는 것이다.

--wheels  mesh     : 로버 모델 그대로 (바퀴 mesh 의 convex hull = 다각형)
          cylinder : 원기둥 (PhysX 가 다시 다각형 convex 로 근사한다)
          capsule  : 캡슐 (굴러가는 방향으로 완전히 둥글다)
          sphere   : 구

--wheel-grip k 는 run_vehicle_v3 의 같은 옵션이 물리에 실제로 들어가는지 보는 시험이다
(같은 planet_physics.apply_planet_physics 를 거친다). 적용되면 구동·제동 유효 마찰이
레골리스 동마찰 × k 로 나와야 한다 (0.52 → ×1.25 는 0.65, ×1.5 는 0.78). k 를 바꿔도
0.52 그대로면 바퀴 재질이 접촉에 쓰이지 않는 것이다. 끝에 기대값과의 비를 찍는다.

    tools/isaac-pypi isaac_sim/rl/probe_traction.py --wheel-grip 1.25

결과 (terrain_00026, 로버 13대, 레골리스 동마찰 0.52):
    mesh 0.31~0.35 · cylinder 0.31~0.39 · capsule 0.50~0.52 · sphere 0.50~0.52
    제동(바퀴 잠김)은 어느 형상이든 0.48~0.54.
다각형 바퀴는 돌 때 꼭짓점이 땅을 치며 떠서 (헛도는 동안 물리 스텝의 82% 가 공중)
접촉이 이어지지 않는다.
"""
import argparse
import json
import os
import sys

_p = argparse.ArgumentParser()
_p.add_argument("--terrain", default="terrain_00026")
_p.add_argument("--wheels", default="capsule",
                choices=["mesh", "cylinder", "capsule", "sphere"])
_p.add_argument("--hz", type=float, default=60.0, help="물리 스텝 주기")
_p.add_argument("--control-hz", type=float, default=30.0)
_p.add_argument("--slips", default="0.15,0.5,1.0,2.0")
_p.add_argument("--out", default="")
_p.add_argument("--wheel-grip", type=float, default=1.0,
                help="바퀴 재질의 마찰 배율 (run_vehicle_v3 --wheel-grip 과 같은 경로). "
                     "유효 마찰이 설정 동마찰 × 이 값으로 나오는지 본다")
args, _ = _p.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from drive_env import REST_HEIGHT, RoverDriveEnv  # noqa: E402

RoverDriveEnv.PHYSICS_HZ = float(args.hz)
RoverDriveEnv.WHEEL_SHAPE = args.wheels
RoverDriveEnv.WHEEL_GRIP = args.wheel_grip

env = RoverDriveEnv(app, terrain=args.terrain, num_envs=16, seed=3, control_hz=args.control_hz)
env.friction_range = (1.0, 1.0)
env.randomize_friction()
n, dt, res, g = env.n, env.dt, env.res, env.gravity
mu_s, mu_d = env._base_friction["regolith"]
zones = [np.array(z["polygon"], float) for z in (env.meta.get("physics_zones") or [])]
cells = env._start_cells[np.random.default_rng(1).permutation(len(env._start_cells))]
starts = []
for c in cells:
    x, y = (c[1] + .5) * res - env.map_w / 2, (c[0] + .5) * res - env.map_h / 2
    if env.slope_near[c[0], c[1]] > 4.0:
        continue
    if any(z[:, 0].min() < x < z[:, 0].max() and z[:, 1].min() < y < z[:, 1].max() for z in zones):
        continue
    if all(np.hypot(x - s[0], y - s[1]) > 5 for s in starts):
        starts.append((x, y, env.hm[c[0], c[1]] + REST_HEIGHT + 0.08,
                       np.random.default_rng(len(starts)).uniform(-3.14, 3.14)))
    if len(starts) == n:
        break
k = len(starts)
starts += [starts[-1]] * (n - k)
print(f"[trac] 바퀴 {args.wheels}, 물리 {args.hz:.0f} Hz (제어 {1 / dt:.0f} Hz), 평지 출발점 {k}곳, "
      f"설정 마찰 정지 {mu_s} / 동 {mu_d}, 바퀴 배율 ×{args.wheel_grip:g} "
      f"(기대 동마찰 {mu_d * args.wheel_grip:.3f}), 중력 {g}", flush=True)

# 로버 모델의 차체 속도는 1.57 m/s 에서 막힌다 (probe 로 확인). 그 전 구간만 가속도 계산에 쓴다.
V_FIT_MAX = 1.1
rows = []
for slip in [float(x) for x in args.slips.split(",")]:
    env.reset_idx(np.arange(n), starts=np.array(starts))
    T_settle, T_drive, T_brake = 1.0, 1.6, 1.2
    ts, vx, vz, pitch = [], [], [], []
    for t in range(int((T_settle + T_drive + T_brake) / dt)):
        T = t * dt
        v = np.zeros(n)
        if T_settle <= T < T_settle + T_drive:
            v[:] = np.clip(env._state["v_b"][:, 0], 0.0, None) + slip
        env.step(command=(v, np.zeros(n)))
        ts.append(T)
        vx.append(env._state["v_b"][:k, 0].copy())
        vz.append(env._state["v_b"][:k, 2].copy())
        pitch.append(np.degrees(np.arcsin(np.clip(env._state["R"][:k, 2, 0], -1, 1))))
    ts, vx, vz, pitch = np.array(ts), np.array(vx), np.array(vz), np.array(pitch)
    a0, a1 = int((T_settle + 0.15) / dt), int((T_settle + T_drive) / dt)
    acc = []
    for q in range(k):                       # 차체 속도가 V_FIT_MAX 에 닿기 전까지만 쓴다
        e = np.nonzero(vx[a0:a1, q] > V_FIT_MAX)[0]
        e = max(int(e[0]) if len(e) else a1 - a0, 4)
        acc.append(np.polyfit(ts[a0:a0 + e], vx[a0:a0 + e, q], 1)[0])
    acc = np.array(acc)
    b0 = a1 + 1
    dec = []
    for q in range(k):                       # 제동: 속도가 처음의 30% 로 줄 때까지
        b = vx[b0:, q]
        e = np.nonzero(b < 0.3 * b[0])[0]
        e = max(int(e[0]) if len(e) else len(b) - 1, 2)
        dec.append(-np.polyfit(ts[b0:b0 + e], b[:e], 1)[0])
    dec = np.array(dec)
    r = {"slip": slip, "mu_drive": round(float(np.median(acc) / g), 3),
         "mu_drive_q25_75": [round(float(v) / g, 3) for v in np.percentile(acc, [25, 75])],
         "mu_brake": round(float(np.median(dec) / g), 3),
         "v_end": round(float(np.median(vx[a1 - 1])), 2),
         "bounce_vz_rms": round(float(np.sqrt(np.mean(vz[a0:a1] ** 2))), 3),
         "pitch_max": round(float(np.median(np.abs(pitch[a0:a1]).max(axis=0))), 1)}
    rows.append(r)
    print("[trac] " + json.dumps(r), flush=True)
# 배율이 물리에 들어갔는가: 잰 값 / (설정 동마찰 × 배율). 둥근 바퀴(capsule·sphere)는 배율 1.0 에서
# 구동 0.96~1.0, 제동 0.92~0.94 가 나온다 (README "바퀴 접지력 교정"). 배율을 줬는데 적용이 안 됐다면
# 이 비가 1/배율 로 떨어진다 (×1.25 → 0.80, ×1.5 → 0.67).
expect = mu_d * args.wheel_grip
drive = float(np.median([r["mu_drive"] for r in rows]))
brake = float(np.median([r["mu_brake"] for r in rows]))
verdict = {"wheel_grip": args.wheel_grip, "mu_expected": round(expect, 3),
           "mu_drive_med": round(drive, 3), "mu_brake_med": round(brake, 3),
           "drive_ratio": round(drive / expect, 3), "brake_ratio": round(brake / expect, 3),
           "ratio_if_not_applied": round(1.0 / args.wheel_grip, 3)}
print("[trac] 배율 확인 " + json.dumps(verdict, ensure_ascii=False), flush=True)
if args.out:
    with open(args.out, "w") as f:
        json.dump({"wheels": args.wheels, "hz": args.hz, "mu_set": [mu_s, mu_d],
                   "wheel_grip": args.wheel_grip, "verdict": verdict, "rows": rows}, f,
                  ensure_ascii=False)
app.close()
