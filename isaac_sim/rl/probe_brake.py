"""내리막 제동 시험 — "속도를 줄이라는 명령이 언제, 얼마나 듣나" 를 잰다 (측정).

    tools/isaac-pypi isaac_sim/rl/probe_brake.py --terrain terrain_00015 --out <json>

경사가 고른 내리막 띠(2.4 m)를 경사 구간별로 찾아, 띠 위 2.2 m 에서 내리막 쪽으로 달려 들어가게
한 뒤 띠의 시작(s = 0)에서 감속 명령을 준다. 명령은 실제 구동부와 같은 제한
(drive_limits: 최고 0.92 m/s, 가속 0.72 m/s²)을 거친다.

  잰 것: 감속 명령 뒤의 차체 속도, 바퀴 속도(관절 속도를 차체 속도로 환산), 제한을 거친 명령,
         멈출 때까지 간 거리.
  --brake  감속 명령 뒤의 목표 속도들 (0.25 = 접근의 느린 구간, 0 = 정지)
"""
import argparse
import json
import math
import os
import sys

_p = argparse.ArgumentParser()
_p.add_argument("--terrain", default="terrain_00015")
_p.add_argument("--out", default="")
_p.add_argument("--num-envs", type=int, default=12)
_p.add_argument("--bins", default="0,6,10,13,16")
_p.add_argument("--brake", default="0.25,0")
_p.add_argument("--speed", type=float, default=0.92)
_p.add_argument("--seed", type=int, default=0)
args, _ = _p.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drive_env import REST_HEIGHT, WHEEL_RADIUS, WHEEL_VEL_GAIN, RoverDriveEnv  # noqa: E402

env = RoverDriveEnv(app, terrain=args.terrain, num_envs=args.num_envs, seed=args.seed)
env.friction_range = (1.0, 1.0)
env.randomize_friction()
n, dt, res = env.n, env.dt, env.res
RUN, BAND = 2.2, 2.4
hm = env.hm
hs = ndimage.uniform_filter(hm, int(round(0.8 / res)))
gy, gx = np.gradient(hs, res)
half_w, half_h = env.map_w / 2, env.map_h / 2
K_WHEEL = WHEEL_VEL_GAIN / (WHEEL_RADIUS * 2.0)          # 관절 속도 = K × 차체 속도 명령


def clear(x, y):
    i, j = env._cell(x, y)
    return (np.abs(x) < half_w - 3.0) & (np.abs(y) < half_h - 3.0) & (env.solid_dist[i, j] > 1.5)


def grade_profile(x, y, ux, uy, s):
    xs = x[:, None] + s[None, :] * ux[:, None]
    ys = y[:, None] + s[None, :] * uy[:, None]
    i, j = env._cell(xs, ys)
    g = np.degrees(np.arctan(np.gradient(hs[i, j], s, axis=1)))
    return ndimage.uniform_filter1d(g, 5, axis=1), clear(xs, ys).all(axis=1)


step = max(1, int(round(0.2 / res)))
ii, jj = np.mgrid[step:hm.shape[0] - step:step, step:hm.shape[1] - step:step]
ii, jj = ii.ravel(), jj.ravel()
cx, cy = (jj + 0.5) * res - half_w, (ii + 0.5) * res - half_h
norm = np.maximum(np.hypot(gx[ii, jj], gy[ii, jj]), 1e-9)
ux, uy = -gx[ii, jj] / norm, -gy[ii, jj] / norm           # 내리막 방향
g_band, _ = grade_profile(cx, cy, ux, uy, np.arange(0.0, BAND + 1e-6, 0.1))
g_run, ok_all = grade_profile(cx, cy, ux, uy, np.arange(-RUN - 0.6, BAND + 1.0, 0.1))
down = -g_band.mean(axis=1)                               # 띠의 평균 내리막 경사 (도, 양수)
ok_run = ok_all & (np.abs(g_run[:, :int((RUN + 0.6) / 0.1)]).max(axis=1) < 14.0)
rng = np.random.default_rng(args.seed)


def pick(mask, k):
    cand = np.nonzero(mask)[0]
    rng.shuffle(cand)
    out = []
    for c in cand:
        if all(math.hypot(cx[c] - cx[o], cy[c] - cy[o]) > 4.0 for o in out):
            out.append(int(c))
            if len(out) == k:
                break
    return out


def run_trial(sites, v_brake, after_sec=3.2):
    k = len(sites)
    starts = []
    for c in sites:
        x, y = cx[c] - RUN * ux[c], cy[c] - RUN * uy[c]
        i, j = env._cell(x, y)
        r = int(0.65 / res)
        z = hm[max(i - r, 0):i + r + 1, max(j - r, 0):j + r + 1].max() + REST_HEIGHT + 0.05
        starts.append((x, y, z, math.atan2(uy[c], ux[c])))
    cells = env._start_cells[rng.permutation(len(env._start_cells))]
    for c in cells:                                       # 남는 로버는 떨어진 평지에 세워 둔다
        if len(starts) == n:
            break
        x, y = (c[1] + 0.5) * res - half_w, (c[0] + 0.5) * res - half_h
        if all(math.hypot(x - s[0], y - s[1]) > 5.0 for s in starts):
            starts.append((x, y, hm[c[0], c[1]] + REST_HEIGHT + 0.12, 0.0))
    starts = np.array(starts)
    env.reset_idx(np.arange(n), starts=starts)
    lim = env.baseline_limiter()
    yaws = starts[:k, 3]
    u = np.stack([np.cos(yaws), np.sin(yaws)], axis=1)
    p0 = np.array([[cx[c], cy[c]] for c in sites])
    braking = np.zeros(k, bool)
    t_brake = np.full(k, -1)
    logs = []
    T = int((1.0 + (RUN + 1.0) / args.speed + 2.0 + after_sec) / dt)
    for t in range(T):
        v = np.zeros(n)
        if t * dt >= 1.0:
            v[:k] = np.where(braking, v_brake, args.speed)
        R = env._state["R"][:k]
        err = (yaws - np.arctan2(R[:, 1, 0], R[:, 0, 0]) + np.pi) % (2 * np.pi) - np.pi
        w = np.zeros(n)
        w[:k] = np.clip(1.2 * err, -0.3, 0.3) * (np.abs(v[:k]) > 0.05)
        v_out, w_out = lim(v, w)
        _, _, _, _, info = env.step(command=(v_out, w_out))
        s = np.sum((info["pos"][:k, :2] - p0) * u, axis=1)
        new = (~braking) & (s >= 0.0)
        t_brake[new] = t
        braking |= new
        logs.append((s.copy(), env._state["v_b"][:k, 0].copy(),
                     np.abs(env._state["wheel"][:k]).mean(axis=1) / K_WHEEL, v_out[:k].copy(),
                     np.degrees(np.arcsin(np.clip(env._state["R"][:k, 2, 0], -1, 1)))))
    S, V, W, C, P = (np.array([l[q] for l in logs]) for q in range(5))
    rows = []
    for q in range(k):
        if t_brake[q] < 0:
            continue
        a = int(t_brake[q])
        marks = {}
        for tau in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.6, 2.0, 2.6):
            b = min(a + int(round(tau / dt)), len(V) - 1)
            marks[f"{tau:g}"] = [round(float(V[b, q]), 2), round(float(W[b, q]), 2),
                                 round(float(C[b, q]), 2)]
        slow = np.nonzero(V[a:, q] <= max(v_brake, 0.0) + 0.05)[0]
        rows.append({"site": int(sites[q]), "down_deg": float(down[sites[q]]),
                     "v0": float(V[a, q]), "pitch0": float(P[a, q]), "marks": marks,
                     "t_slow": float(slow[0] * dt) if len(slow) else None,
                     "d_slow": float(S[a + slow[0], q] - S[a, q]) if len(slow) else None})
    return rows


results = []
for b in [float(x) for x in args.bins.split(",")]:
    lo = np.abs(g_band).max(axis=1) < 3.0 if b == 0 else \
        (np.abs(down - b) <= 1.5) & (-g_band.min(axis=1) < b + 5.0) & (-g_band.max(axis=1) > b - 5.0)
    sites = pick(ok_run & lo, n)
    print(f"[brake] 내리막 {b:.0f}°: 후보 {int((ok_run & lo).sum())}곳 중 {len(sites)}곳", flush=True)
    if not sites:
        continue
    for vb in [float(x) for x in args.brake.split(",")]:
        rows = run_trial(sites, vb)
        if not rows:
            continue
        med = {k: [round(float(np.median([r["marks"][k][c] for r in rows])), 2) for c in range(3)]
               for k in rows[0]["marks"]}
        ds = [r["d_slow"] for r in rows if r["d_slow"] is not None]
        r = {"down_bin": b, "v_brake": vb, "n": len(rows),
             "down_deg": round(float(np.mean([x["down_deg"] for x in rows])), 1),
             "v0": round(float(np.median([x["v0"] for x in rows])), 2),
             "차체·바퀴·명령 (뒤 초)": med,
             "d_slow_med": round(float(np.median(ds)), 2) if ds else None,
             "d_slow_max": round(float(np.max(ds)), 2) if ds else None,
             "not_slowed": len(rows) - len(ds)}
        print("[brake] " + json.dumps(r, ensure_ascii=False), flush=True)
        results.append({**r, "rows": rows})

if args.out:
    with open(args.out, "w") as f:
        json.dump({"terrain": args.terrain, "results": results}, f, ensure_ascii=False)
    print(f"[brake] 저장: {args.out}")
app.close()
