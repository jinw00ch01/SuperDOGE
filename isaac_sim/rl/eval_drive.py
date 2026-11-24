"""구동 조절기 평가 — 같은 출발점·같은 목표점에서 제어 방식만 바꿔 비교한다.

    tools/isaac-pypi isaac_sim/rl/eval_drive.py --terrain terrain_00028 \\
        --policy isaac_sim/rl/runs/gov1/policy.npz --minutes 6

시나리오: 로버마다 목표점 목록을 받아 실제 스택의 Navigator 와 같은 식으로
(지형을 모른 채) 목표를 향한다. 넘어지거나 바위에 닿기 직전이면 다음 출발점에서
이어 간다. 비교 대상:

  raw        명령을 그대로 바퀴에 넘김 (원래 프로젝트의 동작)
  fixed      고정 속도·가속 제한 (drive_limits.py, 지금 기본값 = Froude 환산)
  fixed:1.5  같은 방식인데 최고 속도만 1.5 m/s 로 (가속 제한은 그대로)
  fixed:max_lin=2.0+acc_lin=1.5+max_ang=1.5+acc_ang=2.0
             항목별로 지정 — "고정 제한을 손으로 더 풀면 되지 않나" 를 확인하는 비교 대상
  policy     학습한 조절기
"""
import argparse
import json
import os
import sys

_p = argparse.ArgumentParser()
_p.add_argument("--terrain", default="terrain_00026")
_p.add_argument("--policy", default="")
_p.add_argument("--controllers", default="raw,fixed,policy")
_p.add_argument("--num-envs", type=int, default=16)
_p.add_argument("--minutes", type=float, default=6.0, help="제어 방식 하나당 로버 1 대의 주행 시간")
_p.add_argument("--seed", type=int, default=100)
_p.add_argument("--friction", type=float, default=1.0, help="지면 마찰 배율")
_p.add_argument("--steep-goal-frac", type=float, default=0.0,
                help="목표점을 15° 이상 비탈 위에 두는 비율 (비탈 주행을 일부러 늘린 시험)")
_p.add_argument("--nav-turn-decel", type=float, default=0.4,
                help="목표점 추종의 제자리 회전 감속도 (0 = 예전 식)")
_p.add_argument("--out", default="")
args, _ = _p.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drive_env import RoverDriveEnv, gp  # noqa: E402

RoverDriveEnv.NAV_TURN_DECEL = args.nav_turn_decel
env = RoverDriveEnv(app, terrain=args.terrain, num_envs=args.num_envs, seed=args.seed)
env.max_steps = 10 ** 9
env.friction_range = (args.friction, args.friction)
env.randomize_friction()
n = env.n
T = int(args.minutes * 60.0 / env.dt)

rng = np.random.default_rng(args.seed)
S, K = 40, 200
starts = env._sample_starts(n * S).reshape(n, S, 4)
goals = np.zeros((n, K, 2))
for i in range(n):
    x, y = starts[i, 0, 0], starts[i, 0, 1]
    for k in range(K):
        goals[i, k] = env.sample_goal(rng, x, y, args.steep_goal_frac)
        x, y = goals[i, k]

names = [c for c in args.controllers.split(",") if c and (c != "policy" or args.policy)]
for c in names:
    if c not in ("raw", "fixed", "policy") and not c.startswith("fixed:"):
        raise SystemExit(f"알 수 없는 제어 방식: {c}")
policy = gp.GovernorPolicy(args.policy) if "policy" in names else None
results = {}
for name in names:
    env.set_script(starts, goals)
    obs = env.reset_all()
    limiter = env.baseline_limiter()
    if name.startswith("fixed:"):
        spec = name.split(":", 1)[1]
        if "=" in spec:
            limiter.lim = dict(limiter.lim, **{k: float(v) for k, v in
                                               (kv.split("=") for kv in spec.split("+"))})
        else:
            limiter.lim = dict(limiter.lim, max_lin=float(spec))
    log = {k: [] for k in ("tilt", "vx", "vy", "v_out", "v_des", "w_des", "wz", "slope")}
    flips = outs = 0
    dist = 0.0
    hist = []            # 최근 1.5 s 의 (tilt, v_out, vx, v_des, w_des) — 전복 직전 상황 기록용
    flip_events = []
    for t in range(T):
        if name == "policy":
            obs, r, flipped, trunc, info = env.step(policy(obs))
        else:
            settle = env._settle > 0
            v_des = np.where(settle, 0.0, env.v_des)
            w_des = np.where(settle, 0.0, env.w_des)
            cmd = (v_des, w_des) if name == "raw" else limiter(v_des, w_des)
            obs, r, flipped, trunc, info = env.step(command=cmd)
        for k in log:
            log[k].append(info[k])
        dist += float(np.abs(info["vx"]).sum() * env.dt)
        flips += int(flipped.sum())
        hist.append((info["tilt"], info["v_out"], info["vx"], info["v_des"], info["w_des"]))
        hist = hist[-23:]
        for i in np.nonzero(flipped)[0]:
            h = [np.array([row[c][i] for row in hist]) for c in range(5)]
            flip_events.append({
                "slope_deg": round(float(info["slope"][i]), 1),
                "tilt_1s_before": round(float(h[0][max(len(hist) - 16, 0)]), 1),
                "v_out_mean": round(float(h[1][:-3].mean()), 2) if len(hist) > 3 else None,
                "vx_max": round(float(np.abs(h[2]).max()), 2),
                "v_des": round(float(h[3][-1]), 2), "w_des": round(float(h[4][-1]), 2),
                "steps_since_reset": int(env.steps[i])})
        outs += int(trunc.sum())
        done = flipped | trunc
        if done.any():
            idx = np.nonzero(done)[0]
            limiter.reset(idx)
            obs = env.reset_idx(idx)
    L = {k: np.concatenate(v) for k, v in log.items()}
    moving = np.abs(L["v_des"]) > 0.05
    hours = n * T * env.dt / 3600.0
    res = {
        "rover_minutes": round(hours * 60.0, 1),
        "distance_m": round(dist, 1),
        "goals": int(env.goals_reached.sum()),
        "goals_per_min": round(env.goals_reached.sum() / (hours * 60.0), 2),
        "flips": flips,
        "flips_per_km": round(flips / max(dist / 1000.0, 1e-9), 2),
        "flips_per_hour": round(flips / hours, 1),
        "rock_stops": outs,
        "speed_mean": round(float(np.clip(L["vx"], 0, None)[moving].mean()), 3),
        "slip_mean": round(float(np.abs(L["v_out"] - L["vx"])[moving].mean()), 3),
        "lateral_mean": round(float(np.abs(L["vy"]).mean()), 3),
        "tilt_p50": round(float(np.percentile(L["tilt"], 50)), 1),
        "tilt_p99": round(float(np.percentile(L["tilt"], 99)), 1),
        "tilt_over_30_pct": round(100.0 * float((L["tilt"] > 30.0).mean()), 2),
    }
    for lo, hi in ((0, 10), (10, 20), (20, 90)):
        m = moving & (L["slope"] >= lo) & (L["slope"] < hi)
        res[f"speed_slope_{lo}_{hi}"] = round(float(np.clip(L["vx"], 0, None)[m].mean()), 3) if m.any() else None
        res[f"share_slope_{lo}_{hi}_pct"] = round(100.0 * float(m.sum()) / max(int(moving.sum()), 1), 1)
    res["flip_events"] = flip_events
    results[name] = res
    print(f"[eval] {name}: " + json.dumps({k: v for k, v in res.items() if k != "flip_events"},
                                           ensure_ascii=False), flush=True)

keys = ["goals_per_min", "distance_m", "flips", "flips_per_km", "flips_per_hour", "rock_stops",
        "speed_mean", "slip_mean", "lateral_mean", "tilt_p99", "tilt_over_30_pct",
        "speed_slope_0_10", "speed_slope_10_20", "speed_slope_20_90"]
print(f"\n[eval] {args.terrain} · 마찰 ×{args.friction} · 로버 {n}대 × {args.minutes:.0f}분")
short = [c if len(c) <= 11 else f"fixed#{i}" for i, c in enumerate(names)]
for c, sname in zip(names, short):
    if c != sname:
        print(f"  {sname} = {c}")
print(f"{'':20s}" + "".join(f"{c:>12s}" for c in short))
for k in keys:
    print(f"{k:20s}" + "".join(f"{str(results[c][k]):>12s}" for c in names))
if args.out:
    with open(args.out, "w") as f:
        json.dump({"terrain": args.terrain, "friction": args.friction, "results": results}, f,
                  ensure_ascii=False, indent=1)
app.close()
