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
_p.add_argument("--speed-cap", default="off", choices=["off", "model"],
                help="link 선속도 상한. model = 로버 모델 그대로 (1.5 m/s)")
_p.add_argument("--wheels", default="capsule", choices=["capsule", "mesh"],
                help="바퀴 충돌 형상. mesh = 로버 모델 그대로 (예전 결과 재현용)")
_p.add_argument("--seed", type=int, default=100)
_p.add_argument("--friction", type=float, default=1.0, help="지면 마찰 배율")
_p.add_argument("--steep-goal-frac", type=float, default=0.0,
                help="목표점을 15° 이상 비탈 위에 두는 비율 (비탈 주행을 일부러 늘린 시험)")
_p.add_argument("--balance", action="store_true",
                help="--policy 가 균형 정책(balance_env)일 때. 비교 기준 제어기는 팔을 home 에 둔다")
_p.add_argument("--nav-turn-decel", type=float, default=0.4,
                help="목표점 추종의 제자리 회전 감속도 (0 = 예전 식)")
_p.add_argument("--guard-pitch", type=float, default=30.0,
                help="guard 제어기: 앞이 이만큼(도) 들리면 밀기를 멈추고 물러난다")
_p.add_argument("--guard-roll", type=float, default=32.0)
_p.add_argument("--guard-back", default="0.5,4.0", help="guard: 후진 속도(m/s), 최대 시간(s)")
_p.add_argument("--guard-back-dist", type=float, default=0.5,
                help="guard: 가장 멀리 나간 자리에서 이만큼(m) 물러나면 끝. 0 = 시간만으로 "
                     "(예전 방식: --guard-back 0.5,1.5 --guard-back-dist 0)")
_p.add_argument("--out", default="")
args, _ = _p.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drive_env import RoverDriveEnv, gp  # noqa: E402
if args.balance:
    from balance_env import BalanceEnv as RoverDriveEnv  # noqa: E402,F811

RoverDriveEnv.NAV_TURN_DECEL = args.nav_turn_decel
RoverDriveEnv.WHEEL_SHAPE = args.wheels
RoverDriveEnv.LIFT_SPEED_CAP = args.speed_cap == "off"
env = RoverDriveEnv(app, terrain=args.terrain, num_envs=args.num_envs, seed=args.seed)
from scipy import ndimage as _ndi  # noqa: E402
GY, GX = np.gradient(_ndi.uniform_filter(env.hm.astype(np.float64), 16), env.res)   # 차체 길이로 고른 경사
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
    if c not in ("raw", "fixed", "policy", "guard") and not c.startswith("fixed:"):
        raise SystemExit(f"알 수 없는 제어 방식: {c}")
policy = (gp.GovernorPolicy(args.policy, *((24, 5) if args.balance else ()))
          if "policy" in names else None)
results = {}
for name in names:
    env.set_script(starts, goals)
    obs = env.reset_all()
    limiter = env.baseline_limiter()
    g_left = np.zeros(env.n, dtype=np.int64)        # guard: 남은 후진 스텝
    g_count = 0
    g_p0 = np.zeros((env.n, 2))                     # guard: 발동한 자리
    g_u = np.zeros((env.n, 2))                      # guard: 그때의 앞 방향 (수평 단위 벡터)
    g_far = np.zeros(env.n)                         # guard: 발동 뒤 앞으로 가장 멀리 나간 거리
    G_BACK_V, G_BACK_T = [float(x) for x in args.guard_back.split(",")]
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
    hist = []            # 최근 1.5 s 의 (tilt, v_out, vx, v_des, w_des, pitch, roll, 진행방향 경사, z)
    flip_events = []
    for t in range(T):
        if name == "policy":
            obs, r, flipped, trunc, info = env.step(policy(obs))
        else:
            settle = env._settle > 0
            v_des = np.where(settle, 0.0, env.v_des)
            w_des = np.where(settle, 0.0, env.w_des)
            if name.startswith("guard"):
                # 기울기 반사: 앞이 들리거나 옆으로 기울면 밀기를 멈추고 물러난 뒤 다른 목표로.
                R0 = env._state["R"]
                pitch = np.degrees(np.arcsin(np.clip(R0[:, 2, 0], -1, 1)))
                roll = np.degrees(np.arcsin(np.clip(R0[:, 2, 1], -1, 1)))
                trig = (g_left == 0) & ~settle & ((pitch > args.guard_pitch)
                                                  | (np.abs(roll) > args.guard_roll))
                g_left[trig] = int(G_BACK_T / env.dt)
                g_count += int(trig.sum())
                pos_xy = env._state["pos"][:, :2]
                fwd = R0[:, :2, 0] / np.maximum(np.linalg.norm(R0[:, :2, 0], axis=1, keepdims=True), 1e-6)
                g_p0[trig], g_u[trig], g_far[trig] = pos_xy[trig], fwd[trig], 0.0
                backing = g_left > 0
                v_des = np.where(backing, -G_BACK_V, v_des)
                w_des = np.where(backing, 0.0, w_des)
                if args.guard_back_dist > 0.0:
                    # 구동 제한(가속 0.72 m/s²) 때문에 후진 명령을 내도 한동안은 앞으로 간다.
                    # 시간이 아니라 실제로 물러난 거리로 끝낸다.
                    along = np.einsum("ij,ij->i", pos_xy - g_p0, g_u)
                    g_far = np.where(backing, np.maximum(g_far, along), g_far)
                    done_back = backing & (g_far - along >= args.guard_back_dist)
                    g_left[done_back] = 1
                for i in np.nonzero(g_left == 1)[0]:
                    if env.mode[i] == 1:
                        env._next_goal(int(i))            # 실제 스택에서는 "위험 표시 + 다시 계획"
                g_left = np.maximum(g_left - 1, 0)
            cmd = (v_des, w_des) if name == "raw" else limiter(v_des, w_des)
            obs, r, flipped, trunc, info = env.step(command=cmd)
        for k in log:
            log[k].append(info[k])
        dist += float(np.abs(info["vx"]).sum() * env.dt)
        flips += int(flipped.sum())
        R_ = env._state["R"]
        ci, cj = env._cell(info["pos"][:, 0], info["pos"][:, 1])
        hx, hy = R_[:, 0, 0], R_[:, 1, 0]
        hn = np.maximum(np.hypot(hx, hy), 1e-6)
        grade = np.degrees(np.arctan((GX[ci, cj] * hx + GY[ci, cj] * hy) / hn))
        hist.append((info["tilt"], info["v_out"], info["vx"], info["v_des"], info["w_des"],
                     np.degrees(np.arcsin(np.clip(R_[:, 2, 0], -1, 1))),
                     np.degrees(np.arcsin(np.clip(R_[:, 2, 1], -1, 1))), grade,
                     info["pos"][:, 2].copy()))
        hist = hist[-23:]
        for i in np.nonzero(flipped)[0]:
            h = [np.array([row[c][i] for row in hist]) for c in range(9)]
            b1 = max(len(hist) - 16, 0)          # 1 s 전
            b0 = max(len(hist) - 4, 0)           # 0.2 s 전
            flip_events.append({
                "slope_deg": round(float(info["slope"][i]), 1),
                "tilt_1s_before": round(float(h[0][max(len(hist) - 16, 0)]), 1),
                "v_out_mean": round(float(h[1][:-3].mean()), 2) if len(hist) > 3 else None,
                "vx_max": round(float(np.abs(h[2]).max()), 2),
                "v_des": round(float(h[3][-1]), 2), "w_des": round(float(h[4][-1]), 2),
                "pos": [round(float(info["pos"][i, 0]), 2), round(float(info["pos"][i, 1]), 2)],
                "pitch_1s": round(float(h[5][b1]), 0), "roll_1s": round(float(h[6][b1]), 0),
                "pitch_02s": round(float(h[5][b0]), 0), "roll_02s": round(float(h[6][b0]), 0),
                "grade_1s": round(float(h[7][b1]), 0),      # 진행 방향 지형 경사 (+ 오르막)
                "vx_1s": round(float(h[2][b1]), 2),
                "drop_1s": round(float(h[8][b1] - h[8][-1]), 2),   # 마지막 1 s 동안 내려간 높이
                "steps_since_reset": int(env.steps[i])})
        outs += int(trunc.sum())
        done = flipped | trunc
        if done.any():
            idx = np.nonzero(done)[0]
            limiter.reset(idx)
            g_left[idx] = 0
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
        "guard_triggers": g_count,
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
    # 가파른 구간을 더 잘게: 마찰 한계(미끄럼 27.5°, 정지 30°) 앞뒤로 얼마나 가는지.
    for lo, hi in ((20, 25), (25, 28), (28, 90)):
        m = moving & (L["slope"] >= lo) & (L["slope"] < hi)
        res[f"speed_slope_{lo}_{hi}"] = round(float(np.clip(L["vx"], 0, None)[m].mean()), 3) if m.any() else None
        res[f"slipfrac_slope_{lo}_{hi}"] = (round(float(np.mean(
            1.0 - np.clip(L["vx"][m], 0, None) / np.maximum(np.abs(L["v_out"][m]), 0.05))), 2)
            if m.any() else None)
        res[f"share_slope_{lo}_{hi}_pct"] = round(100.0 * float(m.sum()) / max(int(moving.sum()), 1), 2)
    res["flip_events"] = flip_events
    results[name] = res
    print(f"[eval] {name}: " + json.dumps({k: v for k, v in res.items() if k != "flip_events"},
                                           ensure_ascii=False), flush=True)

keys = ["goals_per_min", "distance_m", "flips", "flips_per_km", "flips_per_hour", "rock_stops",
        "speed_mean", "slip_mean", "lateral_mean", "tilt_p99", "tilt_over_30_pct",
        "speed_slope_0_10", "speed_slope_10_20", "speed_slope_20_90",
        "speed_slope_20_25", "speed_slope_25_28", "speed_slope_28_90",
        "slipfrac_slope_25_28", "slipfrac_slope_28_90", "share_slope_25_28_pct",
        "share_slope_28_90_pct"]
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
