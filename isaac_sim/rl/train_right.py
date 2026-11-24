"""뒤집힌 로버 다시 세우기 — 학습과 평가.

    tools/isaac-pypi isaac_sim/rl/train_right.py --out isaac_sim/rl/runs/right1
    tools/isaac-pypi isaac_sim/rl/train_right.py --out <dir> --eval <dir>/model.pt --episodes 200

--arm-effort "150,150,100,40,40,40" 처럼 팔 관절 구동력 상한(N·m)을 줄 수 있다.
평가는 성공한 뒤 팔을 천천히 접어 넣었을 때도 서 있는지까지 본다.
"""
import argparse
import json
import os
import sys
import time

_p = argparse.ArgumentParser()
_p.add_argument("--terrain", default="terrain_00026")
_p.add_argument("--out", required=True)
_p.add_argument("--num-envs", type=int, default=16)
_p.add_argument("--samples", type=float, default=2.0e6)
_p.add_argument("--horizon", type=int, default=128)
_p.add_argument("--lr", type=float, default=3e-4)
_p.add_argument("--gamma", type=float, default=0.99)
_p.add_argument("--lam", type=float, default=0.95)
_p.add_argument("--seed", type=int, default=0)
_p.add_argument("--resume", default="")
_p.add_argument("--arm-effort", default="")
_p.add_argument("--eval", default="", help="model.pt — 주면 학습하지 않고 평가만")
_p.add_argument("--episodes", type=int, default=200)
args, _ = _p.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from ppo import ActorCritic, RolloutBuffer, ppo_update  # noqa: E402
from righting_env import ACT_DIM, OBS_DIM, RightingEnv, rp  # noqa: E402

torch.set_num_threads(2)
torch.manual_seed(args.seed)
os.makedirs(args.out, exist_ok=True)
effort = [float(v) for v in args.arm_effort.split(",")] if args.arm_effort else None
env = RightingEnv(app, arm_effort=effort, terrain=args.terrain, num_envs=args.num_envs,
                  seed=args.seed)
model = ActorCritic(OBS_DIM, ACT_DIM)

if args.eval:
    model.load_state_dict(torch.load(args.eval))
    obs = env.reset_all()
    eps = []                                   # (출발 기울기, 성공, 걸린 시간, 접은 뒤 기울기)
    retract = np.zeros(env.n, dtype=np.int64)  # >0: 성공 후 팔 접는 중 (남은 스텝)
    t_done = np.zeros(env.n)
    RET = int(4.0 / env.dt)
    while len(eps) < args.episodes:
        act, _, _ = model.act(obs, deterministic=True)
        hold = retract > 0
        if hold.any():                         # 접는 중인 로버: 목표를 home 쪽으로 직접 옮긴다
            step = np.clip(rp.ARM_HOME - env.arm_target[hold], -0.04, 0.04)
            env.arm_target[hold] += step
            act[hold] = 0.0
        obs, _, success, truncated, info = env.step(act)
        finished = []
        for i in np.nonzero(hold)[0]:
            retract[i] -= 1
            if retract[i] == 0:
                eps.append((float(info["start_tilt"][i]), True, t_done[i], float(info["tilt"][i])))
                finished.append(i)
        for i in np.nonzero(success & ~hold)[0]:
            retract[i] = RET
            t_done[i] = info["steps"][i] * env.dt
            env.hold[i] = 0
        for i in np.nonzero(truncated & ~hold)[0]:
            eps.append((float(info["start_tilt"][i]), False, float("nan"), float(info["tilt"][i])))
            finished.append(i)
        if finished:
            obs = env.reset_idx(np.array(finished))
    e = np.array(eps, dtype=float)
    flipped = e[:, 0] > rp.FLIPPED_DEG
    res = {"episodes": len(e), "arm_effort": effort}
    for name, m in (("전체", np.ones(len(e), bool)), ("넘어진 채 출발", flipped),
                    ("60~120° (옆으로)", (e[:, 0] > 60) & (e[:, 0] <= 120)),
                    ("120° 초과 (등으로)", e[:, 0] > 120)):
        if m.sum() == 0:
            continue
        ok = e[m, 1] > 0
        stay = ok & (e[m, 3] < 20.0)
        res[name] = {"n": int(m.sum()), "success": float(ok.mean()),
                     "success_after_retract": float(stay.mean()),
                     "time_p50": float(np.nanmedian(e[m, 2])) if ok.any() else None,
                     "time_p90": float(np.nanpercentile(e[m, 2], 90)) if ok.any() else None}
        print(f"[eval] {name}: {int(m.sum())}회, 일어섬 {100 * ok.mean():.0f}%, "
              f"팔 접은 뒤에도 서 있음 {100 * stay.mean():.0f}%, "
              f"걸린 시간 중앙 {res[name]['time_p50']} s", flush=True)
    with open(os.path.join(args.out, "eval.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    app.close()
    sys.exit(0)

if args.resume:
    model.load_state_dict(torch.load(args.resume))
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
buf = RolloutBuffer(args.horizon, env.n, OBS_DIM, ACT_DIM)
n_iter = int(args.samples // (args.horizon * env.n))
log = open(os.path.join(args.out, "log.csv"), "a")
if log.tell() == 0:
    log.write("iter,samples,minutes,reward,episodes,success,tilt_mean,std,kl\n")
obs = env.reset_all()
t0 = time.time()
recent = []                                    # 최근 에피소드의 성공 여부 (넘어진 채 출발한 것만)
for it in range(n_iter):
    if it % 4 == 0:
        env.randomize_friction()
    for g in opt.param_groups:
        g["lr"] = args.lr * max(0.1, 1.0 - it / n_iter)
    tilts = []
    vmax = 0.0
    for t in range(args.horizon):
        act, logp, val = model.act(obs)
        nxt, rew, success, truncated, info = env.step(act)
        if truncated.any():
            with torch.no_grad():
                v_next = model.value(torch.as_tensor(nxt[truncated])).numpy()
            rew = rew.copy()
            rew[truncated] += args.gamma * v_next
        done = success | truncated
        buf.obs[t], buf.act[t], buf.logp[t] = obs, act, logp
        buf.rew[t], buf.val[t], buf.done[t] = rew, val, done
        tilts.append(info["tilt"][info["live"]])
        vmax = max(vmax, float(info["speed"][info["live"]].max()) if info["live"].any() else 0.0)
        if done.any():
            for i in np.nonzero(done)[0]:
                if info["start_tilt"][i] > rp.FLIPPED_DEG:
                    recent.append(bool(success[i]))
            nxt = env.reset_idx(np.nonzero(done)[0])
        obs = nxt
    recent = recent[-200:]
    with torch.no_grad():
        last_val = model.value(torch.as_tensor(obs)).numpy()
    adv, ret = buf.advantages(last_val, args.gamma, args.lam)
    st = ppo_update(model, opt, buf, adv, ret)
    tilt = np.concatenate(tilts) if tilts else np.zeros(1)
    row = (it, (it + 1) * args.horizon * env.n, (time.time() - t0) / 60.0, float(buf.rew.mean()),
           len(recent), float(np.mean(recent)) if recent else 0.0, float(tilt.mean()),
           float(model.log_std.clamp(-3, 0).exp().mean()), st["kl"])
    log.write(",".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + "\n")
    log.flush()
    if it % 10 == 0 or it == n_iter - 1:
        print(f"[train] {it:4d}/{n_iter} {row[1] / 1e3:6.0f}k {row[2]:5.1f}분 보상 {row[3]:+.3f} "
              f"최근 성공률 {100 * row[5]:.0f}% (n={row[4]}) 평균 기울기 {row[6]:.0f}° "
              f"std {row[7]:.2f} 최고 속도 {vmax:.1f} m/s", flush=True)
    if it % 20 == 0 or it == n_iter - 1:
        torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
        model.export_npz(os.path.join(args.out, "policy.npz"), samples=np.int64(row[1]),
                         terrain=np.str_(args.terrain), control_hz=np.float64(1.0 / env.dt),
                         arm_effort=np.asarray(effort or [0.0]))
log.close()
print(f"[train] 끝: {args.out}/policy.npz ({(time.time() - t0) / 60:.1f} 분)")
app.close()
