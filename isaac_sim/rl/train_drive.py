"""구동 조절기 학습 (PPO).

    tools/isaac-pypi isaac_sim/rl/train_drive.py --out isaac_sim/rl/runs/gov1

약 150 만 샘플(제어 스텝 × 로버 수)에 50 분쯤 걸린다 (16 대, 초당 약 500 샘플).
결과: <out>/policy.npz (ROS 노드가 읽는 가중치), <out>/model.pt, <out>/log.csv
"""
import argparse
import os
import sys
import time

_p = argparse.ArgumentParser()
_p.add_argument("--terrain", default="terrain_00026")
_p.add_argument("--out", required=True)
_p.add_argument("--num-envs", type=int, default=16)
_p.add_argument("--samples", type=float, default=1.5e6)
_p.add_argument("--horizon", type=int, default=128, help="한 번 갱신에 쓰는 제어 스텝 수")
_p.add_argument("--lr", type=float, default=3e-4)
_p.add_argument("--gamma", type=float, default=0.98)
_p.add_argument("--lam", type=float, default=0.95)
_p.add_argument("--wheels", default="capsule", choices=["capsule", "mesh"],
                help="바퀴 충돌 형상. mesh = 로버 모델 그대로 (예전 결과 재현용)")
_p.add_argument("--seed", type=int, default=0)
_p.add_argument("--resume", default="")
_p.add_argument("--steep-goal-frac", type=float, default=0.0,
                help="목표점 추종 에피소드에서 목표를 15° 이상 비탈에 두는 비율")
_p.add_argument("--friction-every", type=int, default=4,
                help="몇 번 갱신마다 지면 마찰 배율을 새로 뽑을지")
_p.add_argument("--balance", action="store_true",
                help="균형 정책(balance_env: 뒤로 빼기 + 팔 균형추)을 학습한다")
args, _ = _p.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if args.balance:
    from balance_env import ACT_DIM, OBS_DIM  # noqa: E402
    from balance_env import BalanceEnv as RoverDriveEnv  # noqa: E402
else:
    from drive_env import ACT_DIM, OBS_DIM, RoverDriveEnv  # noqa: E402
from ppo import ActorCritic, RolloutBuffer, ppo_update  # noqa: E402

torch.set_num_threads(2)
torch.manual_seed(args.seed)
os.makedirs(args.out, exist_ok=True)

RoverDriveEnv.WHEEL_SHAPE = args.wheels
env = RoverDriveEnv(app, terrain=args.terrain, num_envs=args.num_envs, seed=args.seed,
                    steep_goal_frac=args.steep_goal_frac)
model = ActorCritic(OBS_DIM, ACT_DIM)
if args.resume:
    model.load_state_dict(torch.load(args.resume))
    print(f"[train] {args.resume} 에서 이어서 학습")
opt = torch.optim.Adam(model.parameters(), lr=args.lr)
buf = RolloutBuffer(args.horizon, env.n, OBS_DIM, ACT_DIM)

n_iter = int(args.samples // (args.horizon * env.n))
log = open(os.path.join(args.out, "log.csv"), "a")
if log.tell() == 0:
    log.write("iter,samples,minutes,reward,flips_per_1k,progress_ratio,slip,lateral,"
              "tilt_mean,tilt_p99,v_scale,std,kl,friction\n")
obs = env.reset_all()
t0 = time.time()
for it in range(n_iter):
    if it % args.friction_every == 0:
        env.randomize_friction()
    lr = args.lr * max(0.1, 1.0 - it / n_iter)
    for g in opt.param_groups:
        g["lr"] = lr
    acc = {k: [] for k in ("flip", "prog", "goal", "slip", "lat", "tilt", "scale")}
    for t in range(args.horizon):
        act, logp, val = model.act(obs)
        nxt, rew, flipped, truncated, info = env.step(act)
        # 시간 초과·경계 이탈로 끊긴 경우는 "그 뒤가 없다" 가 아니므로 가치로 잇는다.
        if truncated.any():
            with torch.no_grad():
                v_next = model.value(torch.as_tensor(nxt[truncated])).numpy()
            rew = rew.copy()
            rew[truncated] += args.gamma * v_next
        done = flipped | truncated
        buf.obs[t], buf.act[t], buf.logp[t] = obs, act, logp
        buf.rew[t], buf.val[t], buf.done[t] = rew, val, done
        moving = np.abs(info["v_des"]) > 0.05
        acc["flip"].append(flipped.mean())
        acc["prog"].append(np.clip(info["vx"] * np.sign(info["v_des"]), 0, None)[moving].sum())
        acc["goal"].append(np.abs(np.clip(info["v_des"], -2.0, 2.0))[moving].sum())
        acc["slip"].append(np.abs(info["v_out"] - info["vx"])[moving].mean() if moving.any() else 0.0)
        acc["lat"].append(np.abs(info["vy"]).mean())
        acc["tilt"].append(info["tilt"])
        acc["scale"].append(0.5 * (np.clip(act[:, 0], -1, 1) + 1.0).mean())
        if done.any():
            nxt = env.reset_idx(np.nonzero(done)[0])
        obs = nxt
    with torch.no_grad():
        last_val = model.value(torch.as_tensor(obs)).numpy()
    adv, ret = buf.advantages(last_val, args.gamma, args.lam)
    st = ppo_update(model, opt, buf, adv, ret)
    tilt = np.concatenate(acc["tilt"])
    row = (it, (it + 1) * args.horizon * env.n, (time.time() - t0) / 60.0,
           float(buf.rew.mean()), 1000.0 * float(np.mean(acc["flip"])),
           float(np.sum(acc["prog"]) / max(np.sum(acc["goal"]), 1e-6)),
           float(np.mean(acc["slip"])), float(np.mean(acc["lat"])),
           float(tilt.mean()), float(np.percentile(tilt, 99)),
           float(np.mean(acc["scale"])), float(model.log_std.clamp(-3, 0).exp().mean()),
           st["kl"], env.friction_scale)
    log.write(",".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + "\n")
    log.flush()
    if it % 5 == 0 or it == n_iter - 1:
        print(f"[train] {it:4d}/{n_iter} {row[1] / 1e3:6.0f}k {row[2]:5.1f}분 "
              f"보상 {row[3]:+.3f} 전복/1k {row[4]:.2f} 진행률 {row[5]:.2f} "
              f"헛돎 {row[6]:.2f} 기울기 p99 {row[9]:.0f}° 배율 {row[10]:.2f} "
              f"std {row[11]:.2f} μ×{row[13]:.2f}", flush=True)
    if it % 20 == 0 or it == n_iter - 1:
        torch.save(model.state_dict(), os.path.join(args.out, "model.pt"))
        model.export_npz(os.path.join(args.out, "policy.npz"),
                         samples=np.int64(row[1]), terrain=np.str_(args.terrain),
                         control_hz=np.float64(1.0 / env.dt))
log.close()
print(f"[train] 끝: {args.out}/policy.npz ({(time.time() - t0) / 60:.1f} 분)")
app.close()
