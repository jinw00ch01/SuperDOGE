"""작은 PPO (torch 만 사용 — Isaac Sim venv 에 이미 들어 있다).

rsl_rl / stable-baselines 를 새로 깔지 않으려고 직접 쓴 것이다. 정책이 작고
(입력 20 → 64 → 64 → 출력 3) 환경이 초당 500 샘플 정도라 이 정도로 충분하다.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden=(64, 64), init_std: float = 0.4):
        super().__init__()

        def mlp(out_dim):
            layers, d = [], obs_dim
            for h in hidden:
                layers += [nn.Linear(d, h), nn.Tanh()]
                d = h
            layers.append(nn.Linear(d, out_dim))
            return nn.Sequential(*layers)

        self.actor = mlp(act_dim)
        self.critic = mlp(1)
        self.log_std = nn.Parameter(torch.full((act_dim,), float(np.log(init_std))))
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)

    def mean(self, obs):
        return torch.tanh(self.actor(obs))

    def dist(self, obs):
        std = self.log_std.clamp(-3.0, 0.0).exp()
        return torch.distributions.Normal(self.mean(obs), std)

    def value(self, obs):
        return self.critic(obs).squeeze(-1)

    @torch.no_grad()
    def act(self, obs_np: np.ndarray, deterministic: bool = False):
        obs = torch.as_tensor(obs_np, dtype=torch.float32)
        d = self.dist(obs)
        a = d.mean if deterministic else d.sample()
        return a.numpy(), d.log_prob(a).sum(-1).numpy(), self.value(obs).numpy()

    def export_npz(self, path: str, **meta) -> None:
        """추론용 가중치 (governor_policy.GovernorPolicy 가 읽는다)."""
        out, k = {}, 0
        for m in self.actor:
            if isinstance(m, nn.Linear):
                out[f"w{k}"] = m.weight.detach().numpy().astype(np.float32)
                out[f"b{k}"] = m.bias.detach().numpy().astype(np.float32)
                k += 1
        out.update(meta)
        np.savez(path, **out)


class RolloutBuffer:
    def __init__(self, steps: int, n: int, obs_dim: int, act_dim: int):
        self.obs = np.zeros((steps, n, obs_dim), np.float32)
        self.act = np.zeros((steps, n, act_dim), np.float32)
        self.logp = np.zeros((steps, n), np.float32)
        self.rew = np.zeros((steps, n), np.float32)
        self.val = np.zeros((steps, n), np.float32)
        self.done = np.zeros((steps, n), np.float32)     # 다음 스텝이 새 에피소드
        self.steps, self.n = steps, n

    def advantages(self, last_val: np.ndarray, gamma: float, lam: float):
        adv = np.zeros_like(self.rew)
        nxt_adv = np.zeros(self.n, np.float32)
        nxt_val = last_val
        for t in reversed(range(self.steps)):
            keep = 1.0 - self.done[t]
            delta = self.rew[t] + gamma * nxt_val * keep - self.val[t]
            nxt_adv = delta + gamma * lam * keep * nxt_adv
            adv[t] = nxt_adv
            nxt_val = self.val[t]
        return adv, adv + self.val


def ppo_update(model: ActorCritic, opt, buf: RolloutBuffer, adv, ret, *, epochs=5,
               minibatch=512, clip=0.2, vf_coef=0.5, ent_coef=0.003, max_grad=0.5,
               target_kl=0.02):
    f = lambda a: torch.as_tensor(a.reshape(-1, *a.shape[2:]))
    obs, act, logp_old, adv_t, ret_t = f(buf.obs), f(buf.act), f(buf.logp), f(adv), f(ret)
    adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
    total = len(obs)
    stats = {"pi": 0.0, "v": 0.0, "kl": 0.0, "n": 0}
    for _ in range(epochs):
        perm = torch.randperm(total)
        stop = False
        for s in range(0, total, minibatch):
            idx = perm[s:s + minibatch]
            d = model.dist(obs[idx])
            logp = d.log_prob(act[idx]).sum(-1)
            ratio = (logp - logp_old[idx]).exp()
            pi_loss = -torch.min(ratio * adv_t[idx],
                                 ratio.clamp(1 - clip, 1 + clip) * adv_t[idx]).mean()
            v_loss = (model.value(obs[idx]) - ret_t[idx]).pow(2).mean()
            loss = pi_loss + vf_coef * v_loss - ent_coef * d.entropy().sum(-1).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad)
            opt.step()
            kl = float((logp_old[idx] - logp).mean())
            stats["pi"] += float(pi_loss); stats["v"] += float(v_loss)
            stats["kl"] += kl; stats["n"] += 1
            if kl > 1.5 * target_kl:
                stop = True
                break
        if stop:
            break
    n = max(stats.pop("n"), 1)
    return {k: v / n for k, v in stats.items()}
