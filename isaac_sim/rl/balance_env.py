"""비탈에서 균형을 지키는 정책의 학습 환경.

RoverDriveEnv(구동 조절기 학습 환경)와 장면·명령·기본 보상은 같고, 다음을 더한다.
  · 행동 5 개: 전진 배율(뒤로 빼기 포함), 회전 배율, 조향 보정, 팔 균형추 벡터 2 개
  · 보상: 안정 여유 — 무게중심에서 중력 방향으로 내린 선이 바퀴 접지 사각형의
    가장자리에서 얼마나 안쪽에 있는가 (m). 0.18 m 아래로 줄면 벌점. 밖으로 나가면 넘어진다.
무게중심은 매 스텝 각 링크의 위치와 질량으로 계산한다 (팔을 뻗으면 옮겨 간다).
"""
from __future__ import annotations

import numpy as np

from drive_env import DRIVE_NAMES, RoverDriveEnv
import balance_policy as bp           # drive_env 가 경로를 잡아 둔다

OBS_DIM, ACT_DIM = bp.OBS_DIM, bp.ACT_DIM
WHEEL_RADIUS = 0.15
SAFE_MARGIN = 0.18


class BalanceEnv(RoverDriveEnv):
    def __init__(self, app, **kw):
        self._ready = False
        super().__init__(app, **kw)
        self.arm_idx = [self.art.get_dof_index(n) for n in bp.ARM_JOINTS]
        names = list(self.art.body_names)
        self._mass = np.asarray(self.art.get_body_masses())[0].astype(np.float64)
        self._wheel_links = [names.index(n.replace("_Continuous", "")) for n in DRIVE_NAMES]
        self.arm_target = np.tile(bp.ARM_HOME, (self.n, 1))
        self.weight = np.zeros((self.n, 2))
        self.prev5 = np.zeros((self.n, ACT_DIM))
        self._ready = True
        self._read_state()
        w = self._state["wheels_b"]
        print(f"[balance_env] 접지 사각형 x {w[:, :, 0].min():+.2f}~{w[:, :, 0].max():+.2f}, "
              f"y {w[:, :, 1].min():+.2f}~{w[:, :, 1].max():+.2f}, "
              f"평지 안정 여유 {np.median(self._state['margin']):.2f} m")
        self.reset_all()

    def _read_state(self) -> None:
        super()._read_state()
        if not self._ready:
            return
        s = self._state
        s["q"] = np.asarray(self.art.get_joint_positions(joint_indices=self.arm_idx), np.float64)
        T = np.asarray(self.art._physics_view.get_link_transforms(), np.float64)
        T = T.reshape(self.n, -1, 7)[:, :, :3]
        Rt = np.transpose(s["R"], (0, 2, 1))
        com = (T * self._mass[None, :, None]).sum(1) / self._mass.sum()
        com_b = np.einsum("nij,nj->ni", Rt, com - s["pos"])
        wheels_b = np.einsum("nij,nkj->nki", Rt, T[:, self._wheel_links] - s["pos"][:, None, :])
        s["wheels_b"] = wheels_b
        g = s["g_b"]
        z_plane = wheels_b[:, :, 2].mean(1) - WHEEL_RADIUS
        gz = np.minimum(g[:, 2], -0.05)
        t = (z_plane - com_b[:, 2]) / gz
        px = com_b[:, 0] + t * g[:, 0]
        py = com_b[:, 1] + t * g[:, 1]
        m = np.minimum.reduce([px - wheels_b[:, :, 0].min(1), wheels_b[:, :, 0].max(1) - px,
                               py - wheels_b[:, :, 1].min(1), wheels_b[:, :, 1].max(1) - py])
        s["margin"] = np.where(g[:, 2] > -0.05, -1.0, m)

    def _obs(self) -> np.ndarray:
        if not self._ready:
            return super()._obs()
        s = self._state
        settle = self._settle > 0
        return bp.build_obs(np.where(settle, 0.0, self.v_des), np.where(settle, 0.0, self.w_des),
                            s["g_b"], s["w_b"], s["v_b"], s["wheel"], self.v_out, self.w_out,
                            self.prev_action[:, 2], self.weight, s["q"])

    def reset_idx(self, idx, starts=None):
        out = super().reset_idx(idx, starts)
        if self._ready and len(idx):
            idx = np.asarray(idx, dtype=np.int64)
            self.arm_target[idx] = bp.ARM_HOME
            self.weight[idx] = 0.0
            self.prev5[idx] = 0.0
            self.art.set_joint_position_targets(self.arm_target[idx], indices=idx,
                                                joint_indices=self.arm_idx)
            self._read_state()
            return self._obs()
        return out

    def step(self, action=None, command=None):
        if command is not None:               # 비교 기준 제어기: 팔은 home 그대로
            self.weight[:] = 0.0
            self.arm_target = bp.arm_targets(self.arm_target, self.weight, np.zeros(self.n))
            self.art.set_joint_position_targets(self.arm_target, joint_indices=self.arm_idx)
            obs, rew, flipped, truncated, info = super().step(command=command)
            info["margin"] = self._state["margin"].copy()
            return obs, rew, flipped, truncated, info
        a = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        settle = self._settle > 0
        v, w = bp.action_to_command(a, np.where(settle, 0.0, self.v_des),
                                    np.where(settle, 0.0, self.w_des))
        g = np.where(settle, 0.0, bp.gate(self._state["g_b"]))
        self.weight = bp.weight_vector(a)
        self.arm_target = bp.arm_targets(self.arm_target, self.weight, g)
        self.art.set_joint_position_targets(self.arm_target, joint_indices=self.arm_idx)
        self.prev_action = a[:, :3].copy()
        obs, rew, flipped, truncated, info = super().step(command=(v, w))
        margin = self._state["margin"]
        short = np.clip((SAFE_MARGIN - margin) / SAFE_MARGIN, 0.0, 1.5)
        rew = rew - 1.5 * short ** 2 - 0.05 * np.sum((a - self.prev5) ** 2, axis=1) \
            - 0.02 * np.sum(self.weight ** 2, axis=1) * (g < 0.05)
        self.prev5 = a.copy()
        info["margin"] = margin.copy()
        info["arm_used"] = np.linalg.norm(self.weight, axis=1) * g
        return obs, rew.astype(np.float32), flipped, truncated, info
