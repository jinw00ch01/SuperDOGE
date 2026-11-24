"""뒤집힌 로버를 팔로 다시 세우는 학습 환경.

RoverDriveEnv 의 장면(지형 + 로버 N 대, 물리만)을 그대로 쓰고, 리셋과 보상만 바꾼다.
  리셋: 평평한 곳 위 공중에서 뒤집힌 자세로 떨어뜨려 2 초 동안 자리 잡게 둔다
        (등으로 눕거나, 옆으로 눕거나, 팔에 걸쳐 있거나 — 떨어진 결과가 출발 상태).
  행동: 팔 관절 6 개의 목표 각도 변화. 바퀴는 제동.
  보상: 차체가 서는 쪽으로 간 만큼 + 서 있는 정도 − 과격한 움직임. 일어서서 1 초
        가만히 있으면 성공으로 끝낸다.
팔 구동력: 시뮬레이터 기본값은 9600 N·m 로 사실상 무한이다. arm_effort 로 상한을
줄 수 있다 (실제 팔 사양을 확인한 값이 아니라 가정값).
"""
from __future__ import annotations

import math

import numpy as np
from pxr import PhysxSchema

from drive_env import RoverDriveEnv, quat_to_R
import righting_policy as rp          # drive_env 가 경로를 잡아 둔다

OBS_DIM, ACT_DIM = rp.OBS_DIM, rp.ACT_DIM


def _quat_mul(a, b):
    w0, x0, y0, z0 = a.T
    w1, x1, y1, z1 = b.T
    return np.stack([w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
                     w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
                     w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
                     w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1], axis=1)


def _axis_quat(axis, ang):
    q = np.zeros((len(ang), 4))
    q[:, 0] = np.cos(ang / 2)
    q[:, 1 + axis] = np.sin(ang / 2)
    return q


class RightingEnv(RoverDriveEnv):
    SETTLE_SEC = 2.0
    DROP_HEIGHT = 0.75          # 받침대를 평소 높이보다 이만큼 위에서 놓는다
    HOLD_STEPS = 15             # 서서 이만큼(1 s) 가만히 있으면 성공
    SELF_COLLISION = True       # 팔이 차체를 통과해 땅을 미는 꼼수를 막는다

    def __init__(self, app, arm_effort=None, episode_sec: float = 14.0, **kw):
        self._arm_ready = False
        super().__init__(app, episode_sec=episode_sec, **kw)
        self.arm_idx = [self.art.get_dof_index(n) for n in rp.ARM_JOINTS]
        if arm_effort is not None:
            self.art.set_max_efforts(np.tile(np.asarray(arm_effort, float), (self.n, 1)),
                                     joint_indices=self.arm_idx)
        root = self.stage.GetPrimAtPath(self.art.prim_paths[0])
        sc = PhysxSchema.PhysxArticulationAPI(root).GetEnabledSelfCollisionsAttr()
        eff = self.art.get_max_efforts(joint_indices=self.arm_idx)[0]
        print(f"[right_env] 팔 구동력 상한 {np.round(eff, 0).tolist()} N·m, "
              f"자기 충돌 {sc.Get() if sc else None}")
        self.arm_target = np.tile(rp.ARM_HOME, (self.n, 1))
        self.u_prev = np.zeros(self.n)
        self.hold = np.zeros(self.n, dtype=np.int64)
        self.start_tilt = np.zeros(self.n)
        self._arm_ready = True
        self.reset_all()

    # ── 리셋 ─────────────────────────────────────────────────────────────
    def reset_idx(self, idx, starts=None):
        idx = np.asarray(idx, dtype=np.int64)
        if not self._arm_ready:                 # 부모 생성자 안에서의 첫 호출
            return super().reset_idx(idx, starts)
        k = len(idx)
        if k == 0:
            return self._obs()
        s = self._sample_starts(k)
        kind = self.rng.random(k)
        roll = np.where(kind < 0.65, self.rng.uniform(math.radians(95), math.radians(265), k),
                        self.rng.uniform(-math.pi, math.pi, k))
        pitch = np.where(kind < 0.65, self.rng.uniform(-0.4, 0.4, k),
                         np.where(kind < 0.9, self.rng.uniform(math.radians(95), math.radians(180), k)
                                  * self.rng.choice([-1.0, 1.0], k),
                                  self.rng.uniform(-math.pi, math.pi, k)))
        quat = _quat_mul(_axis_quat(2, s[:, 3]),
                         _quat_mul(_axis_quat(1, pitch), _axis_quat(0, roll)))
        pos = s[:, :3].copy()
        pos[:, 2] += self.DROP_HEIGHT
        self.art.set_world_poses(positions=pos, orientations=quat, indices=idx)
        self.art.set_velocities(np.zeros((k, 6)), indices=idx)
        q0 = self.default_q[idx].copy()
        self.art.set_joint_positions(q0, indices=idx)
        self.art.set_joint_velocities(np.zeros_like(q0), indices=idx)
        self.art.set_joint_velocity_targets(np.zeros((k, 6)), indices=idx,
                                            joint_indices=self.drive_idx)
        self.art.set_joint_position_targets(np.zeros((k, 4)), indices=idx,
                                            joint_indices=self.steer_idx)
        self.arm_target[idx] = rp.ARM_HOME
        self.art.set_joint_position_targets(self.arm_target[idx], indices=idx,
                                            joint_indices=self.arm_idx)
        self.steps[idx] = 0
        self.hold[idx] = 0
        self._settle[idx] = int(self.SETTLE_SEC / self.dt)
        self._read_state()
        self._state["R"][idx] = quat_to_R(quat)
        self._state["g_b"][idx] = -self._state["R"][idx][:, 2, :]
        self.u_prev[idx] = -self._state["g_b"][idx, 2]
        return self._obs()

    # ── 관측·스텝 ────────────────────────────────────────────────────────
    def _read_state(self) -> None:
        super()._read_state()
        if self._arm_ready:
            self._state["q"] = np.asarray(
                self.art.get_joint_positions(joint_indices=self.arm_idx), dtype=np.float64)
            self._state["dq"] = np.asarray(
                self.art.get_joint_velocities(joint_indices=self.arm_idx), dtype=np.float64)

    def _obs(self) -> np.ndarray:
        if not self._arm_ready:
            return super()._obs()
        s = self._state
        return rp.build_obs(s["g_b"], s["w_b"], s["q"], s["dq"], self.arm_target)

    def step(self, action):
        """return: obs, reward, success, truncated, info"""
        act = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        live = self._settle <= 0
        tgt = rp.next_target(self.arm_target, self._state["q"], act)
        self.arm_target[live] = tgt[live]
        self.art.set_joint_position_targets(self.arm_target, joint_indices=self.arm_idx)
        for _ in range(self.decimation):
            self.world.step(render=False)
        self._read_state()
        s = self._state
        u = -s["g_b"][:, 2]                       # 1 = 바로 섬, −1 = 완전히 뒤집힘
        spin = np.linalg.norm(s["w_b"], axis=1)
        speed = np.linalg.norm(s["v_b"], axis=1)
        just_settled = (self._settle == 1)
        self._settle = np.maximum(self._settle - 1, 0)
        self.start_tilt[just_settled] = s["tilt"][just_settled]
        reward = (4.0 * (u - self.u_prev) + 0.05 * (u - 1.0)
                  - 0.003 * (act ** 2).sum(axis=1)
                  - 0.02 * np.clip(spin - 2.0, 0.0, None) ** 2
                  - 0.02 * np.clip(speed - 1.5, 0.0, None) ** 2)
        reward = np.where(live, reward, 0.0)
        self.u_prev = u
        upright = (s["tilt"] < rp.UPRIGHT_DEG) & (spin < 0.4) & (speed < 0.3)
        self.hold = np.where(upright & live, self.hold + 1, 0)
        success = self.hold >= self.HOLD_STEPS
        reward = reward + 10.0 * success
        self.steps += live.astype(np.int64)
        self.total_steps += 1
        x, y = s["pos"][:, 0], s["pos"][:, 1]
        off = (np.abs(x) > self.map_w / 2 - 1.5) | (np.abs(y) > self.map_h / 2 - 1.5) \
            | (s["pos"][:, 2] < -20.0) | ~np.isfinite(s["pos"]).all(axis=1)
        truncated = ((self.steps >= self.max_steps) | off) & ~success
        info = {"speed": speed, "tilt": s["tilt"].copy(), "start_tilt": self.start_tilt.copy(),
                "steps": self.steps.copy(), "live": live}
        return self._obs(), reward.astype(np.float32), success, truncated, info
