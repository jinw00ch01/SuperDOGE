"""학습된 구동 조절기(drive governor)의 입출력 규약과 추론 — 순수 numpy.

학습 환경(isaac_sim/rl/drive_env.py)과 ROS 노드(drive_governor_node)가 이 파일
하나를 같이 쓴다. 관측을 만드는 식과 출력을 바퀴 명령으로 바꾸는 식이 두 곳에서
어긋나면 정책이 조용히 틀린 동작을 하기 때문이다.

입력 (20 개)
    명령 v, w (2) · 차체 기준 중력 방향 (3) · 차체 각속도 (3) · 차체 선속도 (3)
    · 바퀴 6 개의 접선 속도 (6) · 직전에 내보낸 v, w (2) · 직전 조향 보정 (1)
출력 (3 개, 각각 −1 ~ 1)
    a0: 선속도를 명령의 몇 배로 낼지 (0 ~ 1 배)
    a1: 각속도를 명령의 몇 배로 낼지 (0 ~ 1 배)
    a2: 전진 중 조향 보정 (±STEER_FIX rad/s) — 비탈에서 옆으로 흐를 때 맞조향

정책은 명령을 깎기만 한다. 명령보다 빠르게 가거나 반대로 돌 수 없다.
"""
from __future__ import annotations

import numpy as np

V_CMD_SCALE = 3.0      # 관측 정규화용 (상위 노드의 최대 명령)
V_CAP = 2.0            # 실제로 낼 수 있는 최고 선속도 (m/s)
W_CAP = 1.5            # 최고 각속도 (rad/s)
STEER_FIX = 0.3        # 조향 보정 범위 (rad/s)
WHEEL_RADIUS = 0.1
OBS_DIM = 20
ACT_DIM = 3

DRIVE_JOINTS = ["FL_Drive_Continuous", "FR_Drive_Continuous", "CL_Drive_Continuous",
                "CR_Drive_Continuous", "RL_Drive_Continuous", "RR_Drive_Continuous"]


def build_obs(v_des, w_des, g_b, w_b, v_b, wheel_rad_s, v_out, w_out, steer_fix_prev):
    """모든 인자는 앞 차원이 로버 수 N 인 배열 (스칼라 묶음은 (N,), 벡터는 (N, k))."""
    v_des = np.atleast_1d(np.asarray(v_des, dtype=np.float64))
    n = len(v_des)

    def col(a):
        return np.asarray(a, dtype=np.float64).reshape(n, 1)

    def mat(a, k):
        return np.asarray(a, dtype=np.float64).reshape(n, k)

    return np.concatenate([
        col(v_des) / V_CMD_SCALE, col(w_des) / W_CAP,
        mat(g_b, 3), mat(w_b, 3), mat(v_b, 3),
        mat(wheel_rad_s, 6) * WHEEL_RADIUS,
        col(v_out) / V_CAP, col(w_out) / W_CAP,
        col(steer_fix_prev),
    ], axis=1).astype(np.float32)


def action_to_command(action, v_des, w_des):
    """action (N, 3) ∈ [−1, 1] → 실제로 내보낼 (v, w)."""
    a = np.clip(np.asarray(action, dtype=np.float64).reshape(-1, ACT_DIM), -1.0, 1.0)
    v_c = np.clip(np.atleast_1d(np.asarray(v_des, dtype=np.float64)), -V_CAP, V_CAP)
    w_c = np.clip(np.atleast_1d(np.asarray(w_des, dtype=np.float64)), -W_CAP, W_CAP)
    v = 0.5 * (a[:, 0] + 1.0) * v_c
    w = 0.5 * (a[:, 1] + 1.0) * w_c + STEER_FIX * a[:, 2] * (np.abs(v_c) > 0.05)
    return v, w


def gravity_in_body(quat_xyzw):
    """world←body 자세(x, y, z, w) → 차체 기준 중력 방향 단위벡터 (아래가 −z)."""
    x, y, z, w = [float(c) for c in quat_xyzw]
    # R 의 셋째 행 (world z 축을 body 좌표로 본 것) 에 −1.
    return np.array([-(2 * (x * z - y * w)), -(2 * (y * z + x * w)),
                     -(1 - 2 * (x * x + y * y))])


def rotate_world_to_body(quat_xyzw, vec):
    x, y, z, w = [float(c) for c in quat_xyzw]
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return R.T @ np.asarray(vec, dtype=np.float64)


class GovernorPolicy:
    """학습 결과(.npz)를 읽어 평균 행동을 낸다. 층: Linear → tanh → … → Linear → tanh."""

    def __init__(self, path: str):
        data = np.load(path)
        self.layers = []
        k = 0
        while f"w{k}" in data:
            self.layers.append((data[f"w{k}"].astype(np.float64),
                                data[f"b{k}"].astype(np.float64)))
            k += 1
        if not self.layers:
            raise ValueError(f"{path}: 정책 가중치(w0, b0, …)가 없습니다")
        if self.layers[0][0].shape[1] != OBS_DIM or self.layers[-1][0].shape[0] != ACT_DIM:
            raise ValueError(
                f"{path}: 입출력 크기가 다릅니다 (입력 {self.layers[0][0].shape[1]} / "
                f"출력 {self.layers[-1][0].shape[0]}, 필요 {OBS_DIM} / {ACT_DIM})")
        self.meta = {k: data[k].item() if data[k].shape == () else data[k]
                     for k in data.files if not (k[0] in "wb" and k[1:].isdigit())}

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        h = np.asarray(obs, dtype=np.float64).reshape(-1, self.layers[0][0].shape[1])
        for w, b in self.layers:
            h = np.tanh(h @ w.T + b)
        return h
