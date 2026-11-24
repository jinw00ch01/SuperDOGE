"""비탈에서 균형을 지키는 정책 — 학습 환경과 ROS 노드가 같이 쓰는 부분.

구동 조절기(governor_policy)에 두 가지를 더한 것이다.
  1. 물러서기: 전진 명령을 깎는 것(0~1 배)뿐 아니라 뒤로 빼는 것(−0.25 배까지)도 할 수 있다.
  2. 팔 균형추: 팔(13.5 kg, 로버 전체 62.6 kg)을 기울어진 반대쪽으로 뻗어 무게중심을
     옮긴다. 실측: 앞 +5 cm, 뒤 −13 cm, 좌우 ±8 cm, 높이 −2~5 cm.
     팔은 차체가 GATE_START~GATE_FULL 도 넘게 기울었을 때만 움직인다 (평지에서는 home).

관측 24 개: 구동 조절기의 20 개 + 지난번 균형추 벡터(2) + 팔이 실제로 뻗은 정도(2).
행동 5 개(−1~1): 전진 배율, 회전 배율, 조향 보정, 균형추 벡터 (차체 기준 x, y).
"""
from __future__ import annotations

import numpy as np

try:
    from . import governor_policy as gp
except ImportError:
    import governor_policy as gp

GovernorPolicy = gp.GovernorPolicy
ARM_JOINTS = [f"joint_{k}" for k in range(1, 7)]
ARM_HOME = np.array([0.0, 0.0, 1.5708, 0.0, 1.5708, 0.0])
OBS_DIM = 24
ACT_DIM = 5
REVERSE = 0.25         # 뒤로 뺄 때 낼 수 있는 속도 = 명령(최대 2 m/s)의 이 배율
GATE_START, GATE_FULL = 10.0, 20.0     # 팔을 쓰기 시작하는 / 다 쓰는 기울기 (도)
LEAN = 1.3             # joint_2 를 최대 이만큼 눕힌다 (rad)
EXTEND = 1.2           # joint_3 를 최대 이만큼 편다 (rad)
ARM_RATE = 0.10        # 팔 목표가 한 스텝(1/15 s)에 움직이는 최대 각 (rad)


def tilt_deg(g_b) -> np.ndarray:
    g = np.atleast_2d(np.asarray(g_b, dtype=np.float64))
    return np.degrees(np.arccos(np.clip(-g[:, 2], -1.0, 1.0)))


def gate(g_b) -> np.ndarray:
    return np.clip((tilt_deg(g_b) - GATE_START) / (GATE_FULL - GATE_START), 0.0, 1.0)


def weight_vector(action) -> np.ndarray:
    """행동의 뒤 두 값 → 균형추 벡터 (크기 1 이하)."""
    c = np.atleast_2d(np.asarray(action, dtype=np.float64))[:, 3:5]
    n = np.linalg.norm(c, axis=1, keepdims=True)
    return c / np.maximum(n, 1.0)


def action_to_command(action, v_des, w_des):
    """→ 실제로 내보낼 (v, w). 전진 배율은 −REVERSE ~ 1."""
    a = np.clip(np.atleast_2d(np.asarray(action, dtype=np.float64)), -1.0, 1.0)
    v_c = np.clip(np.atleast_1d(np.asarray(v_des, dtype=np.float64)), -gp.V_CAP, gp.V_CAP)
    w_c = np.clip(np.atleast_1d(np.asarray(w_des, dtype=np.float64)), -gp.W_CAP, gp.W_CAP)
    scale = 0.5 * (1.0 + REVERSE) * a[:, 0] + 0.5 * (1.0 - REVERSE)
    v = scale * v_c
    w = 0.5 * (a[:, 1] + 1.0) * w_c + gp.STEER_FIX * a[:, 2] * (np.abs(v_c) > 0.05)
    return v, w


def arm_targets(prev, weight, g) -> np.ndarray:
    """균형추 벡터·기울기 문턱 → 팔 관절 목표 (N, 6). prev 에서 ARM_RATE 씩만 움직인다."""
    prev = np.atleast_2d(np.asarray(prev, dtype=np.float64))
    weight = np.atleast_2d(weight)
    amount = np.linalg.norm(weight, axis=1) * np.atleast_1d(g)
    want = np.tile(ARM_HOME, (len(prev), 1))
    aim = np.arctan2(weight[:, 1], weight[:, 0])       # joint_1 = 팔이 향하는 방향 (0 = 앞, +90° = 왼쪽)
    want[:, 0] = np.where(amount > 0.1, aim, prev[:, 0])
    want[:, 1] = LEAN * amount
    want[:, 2] = ARM_HOME[2] - EXTEND * amount
    # 접힌 채로는 어느 쪽을 봐도 무게중심이 거의 같으므로, 눕히기 전에 방향부터 맞춘다.
    off = np.abs(want[:, 0] - prev[:, 0])
    hold = (off > 0.6) & (prev[:, 1] < 0.3)
    want[:, 1] = np.where(hold, np.minimum(want[:, 1], prev[:, 1]), want[:, 1])
    want[:, 2] = np.where(hold, np.maximum(want[:, 2], prev[:, 2]), want[:, 2])
    return prev + np.clip(want - prev, -ARM_RATE, ARM_RATE)


def build_obs(v_des, w_des, g_b, w_b, v_b, wheel, v_out, w_out, steer_fix, weight, q_arm):
    base = gp.build_obs(v_des, w_des, g_b, w_b, v_b, wheel, v_out, w_out, steer_fix)
    q = np.atleast_2d(np.asarray(q_arm, dtype=np.float64))
    used = np.stack([q[:, 1] / LEAN, (ARM_HOME[2] - q[:, 2]) / EXTEND], axis=1)
    return np.concatenate([base, np.atleast_2d(weight), used], axis=1).astype(np.float32)
