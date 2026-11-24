"""뒤집힌 로버를 팔로 다시 세우는 정책 — 학습 환경과 ROS 노드가 같이 쓰는 부분.

관측 24 개: 차체 기준 중력 방향(3), 차체 각속도(3), 팔 관절 각도(6), 팔 관절
속도(6), 지금 내보내고 있는 팔 관절 목표(6).
행동 6 개(−1~1): 팔 관절 목표를 한 제어 스텝에 ±DELTA rad 씩 움직인다.
바퀴는 건드리지 않는다 (정지 명령 = 제동 상태).
"""
from __future__ import annotations

import numpy as np

try:                                   # ROS 패키지로 쓸 때
    from .governor_policy import GovernorPolicy  # noqa: F401  (같은 MLP)
except ImportError:                    # 학습 스크립트가 파일을 직접 읽을 때
    from governor_policy import GovernorPolicy  # noqa: F401

ARM_JOINTS = [f"joint_{k}" for k in range(1, 7)]
ARM_HOME = np.array([0.0, 0.0, 1.5708, 0.0, 1.5708, 0.0])
ARM_LOW = np.array([-3.14, -2.2, -2.6, -3.14, -2.3, -3.14])
ARM_HIGH = -ARM_LOW
OBS_DIM = 24
ACT_DIM = 6
CONTROL_HZ = 15.0
DELTA = 0.12          # 한 스텝에 목표가 움직이는 최대 각 (rad) → 1.8 rad/s
MAX_GAP = 0.5         # 목표가 실제 각도보다 앞서 갈 수 있는 최대 (rad) — 땅을 미는 힘의 상한
UPRIGHT_DEG = 15.0    # 이보다 덜 기울고 가만히 있으면 "일어섰다"
FLIPPED_DEG = 60.0    # 이보다 더 기울어 멈춰 있으면 "넘어졌다"


def build_obs(g_b, w_b, q, dq, target) -> np.ndarray:
    """모두 (N, k) 또는 (k,). return (N, 24) float32."""
    g_b, w_b, q, dq, target = (np.atleast_2d(np.asarray(a, dtype=np.float64))
                               for a in (g_b, w_b, q, dq, target))
    return np.concatenate([g_b, np.clip(w_b / 3.0, -2.0, 2.0), q / np.pi,
                           np.clip(dq / 3.0, -2.0, 2.0), target / np.pi],
                          axis=1).astype(np.float32)


def next_target(target, q, action) -> np.ndarray:
    """행동 → 새 팔 관절 목표."""
    t = np.asarray(target, dtype=np.float64) + DELTA * np.clip(action, -1.0, 1.0)
    q = np.asarray(q, dtype=np.float64)
    t = np.clip(t, q - MAX_GAP, q + MAX_GAP)
    return np.clip(t, ARM_LOW, ARM_HIGH)
