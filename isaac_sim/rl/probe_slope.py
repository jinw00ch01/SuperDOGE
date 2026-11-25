"""비탈 등판 시험 — "몇 도까지, 어떻게 하면 올라가나" 를 잰다 (학습 아님, 측정).

    tools/isaac-pypi isaac_sim/rl/probe_slope.py --terrain terrain_00015 --out <json>

지형에서 경사가 고른 오르막 띠(--length, 기본 1.2 m)를 경사 구간별로 찾아 로버를 올려 보낸다.
생성 지형의 급경사는 짧다 (terrain_00015: 23° 넘는 띠의 길이 중앙 1.0 m, 최대 2 m, 높이차 0.6 m).
그래서 "꾸준히 오르는 속도" 가 아니라 "그 띠를 넘어 1 m 더 가는가" 를 본다.
  rest  : 띠의 아래 끝에 정지해 있다가 출발 (비탈에 걸려 선 상황)
  runup : 띠 아래 2.5 m 쯤에서 속도를 붙여 진입
명령은 지형을 모르는 일정한 전진 명령(가속 1.5 m/s²)이고 방향만 오르막 쪽으로 잡아 준다.

--wheels mesh 로 원본 바퀴 형상과 비교한다 (같은 seed 면 같은 자리를 쓴다). terrain_00015,
명령 0.92 m/s 에서 넘은 비율 (원본 → 캡슐): 정지 출발 20° 3/9 → 6/9, 23° 0/6 → 2/6,
달려서 진입 23° 0/3 → 3/3, 27° 이상은 양쪽 다 0 (마찰 한계 atan 0.52 = 27.5°).
--slip-caps 는 "바퀴를 차체보다 조금만 빨리 돌리기" 시험이다 (원본 바퀴의 정지 출발에는
조금 도움이 됐지만 달려서 진입할 때는 탄력을 죽여 전부 실패 — 채택하지 않음).

--compare 는 "수동 W 로는 오르는데 자율주행은 못 오른다" 를 가르는 시험이다. 같은 자리에
같은 출발 조건으로 명령만 바꿔 넣는다. 모든 명령은 시뮬레이터의 구동 제한기와 같은 것
(drive_env.SlewLimiter = drive_limits._ack_limit) 을 거친다.
  manual<v>  : HUD 에서 W 만 누른 것 — lin = v (0.3 / 0.6 / 0.9), ang = 0, 방향 보정 없음
  nav        : 탐색 노드의 실제 Navigator (online 지도 설정: 제자리 회전 25°, max_lin 3.0,
               max_ang 1.5, kp 2.0) 가 띠를 곧장 넘는 경로를 따라감
  nav_off<d> : 같은데 로버가 띠와 d° 어긋난 방향으로 서 있음 (25° 넘으면 제자리 회전 뒤 출발)
  nav_kink   : 경로가 띠 중간에서 옆으로 0.6 m 꺾임 (재계획·경유점 때문에 비탈 위에서 방향을
               트는 경우)
  approach<v>: 광물 접근(mission_manager._approach_twist) 과 같은 꼴 — lin = v, ang = 1.5·err
               (오차 0.4 rad 넘으면 lin × 0.3), 목표는 띠 너머
               (mvp_multi 의 approach_lin_speed 는 1.2 지만 구동 제한기가 0.92 로 자른다)
출발은 rest(띠 아래 끝에 정지) 와 runup(2.5 m 아래에서 출발) 둘 다 본다.
시도마다 기울기 반사(mission_manager tilt_guard: 앞 30° / 옆 32°) 가 자율 모드였다면
넘기 전에 발동했을지(guard_in_reached) 도 센다 — manual 이 넘은 시도 중 이 값이 크면
"수동은 오르는데 자율은 반사로 되돌아온다" 가 원인이다.

    tools/isaac-pypi isaac_sim/rl/probe_slope.py --terrain terrain_00015 --bins 20,23,25 \
        --compare manual0.6,manual0.9,nav,nav_off18,nav_off30,nav_kink,approach0.92 --out <json>
"""
import argparse
import json
import math
import os
import sys

_p = argparse.ArgumentParser()
_p.add_argument("--terrain", default="terrain_00015")
_p.add_argument("--out", default="")
_p.add_argument("--num-envs", type=int, default=16)
_p.add_argument("--friction", type=float, default=1.0)
_p.add_argument("--bins", default="15,20,23,25,27,29,31,34")
_p.add_argument("--rest-speeds", default="0.3,0.92,1.5")
_p.add_argument("--runup-speeds", default="0.92,1.2,1.5,2.0")
_p.add_argument("--length", type=float, default=1.2, help="경사가 고른 띠의 길이 (m)")
_p.add_argument("--beyond", type=float, default=1.0, help="띠를 지나 이만큼 더 가면 넘은 것 (m)")
_p.add_argument("--slip-caps", default="0",
                help="바퀴 속도를 '차체 속도 + 이 값(m/s)' 로 묶는 시험 (헛돎 제한). 0 = 묶지 않음")
_p.add_argument("--wheels", default="capsule",
                choices=["mesh", "capsule", "sphere", "cylinder"])
_p.add_argument("--seed", type=int, default=0)
_p.add_argument("--wheel-grip", type=float, default=1.0,
                help="바퀴 재질의 마찰 배율 (run_vehicle_v3 --wheel-grip 과 같은 경로)")
_p.add_argument("--compare", default="",
                help="명령 방식 비교 (쉼표 구분): manual<v>, nav, nav_off<deg>, nav_kink, "
                     "approach<v>. 주면 기존 rest/runup 속도 시험 대신 이것만 돈다")
args, _ = _p.parse_known_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})

import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drive_env import REST_HEIGHT, RoverDriveEnv  # noqa: E402
from navigator import Navigator  # noqa: E402  (drive_env 가 isaac_drive/navigation 을 path 에 넣는다)

RoverDriveEnv.WHEEL_SHAPE = args.wheels
RoverDriveEnv.WHEEL_GRIP = args.wheel_grip
env = RoverDriveEnv(app, terrain=args.terrain, num_envs=args.num_envs, seed=args.seed)
env.friction_range = (args.friction, args.friction)
env.randomize_friction()
n, dt, res = env.n, env.dt, env.res
L = float(args.length)
GOAL = L + float(args.beyond)
hm = env.hm
hs = ndimage.uniform_filter(hm, int(round(0.8 / res)))          # 차체 길이쯤으로 고른 높이
gy, gx = np.gradient(hs, res)
half_w, half_h = env.map_w / 2, env.map_h / 2
zones = [np.array(z["polygon"], dtype=float) for z in (env.meta.get("physics_zones") or [])]


def height(x, y, smooth=True):
    i, j = env._cell(x, y)
    return (hs if smooth else hm)[i, j]


def clear(x, y):
    """경계·구조물에서 떨어져 있고 기본 레골리스 위인가."""
    i, j = env._cell(x, y)
    ok = (np.abs(x) < half_w - 3.0) & (np.abs(y) < half_h - 3.0) & (env.solid_dist[i, j] > 1.5)
    for z in zones:
        ok &= ~((x > z[:, 0].min()) & (x < z[:, 0].max())
                & (y > z[:, 1].min()) & (y < z[:, 1].max()))
    return ok


def grade_profile(x, y, ux, uy, s):
    """(K,) 출발점·방향, s (M,) → 진행 방향 경사 (K, M) [도] 와 통과 가능 여부."""
    xs = x[:, None] + s[None, :] * ux[:, None]
    ys = y[:, None] + s[None, :] * uy[:, None]
    h = height(xs, ys)
    g = np.degrees(np.arctan(np.gradient(h, s, axis=1)))
    g = ndimage.uniform_filter1d(g, 5, axis=1)
    return g, clear(xs, ys).all(axis=1)


# ── 시험할 자리 찾기 ──────────────────────────────────────────
step = max(1, int(round(0.2 / res)))
ii, jj = np.mgrid[step:hm.shape[0] - step:step, step:hm.shape[1] - step:step]
ii, jj = ii.ravel(), jj.ravel()
cx = (jj + 0.5) * res - half_w
cy = (ii + 0.5) * res - half_h
norm = np.hypot(gx[ii, jj], gy[ii, jj])
keep = norm > math.tan(math.radians(4.0))
cx, cy, ux, uy = cx[keep], cy[keep], (gx[ii, jj] / np.maximum(norm, 1e-9))[keep], \
    (gy[ii, jj] / np.maximum(norm, 1e-9))[keep]
s_on = np.arange(0.0, L + 1e-6, 0.1)
g_on, _ = grade_profile(cx, cy, ux, uy, s_on)
_, ok_on = grade_profile(cx, cy, ux, uy, np.arange(-1.0, GOAL + 1.0, 0.1))
g_mean, g_lo, g_hi = g_on.mean(axis=1), g_on.min(axis=1), g_on.max(axis=1)
rng = np.random.default_rng(args.seed)


def pick(mask, k):
    """서로 3 m 이상 떨어진 자리 k 개."""
    cand = np.nonzero(mask)[0]
    rng.shuffle(cand)
    out = []
    for c in cand:
        if all(math.hypot(cx[c] - cx[o], cy[c] - cy[o]) > 3.0 for o in out):
            out.append(int(c))
            if len(out) == k:
                break
    return out


def runup_start(c):
    """띠 아래 2.5 m 가 달려올 만한가 (내리막 25° ~ 오르막 15°: 골로 내려왔다 오르는 경우 포함)."""
    s = np.arange(-2.8, -0.4, 0.1)
    g, ok = grade_profile(cx[[c]], cy[[c]], ux[[c]], uy[[c]], s)
    if ok[0] and g[0].max() < 15.0 and g[0].min() > -25.0:
        return -2.5
    return None


def quat_mul(a, b):
    """Hamilton 곱 (w, x, y, z)."""
    return np.array([a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
                     a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
                     a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
                     a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0]])


def quat_from_R(R):
    w = math.sqrt(max(0.0, 1.0 + R[0, 0] + R[1, 1] + R[2, 2])) / 2.0
    if w > 1e-6:
        return np.array([w, (R[2, 1] - R[1, 2]) / (4 * w), (R[0, 2] - R[2, 0]) / (4 * w),
                         (R[1, 0] - R[0, 1]) / (4 * w)])
    return np.array([0.0, 1.0, 0.0, 0.0])


def on_ground_pose(x, y, yaw):
    """(x, y) 의 지면에 비탈을 따라 기울여 올려놓는 자세 → 위치, quat.

    수평 자세로 떨어뜨리면 가파른 곳에서는 뒤쪽이 0.5 m 넘게 떨어지며 뒤로 넘어간다
    (처음 시험에서 27° 구간이 출발 전에 전부 넘어진 이유).
    """
    i, j = env._cell(x, y)
    nrm = np.array([-gx[i, j], -gy[i, j], 1.0])
    nrm /= np.linalg.norm(nrm)
    fwd = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    fwd = fwd - nrm * float(fwd @ nrm)
    fwd /= np.linalg.norm(fwd)
    left = np.cross(nrm, fwd)
    R = np.stack([fwd, left, nrm], axis=1)
    pos = np.array([x, y, hs[i, j]]) + nrm * (REST_HEIGHT + 0.05)
    return pos, quat_mul(quat_from_R(R), np.asarray(env.default_quat, dtype=np.float64))


def parking(sites, k):
    cells = env._start_cells[rng.permutation(len(env._start_cells))]
    out = []
    for c in cells:
        x, y = (c[1] + 0.5) * res - half_w, (c[0] + 0.5) * res - half_h
        if all(math.hypot(x - px, y - py) > 6.0 for px, py in sites) and \
                all(math.hypot(x - o[0], y - o[1]) > 2.5 for o in out):
            out.append((x, y, hm[c[0], c[1]] + REST_HEIGHT + 0.12, 0.0))
            if len(out) == k:
                break
    if len(out) < k:
        raise RuntimeError(f"세워 둘 자리가 모자람 ({len(out)}/{k})")
    return out


class _Pose:
    """Navigator 가 부르는 get_pose_2d() — 시험 중인 로버 하나의 자세."""

    def __init__(self, q):
        self.q = q

    def get_pose_2d(self):
        p, R = env._state["pos"][self.q], env._state["R"][self.q]
        return float(p[0]), float(p[1]), float(math.atan2(R[1, 0], R[0, 0]))


def parse_ctrl(name):
    """'manual0.6' → ('manual', 0.6), 'nav_off18' → ('nav', 18°), 'nav_kink' → ('kink', 0)."""
    if name.startswith("manual"):
        return "manual", float(name[6:] or 0.6), 0.0
    if name.startswith("approach"):
        return "approach", float(name[8:] or 0.6), 0.0
    if name == "nav":
        return "nav", 0.0, 0.0
    if name.startswith("nav_off"):
        return "nav", 0.0, math.radians(float(name[7:]))
    if name == "nav_kink":
        return "kink", 0.0, 0.0
    raise ValueError(f"알 수 없는 --compare 항목: {name}")


def run_trial(sites, s0, v_cmd, drive_sec, settle_sec=1.5, slip_cap=0.0, ctrl=None):
    """sites: 자리 index 목록, s0: 자리마다의 출발 s (0 = 비탈 위, 음수 = 아래에서 달려옴).

    ctrl=None 이면 예전 시험 (가속 1.5 m/s² 로 v_cmd 까지, 오르막 방향 유지).
    ctrl=(종류, 속도, 방향 어긋남 rad) 이면 parse_ctrl 의 명령을 구동 제한기를 거쳐 넣는다.
    """
    k = len(sites)
    kind, v_ctrl, off = ctrl if ctrl is not None else (None, 0.0, 0.0)
    starts, yaws = [], []
    for q, (c, s) in enumerate(zip(sites, s0)):
        x, y = cx[c] + s * ux[c], cy[c] + s * uy[c]
        i, j = env._cell(x, y)
        r = int(0.65 / res)
        z = hm[max(i - r, 0):i + r + 1, max(j - r, 0):j + r + 1].max() + REST_HEIGHT + 0.05
        yaw = math.atan2(uy[c], ux[c])
        f = yaw + (off if q % 2 == 0 else -off)          # 어긋나는 방향은 번갈아
        starts.append((x, y, z, f))
        yaws.append(yaw)
    park = parking([(st[0], st[1]) for st in starts] + [(cx[c], cy[c]) for c in sites], n - k)
    starts = np.array(starts + park)
    env.reset_idx(np.arange(n), starts=starts)
    poses = [on_ground_pose(st[0], st[1], st[3]) for st in starts[:k]]
    env.art.set_world_poses(positions=np.array([p for p, _ in poses]),
                            orientations=np.array([q for _, q in poses]), indices=np.arange(k))
    yaws = np.array(yaws)
    u = np.stack([np.cos(yaws), np.sin(yaws)], axis=1)
    p0 = np.array([[cx[c], cy[c]] for c in sites])
    goal = p0 + (GOAL + 1.0) * u                      # 띠를 넘어 판정선보다 1 m 더
    navs = []
    if kind in ("nav", "kink"):
        for q in range(k):
            sx, sy = starts[q][0], starts[q][1]
            path = []
            # 경유점은 A* + 단순화가 내는 것처럼 최대 2 m 간격
            d0 = float(np.dot(np.array([sx, sy]) - p0[q], u[q]))
            for d in np.arange(d0 + 2.0, GOAL + 1.0, 2.0):
                path.append(tuple(p0[q] + d * u[q]))
            if kind == "kink":                       # 띠 가운데에서 옆으로 0.6 m
                side = np.array([-u[q, 1], u[q, 0]]) * (0.6 if q % 2 == 0 else -0.6)
                path = [w for w in path if np.dot(np.array(w) - p0[q], u[q]) < 0.0]
                path += [tuple(p0[q] + 0.5 * L * u[q] + side)]
            path.append(tuple(goal[q]))
            nav = Navigator(_Pose(q), waypoint_tol=0.4, final_tol=0.8, kp_ang=2.0,
                            max_lin=3.0, max_ang=1.5, point_turn_deg=25.0)
            nav.set_path(path)
            navs.append(nav)
    limiter = env.baseline_limiter() if kind else None
    v = np.zeros(n)
    log_s, log_tilt, log_pitch, log_z, log_roll = [], [], [], [], []
    flipped_any = np.zeros(k, bool)
    T = int((settle_sec + drive_sec) / dt)
    w_log = []
    for t in range(T):
        go = t * dt >= settle_sec
        R = env._state["R"][:k]
        w = np.zeros(n)
        if kind is None:
            v[:k] = np.minimum(v[:k] + 1.5 * dt, v_cmd) if go else 0.0
            err = (yaws - np.arctan2(R[:, 1, 0], R[:, 0, 0]) + np.pi) % (2 * np.pi) - np.pi
            w[:k] = np.where(v[:k] > 0.2, np.clip(1.2 * err, -0.3 * v[:k], 0.3 * v[:k]), 0.0)
            v_send = v.copy()
            if slip_cap > 0.0:             # 헛돎 제한: 바퀴가 차체보다 slip_cap 넘게 빨리 돌지 않게
                vx_now = np.clip(env._state["v_b"][:k, 0], 0.0, None)
                v_send[:k] = np.minimum(v[:k], vx_now + slip_cap)
        else:
            v[:] = 0.0
            if go:
                for q in range(k):
                    if flipped_any[q]:
                        continue
                    if kind == "manual":
                        v[q] = v_ctrl
                    elif kind == "approach":   # mission_manager._approach_twist 와 같은 식
                        px, py, yaw_q = _Pose(q).get_pose_2d()
                        e = math.atan2(goal[q, 1] - py, goal[q, 0] - px) - yaw_q
                        e = (e + math.pi) % (2 * math.pi) - math.pi
                        v[q] = v_ctrl * (0.3 if abs(e) > 0.4 else 1.0)
                        cap = 1.2 if abs(e) <= 0.4 else min(1.2, math.sqrt(2 * 0.4 * abs(e)) / 1.9)
                        w[q] = max(-cap, min(cap, 1.5 * e))
                    else:
                        lin, ang, _adv = navs[q].step()
                        v[q], w[q] = lin, ang
            v_send, w = limiter(v, w)              # 시뮬레이터 구동 제한기와 같은 것
            if go:
                w_log.append((v_send[:k].copy(), w[:k].copy()))
        _, _, flipped, _, info = env.step(command=(v_send, w))
        pos = info["pos"][:k]
        log_s.append(np.sum((pos[:, :2] - p0) * u, axis=1))
        log_z.append(pos[:, 2].copy())
        log_tilt.append(info["tilt"][:k].copy())
        log_pitch.append(np.degrees(np.arcsin(np.clip(env._state["R"][:k, 2, 0], -1, 1))))
        log_roll.append(np.degrees(np.arcsin(np.clip(env._state["R"][:k, 2, 1], -1, 1))))
        flipped_any |= flipped[:k]
        v[:k] = np.where(flipped_any, 0.0, v[:k])
    S, TL, Z = np.array(log_s), np.array(log_tilt), np.array(log_z)
    i_go = int(settle_sec / dt)
    out = []
    for q in range(k):
        s = S[:, q]
        over = np.nonzero(s >= GOAL)[0]
        if len(over):                      # 넘어감: 띠의 아래 끝(s=0) → GOAL 에 걸린 시간으로 속도
            a = max(int(np.argmax(s >= 0.0)), i_go)
            speed = (s[over[0]] - s[a]) / max((over[0] - a) * dt, 1e-6)
        else:                              # 못 감: 마지막 3 s 의 속도
            a = max(len(s) - int(3.0 / dt), i_go)
            speed = (s[-1] - s[a]) / max((len(s) - 1 - a) * dt, 1e-6)
        # 임무 스택의 기울기 반사(tilt_guard: 앞 30° · 옆 32°, AUTO 에서만 켜짐 — 수동에서는 꺼짐)가
        # 이 시도 중에 발동했을 것인가. 넘은 경우는 넘을 때까지만 본다.
        e_g = int(over[0]) + 1 if len(over) else len(s)
        P = np.array(log_pitch)[i_go:e_g, q]
        Rl = np.abs(np.array(log_roll)[i_go:e_g, q])
        guard = bool(len(P) and ((P > 30.0).any() or (Rl > 32.0).any()))
        extra = {}
        if w_log:
            # 띠 위(0 <= s < GOAL)에 있던 동안 바퀴에 들어간 명령: 평균 |회전 명령|, 전진 명령이
            # 0 이었던 시간(제자리 회전·정지), 전진 명령 평균. 넘었으면 넘을 때까지만 본다.
            Vs = np.array([a[q] for a, _ in w_log])
            Ws = np.array([b[q] for _, b in w_log])
            sb = s[i_go:i_go + len(Vs)]
            end = int(over[0]) - i_go if len(over) else len(sb)
            on = (sb[:max(end, 0)] >= 0.0)
            if on.any():
                extra = {"band_sec": float(on.sum() * dt),
                         "band_w_abs": float(np.abs(Ws[:len(on)][on]).mean()),
                         "band_v_cmd": float(Vs[:len(on)][on].mean()),
                         "band_stop_sec": float((np.abs(Vs[:len(on)][on]) < 0.05).sum() * dt)}
        out.append({**extra, "site": int(sites[q]), "grade": float(g_mean[sites[q]]),
                    "reached": bool(len(over)), "speed": float(speed),
                    "s_max": float(s[i_go:].max()), "s_end": float(s[-1]),
                    "s_at_go": float(s[i_go]), "climb_m": float(Z[i_go:, q].max() - Z[i_go, q]),
                    "tilt_max": float(TL[i_go:, q].max()),
                    "pitch_max": float(np.array(log_pitch)[i_go:, q].max()),
                    "flipped": bool(flipped_any[q]), "guard_would_fire": guard,
                    "bad_start": bool(TL[i_go, q] > g_mean[sites[q]] + 15.0)})
    return out


def summarize(tag, b, v, rows):
    bad = sum(x["bad_start"] for x in rows)
    rows = [x for x in rows if not x["bad_start"]]
    if not rows:
        return None
    r = {"mode": tag, "bin": b, "v_cmd": v, "n": len(rows),
         "grade": round(float(np.mean([x["grade"] for x in rows])), 1),
         "reached_pct": round(100.0 * float(np.mean([x["reached"] for x in rows])), 0),
         "speed_med": round(float(np.median([x["speed"] for x in rows])), 2),
         "s_max_med": round(float(np.median([x["s_max"] for x in rows])), 2),
         "climb_med": round(float(np.median([x["climb_m"] for x in rows])), 2),
         "pitch_max_med": round(float(np.median([x["pitch_max"] for x in rows])), 0),
         "flips": int(np.sum([x["flipped"] for x in rows])), "bad_start": int(bad),
         # 넘은 시도 중 AUTO 였다면 기울기 반사가 끊었을 것의 수 / 넘은 수
         "guard_in_reached": f'{sum(x["reached"] and x["guard_would_fire"] for x in rows)}'
                             f'/{sum(x["reached"] for x in rows)}'}
    for key in ("band_w_abs", "band_v_cmd", "band_stop_sec"):
        vals = [x[key] for x in rows if key in x]
        if vals:
            r[key + "_med"] = round(float(np.median(vals)), 2)
    print("[probe] " + json.dumps(r, ensure_ascii=False), flush=True)
    return r


results, raw = [], []
bins = [float(b) for b in args.bins.split(",")]
for b in bins:
    hw = 1.5 if b < 21 else 1.0
    mask = ok_on & (np.abs(g_mean - b) <= hw) & (g_lo > g_mean - 4.0)
    sites = pick(mask, n)
    print(f"[probe] 경사 {b:.0f}° 구간: 후보 {int(mask.sum())}곳 중 {len(sites)}곳 사용", flush=True)
    if not sites:
        continue
    if args.compare:
        ru = [(c, runup_start(c)) for c in sites]
        ru = [(c, s) for c, s in ru if s is not None]
        print(f"[probe] 경사 {b:.0f}° 구간: 비교 시험 — 정지 출발 {len(sites)}곳, "
              f"달려서 진입 {len(ru)}곳", flush=True)
        for name in [x.strip() for x in args.compare.split(",") if x.strip()]:
            ctrl = parse_ctrl(name)
            sec = 14.0 + (6.0 if ctrl[0] in ("nav", "kink") else 0.0) \
                + (GOAL + 2.5) / max(ctrl[1] or 0.6, 0.3)
            rows = run_trial(sites, [0.0] * len(sites), 0.0, drive_sec=sec, ctrl=ctrl)
            raw.append({"mode": f"rest:{name}", "bin": b, "v_cmd": ctrl[1], "rows": rows})
            results.append(summarize(f"rest:{name}", b, ctrl[1], rows))
            if ru:
                rows = run_trial([c for c, _ in ru], [s for _, s in ru], 0.0, drive_sec=sec,
                                 settle_sec=1.0, ctrl=ctrl)
                raw.append({"mode": f"runup:{name}", "bin": b, "v_cmd": ctrl[1], "rows": rows})
                results.append(summarize(f"runup:{name}", b, ctrl[1], rows))
        continue
    caps = [float(x) for x in args.slip_caps.split(",")]
    for v in [float(x) for x in args.rest_speeds.split(",")]:
        for cap in caps:
            tag = "rest" if cap == 0 else f"rest+cap{cap:g}"
            rows = run_trial(sites, [0.0] * len(sites), v, slip_cap=cap,
                             drive_sec=max(8.0, GOAL / v + 4.0) + (10.0 if cap else 0.0))
            raw.append({"mode": tag, "bin": b, "v_cmd": v, "rows": rows})
            results.append(summarize(tag, b, v, rows))
    ru = [(c, runup_start(c)) for c in sites]
    ru = [(c, s) for c, s in ru if s is not None]
    print(f"[probe] 경사 {b:.0f}° 구간: 아래에 완만한 진입로가 있는 곳 {len(ru)}곳", flush=True)
    if not ru:
        continue
    for v in [float(x) for x in args.runup_speeds.split(",")]:
        for cap in caps:
            tag = "runup" if cap == 0 else f"runup+cap{cap:g}"
            rows = run_trial([c for c, _ in ru], [s for _, s in ru], v, slip_cap=cap,
                             drive_sec=max(8.0, (GOAL + 2.5) / v + 4.0) + (6.0 if cap else 0.0),
                             settle_sec=1.0)
            raw.append({"mode": tag, "bin": b, "v_cmd": v, "rows": rows})
            results.append(summarize(tag, b, v, rows))

if args.out:
    with open(args.out, "w") as f:
        json.dump({"terrain": args.terrain, "friction": args.friction, "length_m": L,
                   "wheel_grip": args.wheel_grip, "compare": args.compare,
                   "goal_m": GOAL,
                   "summary": [r for r in results if r], "raw": raw}, f, ensure_ascii=False)
    print(f"[probe] 저장: {args.out}")
app.close()
