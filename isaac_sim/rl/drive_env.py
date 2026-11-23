"""구동 조절기(drive governor) 학습 환경 — 로버 여러 대를 한 장면에서 물리만 돌린다.

SimulationApp 기동 후에만 import 할 것.

무엇을 배우나
--------------
상위 노드(coverage / mission manager)는 지형을 모른 채 "v 로 가라, w 로 돌아라"
를 계단 입력으로 보낸다 (최대 3 m/s · 1.5 rad/s). 화성 중력에서는 이 명령을
그대로 바퀴에 넘기면 넘어지고, 지금은 고정된 속도·가속 제한(drive_limits.py)
으로 막고 있다 — 평지든 30° 비탈이든 같은 값이다.

여기서 배우는 정책은 그 자리에 들어간다:

    상위 명령 (v_des, w_des) ──▶ [정책] ──▶ 실제 바퀴 명령 (v_out, w_out)
                                   ▲
            기울기 · 각속도 · 차체 속도 · 바퀴 속도 (IMU, 엔코더, 주행계)

정책은 명령을 얼마나 깎을지(0~1 배)와 조향 보정(±0.3 rad/s)만 정한다. 명령보다
빠르게 가거나 방향을 바꾸지는 못한다. 지형을 미리 보지 않고 몸으로 느끼는
것만 쓴다 (카메라·지도 없음).

보상
----
  + 명령한 방향으로 실제로 나아간 속도 (명령 속도까지만 인정)
  + 명령한 회전 속도를 따라감
  − 기울기 12° 초과분, roll/pitch 각속도 (전복 위험)
  − 바퀴에 시킨 속도와 실제 속도의 차이 (헛돎), 옆으로 미끄러진 속도
  − 출력의 급변
  − 전복(60° 초과) 시 큰 벌점 + 에피소드 종료

장면
----
vehicle_v3 를 N 대 올리고 OmniGraph(ROS)·카메라는 끈다. 로버끼리는 충돌하지
않게 하고(같은 지형을 공유), 광물은 치운다. 상태는 PhysX tensor API 로 직접
읽으므로 USD 갱신도 끈다 (이것만으로 4.5 배 빨라진다: 16 대에서 초당 물리
2000 스텝).
"""
from __future__ import annotations

import json
import math
import os
import sys

import carb
import numpy as np
import omni.usd
from isaacsim.core.api import World
from isaacsim.core.prims import Articulation
from isaacsim.core.utils.stage import add_reference_to_stage
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ISAAC_SIM = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ISAAC_SIM, "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(ISAAC_SIM), "isaac_drive",
                                "isaac_drive", "navigation"))
from planet_physics import apply_planet_physics  # noqa: E402
from drive_limits import limits_for_gravity      # noqa: E402
import governor_policy as gp                     # noqa: E402

STEER_NAMES = ["FL_Steer_Revolute", "FR_Steer_Revolute",
               "RL_Steer_Revolute", "RR_Steer_Revolute"]
DRIVE_NAMES = ["FL_Drive_Continuous", "FR_Drive_Continuous", "CL_Drive_Continuous",
               "CR_Drive_Continuous", "RL_Drive_Continuous", "RR_Drive_Continuous"]
# Ackermann 상수 — vehicle_v3 의 ScriptNode (drive_limits.ACK_LIMITED_TPL) 와 같다.
WHEELBASE = 0.849
TRACK = 0.894
WHEEL_RADIUS = gp.WHEEL_RADIUS
ACK_OFFSET = -0.0135
STEER_GAIN = 1.1
WHEEL_VEL_GAIN = 2.0

V_CAP, W_CAP = gp.V_CAP, gp.W_CAP     # 정책이 낼 수 있는 최고 속도 (명령이 더 커도 여기까지)
OBS_DIM, ACT_DIM = gp.OBS_DIM, gp.ACT_DIM
REST_HEIGHT = 0.50     # 받침대(base_link)가 지면 위에 놓이는 높이 (m)


def ackermann(lin: np.ndarray, ang: np.ndarray):
    """(N,) 선속도·각속도 → 조향각 (N,4: FL FR RL RR), 바퀴 각속도 (N,6: FL FR CL CR RL RR)."""
    direction = np.where(lin >= 0.0, 1.0, -1.0)
    turn = np.sign(ang)
    a_lin, a_ang = np.abs(lin), np.abs(ang)
    with np.errstate(divide="ignore", invalid="ignore"):
        radius = np.where(a_ang > 0.0, a_lin / np.maximum(a_ang, 1e-9), np.inf)
    half = WHEELBASE / 2.0
    r_ml = radius - (TRACK / 2.0) * turn
    r_mr = radius + (TRACK / 2.0) * turn
    r_fl, r_fr = np.hypot(r_ml, half), np.hypot(r_mr, half)
    spin = radius < TRACK * 0.8                      # 제자리 회전
    turning = a_ang > 0.0

    def wheel(r):
        return np.where(turning, r * a_ang, a_lin) * direction

    with np.errstate(invalid="ignore"):
        v_fl, v_fr, v_ml, v_mr = wheel(r_fl), wheel(r_fr), wheel(r_ml), wheel(r_mr)
        th_fl = np.arctan2(half - ACK_OFFSET, r_fl) * turn
        th_fr = np.arctan2(half - ACK_OFFSET, r_fr) * turn
        th_rl = np.arctan2(half + ACK_OFFSET, r_fl) * -turn
        th_rr = np.arctan2(half + ACK_OFFSET, r_fr) * -turn
    s = a_ang * turn
    vel = np.stack([np.where(spin, -s, v_fl), np.where(spin, s, v_fr),
                    np.where(spin, -s, v_ml), np.where(spin, s, v_mr),
                    np.where(spin, -s, v_fl), np.where(spin, s, v_fr)], axis=1)
    q = math.pi / 4.0
    steer = np.stack([np.where(spin, -q, th_fl), np.where(spin, q, th_fr),
                      np.where(spin, q, th_rl), np.where(spin, -q, th_rr)], axis=1)
    k = WHEEL_VEL_GAIN / (WHEEL_RADIUS * 2.0)
    return np.nan_to_num(STEER_GAIN * steer), np.nan_to_num(k * vel)


def quat_to_R(q: np.ndarray) -> np.ndarray:
    """(N,4) wxyz → (N,3,3)."""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = 1 - 2 * (y * y + z * z); R[:, 0, 1] = 2 * (x * y - z * w); R[:, 0, 2] = 2 * (x * z + y * w)
    R[:, 1, 0] = 2 * (x * y + z * w); R[:, 1, 1] = 1 - 2 * (x * x + z * z); R[:, 1, 2] = 2 * (y * z - x * w)
    R[:, 2, 0] = 2 * (x * z - y * w); R[:, 2, 1] = 2 * (y * z + x * w); R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


class SlewLimiter:
    """drive_limits.py 의 고정 제한과 같은 것 (비교 기준용)."""

    def __init__(self, n: int, limits: dict, dt: float):
        self.lim, self.dt = limits, dt
        self.lin = np.zeros(n)
        self.ang = np.zeros(n)

    def reset(self, idx):
        self.lin[idx] = 0.0
        self.ang[idx] = 0.0

    def __call__(self, v_des, w_des):
        L = self.lim
        v = np.clip(v_des, -L["max_lin"], L["max_lin"])
        w = np.clip(w_des, -L["max_ang"], L["max_ang"])
        self.lin += np.clip(v - self.lin, -L["acc_lin"] * self.dt, L["acc_lin"] * self.dt)
        self.ang += np.clip(w - self.ang, -L["acc_ang"] * self.dt, L["acc_ang"] * self.dt)
        return self.lin.copy(), self.ang.copy()


class RoverDriveEnv:
    def __init__(self, app, terrain: str = "terrain_00026", num_envs: int = 16,
                 control_hz: float = 15.0, episode_sec: float = 20.0, seed: int = 0,
                 friction_range=(0.6, 1.1), planet: str = "mars",
                 goal_frac: float = 0.5, steep_goal_frac: float = 0.0):
        self.app = app
        self.n = int(num_envs)
        self.rng = np.random.default_rng(seed)
        self.physics_dt = 1.0 / 60.0
        self.decimation = max(1, int(round(60.0 / control_hz)))
        self.dt = self.physics_dt * self.decimation
        self.max_steps = int(round(episode_sec / self.dt))
        self.friction_range = friction_range

        st = carb.settings.get_settings()
        for key in ("/physics/updateToUsd", "/physics/updateVelocitiesToUsd",
                    "/physics/updateParticlesToUsd", "/physics/updateForceSensorsToUsd"):
            st.set_bool(key, False)
        st.set_bool("/physics/disableContactProcessing", True)

        self.world = World(stage_units_in_meters=1.0, physics_dt=self.physics_dt,
                           rendering_dt=self.physics_dt)
        add_reference_to_stage(usd_path=f"{ISAAC_SIM}/worlds/{terrain}.usd",
                               prim_path="/World/MarsScene")
        tdir = f"{ISAAC_SIM}/assets/generated_terrains/{terrain}"
        with open(f"{tdir}/meta.json") as f:
            self.meta = json.load(f)
        self._load_maps(tdir)
        stage = omni.usd.get_context().get_stage()
        self.stage = stage

        roots = []
        start = self._sample_starts(self.n)
        for k in range(self.n):
            path = f"/World/Rover_{k}"
            add_reference_to_stage(usd_path=f"{ISAAC_SIM}/assets/vehicle/vehicle_v3.usd",
                                   prim_path=path)
            xf = UsdGeom.Xformable(stage.GetPrimAtPath(path))
            ops = [o for o in xf.GetOrderedXformOps()
                   if o.GetOpType() == UsdGeom.XformOp.TypeTranslate]
            (ops[0] if ops else xf.AddTranslateOp()).Set(
                Gf.Vec3d(float(start[k, 0]), float(start[k, 1]), float(start[k, 2])))
            roots.append(path)
        for _ in range(10):
            app.update()
        for r in roots:                       # ROS 그래프·카메라 끄기
            for prim in Usd.PrimRange(stage.GetPrimAtPath(r)):
                if prim.GetTypeName() in ("OmniGraph", "Camera"):
                    prim.SetActive(False)
        for prim in stage.Traverse():         # 광물 치우기
            if prim.GetName() == "Minerals":
                prim.SetActive(False)
        self.physics_info = apply_planet_physics(
            stage, self.world, planet, meta=self.meta,
            scene_root="/World/MarsScene", rover_roots=roots)
        self.gravity = float(self.physics_info["gravity"] or 9.81)
        self._base_friction = {
            name: (v["static_friction"], v["dynamic_friction"])
            for name, v in (self.physics_info.get("materials") or {}).items()
            if name != "wheel"}
        self.friction_scale = 1.0
        grp = UsdPhysics.CollisionGroup.Define(stage, "/World/RoverGroup")
        coll = Usd.CollectionAPI.Apply(grp.GetPrim(), "colliders")
        for r in roots:
            coll.CreateIncludesRel().AddTarget(Sdf.Path(r))
        grp.CreateFilteredGroupsRel().AddTarget(Sdf.Path("/World/RoverGroup"))

        art_root = None
        for prim in Usd.PrimRange(stage.GetPrimAtPath(roots[0])):
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                art_root = str(prim.GetPath())
                break
        rel = art_root[len(roots[0]):]
        self.art = Articulation(prim_paths_expr=f"/World/Rover_.*{rel}", name="rovers")
        self.world.scene.add(self.art)
        self.world.reset()
        self.steer_idx = [self.art.get_dof_index(n) for n in STEER_NAMES]
        self.drive_idx = [self.art.get_dof_index(n) for n in DRIVE_NAMES]
        self.default_q = self.art.get_joint_positions().copy()
        self.default_quat = np.array([1.0, 0.0, 0.0, 0.0])   # 받침대: x 앞, z 위

        n = self.n
        self.v_des = np.zeros(n)
        self.w_des = np.zeros(n)
        self.cmd_left = np.zeros(n, dtype=np.int64)
        self.v_out = np.zeros(n)
        self.w_out = np.zeros(n)
        self.prev_action = np.zeros((n, ACT_DIM))
        self.steps = np.zeros(n, dtype=np.int64)
        self.total_steps = 0
        self.mode = np.zeros(n, dtype=np.int64)      # 0: 무작위 명령, 1: 목표점 추종
        self.goal = np.zeros((n, 2))
        self.goal_frac = goal_frac
        self.steep_goal_frac = steep_goal_frac   # 목표점을 일부러 비탈(15° 이상)에 두는 비율
        self.goals_reached = np.zeros(n, dtype=np.int64)
        self.script = None              # 평가용 고정 시나리오 (set_script)
        self._settle = np.zeros(n, dtype=np.int64)
        self._state = None
        self.reset_all()
        print(f"[drive_env] {terrain}: 로버 {n}대, 제어 {1 / self.dt:.0f} Hz "
              f"(물리 {self.decimation} 스텝), 에피소드 {self.max_steps} 스텝, "
              f"중력 {self.gravity:.2f}, 출발 가능 칸 {len(self._start_cells)}")

    # ── 지형 정보 (리셋 위치와 종료 판정에만 쓴다. 정책 입력 아님) ─────────
    def _load_maps(self, tdir: str) -> None:
        self.res = float(self.meta["resolution_m"])
        self.map_w, self.map_h = [float(v) for v in self.meta["size_m"]]
        self.hm = np.load(f"{tdir}/heightmap.npy").astype(np.float64)
        og = np.load(f"{tdir}/obstacle_grid.npy") > 0
        gy, gx = np.gradient(self.hm, self.res)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        self.slope = slope
        self.slope_near = ndimage.maximum_filter(slope, 21)     # 주변 1 m 안의 최대 경사
        # 정답 지도의 장애물 중 경사가 완만한 곳 = 바위·구조물·기지 (급경사는 남긴다).
        solid = og & (ndimage.uniform_filter(slope, 9) < 18.0)
        solid = ndimage.binary_opening(solid, iterations=2)
        self.solid_dist = ndimage.distance_transform_edt(~solid) * self.res
        ii, jj = np.mgrid[0:og.shape[0], 0:og.shape[1]]
        xs = (jj + 0.5) * self.res - self.map_w / 2
        ys = (ii + 0.5) * self.res - self.map_h / 2
        ok = ((self.solid_dist > 2.5) & (self.slope_near < 15.0)
              & (np.abs(xs) < self.map_w / 2 - 4.0) & (np.abs(ys) < self.map_h / 2 - 4.0))
        self._start_cells = np.argwhere(ok)

    def _cell(self, x, y):
        j = np.clip(((np.asarray(x) + self.map_w / 2) / self.res).astype(int), 0, self.hm.shape[1] - 1)
        i = np.clip(((np.asarray(y) + self.map_h / 2) / self.res).astype(int), 0, self.hm.shape[0] - 1)
        return i, j

    def _sample_starts(self, k: int) -> np.ndarray:
        """(k, 4): x, y, z, yaw."""
        c = self._start_cells[self.rng.integers(0, len(self._start_cells), k)]
        x = (c[:, 1] + 0.5) * self.res - self.map_w / 2
        y = (c[:, 0] + 0.5) * self.res - self.map_h / 2
        z = self.hm[c[:, 0], c[:, 1]] + REST_HEIGHT + 0.12
        return np.stack([x, y, z, self.rng.uniform(-math.pi, math.pi, k)], axis=1)

    # ── 명령 생성 ───────────────────────────────────────────────────────
    # 두 가지를 섞는다.
    #   무작위: 몇 초마다 바뀌는 (v, w) 계단 입력 — 넓은 범위를 고루 겪게.
    #   목표점: 실제 스택의 Navigator 와 같은 식 (방향 오차가 크면 제자리 회전,
    #           아니면 최대 속도 × cos) 으로 지형을 모른 채 목표점을 향한다.
    NAV_MAX_LIN, NAV_MAX_ANG, NAV_KP, NAV_POINT_TURN = 3.0, 1.5, 2.0, math.radians(25.0)
    GOAL_TOL = 0.6

    def _sample_commands(self, idx) -> None:
        k = len(idx)
        if k == 0:
            return
        mode = self.rng.random(k)
        v = np.zeros(k)
        w = np.zeros(k)
        fwd = mode < 0.55                       # 직진 + 약한 조향
        arc = (mode >= 0.55) & (mode < 0.72)    # 호를 그리며 전진
        spin = (mode >= 0.72) & (mode < 0.90)   # 제자리 회전
        back = (mode >= 0.90) & (mode < 0.94)   # 후진 (지금 상위 노드는 안 쓰지만 대비)
        v[fwd] = self.rng.uniform(0.3, 3.0, fwd.sum())
        w[fwd] = self.rng.uniform(-0.5, 0.5, fwd.sum())
        v[arc] = self.rng.uniform(0.3, 1.5, arc.sum())
        w[arc] = self.rng.uniform(0.4, 1.5, arc.sum()) * self.rng.choice([-1, 1], arc.sum())
        w[spin] = self.rng.uniform(0.4, 1.5, spin.sum()) * self.rng.choice([-1, 1], spin.sum())
        v[back] = -self.rng.uniform(0.2, 1.0, back.sum())
        self.v_des[idx], self.w_des[idx] = v, w
        self.cmd_left[idx] = self.rng.integers(int(2.0 / self.dt), int(6.0 / self.dt), k)

    def sample_goal(self, rng, x, y, steep_frac=None) -> np.ndarray:
        """(x, y) 에서 6~15 m 떨어진, 바위·구조물 위가 아닌 점. 경사는 가리지 않는다
        (steep_frac 의 확률로는 일부러 15° 이상 비탈 위의 점을 고른다)."""
        lim_x, lim_y = self.map_w / 2 - 3.0, self.map_h / 2 - 3.0
        frac = self.steep_goal_frac if steep_frac is None else steep_frac
        want_steep = rng.random() < frac
        for _ in range(80):
            d, th = rng.uniform(6.0, 15.0), rng.uniform(-math.pi, math.pi)
            gx, gy = x + d * math.cos(th), y + d * math.sin(th)
            if abs(gx) > lim_x or abs(gy) > lim_y:
                continue
            i, j = self._cell(gx, gy)
            if want_steep and self.slope_near[i, j] < 15.0:
                continue
            if self.solid_dist[i, j] > 1.5:
                return np.array([gx, gy])
        return np.array([np.clip(x, -lim_x, lim_x), np.clip(y, -lim_y, lim_y)])

    def _next_goal(self, i: int) -> None:
        if self.script is not None:
            seq = self.script["goals"][i]
            self.goal[i] = seq[self._goal_k[i] % len(seq)]
            self._goal_k[i] += 1
        else:
            p = self._state["pos"][i]
            self.goal[i] = self.sample_goal(self.rng, p[0], p[1])

    def _update_commands(self, only=None) -> None:
        """다음 스텝에 쓸 명령을 정한다 (관측에 들어가는 것과 같은 값)."""
        sel = np.ones(self.n, dtype=bool)
        if only is not None:
            sel[:] = False
            sel[only] = True
        rnd = sel & (self.mode == 0)
        self.cmd_left[rnd] -= 1
        self._sample_commands(np.nonzero(rnd & (self.cmd_left <= 0))[0])
        gi = np.nonzero(sel & (self.mode == 1))[0]
        if len(gi) == 0:
            return
        s = self._state
        d = self.goal[gi] - s["pos"][gi, :2]
        for k in np.nonzero(np.hypot(d[:, 0], d[:, 1]) < self.GOAL_TOL)[0]:
            self.goals_reached[gi[k]] += 1
            self._next_goal(int(gi[k]))
        d = self.goal[gi] - s["pos"][gi, :2]
        yaw = np.arctan2(s["R"][gi, 1, 0], s["R"][gi, 0, 0])
        err = np.arctan2(d[:, 1], d[:, 0]) - yaw
        err = (err + math.pi) % (2.0 * math.pi) - math.pi
        self.w_des[gi] = np.clip(self.NAV_KP * err, -self.NAV_MAX_ANG, self.NAV_MAX_ANG)
        self.v_des[gi] = np.where(np.abs(err) > self.NAV_POINT_TURN, 0.0,
                                  self.NAV_MAX_LIN * np.cos(err))

    def set_script(self, starts: np.ndarray, goals: np.ndarray) -> None:
        """평가용: 로버마다 출발점 목록 (n, S, 4) 과 목표점 목록 (n, K, 2) 을 고정한다.
        넘어지거나 경계를 벗어나면 다음 출발점에서 이어 간다."""
        self.script = {"starts": np.asarray(starts), "goals": np.asarray(goals)}
        self._start_k = np.zeros(self.n, dtype=np.int64)
        self._goal_k = np.zeros(self.n, dtype=np.int64)
        self.goals_reached[:] = 0

    # ── 리셋 ────────────────────────────────────────────────────────────
    def reset_all(self):
        return self.reset_idx(np.arange(self.n))

    def reset_idx(self, idx, starts=None):
        idx = np.asarray(idx, dtype=np.int64)
        if len(idx) == 0:
            return self._obs()
        if starts is not None:
            s = np.asarray(starts)
        elif self.script is not None:
            st = self.script["starts"]
            s = st[idx, self._start_k[idx] % st.shape[1]]
            self._start_k[idx] += 1
        else:
            s = self._sample_starts(len(idx))
        half = 0.5 * s[:, 3]
        qz = np.stack([np.cos(half), np.zeros(len(idx)), np.zeros(len(idx)), np.sin(half)], axis=1)
        d = self.default_quat                       # world←base (spawn 자세)
        quat = np.stack([                           # qz ⊗ d
            qz[:, 0] * d[0] - qz[:, 3] * d[3],
            qz[:, 0] * d[1] - qz[:, 3] * d[2],
            qz[:, 0] * d[2] + qz[:, 3] * d[1],
            qz[:, 0] * d[3] + qz[:, 3] * d[0]], axis=1)
        self.art.set_world_poses(positions=s[:, :3], orientations=quat, indices=idx)
        self.art.set_velocities(np.zeros((len(idx), 6)), indices=idx)
        self.art.set_joint_positions(self.default_q[idx], indices=idx)
        self.art.set_joint_velocities(np.zeros_like(self.default_q[idx]), indices=idx)
        self.art.set_joint_velocity_targets(np.zeros((len(idx), 6)), indices=idx,
                                            joint_indices=self.drive_idx)
        self.art.set_joint_position_targets(np.zeros((len(idx), 4)), indices=idx,
                                            joint_indices=self.steer_idx)
        self.v_out[idx] = 0.0
        self.w_out[idx] = 0.0
        self.prev_action[idx] = 0.0
        self.steps[idx] = 0
        # 처음 0.6 s 는 정지 명령 (내려놓은 직후 자리 잡는 시간).
        self._settle[idx] = int(0.6 / self.dt)
        self._read_state()
        # 방금 써 넣은 자세가 다음 물리 스텝 전에도 읽히도록 직접 채운다.
        self._state["pos"][idx] = s[:, :3]
        self._state["R"][idx] = quat_to_R(quat)
        if self.script is not None:
            self.mode[idx] = 1
        else:
            self.mode[idx] = (self.rng.random(len(idx)) < self.goal_frac).astype(np.int64)
        for i in idx[self.mode[idx] == 1]:
            self._next_goal(int(i))
        self.cmd_left[idx] = 0
        self._update_commands(idx)
        return self._obs()

    # ── 상태 읽기 ───────────────────────────────────────────────────────
    def _read_state(self) -> None:
        pos, quat = self.art.get_world_poses()
        vel = self.art.get_velocities()
        R = quat_to_R(np.asarray(quat, dtype=np.float64))
        Rt = np.transpose(R, (0, 2, 1))
        v_b = np.einsum("nij,nj->ni", Rt, vel[:, :3])
        w_b = np.einsum("nij,nj->ni", Rt, vel[:, 3:])
        g_b = -Rt[:, :, 2]                        # body frame 의 중력 방향 (단위)
        wheel = self.art.get_joint_velocities(joint_indices=self.drive_idx)
        tilt = np.degrees(np.arccos(np.clip(-g_b[:, 2], -1.0, 1.0)))
        self._state = {"pos": np.asarray(pos, dtype=np.float64), "v_b": v_b, "w_b": w_b,
                       "g_b": g_b, "wheel": np.asarray(wheel, dtype=np.float64),
                       "tilt": tilt, "R": R}

    def _obs(self) -> np.ndarray:
        s = self._state
        settle = self._settle > 0
        v_des = np.where(settle, 0.0, self.v_des)
        w_des = np.where(settle, 0.0, self.w_des)
        return gp.build_obs(v_des, w_des, s["g_b"], s["w_b"], s["v_b"], s["wheel"],
                            self.v_out, self.w_out, self.prev_action[:, 2])

    # ── 정책 출력 → 바퀴 명령 ───────────────────────────────────────────
    def action_to_command(self, action: np.ndarray):
        """action ∈ [-1,1]^3 → (v_out, w_out). 명령을 깎기만 한다."""
        settle = self._settle > 0
        return gp.action_to_command(action, np.where(settle, 0.0, self.v_des),
                                    np.where(settle, 0.0, self.w_des))

    def apply_command(self, v: np.ndarray, w: np.ndarray) -> None:
        steer, wheel = ackermann(v, w)
        self.art.set_joint_position_targets(steer, joint_indices=self.steer_idx)
        self.art.set_joint_velocity_targets(wheel, joint_indices=self.drive_idx)
        self.v_out, self.w_out = v.copy(), w.copy()

    # ── 한 스텝 ─────────────────────────────────────────────────────────
    def step(self, action=None, command=None):
        """action (정책) 또는 command=(v, w) (비교 기준 제어기) 중 하나를 준다.

        return: obs, reward, done(전복), truncated(시간·경계), info
        """
        if command is None:
            v, w = self.action_to_command(action)
            act = np.clip(action, -1.0, 1.0)
        else:
            v, w = command
            act = self.prev_action
        self.apply_command(np.asarray(v, float), np.asarray(w, float))
        for _ in range(self.decimation):
            self.world.step(render=False)
        self._read_state()
        s = self._state
        settle = self._settle > 0
        v_des = np.where(settle, 0.0, self.v_des)
        w_des = np.where(settle, 0.0, self.w_des)
        vx, vy, wz = s["v_b"][:, 0], s["v_b"][:, 1], s["w_b"][:, 2]

        # 보상
        v_goal = np.abs(np.clip(v_des, -V_CAP, V_CAP))
        progress = np.clip(vx * np.where(v_des >= 0.0, 1.0, -1.0), 0.0, None)
        r_track = np.minimum(progress, v_goal) - 0.5 * np.clip(progress - v_goal, 0.0, None)
        r_turn = 0.4 * np.exp(-((wz - w_des) / 0.35) ** 2) * (np.abs(w_des) > 0.05)
        r_still = (0.3 * np.exp(-(np.abs(vx) + np.abs(wz)) / 0.1)
                   * ((np.abs(v_des) < 0.05) & (np.abs(w_des) < 0.05)))
        p_tilt = 2.0 * (np.clip(s["tilt"] - 12.0, 0.0, None) / 20.0) ** 2
        p_rate = 0.05 * (s["w_b"][:, 0] ** 2 + s["w_b"][:, 1] ** 2)
        p_slip = 0.3 * np.abs(self.v_out - vx) + 0.5 * np.abs(vy)
        p_jerk = 0.05 * np.sum((act - self.prev_action) ** 2, axis=1)
        reward = r_track + r_turn + r_still - p_tilt - p_rate - p_slip - p_jerk

        flipped = s["tilt"] > 60.0
        reward = np.where(flipped, reward - 30.0, reward)
        self.steps += 1
        self.total_steps += self.n
        self._settle = np.maximum(self._settle - 1, 0)
        i, j = self._cell(s["pos"][:, 0], s["pos"][:, 1])
        out = ((np.abs(s["pos"][:, 0]) > self.map_w / 2 - 1.5)
               | (np.abs(s["pos"][:, 1]) > self.map_h / 2 - 1.5)
               | (self.solid_dist[i, j] < 0.9))
        truncated = (~flipped) & (out | (self.steps >= self.max_steps))
        info = {"tilt": s["tilt"].copy(), "vx": vx.copy(), "vy": vy.copy(), "wz": wz.copy(),
                "v_des": v_des.copy(), "w_des": w_des.copy(), "v_out": self.v_out.copy(),
                "flipped": flipped.copy(), "slope": self.slope[i, j].copy(),
                "pos": s["pos"].copy(), "out": out.copy()}
        self.prev_action = act.copy()

        self._update_commands()
        obs = self._obs()
        return obs, reward.astype(np.float32), flipped, truncated, info

    # ── 마찰 무작위화 (지면 전체, 주기적으로) ───────────────────────────
    def randomize_friction(self) -> float:
        lo, hi = self.friction_range
        self.friction_scale = float(self.rng.uniform(lo, hi))
        for name, (mu_s, mu_d) in self._base_friction.items():
            usd_name = {"regolith": "Regolith", "rock": "Rock"}.get(name, f"Zone_{name}")
            prim = self.stage.GetPrimAtPath(f"/World/PhysicsMaterials/{usd_name}")
            if not prim or not prim.IsValid():
                continue
            api = UsdPhysics.MaterialAPI(prim)
            api.GetStaticFrictionAttr().Set(float(mu_s * self.friction_scale))
            api.GetDynamicFrictionAttr().Set(float(mu_d * self.friction_scale))
        return self.friction_scale

    def baseline_limiter(self) -> SlewLimiter:
        return SlewLimiter(self.n, limits_for_gravity(self.gravity), self.dt)
