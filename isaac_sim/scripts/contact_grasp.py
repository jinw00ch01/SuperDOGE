"""접촉 조건부 부착 — 그리퍼 손끝이 광물에 닿았을 때만 FixedJoint 를 건다.

SimulationApp 기동 후에만 import 할 것.

예전 방식(magic grasp)과의 차이
------------------------------
예전 GraspScript 는 pickup 명령이 오면 팔 받침대에서 2.5 m 안의 가장 가까운
광물을 손가락 원점으로 순간이동시켜 붙였다. 팔이 광물에 닿았는지, 인식 좌표가
맞는지는 결과에 아무 영향이 없었다.

여기서는
  · 손끝(inner finger 끝) 주변의 작은 구가 광물 collider 와 겹칠 때만 붙인다.
    구의 여유(손끝 너머 약 1.6 cm)는 PhysX 가 접촉을 만들기 시작하는 거리
    (contact offset, 약 2 cm) 수준이다.
  · 붙이는 순간의 상대 자세를 그대로 유지한다 (순간이동 없음).
그래서 팔이 실제로 광물까지 가야 하고, 인식 오차가 크면 pick 은 실패한다.

아직 남은 비현실
----------------
손가락이 광물을 조여서 마찰로 드는 것이 아니다. RG2-FT 의 최대 벌림은 약
10 cm, 광물은 약 17 × 19 × 23 cm 라 물리적으로 쥘 수 없다. 닿은 뒤의 고정은
여전히 FixedJoint 이고, 들고 있는 동안 광물 collider 는 꺼 둔다 (예전과 같음).
조여서 드는 방식은 wide_gripper.py (--grasp friction, 기본값) 에 있다.

명령 규약 (arm_executor 와 공유)
-------------------------------
  /<ns>/grasp/command (Twist): angular.x > 0.5 = pickup, < -0.5 = release.
  pickup 은 "지금부터 arm_window 초 동안 접촉하면 붙여라" 는 뜻이다. 팔은
  내려가는 동안 pickup 을 계속 보내고, /<ns>/grasp/status 에 광물 이름이
  뜨면 멈춘다.
"""
from __future__ import annotations

import carb
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics

# link_6 프레임에서 본 손끝 탐지 구의 중심 (vehicle_v3.usd 실측: inner finger
# collider 는 x ±0.050~0.073, z 0.134~0.194 를 차지한다).
TIP_IN_LINK6 = {
    "left_inner_finger": (0.0615, 0.0, 0.185),
    "right_inner_finger": (-0.0615, 0.0, 0.185),
}
TIP_RADIUS = 0.025
ATTACH_LINK = "right_inner_finger"      # joint body0 — 기존 규약과 동일


def _mat(gf_matrix) -> np.ndarray:
    """Gf.Matrix4d (row-vector 규약) → column-vector 4x4."""
    return np.array(gf_matrix, dtype=np.float64).T


def _quat_to_R(x, y, z, w) -> np.ndarray:
    n = (x * x + y * y + z * z + w * w) ** 0.5
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _R_to_quat(R: np.ndarray) -> Gf.Quatf:
    m = Gf.Matrix3d(*[float(v) for v in R.T.reshape(-1)])   # Gf 는 row-vector
    q = m.ExtractRotation().GetQuat()
    im = q.GetImaginary()
    return Gf.Quatf(float(q.GetReal()), float(im[0]), float(im[1]), float(im[2]))


class ContactGrasp:
    def __init__(self, stage, rovers: dict, arm_window: float = 0.5):
        """rovers: {ns: (rover root prim path, grasp FixedJoint path)}."""
        import omni.physx
        self._stage = stage
        self._physx = omni.physx.get_physx_interface()
        self._query = omni.physx.get_physx_scene_query_interface()
        self._window = float(arm_window)
        self._armed: dict = {}          # ns → 만료 sim time
        self._last_miss_log: dict = {}
        self._rovers: dict = {}
        cache = UsdGeom.XformCache()
        for ns, (root, joint_path) in rovers.items():
            rp = stage.GetPrimAtPath(root)
            if not rp or not rp.IsValid():
                continue
            links = {}
            for prim in Usd.PrimRange(rp):
                name = prim.GetName()
                path = str(prim.GetPath())
                if name == "link_6" and "m0609" in path:
                    links["link_6"] = prim
                elif name in TIP_IN_LINK6 and "onrobot" in path:
                    links[name] = prim
            if len(links) != 1 + len(TIP_IN_LINK6):
                print(f"[grasp] ⚠ {root}: 그리퍼 링크를 못 찾음 {sorted(links)} "
                      f"— 이 로버는 접촉 부착 불가")
                continue
            # 손끝 점을 각 손가락 링크의 local 좌표로 (정지 자세 기준, 한 번만).
            T6 = _mat(cache.GetLocalToWorldTransform(links["link_6"]))
            tips = {}
            for name, p6 in TIP_IN_LINK6.items():
                Tf = _mat(cache.GetLocalToWorldTransform(links[name]))
                world = T6 @ np.array([*p6, 1.0])
                tips[str(links[name].GetPath())] = (np.linalg.inv(Tf) @ world)[:3]
            attach = str(links[ATTACH_LINK].GetPath())
            self._rovers[ns] = {"root": root, "joint": joint_path,
                                "tips": tips, "attach": attach}
        print(f"[grasp] 접촉 조건부 부착 준비: {sorted(self._rovers)} "
              f"(손끝 구 반지름 {TIP_RADIUS * 100:.1f} cm, 창 {self._window:.1f} s)")

    # ── 명령 ────────────────────────────────────────────────────────────
    def arm(self, ns: str, now: float) -> None:
        if ns in self._rovers:
            self._armed[ns] = now + self._window

    def disarm(self, ns: str) -> None:
        self._armed.pop(ns, None)

    def holding(self, ns: str) -> bool:
        r = self._rovers.get(ns)
        return bool(r) and self._stage.GetPrimAtPath(r["joint"]).IsValid()

    # ── 매 physics step ─────────────────────────────────────────────────
    def step(self, now: float) -> None:
        for ns in list(self._armed):
            if now > self._armed[ns]:
                del self._armed[ns]
                continue
            if self.holding(ns):
                continue
            self._try_attach(ns, now)

    def _body_pose(self, path: str):
        t = self._physx.get_rigidbody_transformation(path)
        if not t or not t.get("ret_val"):
            return None
        p, q = t["position"], t["rotation"]
        return np.array([p[0], p[1], p[2]]), _quat_to_R(q[0], q[1], q[2], q[3])

    def _mineral_of(self, body_path: str):
        """overlap 이 돌려준 rigid body path → Minerals 바로 아래 prim path."""
        prim = self._stage.GetPrimAtPath(body_path)
        while prim and prim.IsValid() and str(prim.GetPath()) != "/":
            parent = prim.GetParent()
            if parent and parent.GetName() == "Minerals":
                return str(prim.GetPath())
            prim = parent
        return None

    def _held_minerals(self) -> set:
        held = set()
        for r in self._rovers.values():
            jp = self._stage.GetPrimAtPath(r["joint"])
            if jp and jp.IsValid():
                for t in UsdPhysics.FixedJoint(jp).GetBody1Rel().GetTargets():
                    held.add(str(t))
        return held

    def _try_attach(self, ns: str, now: float) -> None:
        r = self._rovers[ns]
        touched: dict = {}                       # mineral path → 닿은 손끝 수
        tips_world = []
        for finger_path, tip_local in r["tips"].items():
            pose = self._body_pose(finger_path)
            if pose is None:
                return
            p, R = pose
            tip = p + R @ tip_local
            tips_world.append(tip)
            hits = set()

            def _report(hit, _hits=hits):
                m = self._mineral_of(str(hit.rigid_body))
                if m:
                    _hits.add(m)
                return True

            self._query.overlap_sphere(
                TIP_RADIUS, carb.Float3(float(tip[0]), float(tip[1]), float(tip[2])),
                _report, False)
            for m in hits:
                touched[m] = touched.get(m, 0) + 1
        for m in self._held_minerals():
            touched.pop(m, None)
        if not touched:
            if now - self._last_miss_log.get(ns, -1e9) > 1.0:
                self._last_miss_log[ns] = now
                c = np.mean(tips_world, axis=0)
                print(f"[grasp] {ns} 접촉 없음 — 손끝 중심 "
                      f"({c[0]:.3f},{c[1]:.3f},{c[2]:.3f})")
            return
        mineral = max(touched, key=touched.get)
        self._attach(ns, mineral, touched[mineral], tips_world)

    def _attach(self, ns: str, mineral_path: str, n_tips: int, tips_world) -> None:
        r = self._rovers[ns]
        stage = self._stage
        link = self._body_pose(r["attach"])
        obj = self._body_pose(mineral_path)
        if link is None or obj is None:
            return
        (pl, Rl), (pm, Rm) = link, obj
        # 붙이는 순간의 상대 자세를 그대로 joint frame 으로 — 광물이 튀지 않는다.
        rel_p = Rl.T @ (pm - pl)
        rel_R = Rl.T @ Rm
        joint = UsdPhysics.FixedJoint.Define(stage, r["joint"])
        joint.CreateBody0Rel().SetTargets([Sdf.Path(r["attach"])])
        joint.CreateBody1Rel().SetTargets([Sdf.Path(mineral_path)])
        joint.CreateLocalPos0Attr().Set(Gf.Vec3f(*[float(v) for v in rel_p]))
        joint.CreateLocalRot0Attr().Set(_R_to_quat(rel_R))
        joint.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        joint.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        # 들고 있는 동안은 광물 collider 를 끈다 (차체를 지나 적재함으로 가므로).
        for child in Usd.PrimRange(stage.GetPrimAtPath(mineral_path)):
            if child.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr().Set(False)
        self._armed.pop(ns, None)
        c = np.mean(tips_world, axis=0)
        print(f"[grasp] {ns} 접촉 부착 — {mineral_path.rsplit('/', 1)[-1]} "
              f"(닿은 손끝 {n_tips}/{len(tips_world)}, 손끝 중심 "
              f"({c[0]:.3f},{c[1]:.3f},{c[2]:.3f}), 광물 원점 "
              f"({pm[0]:.3f},{pm[1]:.3f},{pm[2]:.3f}), "
              f"손끝↔원점 {np.linalg.norm(c - pm) * 100:.1f} cm)")
