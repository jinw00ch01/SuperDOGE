"""넓은 평행 집게 + 마찰 파지 (FixedJoint 없이 손가락이 조여서 든다).

SimulationApp 기동 후에만 import 할 것.

왜 바꾸나
---------
광물 mesh 는 약 0.17 × 0.19 × 0.23 m 인데 OnRobot RG2-FT 는 최대 0.10 m 까지만
벌어진다. 그래서 원본은 광물을 손가락에 "붙이는" 방식(FixedJoint)만 가능했다.
여기서는 RG2 의 본체·손가락을 비활성화하고, 0.27 m 까지 벌어지는 평행 집게를
같은 자리(link_6 끝, 손목 카메라 브래킷 아래)에 만든다.

구조 (link_6 프레임, x = 벌림 방향, z = 공구 방향)
  palm      0.32 × 0.03 × 0.03 m 상자, z 0.067~0.097
  finger ×2 0.012 × 0.06 × 0.17 m 판, palm 아래로 z 0.097~0.267
            palm 에 prismatic joint (x 축), 안쪽 면 간격 0.27 m(열림)~0.03 m(닫힘)
  구동      위치 구동 + 힘 한계 FINGER_FORCE (N) — 닫으면 물체를 이 힘으로 조인다
  패드      마찰 1.5/1.2, combine=max (고무 패드 가정)

들 수 있는 조건은 2·mu·F ≥ m·g 이다. F=40 N, mu=1.2 면 96 N 까지 — 화성 중력에서
약 25 kg 에 해당한다 (광물은 planet_physics 기준 4 kg = 14.9 N).

명령 규약 (arm_executor 와 공유)
  /<ns>/grasp/command (Twist): angular.x > 0.5 = 닫기, < -0.5 = 열기(+적재).
  /<ns>/grasp/status (String): 닫은 손가락이 광물에 막혀 멈춰 있으면 광물
  이름, 아니면 빈 문자열. 판정은 (1) 닫기 명령 상태 (2) 손가락 간격이 완전히
  닫힌 값보다 크고 더는 줄지 않음 (3) 손가락 사이 공간의 overlap 질의 — 물리
  상태를 읽을 뿐 바꾸지 않는다.

적재
  열기 명령이 왔을 때 손가락 사이에 있던 광물은 "적재함에 넣은 것" 으로 보고
  숨기고 물리를 끈다 (원본의 release + hide 와 같은 추상화). 로버 위에 실제
  적재함 collider 가 없기 때문이다.
"""
from __future__ import annotations

import carb
import numpy as np
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

PALM_SIZE = (0.32, 0.03, 0.03)       # y 폭을 좁게 — 손목 카메라(y=0.045) 시야를 가리지 않도록
PALM_Z = 0.082                      # palm 중심 (link_6 z)
FINGER_SIZE = (0.012, 0.06, 0.17)
FINGER_Z = 0.097 + FINGER_SIZE[2] / 2
GAP_OPEN = 0.27                     # 안쪽 면 간격
GAP_CLOSED = 0.03
FINGER_TRAVEL = (GAP_OPEN - GAP_CLOSED) / 2
FINGER_FORCE = 40.0                 # N, 손가락 하나가 미는 최대 힘
PALM_BOTTOM = PALM_Z + PALM_SIZE[2] / 2          # 0.097
TIP_Z = 0.097 + FINGER_SIZE[2]                   # 0.267
PAD_FRICTION = (1.5, 1.2)

# RG2-FT 에서 끄는 것 (quick_changer·angle_bracket·손목 카메라는 남긴다).
RG2_LINKS = ("gripper_body", "right_outer_knuckle", "right_inner_finger",
             "left_inner_knuckle", "left_outer_knuckle", "left_inner_finger",
             "right_inner_knuckle")
RG2_JOINTS = ("gripper_body_joint", "finger_joint", "right_inner_finger_joint",
              "left_inner_knuckle_joint", "left_outer_knuckle_joint",
              "left_inner_finger_joint", "right_inner_knuckle_joint")


def _mat(gf_matrix) -> np.ndarray:
    return np.array(gf_matrix, dtype=np.float64).T


def _gf(m: np.ndarray) -> Gf.Matrix4d:
    return Gf.Matrix4d(*[float(v) for v in m.T.reshape(-1)])


def _box(stage, path: str, world: np.ndarray, parent_world: np.ndarray,
         size, mass: float, color) -> Usd.Prim:
    """rigid body Xform + 그 아래 Cube collider. world = 원하는 world 자세(4x4)."""
    xf = UsdGeom.Xform.Define(stage, path)
    xf.AddTransformOp().Set(_gf(np.linalg.inv(parent_world) @ world))
    prim = xf.GetPrim()
    UsdPhysics.RigidBodyAPI.Apply(prim)
    UsdPhysics.MassAPI.Apply(prim).CreateMassAttr().Set(float(mass))
    rb = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
    rb.CreateSolverPositionIterationCountAttr().Set(32)
    rb.CreateSolverVelocityIterationCountAttr().Set(4)
    cube = UsdGeom.Cube.Define(stage, f"{path}/geom")
    cube.CreateSizeAttr().Set(1.0)
    cube.AddScaleOp().Set(Gf.Vec3f(*[float(v) for v in size]))
    cube.CreateDisplayColorAttr().Set([Gf.Vec3f(*color)])
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    return prim


def install_wide_gripper(stage, rover_root: str) -> dict | None:
    """rover_root 의 RG2 를 넓은 집게로 바꾼다. world.reset() 전에 호출.

    return: {"palm", "finger_l", "finger_r", "joint_l", "joint_r", "link_6"} prim path.
    """
    root = stage.GetPrimAtPath(rover_root)
    link6 = rg2 = None
    for prim in Usd.PrimRange(root):
        path = str(prim.GetPath())
        if prim.GetName() == "link_6" and "m0609" in path:
            link6 = prim
        elif prim.GetName() == "onrobot_rg2ft" and rg2 is None:
            rg2 = prim
    if link6 is None or rg2 is None:
        print(f"[gripper] ⚠ {rover_root}: link_6 / onrobot_rg2ft 를 못 찾음 — 집게 교체 생략")
        return None

    for name in RG2_LINKS:
        p = stage.GetPrimAtPath(rg2.GetPath().AppendChild(name))
        if p and p.IsValid():
            p.SetActive(False)
    for name in RG2_JOINTS:
        p = stage.GetPrimAtPath(rg2.GetPath().AppendChild("joints").AppendChild(name))
        if p and p.IsValid():
            p.SetActive(False)

    cache = UsdGeom.XformCache()
    T6 = _mat(cache.GetLocalToWorldTransform(link6))
    base = rg2.GetParent().GetPath().AppendChild("wide_gripper")
    holder = UsdGeom.Xform.Define(stage, base)
    Tp = _mat(cache.GetLocalToWorldTransform(holder.GetPrim()))

    def at(x, y, z):
        T = np.eye(4)
        T[:3, 3] = (x, y, z)
        return T6 @ T

    pad = UsdShade.Material.Define(stage, base.AppendChild("PadMaterial"))
    mapi = UsdPhysics.MaterialAPI.Apply(pad.GetPrim())
    mapi.CreateStaticFrictionAttr().Set(PAD_FRICTION[0])
    mapi.CreateDynamicFrictionAttr().Set(PAD_FRICTION[1])
    mapi.CreateRestitutionAttr().Set(0.0)
    PhysxSchema.PhysxMaterialAPI.Apply(pad.GetPrim()).CreateFrictionCombineModeAttr().Set("max")

    palm = _box(stage, str(base.AppendChild("palm")), at(0, 0, PALM_Z), Tp,
                PALM_SIZE, 0.5, (0.25, 0.25, 0.28))
    x_open = GAP_OPEN / 2 + FINGER_SIZE[0] / 2
    fingers = {}
    for name, sign in (("finger_l", 1.0), ("finger_r", -1.0)):
        f = _box(stage, str(base.AppendChild(name)), at(sign * x_open, 0, FINGER_Z), Tp,
                 FINGER_SIZE, 0.15, (0.20, 0.20, 0.22))
        UsdShade.MaterialBindingAPI.Apply(f).Bind(
            pad, UsdShade.Tokens.weakerThanDescendants, "physics")
        fingers[name] = (f, sign)

    joints = UsdGeom.Scope.Define(stage, base.AppendChild("joints")).GetPath()
    fixed = UsdPhysics.FixedJoint.Define(stage, joints.AppendChild("palm_fixed"))
    fixed.CreateBody0Rel().SetTargets([link6.GetPath()])
    fixed.CreateBody1Rel().SetTargets([palm.GetPath()])
    fixed.CreateLocalPos0Attr().Set(Gf.Vec3f(0.0, 0.0, PALM_Z))
    fixed.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    fixed.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
    fixed.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

    out = {"palm": str(palm.GetPath()), "link_6": str(link6.GetPath())}
    for name, (f, sign) in fingers.items():
        j = UsdPhysics.PrismaticJoint.Define(stage, joints.AppendChild(f"{name}_slide"))
        j.CreateBody0Rel().SetTargets([palm.GetPath()])
        j.CreateBody1Rel().SetTargets([f.GetPath()])
        j.CreateAxisAttr().Set("X")
        j.CreateLocalPos0Attr().Set(Gf.Vec3f(float(sign * x_open), 0.0,
                                             float(FINGER_Z - PALM_Z)))
        j.CreateLocalRot0Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        j.CreateLocalPos1Attr().Set(Gf.Vec3f(0.0, 0.0, 0.0))
        j.CreateLocalRot1Attr().Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
        # 닫는 방향: 왼쪽(+x) 손가락은 -x, 오른쪽은 +x.
        lo, hi = (-FINGER_TRAVEL, 0.0) if sign > 0 else (0.0, FINGER_TRAVEL)
        j.CreateLowerLimitAttr().Set(float(lo))
        j.CreateUpperLimitAttr().Set(float(hi))
        d = UsdPhysics.DriveAPI.Apply(j.GetPrim(), "linear")
        d.CreateTypeAttr().Set("force")
        d.CreateStiffnessAttr().Set(4000.0)
        d.CreateDampingAttr().Set(200.0)
        d.CreateMaxForceAttr().Set(FINGER_FORCE)
        d.CreateTargetPositionAttr().Set(0.0)
        out[name] = str(f.GetPath())
        out["joint_" + name[-1]] = str(j.GetPath())
    print(f"[gripper] {rover_root}: RG2 → 넓은 집게 (벌림 {GAP_CLOSED:.2f}~{GAP_OPEN:.2f} m, "
          f"손가락 길이 {FINGER_SIZE[2]:.2f} m, 조이는 힘 {FINGER_FORCE:.0f} N, "
          f"패드 마찰 {PAD_FRICTION[0]}/{PAD_FRICTION[1]})")
    return out


class FrictionGrasp:
    """닫기/열기 명령을 손가락 구동 목표로 바꾸고, 물리 상태에서 '쥠' 을 읽는다."""

    def __init__(self, stage, grippers: dict):
        """grippers: {ns: install_wide_gripper() 의 결과}."""
        import omni.physx
        self._stage = stage
        self._physx = omni.physx.get_physx_interface()
        self._query = omni.physx.get_physx_scene_query_interface()
        self._g = {ns: g for ns, g in grippers.items() if g}
        self._closed = {ns: False for ns in self._g}
        self._gap_hist = {ns: [] for ns in self._g}   # held() 호출 때마다의 간격
        self.last_held = {ns: "" for ns in self._g}  # 마지막 held() 결과 (진단용)
        print(f"[grasp] 마찰 파지 준비: {sorted(self._g)}")

    def _set(self, ns: str, closed: bool) -> None:
        g = self._g[ns]
        for key, sign in (("joint_l", 1.0), ("joint_r", -1.0)):
            target = (-sign * FINGER_TRAVEL) if closed else 0.0
            self._stage.GetPrimAtPath(g[key]).GetAttribute(
                "drive:linear:physics:targetPosition").Set(float(target))
        if closed != self._closed[ns]:
            print(f"[grasp] {ns} 집게 {'닫기' if closed else '열기'}")
            self._gap_hist[ns] = []
        self._closed[ns] = closed

    def close(self, ns: str) -> None:
        if ns in self._g and not self._closed[ns]:
            self._set(ns, True)

    def open(self, ns: str) -> None:
        """집게를 열고, 그 순간 손가락 사이에 있던 광물을 적재 처리(숨김)한다."""
        if ns not in self._g:
            return
        held = self._between(ns)[0]
        self._set(ns, False)
        if held:
            self._stow(ns, held)

    def _pose(self, path: str):
        t = self._physx.get_rigidbody_transformation(path)
        if not t or not t.get("ret_val"):
            return None
        p, q = t["position"], t["rotation"]
        x, y, z, w = q[0], q[1], q[2], q[3]
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        return np.array([p[0], p[1], p[2]]), R, (x, y, z, w)

    def gap(self, ns: str) -> float | None:
        """손가락 안쪽 면 사이 간격 (m)."""
        g = self._g[ns]
        a, b = self._pose(g["finger_l"]), self._pose(g["finger_r"])
        if a is None or b is None:
            return None
        return float(np.linalg.norm(a[0] - b[0])) - FINGER_SIZE[0]

    def held(self, ns: str) -> str:
        """닫은 손가락이 광물에 막혀 멈춰 있으면 그 광물 이름 (없으면 "").

        status 발행 주기(약 0.1 s)마다 한 번씩 불린다고 가정한다 — 최근 4 번의
        간격이 3 mm 안에서 변하지 않아야 "멈췄다" 고 본다. 닫히는 도중에는
        손가락 사이에 광물이 있어도 쥔 것이 아니다.
        """
        path, gap = self._between(ns)
        if ns not in self._g:
            return ""
        hist = self._gap_hist[ns]
        hist.append(gap if gap is not None else -1.0)
        del hist[:-4]
        if not path or len(hist) < 4 or max(hist) - min(hist) > 0.003:
            self.last_held[ns] = ""
        else:
            self.last_held[ns] = path.rsplit("/", 1)[-1]
        return self.last_held[ns]

    def _between(self, ns: str):
        """(닫힌 손가락 사이에 있는 광물 path 또는 "", 간격). 열려 있으면 ("", 간격)."""
        if ns not in self._g:
            return "", None
        g = self._g[ns]
        a, b = self._pose(g["finger_l"]), self._pose(g["finger_r"])
        if a is None or b is None:
            return "", None
        gap = float(np.linalg.norm(a[0] - b[0])) - FINGER_SIZE[0]
        if not self._closed[ns] or gap < GAP_CLOSED + 0.02:   # 끝까지 닫힘 = 못 잡음
            return "", gap
        centre = (a[0] + b[0]) / 2
        half = carb.Float3(max(gap / 2 - 0.005, 0.005), FINGER_SIZE[1] / 2,
                           FINGER_SIZE[2] / 2)
        names = []

        def _report(hit):
            prim = self._stage.GetPrimAtPath(str(hit.rigid_body))
            while prim and prim.IsValid() and str(prim.GetPath()) != "/":
                parent = prim.GetParent()
                if parent and parent.GetName() == "Minerals":
                    names.append(str(prim.GetPath()))
                    break
                prim = parent
            return True

        x, y, z, w = a[2]
        self._query.overlap_box(
            half, carb.Float3(float(centre[0]), float(centre[1]), float(centre[2])),
            carb.Float4(float(x), float(y), float(z), float(w)), _report, False)
        return (names[0] if names else ""), gap

    def _stow(self, ns: str, path: str) -> None:
        prim = self._stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            return
        for child in Usd.PrimRange(prim):
            if child.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr().Set(False)
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            UsdPhysics.RigidBodyAPI(prim).CreateRigidBodyEnabledAttr().Set(False)
        UsdGeom.Imageable(prim).MakeInvisible()
        print(f"[grasp] {ns} 열기 + 적재 처리(숨김) {prim.GetName()}")
