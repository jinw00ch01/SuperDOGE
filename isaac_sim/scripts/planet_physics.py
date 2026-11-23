"""행성 물리 환경(중력 + 접촉 마찰) 을 stage 에 적용한다.

SimulationApp 기동 후에만 import 할 것 (pxr / PhysxSchema 필요).

적용 범위와 한계
----------------
PhysX 는 강체 접촉만 푼다. 그래서 화성 환경 중 여기서 맞출 수 있는 것은
  · 중력 가속도
  · 지면·바위·바퀴의 Coulomb 마찰 계수 (지형 구역별)
  · 반발 계수 (레골리스는 튀지 않음 → 0)
이고, 맞출 수 없는 것은
  · 바퀴 침하·흙 밀림 (terramechanics, Bekker-Wong) — 강체 지형이라 없음
  · 흙의 점착력(cohesion) — Coulomb 마찰에는 항이 없음
  · 대기 — 밀도가 지구의 약 1.5% 라 항력은 무시 (PhysX 에도 항력 없음)
이다. 침하가 없으므로 "마찰 계수 = tan(내부마찰각)" 을 견인 한계로 쓴다.
같은 이유로 등판 한계는 atan(mu) 가 된다 (sand 0.30 → 약 17°).

값의 근거
---------
  · 중력: NASA Mars Fact Sheet 3.73, ESA 3.711 m/s². 프로젝트의 다른
    스크립트와 맞춰 3.72 를 쓴다.
  · 기본 레골리스: Mars sand 의 내부마찰각 약 30° (InSight HP3 분석),
    Pathfinder soil-like deposit 평균 34.3° → tan(30°)=0.58 을 정지 마찰로.
  · sand / rocky 구역: terrain meta.json 의 physics_zones 값을 그대로 쓴다.
    sand 0.30 = tan(16.7°) 는 Viking drift material (15.6~20.4°) 범위이고,
    MSL 시험 차량이 점착 없는 모래에서 10° 는 오르고 17° 에서 파묻힌
    결과와도 맞는다.
  · 광물 질량: 지형 생성기는 모든 광물에 0.3 kg 을 넣는다. 광물 mesh 의
    convex hull 부피는 2.98 L (0.17 × 0.19 × 0.23 m) 이고 결정 다발 모양이라
    실제 부피를 그 절반(약 1.5 L)으로 보면, 암석 밀도 2700 kg/m³ 에서 약
    4 kg 이다 — 0.3 kg 은 스티로폼 수준이다. 화성 모드에서는 4 kg 으로 바꾼다
    (화성 무게 14.9 N = 지구에서 1.5 kg 을 드는 것과 같음).
"""
from __future__ import annotations

import math

import numpy as np
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics, UsdShade, Vt

MATERIAL_ROOT = "/World/PhysicsMaterials"
MINERAL_MASS_KG = 4.0


def apply_mineral_mass(stage, mass: float | None,
                       scene_root: str = "/World/MarsScene") -> int:
    """<scene_root>/**/Minerals 아래 광물의 질량을 mass (kg) 로 바꾼다.

    mass 가 None 이거나 0 이하면 아무것도 바꾸지 않는다. return: 바꾼 개수.
    """
    if not mass or mass <= 0.0:
        print("[physics] 광물 질량: 지형 파일 값 그대로")
        return 0
    root = stage.GetPrimAtPath(scene_root)
    n, before = 0, set()
    if root and root.IsValid():
        for prim in Usd.PrimRange(root):
            if prim.GetName() != "Minerals":
                continue
            for m in prim.GetChildren():
                if not m.HasAPI(UsdPhysics.RigidBodyAPI):
                    continue
                attr = UsdPhysics.MassAPI.Apply(m).CreateMassAttr()
                old = attr.Get()
                if old is not None:
                    before.add(round(float(old), 3))
                attr.Set(float(mass))
                n += 1
    print(f"[physics] 광물 질량: {sorted(before) or '?'} kg → {mass} kg ({n}개)")
    return n

PLANETS = {
    "mars": {
        "gravity": 3.72,
        # 구역 밖 기본 지면.
        "regolith": {"static_friction": 0.58, "dynamic_friction": 0.52},
        # 현무암 바위 표면.
        "rock": {"static_friction": 0.70, "dynamic_friction": 0.60},
    },
    # earth 는 아무것도 바꾸지 않는다 (원래 동작: 9.81, PhysX 기본 마찰 0.5).
    "earth": None,
}


def _make_material(stage, name, static_friction, dynamic_friction, combine):
    mat = UsdShade.Material.Define(stage, f"{MATERIAL_ROOT}/{name}")
    api = UsdPhysics.MaterialAPI.Apply(mat.GetPrim())
    api.CreateStaticFrictionAttr().Set(float(static_friction))
    api.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
    api.CreateRestitutionAttr().Set(0.0)
    px = PhysxSchema.PhysxMaterialAPI.Apply(mat.GetPrim())
    px.CreateFrictionCombineModeAttr().Set(combine)
    px.CreateRestitutionCombineModeAttr().Set("min")
    return mat


def bind_physics(prim, material) -> None:
    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
        material, UsdShade.Tokens.weakerThanDescendants, "physics")


def _points_in_polygon(xy: np.ndarray, polygon) -> np.ndarray:
    """ray casting. xy (N,2) → bool (N,)."""
    x, y = xy[:, 0], xy[:, 1]
    inside = np.zeros(len(xy), dtype=bool)
    n = len(polygon)
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        if y0 == y1:
            continue
        crosses = (y0 > y) != (y1 > y)
        x_at = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
        inside ^= crosses & (x < x_at)
    return inside


def zone_at(x: float, y: float, zones) -> str:
    """world (x, y) 가 속한 구역 이름. 어느 polygon 에도 없으면 'regolith'."""
    pt = np.array([[x, y]], dtype=float)
    for z in zones or []:
        if _points_in_polygon(pt, z["polygon"])[0]:
            return str(z["type"])
    return "regolith"


def _find_terrain_meshes(stage, scene_root: str):
    out = []
    root = stage.GetPrimAtPath(scene_root)
    if not root or not root.IsValid():
        return out
    for prim in Usd.PrimRange(root):
        if prim.GetName() == "TerrainMesh" and prim.IsA(UsdGeom.Mesh):
            out.append(prim)
    return out


def _zone_subsets(stage, mesh_prim, zones, materials) -> dict:
    """지형 mesh 의 face 를 구역 polygon 으로 나눠 GeomSubset + physics material."""
    mesh = UsdGeom.Mesh(mesh_prim)
    pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
    counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    if len(counts) == 0 or not np.all(counts == 3):
        print("[physics] ⚠ 지형 mesh 가 삼각형이 아님 — 구역별 마찰 생략")
        return {}
    M = UsdGeom.XformCache().GetLocalToWorldTransform(mesh_prim)
    R = np.array([[M[i][j] for j in range(3)] for i in range(3)])
    t = np.array([M[3][0], M[3][1], M[3][2]])
    world = pts @ R + t                                 # USD: row vector 규약
    centers = world[idx.reshape(-1, 3)].mean(axis=1)[:, :2]
    taken = np.zeros(len(centers), dtype=bool)
    counts_out = {}
    for z in zones:
        name = str(z["type"])
        inside = _points_in_polygon(centers, z["polygon"]) & ~taken
        taken |= inside
        faces = np.nonzero(inside)[0]
        if len(faces) == 0:
            continue
        subset = UsdGeom.Subset.Define(
            stage, mesh_prim.GetPath().AppendChild(f"PhysicsZone_{name}"))
        subset.CreateElementTypeAttr().Set(UsdGeom.Tokens.face)
        subset.CreateIndicesAttr().Set(Vt.IntArray(faces.tolist()))
        # familyName 을 materialBind 로 두지 않는다 — 렌더 재질 분할로 해석되면
        # 지형의 시각 재질이 구역에서 빠진다. PhysX 는 family 와 무관하게
        # subset 의 physics-purpose binding 을 읽는다.
        subset.CreateFamilyNameAttr().Set("physicsZone")
        bind_physics(subset.GetPrim(), materials[name])
        counts_out[name] = int(len(faces))
    counts_out["regolith"] = int((~taken).sum())
    return counts_out


def apply_planet_physics(stage, world, planet: str, meta: dict | None = None,
                         scene_root: str = "/World/MarsScene",
                         rover_roots=()) -> dict:
    """중력과 접촉 재질을 적용하고 적용 내역을 dict 로 돌려준다.

    world.reset() / play() 전에, 로버를 stage 에 올린 뒤에 호출할 것.
    """
    cfg = PLANETS.get(planet)
    scenes = [p for p in stage.Traverse() if p.IsA(UsdPhysics.Scene)]
    if cfg is None:
        g = None
        for p in scenes:
            g = UsdPhysics.Scene(p).GetGravityMagnitudeAttr().Get()
        info = {"planet": planet, "gravity": g, "materials": {}, "zones": {}}
        print(f"[physics] {planet}: 변경 없음 (PhysicsScene gravity attr={g}, "
              f"접촉 재질 없음 → PhysX 기본 마찰)")
        return info

    # ── 중력 ────────────────────────────────────────────────────────────
    g = float(cfg["gravity"])
    for p in scenes:
        sc = UsdPhysics.Scene(p)
        sc.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
        sc.CreateGravityMagnitudeAttr().Set(g)
    try:
        world.get_physics_context().set_gravity(-g)
    except Exception as exc:
        print(f"[physics] ⚠ physics_context.set_gravity 실패: {exc}")

    # ── 재질 ────────────────────────────────────────────────────────────
    # 지면·바위는 combine=min, 바퀴는 mu=1.0 → 바퀴-지면 유효 마찰 = 지면 값.
    # (PhysX 는 두 재질 중 우선순위 높은 combine mode 를 쓴다:
    #  average < min < multiply < max.)
    zones = list((meta or {}).get("physics_zones") or [])
    mats = {
        "regolith": _make_material(stage, "Regolith", combine="min", **cfg["regolith"]),
        "rock": _make_material(stage, "Rock", combine="min", **cfg["rock"]),
        "wheel": _make_material(stage, "WheelGrouser", 1.0, 1.0, combine="average"),
    }
    table = {"regolith": dict(cfg["regolith"]), "rock": dict(cfg["rock"])}
    for z in zones:
        name = str(z["type"])
        mats[name] = _make_material(
            stage, f"Zone_{name}", z["static_friction"], z["dynamic_friction"],
            combine="min")
        table[name] = {"static_friction": float(z["static_friction"]),
                       "dynamic_friction": float(z["dynamic_friction"])}

    zone_faces = {}
    meshes = _find_terrain_meshes(stage, scene_root)
    for mesh_prim in meshes:
        bind_physics(mesh_prim, mats["regolith"])
        if zones:
            zone_faces = _zone_subsets(stage, mesh_prim, zones, mats)

    n_rocks = 0
    rocks = stage.GetPrimAtPath(f"{scene_root}/Rocks")
    if rocks and rocks.IsValid():
        for prim in rocks.GetChildren():
            bind_physics(prim, mats["rock"])
            n_rocks += 1

    n_wheels = 0
    for root in rover_roots:
        rp = stage.GetPrimAtPath(root)
        if not rp or not rp.IsValid():
            continue
        for prim in Usd.PrimRange(rp):
            if not prim.GetName().endswith("_Drive"):
                continue
            for sub in Usd.PrimRange(prim):
                if sub.HasAPI(UsdPhysics.CollisionAPI):
                    bind_physics(sub, mats["wheel"])
                    n_wheels += 1

    info = {"planet": planet, "gravity": g, "materials": table,
            "zones": zone_faces, "terrain_meshes": len(meshes),
            "rocks": n_rocks, "wheel_colliders": n_wheels}
    climb = {k: round(math.degrees(math.atan(v["static_friction"])), 1)
             for k, v in table.items() if k != "rock"}
    print(f"[physics] {planet}: gravity {g} m/s² (scene {len(scenes)}개), "
          f"지형 mesh {len(meshes)}, 바위 {n_rocks}, 바퀴 collider {n_wheels}")
    print(f"[physics] 마찰(static/dynamic): "
          + ", ".join(f"{k} {v['static_friction']:.2f}/{v['dynamic_friction']:.2f}"
                      for k, v in table.items()))
    print(f"[physics] 구역별 face 수: {zone_faces or '구역 없음'} · "
          f"등판 한계 atan(mu): {climb}")
    return info
