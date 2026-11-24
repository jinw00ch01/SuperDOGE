"""바퀴 충돌 형상을 구동축에 중심을 맞춘 둥근 형상(캡슐)으로 바꾼다.

SimulationApp 기동 후에만 import 할 것 (pxr 필요).

왜 필요한가
-----------
로버 모델(vehicle_v3.usd)의 바퀴 충돌 형상은 바퀴 mesh 의 convex hull, 즉 다각형이다.
바퀴가 돌면 꼭짓점이 땅을 치며 떠서 접촉이 이어지지 않는다 — 헛도는 동안 물리 스텝의
82 % 에서 바퀴에 걸린 법선력이 0 이었고 (접촉 기록), 구동 접지력은 설정 마찰의
60~65 % 만 나왔다 (레골리스 동마찰 0.52 인데 0.31~0.35. isaac_sim/rl/probe_traction.py).
제동(바퀴 잠김)은 형상이 돌지 않으니 정상(0.50)이라, 미끄럼 마찰만 재서는 드러나지 않는다.

  · USD 원기둥으로 바꿔도 PhysX 가 다시 다각형 convex 로 근사해서 같다 (0.31~0.39).
  · 물리 주기를 60 → 240 Hz 로 올려도 같다.
  · mesh 중심이 구동축에서 1.7~4.8 mm 벗어나 있기도 한데 (앞 1.7, 가운데 3.7, 뒤 4.8),
    편심만 없앤 다각형으로는 나아지지 않았다. 둥근지가 전부다.
  · 캡슐·구는 0.50~0.52 로 설정값 그대로 나온다.

캡슐은 축 방향으로 누운 형태라 굴러가는 방향으로는 완전한 원이고, 땅에는 선으로 닿는다.
"""
from __future__ import annotations

import numpy as np
from pxr import Gf, Usd, UsdGeom, UsdPhysics

ROUND_NAME = "RoundCollider"


def round_wheel_colliders(stage, rover_roots, min_radius: float = 0.08,
                          shape: str = "capsule") -> dict:
    """rover_roots 아래 *_Drive link 의 접지면 collider 를 원기둥으로 바꾼다.

    접지면 = 축에서 본 바깥 반지름이 min_radius 보다 큰 mesh collider. 그 mesh 들의
    충돌은 끄고(모양은 그대로 보인다), 같은 폭·같은 평균 반지름의 원기둥을 축 위에 둔다.
    return: {"wheels": 바꾼 바퀴 수, "radius": [...], "eccentricity_mm": [...]}
    """
    cache = UsdGeom.XformCache()
    radii, ecc, count = [], [], 0
    for root in rover_roots:
        rp = stage.GetPrimAtPath(root)
        if not rp or not rp.IsValid():
            continue
        for link in Usd.PrimRange(rp):
            if not link.GetName().endswith("_Drive") or link.GetChild(ROUND_NAME):
                continue
            to_link = np.linalg.inv(np.array(cache.GetLocalToWorldTransform(link)))
            lo, hi, r_min, r_max, treads = np.inf, -np.inf, np.inf, 0.0, []
            for sub in Usd.PrimRange(link):
                if not (sub.IsA(UsdGeom.Mesh) and sub.HasAPI(UsdPhysics.CollisionAPI)):
                    continue
                pts = np.asarray(UsdGeom.Mesh(sub).GetPointsAttr().Get(), dtype=np.float64)
                pts = (np.c_[pts, np.ones(len(pts))]
                       @ np.array(cache.GetLocalToWorldTransform(sub)) @ to_link)[:, :3]
                r = np.hypot(pts[:, 1], pts[:, 2])           # 구동축 = link 의 x 축
                if r.max() < min_radius:
                    continue
                ang = np.arctan2(pts[:, 2], pts[:, 1])
                rim = [r[(ang >= a) & (ang < a + 0.2618)].max()
                       for a in np.arange(-np.pi, np.pi, 0.2618)
                       if ((ang >= a) & (ang < a + 0.2618)).any()]
                treads.append(sub)
                if max(rim) > r_max:                          # 가장 바깥 mesh 가 접지면
                    r_min, r_max = min(rim), max(rim)
                lo, hi = min(lo, pts[:, 0].min()), max(hi, pts[:, 0].max())
            if not treads:
                continue
            radius = 0.5 * (r_min + r_max)
            for sub in treads:
                UsdPhysics.CollisionAPI(sub).CreateCollisionEnabledAttr().Set(False)
            path = link.GetPath().AppendChild(ROUND_NAME)
            if shape == "sphere":
                cyl = UsdGeom.Sphere.Define(stage, path)
            else:
                cyl = (UsdGeom.Capsule if shape == "capsule" else UsdGeom.Cylinder).Define(stage, path)
                cyl.CreateAxisAttr("X")
                # 캡슐은 양 끝이 반구라 옆으로 튀어나온다 → 몸통 길이를 접지 폭의 절반으로.
                cyl.CreateHeightAttr(float(hi - lo) * (0.5 if shape == "capsule" else 1.0))
            cyl.CreateRadiusAttr(float(radius))
            cyl.AddTranslateOp().Set(Gf.Vec3d(float(0.5 * (lo + hi)), 0.0, 0.0))
            cyl.CreatePurposeAttr(UsdGeom.Tokens.guide)       # 그리지 않는다
            UsdPhysics.CollisionAPI.Apply(cyl.GetPrim())
            radii.append(round(float(radius), 4))
            ecc.append(round(1000.0 * 0.5 * float(r_max - r_min), 1))
            count += 1
    return {"wheels": count, "radius": sorted(set(radii)),
            "eccentricity_mm": sorted(set(ecc))}
