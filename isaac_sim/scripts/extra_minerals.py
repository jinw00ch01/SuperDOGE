"""시험용: 지형 파일을 바꾸지 않고 광물을 더 깐다 (`run_vehicle_v3.py --mineral-scale 2`).

왜 필요한가
-----------
지형 하나에 광물이 12 개뿐이라 10 분 임무에서 3~7 개를 모은다. 같은 코드로 돌려도
그만큼 흩어져서, 한 번씩 실행해서는 "고친 것이 나아졌는지" 를 수집 수로 가릴 수 없다.
광물을 배로 깔면 한 번의 실행에 집기·접근 시도가 배로 들어간다.

자리 고르기 (`plan_extra`, numpy 만 씀 — Isaac 없이 시험할 수 있다)
--------------------------------------------------------------------
지형 생성기가 광물을 놓을 때와 같은 조건에, 지도 전체에 고르게 퍼지도록:
  · 정답 장애물(obstacle_grid.npy)에서 clear_m 넘게 떨어진 칸
  · 0.8 m 창으로 본 경사 ≤ slope_deg
  · 기지 중심에서 exclude_basecamp_radius_m 밖, 지도 가장자리에서 edge_m 안쪽
  · 로버 출발 지점에서 spawn_clear_m 밖
  · 다른 광물(원래 것 + 새로 놓은 것)과 min_spacing_m 이상
  · 후보 K 개 중 "가장 가까운 광물까지의 거리" 가 가장 먼 것을 고른다 (빈 곳부터 채움)
종류는 원래 광물의 종류를 그대로 한 번씩 더 쓴다 (배율 2 → 종류별로 정확히 두 배).
씨앗은 지형의 seed 에서 정하므로 같은 지형·같은 배율이면 늘 같은 자리다.

만드는 prim 은 지형 생성기(`mars_terrain_generator_v4.compose_world`)가 만든 것과 같은
구조다: <Minerals>/<type>_<id> (Xform, translate, RigidBody + convexHull 충돌) 아래
Reference 자식이 markers/tier2_mineral/<type>.usd 를 가리킨다.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np

HEIGHT_OFFSET = 0.10        # 원래 광물의 (z − 지형 높이) 는 0.09~0.11 m


def plan_extra(meta: dict, heightmap: np.ndarray, obstacle: np.ndarray, scale: float,
               slope_deg: float = 18.0, clear_m: float = 1.0, edge_m: float = 1.5,
               spawn_clear_m: float = 2.0, candidates: int = 40, starts=None) -> list:
    """return: [{"id", "type", "position": {"x", "y", "z"}}] — 새로 놓을 광물."""
    from scipy import ndimage

    base = list(meta.get("minerals") or [])
    n_extra = int(round(len(base) * (float(scale) - 1.0)))
    if n_extra <= 0 or not base:
        return []
    res = float(meta.get("resolution_m", 0.05))
    ox, oy = float(meta["origin"]["x"]), float(meta["origin"]["y"])
    rows, cols = heightmap.shape
    gp = (meta.get("generation_params") or {}).get("minerals") or {}
    spacing = float(gp.get("min_spacing_m", 3.0))
    base_r = float(gp.get("exclude_basecamp_radius_m", 6.0))

    k = max(1, int(round(0.8 / res)))
    hs = ndimage.uniform_filter(heightmap.astype(np.float64), k)
    gy, gx = np.gradient(hs, res)
    ok = np.hypot(gx, gy) <= math.tan(math.radians(slope_deg))
    ok &= ndimage.distance_transform_edt(obstacle == 0) * res > clear_m
    ii, jj = np.mgrid[0:rows, 0:cols]
    xs, ys = ox + (jj + 0.5) * res, oy + (ii + 0.5) * res
    e = int(round(edge_m / res))
    ok[:e, :] = ok[-e:, :] = False
    ok[:, :e] = ok[:, -e:] = False
    bc = ((meta.get("basecamp") or {}).get("center")) or {}
    if "x" in bc:
        ok &= (xs - float(bc["x"])) ** 2 + (ys - float(bc["y"])) ** 2 > base_r ** 2
    for s in (starts if starts is not None else meta.get("spawn_locations") or []):
        ok &= (xs - float(s["x"])) ** 2 + (ys - float(s["y"])) ** 2 > spawn_clear_m ** 2
    ci, cj = np.nonzero(ok)
    if len(ci) == 0:
        return []
    cx, cy = xs[ci, cj], ys[ci, cj]

    rng = np.random.default_rng(int(meta.get("seed", 0)) * 7919 + int(round(scale * 100)))
    placed = [(float(m["position"]["x"]), float(m["position"]["y"])) for m in base]
    types = [str(m["type"]) for m in base]
    order = [types[i % len(types)] for i in rng.permutation(n_extra) % len(types)] \
        if n_extra != len(types) else [types[i] for i in rng.permutation(len(types))]
    next_id = max(int(m["id"]) for m in base) + 1
    out = []
    for mtype in order:
        pick = rng.choice(len(ci), size=min(int(candidates), len(ci)), replace=False)
        p = np.asarray(placed)
        d = np.sqrt(((cx[pick, None] - p[None, :, 0]) ** 2
                     + (cy[pick, None] - p[None, :, 1]) ** 2).min(axis=1))
        b = int(np.argmax(d))
        if d[b] < spacing:
            continue                     # 이 후보들로는 간격을 못 지킨다 — 이 하나는 건너뜀
        q = pick[b]
        x, y = float(cx[q]), float(cy[q])
        z = float(heightmap[ci[q], cj[q]]) + HEIGHT_OFFSET
        placed.append((x, y))
        out.append({"id": next_id, "type": mtype, "value": None,
                    "position": {"x": round(x, 2), "y": round(y, 2), "z": round(z, 2)}})
        next_id += 1
    return out


def spawn_extra(stage, terrain_dir: str, meta: dict, scale: float,
                scene_root: str = "/World/MarsScene", marker_dir: str = "") -> list:
    """광물을 scale 배로 늘린다 (stage 에 prim 추가). return: 추가한 광물 목록."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    if float(scale) <= 1.0:
        return []
    hm = np.load(os.path.join(terrain_dir, "heightmap.npy"))
    og = np.load(os.path.join(terrain_dir, "obstacle_grid.npy"))
    extra = plan_extra(meta, hm, og, scale)
    scope = None
    root = stage.GetPrimAtPath(scene_root)
    if root and root.IsValid():
        for prim in Usd.PrimRange(root):
            if prim.GetName() == "Minerals":
                scope = prim
                break
    if scope is None:
        print(f"[minerals] {scene_root} 아래에 Minerals 가 없음 — 추가하지 않음")
        return []
    made = []
    for m in extra:
        asset = os.path.join(marker_dir, "tier2_mineral", f"{m['type']}.usd")
        if not os.path.isfile(asset):
            print(f"[minerals] 자산 없음: {asset} — 건너뜀")
            continue
        path = f"{scope.GetPath()}/{m['type']}_{int(m['id']):04d}"
        p = m["position"]
        wrapper = UsdGeom.Xform.Define(stage, path)
        UsdGeom.XformCommonAPI(wrapper.GetPrim()).SetTranslate(
            Gf.Vec3d(float(p["x"]), float(p["y"]), float(p["z"])))
        ref = UsdGeom.Xform.Define(stage, f"{path}/Reference")
        ref.GetPrim().GetReferences().AddReference(asset)
        wprim = wrapper.GetPrim()
        UsdPhysics.RigidBodyAPI.Apply(wprim)
        UsdPhysics.CollisionAPI.Apply(wprim)
        UsdPhysics.MeshCollisionAPI.Apply(wprim).CreateApproximationAttr().Set("convexHull")
        UsdPhysics.MassAPI.Apply(wprim).CreateMassAttr().Set(0.3)   # apply_mineral_mass 가 덮어씀
        made.append(m)
    count = {}
    for m in made:
        count[m["type"]] = count.get(m["type"], 0) + 1
    print(f"[minerals] 광물 ×{float(scale):g}: 원래 {len(meta.get('minerals') or [])}개에 "
          f"{len(made)}개 추가 {count}")
    print("[minerals] 추가한 자리: " + json.dumps(
        [{"id": m["id"], "type": m["type"], **m["position"]} for m in made],
        separators=(",", ":")))
    return made
