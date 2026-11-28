"""로버 여러 대의 출발 자리 — 서로 가장 멀리 떨어지게 (`run_vehicle_v3.py --spawn spread`).

지형 meta 의 spawn_locations 는 순서대로 쓰면 가까이 붙기도 한다 (terrain_00015 의 0 번과
2 번은 3.9 m). 여러 대가 같은 곳에서 출발하면 같은 광물을 보고 같은 길로 간다.

고르는 법 (`pick_spread`, numpy/scipy 만 씀 — Isaac 없이 시험할 수 있다)
  후보: 0.5 m 간격의 칸 중
    · 정답 장애물(obstacle_grid.npy)에서 clear_m 넘게 떨어짐  → 장애물 위·옆에서 출발하지 않는다
    · 0.8 m 창으로 본 경사 ≤ slope_deg
    · 기지 반경 + base_margin_m 밖, 지도 가장자리에서 edge_m 안쪽
    · 광물에서 mineral_clear_m 넘게 떨어짐
    · 로버가 다닐 수 있는 가장 큰 연결 영역 안 (막힌 구석이 아님)
  선택: "가장 가까운 두 대 사이의 거리" 가 가장 커지는 조합 (2~3 대는 전수 탐색,
        4 대부터는 가장 먼 두 자리에서 시작해 하나씩 가장 먼 자리를 더함).
  방향: 지도 중심을 보게 한다.
"""
from __future__ import annotations

import itertools
import math

import numpy as np


def candidates(meta: dict, heightmap: np.ndarray, obstacle: np.ndarray,
               slope_deg: float = 10.0, clear_m: float = 2.0, edge_m: float = 3.0,
               base_margin_m: float = 2.0, mineral_clear_m: float = 2.0,
               step_m: float = 0.5, extra_minerals=()) -> np.ndarray:
    """return: (N, 3) 의 x, y, 지면 높이."""
    from scipy import ndimage

    res = float(meta.get("resolution_m", 0.05))
    ox, oy = float(meta["origin"]["x"]), float(meta["origin"]["y"])
    rows, cols = heightmap.shape
    k = max(1, int(round(0.8 / res)))
    hs = ndimage.uniform_filter(heightmap.astype(np.float64), k)
    gy, gx = np.gradient(hs, res)
    dist = ndimage.distance_transform_edt(obstacle == 0) * res
    ok = (np.hypot(gx, gy) <= math.tan(math.radians(slope_deg))) & (dist > clear_m)
    e = int(round(edge_m / res))
    ok[:e, :] = ok[-e:, :] = False
    ok[:, :e] = ok[:, -e:] = False
    ii, jj = np.mgrid[0:rows, 0:cols]
    xs, ys = ox + (jj + 0.5) * res, oy + (ii + 0.5) * res
    bc = meta.get("basecamp") or {}
    c = bc.get("center") or {}
    if "x" in c:
        r = float(bc.get("radius", 6.0)) + base_margin_m
        ok &= (xs - float(c["x"])) ** 2 + (ys - float(c["y"])) ** 2 > r * r
    minerals = [(float(m["position"]["x"]), float(m["position"]["y"]))
                for m in (meta.get("minerals") or [])]
    minerals += [(float(m["position"]["x"]), float(m["position"]["y"])) for m in extra_minerals]
    for (mx, my) in minerals:
        ok &= (xs - mx) ** 2 + (ys - my) ** 2 > mineral_clear_m ** 2
    # 다닐 수 있는 가장 큰 영역 (로버 반경 0.8 m 로 부풀린 장애물 밖).
    lab, n = ndimage.label(dist > 0.8)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
        ok &= lab == (1 + int(np.argmax(sizes)))
    s = max(1, int(round(step_m / res)))
    sub = np.zeros_like(ok)
    sub[s // 2::s, s // 2::s] = True
    ci, cj = np.nonzero(ok & sub)
    hmax = ndimage.maximum_filter(heightmap, size=max(1, int(round(1.0 / res))))
    return np.stack([xs[ci, cj], ys[ci, cj], hmax[ci, cj]], axis=1)


def pick_spread(meta: dict, heightmap: np.ndarray, obstacle: np.ndarray, n: int,
                extra_minerals=(), exhaustive_pool: int = 160, **kw) -> list:
    """return: [{"x", "y", "z", "yaw"}] n 개 (못 고르면 더 짧다)."""
    cand = candidates(meta, heightmap, obstacle, extra_minerals=extra_minerals, **kw)
    if len(cand) == 0 or n <= 0:
        return []
    if len(cand) <= n:
        idx = list(range(len(cand)))
    else:
        xy = cand[:, :2]
        center = xy.mean(axis=0)
        # 가장 먼 조합은 바깥쪽 후보에서 나온다 — 중심에서 먼 순으로 추려서 전수 탐색.
        pool = np.argsort(-np.hypot(*(xy - center).T))[:exhaustive_pool]
        d = np.hypot(xy[pool, None, 0] - xy[None, pool, 0], xy[pool, None, 1] - xy[None, pool, 1])
        if n == 1:
            idx = [int(pool[0])]
        elif n == 2:
            a, b = np.unravel_index(int(np.argmax(d)), d.shape)
            idx = [int(pool[a]), int(pool[b])]
        elif n == 3:
            best, idx = -1.0, []
            for a, b in itertools.combinations(range(len(pool)), 2):
                if d[a, b] <= best:
                    continue
                m = np.minimum(d[a], d[b])
                m[[a, b]] = 0.0
                c3 = int(np.argmax(m))
                score = min(d[a, b], m[c3])
                if score > best:
                    best, idx = score, [int(pool[a]), int(pool[b]), int(pool[c3])]
        else:
            a, b = np.unravel_index(int(np.argmax(d)), d.shape)
            chosen = [int(a), int(b)]
            while len(chosen) < n:
                m = d[chosen].min(axis=0)
                m[chosen] = 0.0
                chosen.append(int(np.argmax(m)))
            idx = [int(pool[c]) for c in chosen]
    c = (meta.get("basecamp") or {}).get("center") or {}
    cx, cy = float(c.get("x", 0.0)), float(c.get("y", 0.0))
    out = []
    for i in idx:
        x, y, z = (float(v) for v in cand[i])
        out.append({"x": round(x, 2), "y": round(y, 2), "z": round(z, 2),
                    "yaw": round(math.atan2(cy - y, cx - x), 3)})
    # 늘 같은 순서: 방위각 순 (rover_1 이 어느 자리인지가 실행마다 바뀌지 않게).
    out.sort(key=lambda s: math.atan2(s["y"] - cy, s["x"] - cx))
    return out


def min_separation(spots) -> float:
    return min((math.hypot(a["x"] - b["x"], a["y"] - b["y"])
                for a, b in itertools.combinations(spots, 2)), default=float("inf"))
