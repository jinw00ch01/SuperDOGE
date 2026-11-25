"""경사 비용 계획 확인 — 꺼진 곳에서 밖으로 나가는 경로가 얼마나 가파른 곳을 지나는가.

시뮬레이터 없이 돈다 (numpy·scipy). 실제 스택과 같은 코드를 거친다:
  지형 정답 높이 → HazardMap.classify(with_grade) (깊이 카메라가 빠짐없이 본 것으로 침)
  → hazard_map_node 와 같은 인코딩 (안전 칸 = 경사°, 위험 = 100)
  → OnlineObstacleLayer.update_from_msg (로버 반경 부풀림, slope_cost)
  → path_planner.astar(cost=…) → simplify_path(cost=…)
그리고 로버가 실제로 달리는 직선 구간(경유점 사이)을 따라 몸 길이(0.8 m)로 고른 경사를 잰다.

꺼진 곳(pocket_check 와 같은 정의) 마다 바닥에서 바깥 6~12 m 의 점 8 곳으로 계획한다.

    python3 isaac_drive/scripts/slope_route_check.py [--weight 4] [--cell 0.1] [terrain ...|all]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from types import SimpleNamespace

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pocket_check as pc  # noqa: E402

NAV = os.path.join(pc.REPO, "isaac_drive", "isaac_drive", "navigation")
sys.path.insert(0, os.path.dirname(NAV))
from navigation.obstacle_grid import ObstacleGrid  # noqa: E402
from navigation.online_map import OnlineObstacleLayer  # noqa: E402
from navigation.path_planner import astar, simplify_path  # noqa: E402


def hazard_msg(hm, res, size):
    """hazard_map_node._publish 와 같은 값의 OccupancyGrid 흉내."""
    W, H = size
    from navigation.hazard_map import HazardMap
    hz = HazardMap(map_size=(W, H), cell_size=pc.CELL)
    ii, jj = np.mgrid[0:hm.shape[0], 0:hm.shape[1]]
    pts = np.stack([((jj + 0.5) * res - W / 2).ravel(), ((ii + 0.5) * res - H / 2).ravel(),
                    hm.ravel()], axis=1)
    cells = np.full((hz.rows, hz.cols), -1, dtype=np.int8)
    step = int(round(10.0 / res))                    # 10 m 띠로
    pad = int(round(0.2 / res))
    for r0 in range(0, hm.shape[0], step):
        a, b = max(0, r0 - pad), min(hm.shape[0], r0 + step + pad)
        sl = pts.reshape(hm.shape[0], hm.shape[1], 3)[a:b].reshape(-1, 3)
        i, j, bad, deg = hz.classify(sl, with_grade=True)
        lo, hi = int(round(r0 * res / pc.CELL)), int(round((r0 + step) * res / pc.CELL))
        k = (i >= lo) & (i < hi)
        v = np.where(bad[k], 100, np.clip(np.rint(np.nan_to_num(deg[k])), 0, 49))
        cells[i[k], j[k]] = v.astype(np.int8)
    return SimpleNamespace(info=SimpleNamespace(width=hz.cols, height=hz.rows,
                                                resolution=pc.CELL),
                           data=cells.ravel())


def line_grades(waypoints, body, cell):
    """경유점 사이 직선을 0.05 m 간격으로 따라가며 몸 경사(도)."""
    out = []
    for (x0, y0), (x1, y1) in zip(waypoints[:-1], waypoints[1:]):
        n = max(1, int(math.hypot(x1 - x0, y1 - y0) / 0.05))
        t = np.linspace(0.0, 1.0, n + 1)
        xs, ys = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
        j = np.clip(((xs + 25.0) / cell).astype(int), 0, body.shape[1] - 1)
        i = np.clip(((ys + 25.0) / cell).astype(int), 0, body.shape[0] - 1)
        out.append(body[i, j])
    return np.concatenate(out) if out else np.zeros(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("terrains", nargs="*", default=["all"])
    ap.add_argument("--weight", type=float, default=4.0)
    ap.add_argument("--cell", type=float, default=0.1, help="계획 격자 (coverage 0.1, 접근 0.2)")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    tids = a.terrains
    if tids == ["all"]:
        tids = sorted(d for d in os.listdir(pc.TERRAINS) if d.startswith("terrain_"))
    rows, times = [], {"off": [], "on": []}
    for tid in tids:
        meta = json.load(open(os.path.join(pc.TERRAINS, tid, "meta.json")))
        res, size = float(meta["resolution_m"]), [float(v) for v in meta["size_m"]]
        hm = np.load(os.path.join(pc.TERRAINS, tid, "heightmap.npy")).astype(np.float64)
        og = np.load(os.path.join(pc.TERRAINS, tid, "obstacle_grid.npy")) > 0
        f = int(round(pc.CELL / res))
        gy, gx = np.gradient(hm, res)
        sf = np.degrees(np.arctan(np.hypot(gx, gy)))
        solid = ndimage.binary_opening(og & (ndimage.uniform_filter(sf, 9) < 18.0), iterations=2)
        hs = ndimage.uniform_filter(hm, int(round(pc.BODY_M / res)))
        sy, sx = np.gradient(hs, res)
        body = pc.down(np.degrees(np.arctan(np.hypot(sx, sy))), f, np.max)
        msg = hazard_msg(hm, res, size)
        d = np.asarray(msg.data).reshape(msg.info.height, msg.info.width).copy()
        d[pc.down(solid, f, np.max)[:d.shape[0], :d.shape[1]]] = 100     # 바위·구조물
        msg.data = d.ravel()
        bc = meta.get("basecamp") or {}
        keep = [(bc["center"]["x"], bc["center"]["y"], bc["radius"])] if bc.get("radius") else []
        layers = {}
        for name, w in (("off", 0.0), ("on", a.weight)):
            og_ = ObstacleGrid(map_size=size, cell_size=a.cell, robot_radius=pc.ROBOT_R)
            lay = OnlineObstacleLayer(og_, keep, slope_cost=(w, 15.0, 25.0))
            lay.update_from_msg(msg)
            layers[name] = (og_, lay)
        og0 = layers["off"][0]
        # 꺼진 곳 (0.1 m 높이 격자에서 찾고 계획 격자로 옮김)
        hcell = pc.down(hm, f, np.mean)
        hsm = ndimage.uniform_filter(hcell, 5)
        seed = hsm.copy()
        seed[1:-1, 1:-1] = hsm.max()
        depth = pc._fill(seed, hsm) - hsm
        k = int(round(a.cell / pc.CELL))
        depth_p = pc.down(depth, k, np.max) if k > 1 else depth
        free = og0.grid == 0
        lab, nd = ndimage.label((depth_p > 0.3) & free, structure=np.ones((3, 3)))
        rng = np.random.default_rng(0)
        outside = np.argwhere((depth_p < 0.05) & free)
        for q in range(1, nd + 1):
            msk = lab == q
            if msk.sum() * a.cell ** 2 < 4.0:
                continue
            s = np.unravel_index(int(np.argmax(np.where(msk, depth_p, -1))), msk.shape)
            dist = np.hypot(outside[:, 0] - s[0], outside[:, 1] - s[1]) * a.cell
            cand = outside[(dist > 6) & (dist < 12)]
            if not len(cand):
                continue
            for g in cand[rng.choice(len(cand), min(8, len(cand)), replace=False)]:
                g = (int(g[0]), int(g[1]))
                res_ = {}
                for name in ("off", "on"):
                    og_, lay = layers[name]
                    t0 = time.perf_counter()
                    cells = astar(og_.grid, s, g, cost=lay.cost)
                    if cells is None:
                        break
                    cells = simplify_path(og_.grid, cells, max_skip=20, cost=lay.cost)
                    times[name].append(time.perf_counter() - t0)
                    wps = [og_.cell_to_world(i, j) for i, j in cells]
                    gr = line_grades(wps, body, pc.CELL)
                    L = sum(math.hypot(x1 - x0, y1 - y0)
                            for (x0, y0), (x1, y1) in zip(wps[:-1], wps[1:]))
                    res_[name] = (float(gr.max()), float(np.mean(gr > 20.0)), L, len(wps))
                if len(res_) == 2:
                    rows.append({"terrain": tid, "pit": q, **{f"{n}_{key}": res_[n][i]
                                 for n in res_ for i, key in enumerate(
                                     ("max_deg", "frac_over20", "len_m", "waypoints"))}})
        print(f"{tid}: 경로 {sum(r['terrain'] == tid for r in rows)}개", flush=True)
    if not rows:
        print("경로 없음")
        return
    A = {k: np.array([r[k] for r in rows]) for k in rows[0] if k not in ("terrain", "pit")}
    pits = len({(r["terrain"], r["pit"]) for r in rows})
    print(f"\n꺼진 곳 {pits}곳, 경로 {len(rows)}개 (계획 격자 {a.cell} m, 경사 비용 ×{a.weight:g})")
    for n, label in (("off", "균일 비용 (지금)"), ("on", "경사 비용")):
        mx = A[f"{n}_max_deg"]
        print(f"  {label}: 지나는 최대 경사 중앙 {np.median(mx):.1f}°, 90 % {np.percentile(mx, 90):.1f}°"
              f" · 20° 넘는 곳을 지나는 경로 {np.mean(mx > 20) * 100:.0f} %"
              f" · 23° 넘는 {np.mean(mx > 23) * 100:.0f} %"
              f" · 길이 중앙 {np.median(A[f'{n}_len_m']):.1f} m"
              f" · 계획 시간 중앙 {np.median(times[n]) * 1000:.0f} ms (최대 {max(times[n]) * 1000:.0f})")
    ratio = A["on_len_m"] / np.maximum(A["off_len_m"], 1e-6)
    print(f"  길이 비 (경사/균일) 중앙 {np.median(ratio):.2f}, 90 % {np.percentile(ratio, 90):.2f}, "
          f"최대 {ratio.max():.2f}")
    worse = np.sum(A["on_max_deg"] > A["off_max_deg"] + 0.5)
    print(f"  경사 비용 쪽 최대 경사가 0.5° 넘게 더 큰 경로 {worse}개")
    if a.out:
        json.dump(rows, open(a.out, "w"), ensure_ascii=False)


if __name__ == "__main__":
    main()
