"""지도상 갇힌 저지대 찾기 — 자율주행이 '못 올라온다' 가 접지력 탓인지 계획 탓인지 가른다.

시뮬레이터 없이 돈다 (numpy·scipy). 지형의 정답 높이(heightmap.npy)를 깊이 카메라가 빠짐없이
본 것처럼 점으로 바꿔 hazard_map_node 와 **같은 판정 코드**(navigation/hazard_map.py
HazardMap.classify: 0.3 m 평면 기울기 25°, 칸 안 단차 12 cm, 거칠기 4 cm)에 넣고, 바위·구조물은
정답 지도(obstacle_grid.npy)에서 가져온 뒤, 경로 계획과 같이 로버 반경 0.8 m 만큼 부풀린다.
그 격자에서 큰 영역과 이어지지 않는 빈 칸 덩어리 = 계획기가 "나갈 길이 없다" 고 보는 곳이다.

각 덩어리에 대해
  · exit_plan_deg : 경사 문턱을 몇 도까지 올려야 큰 영역과 이어지는가 (계획기 기준 출구 경사)
  · exit_body_deg : 로버 길이(0.8 m)로 고른 지형에서, 나가는 길 중 가장 완만한 길의 최대 경사
                    (실제로 밀고 올라가야 하는 경사 — 등판 한계 atan 0.52 = 27.5° 와 비교)
  · inflate_only  : 부풀리지 않으면 이어지는가 (좁은 출구가 로버 반경 때문에 막힌 경우)
exit_body_deg 가 27° 아래인데 덩어리로 잡히면 "몸은 나갈 수 있는데 지도가 막은" 곳이다.

그리고 움푹 꺼진 곳(높이를 채우면 0.3 m 넘게 차는 곳, 넓이 4 m² 이상)마다
  · exit_deg : 계획기가 지나가게 두는 칸만 밟아서 나가는 길 중 가장 완만한 길의 최대 경사
               (몸 길이 0.8 m 로 고른 경사). 이만큼은 비탈을 밀고 올라가야 저지대를 벗어난다.
README 의 등판 측정(서 있다 출발: 20° 67~78 %, 23° 0~33 %, 달려와서: 23~25° 100 %)과 맞대 본다.

    python3 isaac_drive/scripts/pocket_check.py terrain_00015 [terrain_00026 ...]
    python3 isaac_drive/scripts/pocket_check.py all
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(REPO, "isaac_drive", "isaac_drive", "navigation"))
from hazard_map import HazardMap  # noqa: E402

TERRAINS = os.path.join(REPO, "isaac_sim", "assets", "generated_terrains")
CELL = 0.1
ROBOT_R = 0.8          # coverage_node robot_radius, mission_manager path_robot_radius
BODY_M = 0.8           # 경사를 "몸으로" 잴 때 고르는 길이
MIN_POCKET_M2 = 2.0


def hazard_from_heightmap(hm, res, size, slope_deg=25.0):
    """정답 높이를 빠짐없이 본 깊이 점으로 보고 HazardMap.classify 로 판정 → (rows, cols) bool."""
    W, H = size
    hz = HazardMap(map_size=(W, H), cell_size=CELL, slope_deg=slope_deg)
    ii, jj = np.mgrid[0:hm.shape[0], 0:hm.shape[1]]
    x = (jj + 0.5) * res - W / 2.0
    y = (ii + 0.5) * res - H / 2.0
    pts = np.stack([x.ravel(), y.ravel(), hm.ravel()], axis=1)
    out = np.zeros((hz.rows, hz.cols), dtype=bool)
    seen = np.zeros_like(out)
    # 메모리 때문에 띠로 나눠 판정 (경계 한 줄은 겹치게)
    band = 100                                   # 셀 행
    step_rows = int(round(band * CELL / res))
    pad = int(round(2 * CELL / res))
    for r0 in range(0, hm.shape[0], step_rows):
        a, b = max(0, r0 - pad), min(hm.shape[0], r0 + step_rows + pad)
        sl = pts.reshape(hm.shape[0], hm.shape[1], 3)[a:b].reshape(-1, 3)
        i, j, bad = hz.classify(sl)
        lo = int(math.floor((r0 * res) / CELL))
        hi = int(math.ceil(((r0 + step_rows) * res) / CELL))
        keep = (i >= lo) & (i < hi)
        out[i[keep], j[keep]] = bad[keep]
        seen[i[keep], j[keep]] = True
    return out, seen


def down(a, f, how):
    R, C = a.shape[0] // f, a.shape[1] // f
    a = a[:R * f, :C * f].reshape(R, f, C, f)
    return how(a, axis=(1, 3))


def analyse(tid: str) -> dict:
    tdir = os.path.join(TERRAINS, tid)
    meta = json.load(open(os.path.join(tdir, "meta.json")))
    res = float(meta["resolution_m"])
    size = [float(v) for v in meta["size_m"]]
    hm = np.load(os.path.join(tdir, "heightmap.npy")).astype(np.float64)
    og = np.load(os.path.join(tdir, "obstacle_grid.npy")) > 0
    f = int(round(CELL / res))
    # 바위·구조물: 정답 지도의 막힌 칸 중 완만한 곳 (drive_env._load_maps 와 같은 규칙)
    gy, gx = np.gradient(hm, res)
    slope_fine = np.degrees(np.arctan(np.hypot(gx, gy)))
    solid = og & (ndimage.uniform_filter(slope_fine, 9) < 18.0)
    solid = ndimage.binary_opening(solid, iterations=2)
    solid = down(solid, f, np.max)
    # 몸 길이로 고른 경사 (0.1 m 칸)
    hs = ndimage.uniform_filter(hm, int(round(BODY_M / res)))
    sy, sx = np.gradient(hs, res)
    body = down(np.degrees(np.arctan(np.hypot(sx, sy))), f, np.max)
    rows, cols = body.shape
    border = np.zeros((rows, cols), bool)
    m = int(round(ROBOT_R / CELL))
    border[:m, :] = border[-m:, :] = border[:, :m] = border[:, -m:] = True
    # 기지(사전에 아는 접근 금지)
    ii, jj = np.mgrid[0:rows, 0:cols]
    xs, ys = (jj + 0.5) * CELL - size[0] / 2, (ii + 0.5) * CELL - size[1] / 2
    bc = (meta.get("basecamp") or {})
    keep = np.zeros((rows, cols), bool)
    if bc.get("radius"):
        keep = (xs - bc["center"]["x"]) ** 2 + (ys - bc["center"]["y"]) ** 2 <= bc["radius"] ** 2

    def free_grid(slope_deg, inflate=True):
        haz, _ = hazard_from_heightmap(hm, res, size, slope_deg)
        raw = haz[:rows, :cols] | solid | keep
        if inflate:
            blocked = ndimage.distance_transform_edt(~raw) * CELL <= ROBOT_R
        else:
            blocked = raw
        return ~(blocked | border)

    base = free_grid(25.0)
    lab, n = ndimage.label(base, structure=np.ones((3, 3)))
    sizes = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
    main = int(np.argmax(sizes)) + 1
    pockets = [k + 1 for k in range(n)
               if k + 1 != main and sizes[k] * CELL * CELL >= MIN_POCKET_M2]
    # 출구 경사 (계획기 기준): 문턱을 올려 가며 큰 영역과 이어지는지
    exit_plan = {p: None for p in pockets}
    main_pt = np.argwhere(lab == main)[0]
    for th in np.arange(26.0, 45.0, 1.0):
        if all(v is not None for v in exit_plan.values()):
            break
        g = free_grid(float(th))
        l2, _ = ndimage.label(g, structure=np.ones((3, 3)))
        mk = l2[tuple(main_pt)]
        for p in pockets:
            if exit_plan[p] is None:
                cells = np.argwhere((lab == p) & g)
                if len(cells) and np.any(l2[cells[:, 0], cells[:, 1]] == mk):
                    exit_plan[p] = float(th)
    # 부풀리기만 빼면 이어지는가
    g0 = free_grid(25.0, inflate=False)
    l0, _ = ndimage.label(g0, structure=np.ones((3, 3)))
    mk0 = l0[tuple(main_pt)]
    # 몸 기준 출구 경사: 바위·구조물·경계만 막고, 몸 경사의 minimax 경로 (문턱 올려 가며)
    hard = ndimage.distance_transform_edt(~(solid | keep)) * CELL <= ROBOT_R
    hard |= border
    exit_body = {p: None for p in pockets}
    for th in np.arange(10.0, 60.0, 0.5):
        if all(v is not None for v in exit_body.values()):
            break
        g = (body <= th) & ~hard
        l3, _ = ndimage.label(g, structure=np.ones((3, 3)))
        mk = l3[tuple(main_pt)] if g[tuple(main_pt)] else -1
        for p in pockets:
            if exit_body[p] is None:
                cells = np.argwhere((lab == p) & g)
                if len(cells) and mk > 0 and np.any(l3[cells[:, 0], cells[:, 1]] == mk):
                    exit_body[p] = float(th)
    hcell = down(hm, f, np.mean)[:rows, :cols]
    # ── 움푹 꺼진 곳과 그 출구 경사 ─────────────────────────────────────
    hsm = ndimage.uniform_filter(hcell, 5)
    seed = hsm.copy()
    seed[1:-1, 1:-1] = hsm.max()
    filled = _fill(seed, hsm)
    depth = filled - hsm
    dep_lab, nd = ndimage.label((depth > 0.3) & base, structure=np.ones((3, 3)))
    outside = (depth < 0.05) & base
    deps = []
    for k in range(1, nd + 1):
        msk = dep_lab == k
        if msk.sum() * CELL * CELL < 4.0:
            continue
        ex = None
        for th in np.arange(5.0, 45.0, 0.5):
            g = base & (body <= th)
            l4, _ = ndimage.label(g, structure=np.ones((3, 3)))
            ids = np.unique(l4[msk & g])
            ids = ids[ids > 0]
            if len(ids) and np.any(np.isin(l4[outside & g], ids)):
                ex = float(th)
                break
        cy_, cx_ = np.argwhere(msk).mean(axis=0)
        deps.append({"center": [round(float((cx_ + 0.5) * CELL - size[0] / 2), 1),
                                round(float((cy_ + 0.5) * CELL - size[1] / 2), 1)],
                     "area_m2": round(float(msk.sum() * CELL * CELL), 1),
                     "depth_m": round(float(depth[msk].max()), 2), "exit_deg": ex})
    deps.sort(key=lambda d: -d["area_m2"])
    out = []
    for p in pockets:
        msk = lab == p
        rim = ndimage.binary_dilation(msk, iterations=int(1.5 / CELL)) & ~msk
        cy_, cx_ = np.argwhere(msk).mean(axis=0)
        out.append({
            "center": [round(float((cx_ + 0.5) * CELL - size[0] / 2), 1),
                       round(float((cy_ + 0.5) * CELL - size[1] / 2), 1)],
            "area_m2": round(float(msk.sum() * CELL * CELL), 1),
            "depth_m": round(float(np.percentile(hcell[rim], 50) - np.median(hcell[msk])), 2),
            "exit_plan_deg": exit_plan[p],
            "exit_body_deg": exit_body[p],
            "inflate_only": bool(np.any(l0[msk] == mk0)),
        })
    out.sort(key=lambda d: -d["area_m2"])
    free_area = float(base.sum() * CELL * CELL)
    return {"terrain": tid, "free_m2": round(free_area, 0),
            "pocket_m2": round(sum(d["area_m2"] for d in out), 1), "pockets": out,
            "depressions": deps}


def _fill(seed, mask):
    """높이 채우기 (morphological reconstruction by erosion): 바깥으로 물이 빠지는 높이."""
    prev = None
    cur = np.maximum(seed, mask)
    while prev is None or not np.array_equal(cur, prev):
        prev = cur
        cur = np.maximum(ndimage.grey_erosion(cur, size=(3, 3)), mask)
    return cur


def main():
    args = sys.argv[1:] or ["terrain_00015"]
    if args == ["all"]:
        args = sorted(d for d in os.listdir(TERRAINS) if d.startswith("terrain_"))
    for tid in args:
        r = analyse(tid)
        body_ok = [d for d in r["pockets"]
                   if d["exit_body_deg"] is not None and d["exit_body_deg"] < 27.0]
        print(f"{tid}: 빈 땅 {r['free_m2']:.0f} m², 갇힌 덩어리 {len(r['pockets'])}개 "
              f"({r['pocket_m2']} m²), 그중 몸으로는 27° 아래로 나갈 수 있는 것 {len(body_ok)}개",
              flush=True)
        for d in r["pockets"]:
            print("   " + json.dumps(d, ensure_ascii=False), flush=True)
        for d in r["depressions"]:
            print("   꺼진 곳 " + json.dumps(d, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
