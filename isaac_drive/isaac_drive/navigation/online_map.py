"""주행 중에 받은 위험 지도(nav_msgs/OccupancyGrid)를 경로 계획용 격자로 바꾼다.

미리 만든 obstacle_grid.npy 를 읽는 대신 (terrain_loader.load_terrain), 빈 격자로
시작해서 hazard_map_node 가 보내는 지도가 올 때마다 ObstacleGrid.grid 를 갈아
끼운다. 안 본 칸은 "지나갈 수 있다" 로 둔다 (보이면 다시 계획).

사전에 아는 것으로 남긴 것
  · 운용 구역의 크기 (meta.json 의 size_m) — 격자의 범위. 지형 정보가 아니다.
  · 기지(착륙선) 위치와 접근 금지 반경 — 로버가 출발한 곳이므로 안다.
"""
from __future__ import annotations

import json
import os

import numpy as np
from scipy import ndimage

from .fog_map import FogMap
from .obstacle_grid import ObstacleGrid


def blank_terrain(terrain_dir, cell_size=0.1, robot_radius=0.8,
                  reveal_radius=2.0, grid_n=3):
    """load_terrain 과 같은 (meta, ogrid, fog) 를 돌려주되 obstacle_grid.npy 는
    읽지 않는다. 격자는 비어 있고 (외곽만 막힘), OnlineObstacleLayer 가 채운다."""
    with open(os.path.join(terrain_dir, "meta.json")) as f:
        meta = json.load(f)
    W, H = float(meta["size_m"][0]), float(meta["size_m"][1])
    ogrid = ObstacleGrid(map_size=(W, H), cell_size=cell_size,
                         robot_radius=robot_radius)
    fog = FogMap(map_size=(W, H), cell_size=ogrid.cell_size,
                 reveal_radius=reveal_radius, grid_n=grid_n)
    print(f"[online_map] {meta.get('terrain_id', '?')}: {W:.0f}×{H:.0f}m 빈 격자 "
          f"{ogrid.rows}×{ogrid.cols} (cell={ogrid.cell_size:.2f}m) — "
          f"obstacle_grid.npy 사용 안 함")
    return meta, ogrid, fog


def basecamp_keepout(meta: dict):
    """meta 의 기지 위치·반경 → [(x, y, r)]. 없으면 []."""
    bc = (meta or {}).get("basecamp") or {}
    c = bc.get("center") or {}
    if "x" in c and "y" in c and bc.get("radius"):
        return [(float(c["x"]), float(c["y"]), float(bc["radius"]))]
    return []


class OnlineObstacleLayer:
    def __init__(self, ogrid: ObstacleGrid, keepouts=(), check_margin_m: float = 0.2,
                 unknown_margin_m: float = 0.8, slope_cost=(0.0, 15.0, 25.0)):
        """keepouts: [(x, y, radius)] — 처음부터 막아 둘 원 (기지).
        check_margin_m: 이미 가고 있는 경로가 막혔는지 볼 때는 로버 반경보다 이만큼
        덜 부풀린 격자를 쓴다 (경계에서 계획이 계속 바뀌는 것을 막는 여유).
        slope_cost: (weight, 시작 경사°, 끝 경사°). weight > 0 이면 위험 지도가 실어 보낸
        안전 칸의 경사로 칸 비용을 만든다 — 시작 경사 아래는 0, 끝 경사에서 weight
        (그 칸을 지나는 거리가 1 + weight 배). self.cost 로 A* 에 넘긴다. weight 0 = 끔
        (self.cost = None, 예전과 같은 균일 비용)."""
        self.ogrid = ogrid
        self.slope_w, self.slope_lo, self.slope_hi = (float(v) for v in slope_cost)
        self.cost = None
        self.check_margin = float(check_margin_m)
        # 위험 칸에 붙어 있는 "아직 못 본 칸" 은 이 거리까지 위험으로 친다. 장애물은
        # 보인 면보다 더 넓을 수 있다 (둔덕 뒤에 가려진 건물 밑동, 바위의 뒷면).
        # 못 본 칸을 전부 막으면 탐색을 못 하므로 위험 칸 바로 옆만 막는다. 0 = 끔.
        self.unknown_margin = float(unknown_margin_m)
        rows, cols, cs = ogrid.rows, ogrid.cols, ogrid.cell_size
        ii, jj = np.mgrid[0:rows, 0:cols]
        xs = (jj + 0.5) * cs - ogrid.map_w / 2.0
        ys = (ii + 0.5) * cs - ogrid.map_h / 2.0
        self._known = np.zeros((rows, cols), dtype=bool)      # 사전에 아는 장애물
        for (x, y, r) in keepouts:
            self._known |= (xs - x) ** 2 + (ys - y) ** 2 <= r * r
        m = max(1, int(round(ogrid.robot_radius / cs)))
        self._border = np.zeros((rows, cols), dtype=bool)
        self._border[:m, :] = self._border[-m:, :] = True
        self._border[:, :m] = self._border[:, -m:] = True
        self.raw = self._known.copy()            # 부풀리기 전 (planner 해상도)
        self._ext = np.zeros((rows, cols), dtype=bool)   # raw 중 "못 본 칸" 이라 막은 것
        self.check_grid = np.zeros((rows, cols), dtype=np.uint8)
        self.updates = 0
        self._rebuild()

    def _rebuild(self) -> None:
        cs, r = self.ogrid.cell_size, self.ogrid.robot_radius
        if self.raw.any():
            dist = ndimage.distance_transform_edt(~self.raw) * cs
        else:
            dist = np.full(self.raw.shape, np.inf)
        self.ogrid.grid = ((dist <= r) | self._border).astype(np.uint8)
        self.check_grid = ((dist <= max(cs, r - self.check_margin))
                           | self._border).astype(np.uint8)

    def update_from_msg(self, msg) -> np.ndarray:
        """nav_msgs/OccupancyGrid → 격자 갱신. return: planner 해상도의 raw 위험 mask."""
        w, h, res = msg.info.width, msg.info.height, float(msg.info.resolution)
        data = np.asarray(msg.data, dtype=np.int8).reshape(h, w)
        grade = None
        if self.slope_w > 0.0:
            grade = np.where((data >= 0) & (data <= 50), data, 0).astype(np.float32)
        return self.update(data > 50, res, unknown=data < 0, grade=grade)

    def update(self, hazard: np.ndarray, cell: float, unknown=None, grade=None) -> np.ndarray:
        rows, cols = self.ogrid.rows, self.ogrid.cols
        f = max(1, int(round(self.ogrid.cell_size / cell)))

        def to_planner(a, pool):
            if f > 1:
                R2, C2 = a.shape[0] // f, a.shape[1] // f
                a = pool(a[:R2 * f, :C2 * f].reshape(R2, f, C2, f), axis=(1, 3))
            if a.shape != (rows, cols):
                fixed = np.zeros((rows, cols), dtype=a.dtype)
                r, c = min(rows, a.shape[0]), min(cols, a.shape[1])
                fixed[:r, :c] = a[:r, :c]
                a = fixed
            return a

        hazard = to_planner(hazard, np.max)
        self._ext = np.zeros((rows, cols), dtype=bool)
        if unknown is not None and self.unknown_margin > 0.0 and hazard.any():
            unknown = to_planner(unknown, np.min)        # 칸 전체를 못 본 경우만
            near = ndimage.distance_transform_edt(~hazard) * self.ogrid.cell_size \
                <= self.unknown_margin
            self._ext = unknown & near & ~hazard
        self.raw = hazard | self._known | self._ext
        if grade is not None and self.slope_w > 0.0:
            g = to_planner(grade, np.max)                 # 칸 안에서 가장 가파른 값
            span = max(self.slope_hi - self.slope_lo, 1e-3)
            self.cost = (self.slope_w * np.clip((g - self.slope_lo) / span, 0.0, 1.0)
                         ).astype(np.float32)
        self._rebuild()
        self.updates += 1
        return self.raw

    # ── 다가가야 하는 대상 (광물) ───────────────────────────────────────
    def grid_excluding(self, x: float, y: float, radius: float) -> np.ndarray:
        """(x, y) 주변 radius 안의 위험 칸을 없는 것으로 치고 부풀린 계획용 격자.

        광물은 깊이 영상에서 '바닥에서 튀어나온 것' 이라 위험 칸으로 잡힌다. 피해 갈
        때는 그게 맞지만, 그 광물을 집으러 갈 때는 목표 칸 자체가 막혀 있으면 안
        된다 (막혀 있으면 계획이 로버 반경만큼 떨어진 곳에서 끝난다). 사전에 아는
        장애물(기지)은 빼지 않는다. 뺄 것이 없으면 원래 격자를 그대로 돌려준다.
        """
        og = self.ogrid
        cs, r = og.cell_size, og.robot_radius
        ci, cj = og.world_to_cell(x, y, clip=True)
        k = int(np.ceil(radius / cs))
        reach = k + int(np.ceil((r + self.unknown_margin) / cs)) + 1   # 뺀 칸의 부풀림이 닿던 범위
        pad = int(np.ceil(r / cs)) + 1                # 창 밖 장애물의 부풀림이 들어오는 폭
        rows, cols = self.raw.shape
        a0, a1 = max(0, ci - reach - pad), min(rows, ci + reach + pad + 1)
        b0, b1 = max(0, cj - reach - pad), min(cols, cj + reach + pad + 1)
        ii, jj = np.mgrid[a0:a1, b0:b1]
        d2 = (ii - ci) ** 2 + (jj - cj) ** 2
        near = (d2 <= k * k) & ~self._known[a0:a1, b0:b1]
        # 광물 뒤 그늘(못 본 칸)을 막아 둔 것도 목표 주변에서는 푼다.
        ke = k + int(np.ceil(self.unknown_margin / cs))
        near |= (d2 <= ke * ke) & self._ext[a0:a1, b0:b1]
        raw = self.raw[a0:a1, b0:b1]
        if not (raw & near).any():
            return og.grid
        raw = raw & ~near
        if raw.any():
            sub = ndimage.distance_transform_edt(~raw) * cs <= r
        else:
            sub = np.zeros(raw.shape, dtype=bool)
        g = og.grid.copy()
        # 창 가장자리 pad 폭은 창 밖 장애물을 못 봐서 믿을 수 없으므로 안쪽만 바꾼다.
        i0, i1 = max(a0, ci - reach), min(a1, ci + reach + 1)
        j0, j1 = max(b0, cj - reach), min(b1, cj + reach + 1)
        g[i0:i1, j0:j1] = (sub[i0 - a0:i1 - a0, j0 - b0:j1 - b0]
                           | self._border[i0:i1, j0:j1]).astype(np.uint8)
        return g

    # ── 경로 점검 ───────────────────────────────────────────────────────
    def path_blocked(self, points, ignore=None) -> bool:
        """points[0] = 로버 현재 위치, 그 뒤 = 남은 경유점. 경로가 새로 알게 된
        장애물을 지나가면 True. 로버가 이미 부풀린 영역 안에 있을 때는 거기서
        빠져나오는 첫 구간(최대 1.5 m)은 막힌 것으로 치지 않는다.
        ignore=(x, y, r): 그 원 안의 구간은 보지 않는다 (다가가는 대상 주변)."""
        if len(points) < 2:
            return False
        g, og = self.check_grid, self.ogrid
        step = og.cell_size * 0.5
        travelled = 0.0
        escaping = True
        for (x0, y0), (x1, y1) in zip(points[:-1], points[1:]):
            seg = float(np.hypot(x1 - x0, y1 - y0))
            n = max(1, int(np.ceil(seg / step)))
            for k in range(n + 1):
                t = k / n
                px, py = x0 + (x1 - x0) * t, y0 + (y1 - y0) * t
                i, j = og.world_to_cell(px, py, clip=True)
                d = travelled + seg * t
                if ignore is not None and \
                        (px - ignore[0]) ** 2 + (py - ignore[1]) ** 2 <= ignore[2] ** 2:
                    continue
                if g[i, j]:
                    if not (escaping and d <= 1.5):
                        return True
                else:
                    escaping = False
            travelled += seg
        return False

    def nearest_free(self, cell, max_radius_m: float = 3.0):
        """막힌 칸에서 가장 가까운 빈 칸 (없으면 None)."""
        g = self.ogrid.grid
        i0, j0 = cell
        r = int(np.ceil(max_radius_m / self.ogrid.cell_size))
        a0, a1 = max(0, i0 - r), min(g.shape[0], i0 + r + 1)
        b0, b1 = max(0, j0 - r), min(g.shape[1], j0 + r + 1)
        fi, fj = np.nonzero(g[a0:a1, b0:b1] == 0)
        if len(fi) == 0:
            return None
        d2 = (fi + a0 - i0) ** 2 + (fj + b0 - j0) ** 2
        k = int(np.argmin(d2))
        if d2[k] > r * r:
            return None
        return int(fi[k] + a0), int(fj[k] + b0)
