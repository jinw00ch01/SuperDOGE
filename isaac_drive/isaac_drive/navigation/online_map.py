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
    def __init__(self, ogrid: ObstacleGrid, keepouts=(), check_margin_m: float = 0.2):
        """keepouts: [(x, y, radius)] — 처음부터 막아 둘 원 (기지).
        check_margin_m: 이미 가고 있는 경로가 막혔는지 볼 때는 로버 반경보다 이만큼
        덜 부풀린 격자를 쓴다 (경계에서 계획이 계속 바뀌는 것을 막는 여유)."""
        self.ogrid = ogrid
        self.check_margin = float(check_margin_m)
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
        hazard = (np.asarray(msg.data, dtype=np.int8).reshape(h, w) > 50)
        return self.update(hazard, res)

    def update(self, hazard: np.ndarray, cell: float) -> np.ndarray:
        rows, cols = self.ogrid.rows, self.ogrid.cols
        f = max(1, int(round(self.ogrid.cell_size / cell)))
        if f > 1:
            R2, C2 = hazard.shape[0] // f, hazard.shape[1] // f
            hazard = hazard[:R2 * f, :C2 * f].reshape(R2, f, C2, f).max(axis=(1, 3))
        if hazard.shape != (rows, cols):
            fixed = np.zeros((rows, cols), dtype=bool)
            r, c = min(rows, hazard.shape[0]), min(cols, hazard.shape[1])
            fixed[:r, :c] = hazard[:r, :c]
            hazard = fixed
        self.raw = hazard | self._known
        self._rebuild()
        self.updates += 1
        return self.raw

    # ── 경로 점검 ───────────────────────────────────────────────────────
    def path_blocked(self, points) -> bool:
        """points[0] = 로버 현재 위치, 그 뒤 = 남은 경유점. 경로가 새로 알게 된
        장애물을 지나가면 True. 로버가 이미 부풀린 영역 안에 있을 때는 거기서
        빠져나오는 첫 구간(최대 1.5 m)은 막힌 것으로 치지 않는다."""
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
                i, j = og.world_to_cell(x0 + (x1 - x0) * t, y0 + (y1 - y0) * t, clip=True)
                d = travelled + seg * t
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
