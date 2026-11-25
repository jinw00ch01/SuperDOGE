"""지형 스캐너(360° 거리 센서)의 점으로 만드는 높이 지도.

ROS 에 의존하지 않는 순수 numpy 모듈 — 노드(hazard_map_node)와 오프라인 평가가
같은 코드를 쓴다. 좌표 규약은 HazardMap / ObstacleGrid 와 같다:
grid[i, j] = [y 행, x 열], world 원점이 지도 중심.

하는 일
-------
  1. world 점을 칸(0.1 m)에 모아 칸마다 높이를 평균으로 쌓는다 (최근 max_count 점의
     가중 평균 — 지형은 변하지 않으므로 오래 쌓을수록 잡음이 준다).
  2. 칸 주변 (2·win+1)² 칸에 평면을 맞춰 기울기와, 그 평면에서 벗어난 정도를 구한다.
       · 기울기 > slope_deg            → 급경사
       · 평면에서 step_m 넘게 솟음/꺼짐 → 바위·턱·구덩이 가장자리
  3. 로버 위치 기준의 요약(앞·뒤·좌·우 높이 차, 주변 최고·최저)을 낸다.

깊이 카메라 쪽(HazardMap)은 자세 오차 때문에 영상마다 따로 판정하지만, 스캐너는
한 번에 3600 점뿐이라 한 장으로는 칸을 채우지 못한다 — 여러 장을 높이로 합친다.
합쳐도 되는지는 정답 높이 지도와 비교해 확인했다 (README '주변 높낮이').
"""
from __future__ import annotations

import math

import numpy as np
from scipy import ndimage


class ElevationMap:
    def __init__(self, map_size=(50.0, 50.0), cell_size: float = 0.1,
                 max_count: int = 40, slope_deg: float = 25.0, step_m: float = 0.10,
                 win: int = 2, min_support: int = 10, min_spread: float = 0.35,
                 max_range: float = 9.5, self_radius: float = 1.0, self_above: float = -0.25):
        self.map_w, self.map_h = float(map_size[0]), float(map_size[1])
        self.cell = float(cell_size)
        self.cols = int(round(self.map_w / self.cell))
        self.rows = int(round(self.map_h / self.cell))
        self.max_count = int(max_count)
        self.slope_tan = math.tan(math.radians(slope_deg))
        self.step_m = float(step_m)
        self.win = int(win)
        self.min_support = int(min_support)
        self.min_spread = float(min_spread)
        self.max_range = float(max_range)
        # 몸체에서 self_radius 안이면서 몸체 원점 − 0.25 m 보다 높은 점은 버린다:
        # 시뮬레이터가 로버 자기 몸은 걸러 주지만 집게가 든 광물은 로버의 일부가 아니다.
        self.self_radius = float(self_radius)
        self.self_above = float(self_above)
        self.z = np.full((self.rows, self.cols), np.nan, dtype=np.float32)
        self.n = np.zeros((self.rows, self.cols), dtype=np.uint16)
        self.scans = 0
        self._cache = None            # (scans, slope, resid, support) — 지도가 바뀔 때까지 재사용

    def world_to_cell(self, x, y):
        j = np.floor((np.asarray(x) + self.map_w / 2.0) / self.cell).astype(np.int64)
        i = np.floor((np.asarray(y) + self.map_h / 2.0) / self.cell).astype(np.int64)
        return i, j

    def clear(self) -> None:
        self.z[:] = np.nan
        self.n[:] = 0
        self._cache = None

    # ── 스캔 한 장 ──────────────────────────────────────────────────────
    def integrate(self, pts_body: np.ndarray, p: np.ndarray, R: np.ndarray) -> int:
        """몸체 좌표의 점 (N, 3) 을 그 시각의 자세 (p, R) 로 world 로 옮겨 쌓는다."""
        if pts_body.size == 0:
            return 0
        pts = np.asarray(pts_body, dtype=np.float64)
        pts = pts[np.linalg.norm(pts, axis=1) < self.max_range]
        pts = pts[~((np.hypot(pts[:, 0], pts[:, 1]) < self.self_radius)
                    & (pts[:, 2] > self.self_above))]
        w = pts @ R.T + p
        i, j = self.world_to_cell(w[:, 0], w[:, 1])
        ok = (i >= 0) & (i < self.rows) & (j >= 0) & (j < self.cols)
        if not ok.any():
            return 0
        flat = i[ok] * self.cols + j[ok]
        cells, inv, cnt = np.unique(flat, return_inverse=True, return_counts=True)
        zs = np.bincount(inv, weights=w[ok, 2]) / cnt
        zf, nf = self.z.reshape(-1), self.n.reshape(-1)
        n_old = nf[cells].astype(np.float64)
        z_old = np.where(n_old > 0, zf[cells], 0.0)
        zf[cells] = ((z_old * n_old + zs * cnt) / (n_old + cnt)).astype(np.float32)
        nf[cells] = np.minimum(n_old + cnt, self.max_count).astype(np.uint16)
        self.scans += 1
        return int(len(cells))

    # ── 기울기·턱 ───────────────────────────────────────────────────────
    def _fit(self):
        if self._cache is not None and self._cache[0] == self.scans:
            return self._cache[1:]
        valid = self.n > 0
        if not valid.any():
            z0 = np.zeros(self.z.shape, np.float32)
            self._cache = (self.scans, z0, z0, z0.astype(np.int32))
            return self._cache[1:]
        # 본 영역만 잘라서 계산.
        ii, jj = np.where(valid)
        m = self.win + 1
        i0, i1 = max(ii.min() - m, 0), min(ii.max() + m + 1, self.rows)
        j0, j1 = max(jj.min() - m, 0), min(jj.max() + m + 1, self.cols)
        v = valid[i0:i1, j0:j1].astype(np.float64)
        # 평균을 빼 두면 수치가 안정된다. 안 본 칸은 0 이어야 한다 (합에 끼지 않게).
        zc = np.where(valid[i0:i1, j0:j1],
                      self.z[i0:i1, j0:j1] - float(np.mean(self.z[valid])), 0.0).astype(np.float64)
        Y, X = np.mgrid[0:v.shape[0], 0:v.shape[1]].astype(np.float64) * self.cell
        k = 2 * self.win + 1

        def box(a):
            return ndimage.uniform_filter(a, size=k, mode="constant") * (k * k)

        S1 = box(v)
        with np.errstate(invalid="ignore", divide="ignore"):
            mx, my, mz = box(v * X) / S1, box(v * Y) / S1, box(zc) / S1
            cxx = box(v * X * X) / S1 - mx * mx
            cyy = box(v * Y * Y) / S1 - my * my
            cxy = box(v * X * Y) / S1 - mx * my
            cxz = box(zc * X) / S1 - mx * mz
            cyz = box(zc * Y) / S1 - my * mz
            det = cxx * cyy - cxy * cxy
            a = (cxz * cyy - cyz * cxy) / det
            b = (cyz * cxx - cxz * cxy) / det
            pred = mz + a * (X - mx) + b * (Y - my)
            # 점들이 퍼진 정도 — xy 공분산의 작은 쪽 고유값. 먼 곳은 고리 한 줄만 걸려서
            # 점이 거의 일직선이고, 그러면 줄에 수직인 기울기는 잡음(1 cm)이 정한다
            # (실측: 정답 7° 인 곳을 41° 로 봄). 창 전체에 고르게 있으면 (창 폭)²/12.
            lam = 0.5 * (cxx + cyy) - np.sqrt((0.5 * (cxx - cyy)) ** 2 + cxy * cxy)
        full = ((2 * self.win + 1) * self.cell) ** 2 / 12.0
        good = (S1 >= self.min_support) & (lam > self.min_spread * full) & (v > 0)
        slope = np.zeros(self.z.shape, np.float32)
        resid = np.zeros(self.z.shape, np.float32)
        supp = np.zeros(self.z.shape, np.int32)
        slope[i0:i1, j0:j1] = np.where(good, np.hypot(a, b), 0.0)
        resid[i0:i1, j0:j1] = np.where(good, zc - pred, 0.0)
        supp[i0:i1, j0:j1] = np.where(good, S1, 0).astype(np.int32)
        self._cache = (self.scans, slope, resid, supp)
        return slope, resid, supp

    def known_mask(self) -> np.ndarray:
        """기울기까지 판단할 수 있는 칸."""
        return self._fit()[2] > 0

    def hazard_mask(self) -> np.ndarray:
        slope, resid, supp = self._fit()
        return (supp > 0) & ((slope > self.slope_tan) | (np.abs(resid) > self.step_m))

    def slope_deg(self) -> np.ndarray:
        return np.degrees(np.arctan(self._fit()[0]))

    # ── 로버 주변 요약 ──────────────────────────────────────────────────
    def height_at(self, x: float, y: float, radius: float = 0.3):
        """(x, y) 둘레 radius 안의 평균 높이. 본 칸이 없으면 None."""
        r = int(math.ceil(radius / self.cell))
        i, j = self.world_to_cell(x, y)
        i, j = int(i), int(j)
        a = self.z[max(i - r, 0):i + r + 1, max(j - r, 0):j + r + 1]
        return float(np.nanmean(a)) if a.size and np.isfinite(a).any() else None

    def around(self, x: float, y: float, yaw: float, ground_z: float,
               dist: float = 2.0, radius: float = 4.0) -> dict:
        """로버 기준 요약: 앞·뒤·좌·우 dist m 지점의 (높이 − 발밑), 주변 radius 의 최고·최저."""
        out = {}
        c, s = math.cos(yaw), math.sin(yaw)
        for name, (dx, dy) in (("front", (c, s)), ("back", (-c, -s)),
                               ("left", (-s, c)), ("right", (s, -c))):
            h = self.height_at(x + dist * dx, y + dist * dy)
            out[name] = None if h is None else round(h - ground_z, 2)
        r = int(round(radius / self.cell))
        i, j = self.world_to_cell(x, y)
        i, j = int(i), int(j)
        i0, j0 = max(i - r, 0), max(j - r, 0)
        a = self.z[i0:i + r + 1, j0:j + r + 1]
        if a.size and np.isfinite(a).any():
            yy, xx = np.mgrid[0:a.shape[0], 0:a.shape[1]]
            disk = np.hypot(yy - (i - i0), xx - (j - j0)) <= r
            vals = np.where(disk & np.isfinite(a), a, np.nan)
            out["hi"] = round(float(np.nanmax(vals)) - ground_z, 2)
            out["lo"] = round(float(np.nanmin(vals)) - ground_z, 2)
            out["known_pct"] = round(100.0 * float(np.isfinite(vals).sum()) / max(1, int(disk.sum())), 0)
        return out
