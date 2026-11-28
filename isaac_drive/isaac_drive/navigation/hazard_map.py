"""깊이 카메라만으로 주행 중에 만드는 위험 지도 (미리 만든 obstacle_grid 대체).

ROS 에 의존하지 않는 순수 numpy 모듈 — 노드(hazard_map_node)와 오프라인 평가
(녹화한 깊이 영상 ↔ 정답 지도 비교)가 같은 코드를 쓴다.

무엇을 위험으로 보나
--------------------
지형 생성기가 만든 정답 지도(obstacle_grid.npy)는 "경사 25° 이상 + 바위 +
구조물" 을 막힌 칸으로 표시한다. 여기서는 같은 것을 로버가 본 깊이 영상에서
직접 계산한다. 영상 한 장마다:

  1. 깊이 → 점 → world 좌표 (영상이 찍힌 시각의 로버 자세로).
  2. 점을 격자 칸에 모으고, 각 칸 주변 3×3 칸(0.3 m)의 점들에 평면을 맞춘다.
  3. 칸이 위험한 경우
       · 평면의 기울기가 slope_deg 를 넘거나           (급경사)
       · 한 칸 안의 높이 차가 step_m 를 넘거나          (바위·턱의 옆면)
       · 평면에서 벗어난 정도(RMS)가 rough_m 를 넘을 때 (울퉁불퉁한 바위)
  4. 영상 여러 장의 판정을 증거 값으로 누적한다 (위험 +, 안전 −). 한 장의
     잡음으로 막히거나 뚫리지 않는다.

높이를 여러 영상에 걸쳐 합치지 않고 영상마다 따로 판정하는 이유: 자세 오차가
영상마다 달라서, 합친 높이에는 실제로 없는 단차가 생긴다. 한 영상 안의 점들은
같은 오차를 공유하므로 상대 높이(=기울기)는 그대로다.

보지 못하는 것
--------------
  · 카메라보다 낮게 꺼진 곳(구덩이 안쪽)과 바위 뒤 — 안 본 칸은 "모름" 이고,
    경로 계획은 모르는 칸을 지나갈 수 있다고 가정한다 (보이면 다시 계획).
  · 카메라 시야는 정면 60° 뿐이다. 후진·제자리 회전 중 옆과 뒤는 못 본다.

좌표 규약은 ObstacleGrid 와 같다: grid[i, j] = [y 행, x 열], world 원점이 지도
중심.
"""
from __future__ import annotations

import math

import numpy as np


def box3(a: np.ndarray) -> np.ndarray:
    """3×3 이웃 합 (가장자리는 있는 칸만)."""
    p = np.pad(a, 1)
    out = np.zeros_like(a)
    for di in range(3):
        for dj in range(3):
            out += p[di:di + a.shape[0], dj:dj + a.shape[1]]
    return out


class HazardMap:
    def __init__(self, map_size=(50.0, 50.0), cell_size: float = 0.1,
                 cam_offset=(0.147, 0.0, -0.312),
                 min_range: float = 0.5, max_range: float = 6.0,
                 slope_deg: float = 25.0, step_m: float = 0.12,
                 rough_m: float = 0.04, min_points: int = 6,
                 hit_gain: int = 3, miss_gain: int = 1,
                 evidence_max: int = 12, evidence_on: int = 3):
        self.map_w, self.map_h = float(map_size[0]), float(map_size[1])
        self.cell = float(cell_size)
        self.cols = int(round(self.map_w / self.cell))
        self.rows = int(round(self.map_h / self.cell))
        self.cam_offset = np.asarray(cam_offset, dtype=np.float64)
        self.min_range = float(min_range)
        self.max_range = float(max_range)
        self.slope_tan = math.tan(math.radians(slope_deg))
        self.step_m = float(step_m)
        self.rough_m = float(rough_m)
        self.min_points = int(min_points)
        self.hit_gain = int(hit_gain)
        self.miss_gain = int(miss_gain)
        self.evidence_max = int(evidence_max)
        self.evidence_on = int(evidence_on)
        # 누적 상태
        self.evidence = np.zeros((self.rows, self.cols), dtype=np.int16)
        self.seen = np.zeros((self.rows, self.cols), dtype=np.uint16)
        self.frames = 0

    # ── 좌표 ────────────────────────────────────────────────────────────
    def world_to_cell(self, x, y):
        j = np.floor((np.asarray(x) + self.map_w / 2.0) / self.cell).astype(np.int64)
        i = np.floor((np.asarray(y) + self.map_h / 2.0) / self.cell).astype(np.int64)
        return i, j

    # ── 영상 한 장 ──────────────────────────────────────────────────────
    def points_world(self, depth: np.ndarray, fx: float, fy: float, cx: float,
                     cy: float, p: np.ndarray, R: np.ndarray, stride: int = 1):
        """깊이 영상(z-depth, m) → world 점 (N, 3). (cx, cy, fx, fy) 는 원본 해상도
        기준이고 stride 는 depth 가 이미 몇 픽셀 간격으로 줄어 있는지다."""
        h, w = depth.shape
        v, u = np.mgrid[0:h, 0:w]
        ok = np.isfinite(depth) & (depth > self.min_range) & (depth < self.max_range)
        z = depth[ok].astype(np.float64)
        xo = (u[ok] * stride - cx) / fx * z          # optical: x 오른쪽
        yo = (v[ok] * stride - cy) / fy * z          #          y 아래
        # 카메라 광축 = 받침대 +x, 기울임 없음.
        base = np.stack([z, -xo, -yo], axis=1) + self.cam_offset
        return base @ R.T + p

    def classify(self, pts: np.ndarray):
        """world 점 → (칸 index i, j, 위험 여부) — 판정할 수 있는 칸만."""
        i, j = self.world_to_cell(pts[:, 0], pts[:, 1])
        ok = (i >= 0) & (i < self.rows) & (j >= 0) & (j < self.cols)
        if not ok.any():
            return (np.zeros(0, np.int64),) * 2 + (np.zeros(0, bool),)
        i, j, pts = i[ok], j[ok], pts[ok]
        # 점이 있는 영역만 잘라서 계산 (전체 격자를 매번 돌지 않게).
        i0, i1, j0, j1 = i.min(), i.max() + 1, j.min(), j.max() + 1
        H, W = i1 - i0, j1 - j0
        flat = (i - i0) * W + (j - j0)
        # 수치 안정: 좌표를 영역 중심 기준으로.
        x = pts[:, 0] - pts[:, 0].mean()
        y = pts[:, 1] - pts[:, 1].mean()
        z = pts[:, 2] - pts[:, 2].mean()

        def acc(wgt):
            return np.bincount(flat, weights=wgt, minlength=H * W).reshape(H, W)

        n = acc(None)
        sums = [acc(a) for a in (x, y, z, x * x, x * y, y * y, x * z, y * z, z * z)]
        zmin = np.full(H * W, np.inf)
        zmax = np.full(H * W, -np.inf)
        np.minimum.at(zmin, flat, z)
        np.maximum.at(zmax, flat, z)
        zrange = (zmax - zmin).reshape(H, W)

        N = box3(n)
        Sx, Sy, Sz, Sxx, Sxy, Syy, Sxz, Syz, Szz = [box3(s) for s in sums]
        with np.errstate(divide="ignore", invalid="ignore"):
            mx, my, mz = Sx / N, Sy / N, Sz / N
            cxx = Sxx / N - mx * mx
            cxy = Sxy / N - mx * my
            cyy = Syy / N - my * my
            cxz = Sxz / N - mx * mz
            cyz = Syz / N - my * mz
            czz = Szz / N - mz * mz
            det = cxx * cyy - cxy * cxy
            gx = (cxz * cyy - cyz * cxy) / det
            gy = (cyz * cxx - cxz * cxy) / det
            resid = czz - gx * cxz - gy * cyz          # 평면 잔차 분산
        # 평면을 맞출 만큼 점이 2 차원으로 퍼져 있는 칸만 (한 줄로 늘어선 점 제외).
        spread_ok = (N >= self.min_points) & (det > (0.02 ** 2) ** 2)
        slope_bad = spread_ok & (np.hypot(gx, gy) > self.slope_tan)
        rough_bad = spread_ok & (np.sqrt(np.clip(resid, 0.0, None)) > self.rough_m)
        step_bad = (n >= 2) & (zrange > self.step_m)
        decided = (n >= 1) & (spread_ok | step_bad)
        hazard = slope_bad | rough_bad | step_bad
        ii, jj = np.nonzero(decided)
        return ii + i0, jj + j0, hazard[ii, jj]

    def integrate(self, depth, fx, fy, cx, cy, p, R, stride: int = 1, exclude=()) -> int:
        """영상 한 장을 지도에 반영. return: 이번 영상에서 판정한 칸 수.

        exclude: [(x, y, r)] — 이 원 안의 점은 버린다 (다른 로버 — 움직이는 물체라
        지도에 남기면 지나간 자리에 가짜 장애물이 쌓인다)."""
        pts = self.points_world(depth, fx, fy, cx, cy, np.asarray(p, float),
                                np.asarray(R, float), stride)
        for (ex, ey, er) in exclude:
            if len(pts) == 0:
                break
            pts = pts[(pts[:, 0] - ex) ** 2 + (pts[:, 1] - ey) ** 2 > er * er]
        if len(pts) < 50:
            return 0
        i, j, hazard = self.classify(pts)
        if len(i) == 0:
            return 0
        delta = np.where(hazard, self.hit_gain, -self.miss_gain).astype(np.int16)
        ev = self.evidence[i, j] + delta
        self.evidence[i, j] = np.clip(ev, -self.evidence_max, self.evidence_max)
        self.seen[i, j] = np.minimum(self.seen[i, j].astype(np.int32) + 1, 65535)
        self.frames += 1
        return int(len(i))

    # ── 결과 ────────────────────────────────────────────────────────────
    def obstacle_mask(self) -> np.ndarray:
        return self.evidence >= self.evidence_on

    def observed_mask(self) -> np.ndarray:
        return self.seen > 0

    def clear_disk(self, x: float, y: float, radius: float) -> None:
        """(x, y) 주변을 안전으로 되돌린다 — 광물처럼 '집을 대상' 인 물체용."""
        r = int(math.ceil(radius / self.cell))
        ci, cj = self.world_to_cell(x, y)
        ci, cj = int(ci), int(cj)
        i0, i1 = max(0, ci - r), min(self.rows, ci + r + 1)
        j0, j1 = max(0, cj - r), min(self.cols, cj + r + 1)
        if i0 >= i1 or j0 >= j1:
            return
        ii, jj = np.mgrid[i0:i1, j0:j1]
        m = (ii - ci) ** 2 + (jj - cj) ** 2 <= (radius / self.cell) ** 2
        sub = self.evidence[i0:i1, j0:j1]
        sub[m] = np.minimum(sub[m], 0)
