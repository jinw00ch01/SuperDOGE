"""Mission FSM: 구역 1→9 순회하며 각 구역 내부 reveal.

State:
  PLAN_SECTOR   현재 구역의 anchor 목록 생성 → 첫 anchor 까지 A* path → DRIVE
  DRIVE         navigator 가 path 따라감
                  · path 끝나면 다음 anchor 로 PLAN_PATH
                  · 구역 reveal >= 임계 → 다음 구역으로 SWITCH
  PLAN_PATH     현재 pose 에서 다음 anchor 까지 A* path → DRIVE
  SWITCH        다음 구역 (current+1). 마지막이면 DONE.
  DONE          정지.
"""
import numpy as np

from .path_planner import astar, simplify_path
from .coverage_planner import sector_visit_order


class Mission:
    # anchor 가 새로 밝힐 수 있는 최소 셀 수. 이 미만이면 skip.
    # 예: 100 셀 (= 1m² @ 0.1m grid). reveal disk 전체는 ~1256 셀.
    MIN_NEW_REVEAL_CELLS = 100

    def __init__(self, fog_map, obstacle_grid, planner, navigator, rover,
                 sector_done_ratio=0.95):
        self.fog = fog_map
        self.ogrid = obstacle_grid
        self.planner = planner
        self.nav = navigator
        self.rover = rover
        self.sector_done_ratio = float(sector_done_ratio)

        # 방문 순서는 spawn 위치 기준 NN 으로 정한다(sector_visit_order).
        # __init__ 시점엔 아직 pose 가 안 들어왔으므로, 첫 update() 에서
        # 실제 pose 로 지연 초기화한다 — _init_visit_order 참고.
        self.visit_order = None
        self.visit_idx = 0
        self.current_sector = 0      # placeholder — _init_visit_order 에서 확정
        self.anchor_queue = []       # 남은 anchor 의 world xy 리스트
        self.state = "PLAN_SECTOR"
        self._last_log = -1
        # 주행 중에 만든 지도에서는 "지금은 길이 없는" anchor 가 생긴다 (몸으로 찍은
        # 위험 표시에 둘러싸였을 때 등). 그런 anchor 를 버리기만 하면 모든 구역이
        # reveal 0% 인 채 '완료' 가 되고 로버는 그 자리에 서 버린다 (logs/guard_t15_force2:
        # 구덩이 안에서 anchor 45 개 전부 도달 불가 → reveal 13% 에서 완료).
        # 도달 불가로 버린 것이 있으면 완료하지 않고 처음부터 다시 돈다.
        self.unreachable = 0         # 이번 바퀴에서 도달 불가로 버린 anchor 수
        self.restarts = 0            # 다시 돈 횟수
        self.max_restarts = 3
        self.escape_requested = False   # coverage_node 가 보고 위험 표시를 지운다

    def is_done(self):
        return self.state == "DONE"

    def _init_visit_order(self):
        """첫 update() 에서 실제 spawn pose 로 섹터 방문 순서를 확정한다.

        coverage_node 는 pose 를 한 번이라도 받은 뒤에야 update() 를
        호출하므로, 이 시점의 get_pose_2d() 는 유효한 spawn 위치다.
        """
        cx, cy, _ = self.rover.get_pose_2d()
        self.visit_order = sector_visit_order(self.fog, (cx, cy))
        self.current_sector = self.visit_order[self.visit_idx]
        order_str = " → ".join(str(s + 1) for s in self.visit_order)
        print(f"[mission] spawn ({cx:+.1f},{cy:+.1f}) → 섹터 "
              f"{self.current_sector + 1} 부터, 방문 순서: {order_str}")

    # ──────────────────────────────────────────────────
    def update(self, step_index):
        """매 시뮬 step 호출. 반환: (lin_vel, ang_vel)."""
        if self.visit_order is None:
            self._init_visit_order()

        if self.state == "PLAN_SECTOR":
            self._plan_sector()
            # 바로 첫 anchor 까지 path 만들고 DRIVE 로 전환
            self._plan_next_path()
            return 0.0, 0.0

        if self.state == "PLAN_PATH":
            self._plan_next_path()
            return 0.0, 0.0

        if self.state == "DRIVE":
            # 구역 진척률 체크 — 조기 종료 가능
            ratio = self.fog.sector_revealed_ratio(self.current_sector)
            if ratio >= self.sector_done_ratio:
                self._log(f"[구역 {self.current_sector + 1}] reveal {ratio*100:.0f}% → 다음 구역")
                self._switch_to_next_sector()
                return 0.0, 0.0

            # 현재 path 따라가기
            lin, ang, advanced = self.nav.step()
            if advanced and self.nav.is_done():
                # 한 anchor 도착 완료 → 다음 anchor 로
                self.state = "PLAN_PATH"
                return 0.0, 0.0
            return lin, ang

        if self.state == "SWITCH":
            self._switch_to_next_sector()
            return 0.0, 0.0

        return 0.0, 0.0

    # ──────────────────────────────────────────────────
    def _plan_sector(self):
        # BCD 분할 → sub-cell zigzag anchor. sub-cell 방문 순서는 로버
        # 현재 위치 기준 Nearest Neighbor (planner 내부에서 처리).
        cx, cy, _ = self.rover.get_pose_2d()
        anchors = self.planner.generate_anchors(self.current_sector, (cx, cy))
        self.anchor_queue = anchors
        print(f"[mission] 구역 {self.current_sector + 1} 시작 — "
              f"BCD anchor {len(anchors)}개 "
              f"(첫 anchor: {anchors[0] if anchors else 'None'})")

    def _plan_next_path(self):
        # 다음 anchor 를 동적으로 재평가해 선택 (Greedy Frontier).
        # 차 네비게이션이 매번 최적 경로를 재구성하듯, 남은 anchor 를
        # 전부 다시 채점해 "단위 거리당 신규 reveal 이 가장 큰" 곳으로 향한다.
        # 이미 충분히 밝혀진 anchor 는 후보에서 영구 제거.
        cx, cy, _ = self.rover.get_pose_2d()

        target = None
        best_score = -1.0
        survivors = []
        for a in self.anchor_queue:
            new_cells = self.fog.potential_new_reveal(a[0], a[1])
            if new_cells < self.MIN_NEW_REVEAL_CELLS:
                continue                       # 이미 밝혀짐 — 후보에서 제거
            survivors.append(a)
            d = float(np.hypot(a[0] - cx, a[1] - cy))
            score = new_cells / (1.0 + d)      # 가깝고 많이 밝히는 곳 우선
            if score > best_score:
                best_score = score
                target = a

        skipped = len(self.anchor_queue) - len(survivors)
        self.anchor_queue = survivors
        if skipped > 0:
            print(f"[mission] anchor {skipped}개 제거 (이미 밝혀짐)")

        # 남은 미방문 anchor 없음 → 구역 종료
        if target is None:
            ratio = self.fog.sector_revealed_ratio(self.current_sector)
            self._log(f"[구역 {self.current_sector + 1}] 미방문 anchor 없음 "
                      f"(reveal={ratio*100:.0f}%) → 다음 구역")
            self._switch_to_next_sector()
            return

        self.anchor_queue.remove(target)       # 선택한 anchor 소비

        start_ij = self.ogrid.world_to_cell(cx, cy, clip=True)
        goal_ij = self.ogrid.world_to_cell(target[0], target[1], clip=True)
        cost = getattr(self.ogrid, "cost", None)       # 기울기 비용 (지형 스캐너가 있을 때)
        if not self.ogrid.is_free(*goal_ij):
            # anchor 자리가 막혀 있다 (바위 옆 등). 버리지 않고 가장 가까운 빈 칸까지만
            # 간다 — 거기서도 anchor 둘레는 밝혀진다. 지형 스캐너를 쓰면 멀리서도 막힌
            # 곳을 알게 되어, 그냥 버리면 구역이 reveal 30% 대에서 끝났다 (anchor 196개 버림).
            near = self._nearest_free_cell(goal_ij, max_radius_m=2.0)
            if near is not None:
                goal_ij = near
                target = self.ogrid.cell_to_world(*near)

        if not self.ogrid.is_free(*start_ij):
            # 로버가 inflate 영역 안에 있음 (시작셀 막혔다고 표시됨). 일시 free 처리.
            # 주행 중에 만든 지도에서는 장애물을 가까이 와서야 알게 되어 깊이
            # 들어와 있을 수 있다 → 가장 가까운 빈 칸으로 먼저 빠져나온다.
            escape = self._nearest_free_cell(start_ij)
            cells = None
            if escape is not None:
                tail = astar(self.ogrid.grid, escape, goal_ij, cost=cost)
                if tail is not None:
                    cells = [start_ij] + tail
            if cells is None:
                cells = astar(self._free_start_grid(start_ij), start_ij, goal_ij, cost=cost)
        else:
            cells = astar(self.ogrid.grid, start_ij, goal_ij, cost=cost)

        if cells is None:
            print(f"[mission] anchor ({target[0]:+.1f},{target[1]:+.1f}) 도달 불가, 스킵")
            self.unreachable += 1
            self.state = "PLAN_PATH"     # 다음 anchor 즉시 시도
            return

        cells = simplify_path(self.ogrid.grid, cells, max_skip=20, cost=cost)
        path = [self.ogrid.cell_to_world(i, j) for (i, j) in cells]
        # 마지막은 정확한 anchor 좌표로 교체 (셀 양자화 오차 제거)
        if path:
            path[-1] = target
        self.nav.set_path(path)
        self.state = "DRIVE"
        self._log(f"[mission] → anchor ({target[0]:+.1f},{target[1]:+.1f}), "
                  f"path {len(path)}점")

    def replan_keep_target(self) -> bool:
        """가고 있던 anchor 를 버리지 않고 경로만 다시 짠다 (지도가 바뀌었을 때)."""
        if self.state != "DRIVE" or not self.nav.path:
            return False
        self.anchor_queue.insert(0, self.nav.path[-1])
        self.nav.set_path([])
        self.state = "PLAN_PATH"
        return True

    def _nearest_free_cell(self, cell, max_radius_m=3.0):
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
        return (int(fi[k] + a0), int(fj[k] + b0)) if d2[k] <= r * r else None

    def _switch_to_next_sector(self):
        self.visit_idx += 1
        if self.visit_idx >= len(self.visit_order):
            if (self.unreachable > 0 and self.restarts < self.max_restarts
                    and self.fog.overall_ratio() < 0.9):
                self.restarts += 1
                print(f"[mission] 길이 없어 버린 anchor 가 {self.unreachable}개 "
                      f"(reveal {self.fog.overall_ratio() * 100:.0f}%) — 완료하지 않고 "
                      f"처음부터 다시 ({self.restarts}/{self.max_restarts})")
                self.unreachable = 0
                self.visit_idx = 0
                self.current_sector = self.visit_order[0]
                self.state = "PLAN_SECTOR"
                self.escape_requested = True
                return
            print("[mission] 모든 구역 완료!")
            self.state = "DONE"
            return
        self.current_sector = self.visit_order[self.visit_idx]
        self.state = "PLAN_SECTOR"

    def _free_start_grid(self, start_ij):
        """시작 셀이 inflate 막힘 영역일 때 임시로 그 주변 풀어주기."""
        g = self.ogrid.grid.copy()
        i, j = start_ij
        for di in range(-2, 3):
            for dj in range(-2, 3):
                ii, jj = i + di, j + dj
                if self.ogrid.in_bounds(ii, jj):
                    g[ii, jj] = 0
        return g

    def _log(self, msg):
        # 너무 자주 반복되는 로그 억제 (간단)
        print(msg)
