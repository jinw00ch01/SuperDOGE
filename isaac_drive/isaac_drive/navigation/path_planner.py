"""8연결 A* 알고리즘 + 경로 단순화.

grid: 2D numpy array uint8 (0=free, 1=blocked)
start, goal: (i, j) 튜플
cost: (선택) grid 와 같은 모양의 float 배열 — 그 칸에 들어갈 때 거리에 더하는 배수
      (칸 비용 = 거리 × (1 + cost)). None 이면 예전과 같은 균일 비용.
      경사 비용(online_map.OnlineObstacleLayer.slope_cost)에 쓴다: 지나갈 수는 있지만
      가파른 칸을 덜 밟는 길을 고르게 한다. heuristic 은 그대로 거리라서 cost >= 0 이면
      여전히 최단(최소 비용) 경로다.
반환: 셀 인덱스의 리스트 (start → goal), 못 찾으면 None
"""
import heapq
import math


_NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1),
              ( 0, -1),          ( 0, 1),
              ( 1, -1), ( 1, 0), ( 1, 1)]


def astar(grid, start, goal, cost=None):
    rows, cols = grid.shape
    si, sj = start
    gi, gj = goal
    if not (0 <= si < rows and 0 <= sj < cols
            and 0 <= gi < rows and 0 <= gj < cols):
        return None
    if grid[si, sj] == 1 or grid[gi, gj] == 1:
        return None
    if start == goal:
        return [start]

    open_set = [(0.0, start)]
    came_from = {}
    g_score = {start: 0.0}

    while open_set:
        _, cur = heapq.heappop(open_set)
        if cur == goal:
            path = [cur]
            while cur in came_from:
                cur = came_from[cur]
                path.append(cur)
            path.reverse()
            return path

        ci, cj = cur
        for di, dj in _NEIGHBORS:
            ni, nj = ci + di, cj + dj
            if ni < 0 or ni >= rows or nj < 0 or nj >= cols:
                continue
            if grid[ni, nj] == 1:
                continue
            # 대각선 이동 시 옆 셀 통해 corner-cut 방지
            if di != 0 and dj != 0:
                if grid[ci + di, cj] == 1 or grid[ci, cj + dj] == 1:
                    continue
            step_cost = math.hypot(di, dj)
            if cost is not None:
                step_cost *= 1.0 + float(cost[ni, nj])
            tg = g_score[cur] + step_cost
            n = (ni, nj)
            if n not in g_score or tg < g_score[n]:
                came_from[n] = cur
                g_score[n] = tg
                h = math.hypot(ni - gi, nj - gj)
                heapq.heappush(open_set, (tg + h, n))

    return None


def simplify_path(grid, path, max_skip=20, cost=None, cost_tol=0.25):
    """Line-of-sight 단순화: 가능하면 멀리있는 셀로 점프.

    A* 출력은 셀 단위라 너무 빽빽함. 직선으로 막힘 없이 도달 가능한
    최대 점프 길이로 줄여서 navigator 가 따라가기 좋게 만듦.

    cost 를 주면 직선이 지나는 칸의 비용이 원래 경로(그 구간)의 가장 큰 비용보다
    cost_tol 넘게 크지 않을 때만 건너뛴다 — 경사를 돌아가는 A* 경로를 직선으로 펴서
    다시 비탈을 곧장 오르게 만들지 않도록.
    """
    if not path or len(path) <= 2:
        return list(path)
    simplified = [path[0]]
    i = 0
    while i < len(path) - 1:
        # path[i] 에서 직선 line-of-sight 가능한 가장 먼 점 찾기
        j_far = min(i + max_skip, len(path) - 1)
        while j_far > i + 1:
            if _line_of_sight(grid, path[i], path[j_far]):
                if cost is None:
                    break
                seg = path[i:j_far + 1]
                allowed = max(float(cost[a, b]) for a, b in seg) + cost_tol
                if _line_max_cost(cost, path[i], path[j_far]) <= allowed:
                    break
            j_far -= 1
        simplified.append(path[j_far])
        i = j_far
    return simplified


def _line_cells(a, b):
    """Bresenham 으로 a→b 사이의 칸들."""
    i0, j0 = a
    i1, j1 = b
    di = abs(i1 - i0)
    dj = abs(j1 - j0)
    si = 1 if i0 < i1 else -1
    sj = 1 if j0 < j1 else -1
    err = di - dj
    ci, cj = i0, j0
    while True:
        yield ci, cj
        if ci == i1 and cj == j1:
            return
        e2 = 2 * err
        if e2 > -dj:
            err -= dj; ci += si
        if e2 < di:
            err += di; cj += sj


def _line_max_cost(cost, a, b):
    return max(float(cost[i, j]) for i, j in _line_cells(a, b))


def _line_of_sight(grid, a, b):
    """Bresenham 으로 a→b 사이에 막힌 셀 있는지."""
    i0, j0 = a
    i1, j1 = b
    di = abs(i1 - i0)
    dj = abs(j1 - j0)
    si = 1 if i0 < i1 else -1
    sj = 1 if j0 < j1 else -1
    err = di - dj
    ci, cj = i0, j0
    while True:
        if grid[ci, cj] == 1:
            return False
        if ci == i1 and cj == j1:
            return True
        e2 = 2 * err
        if e2 > -dj:
            err -= dj; ci += si
        if e2 < di:
            err += di; cj += sj
