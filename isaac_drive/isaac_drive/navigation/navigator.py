"""Path tracker: 웨이포인트 리스트를 헤딩 비례 + Ackermann 으로 따라감.

사용:
    nav = Navigator(rover)
    nav.set_path([(x1, y1), (x2, y2), ...])
    while not nav.is_done():
        lin, ang = nav.step()
        rover.drive(lin, ang)
"""
import numpy as np


class Navigator:
    def __init__(self, rover,
                 waypoint_tol=0.4, final_tol=0.3,
                 kp_ang=2.0, max_lin=3.0, max_ang=1.5,
                 point_turn_deg=45, turn_decel=0.4, spin_gain=1.9):
        self.rover = rover
        self.waypoint_tol  = float(waypoint_tol)
        self.final_tol     = float(final_tol)
        self.kp_ang        = float(kp_ang)
        self.max_lin       = float(max_lin)
        self.max_ang       = float(max_ang)
        self.point_turn_th = np.deg2rad(point_turn_deg)
        # 제자리 회전을 멈출 때 쓸 수 있다고 보는 감속도 (rad/s², 보수적으로) 와
        # 제자리 회전 때 실제 회전 속도 / 명령 비 (vehicle_v3 는 약 1.9 배로 돈다).
        self.turn_decel = float(turn_decel)
        self.spin_gain = float(spin_gain)
        self.path = []
        self.idx = 0

    def set_path(self, path):
        self.path = list(path) if path else []
        self.idx = 0

    def is_done(self):
        return self.idx >= len(self.path)

    @property
    def current_target(self):
        if self.is_done():
            return None
        return self.path[self.idx]

    def step(self):
        """현재 pose 에 맞춰 (lin, ang) 계산. 도달 시 다음 wp 로 advance.

        반환: (lin_vel, ang_vel, advanced_flag)
        """
        if self.is_done():
            return 0.0, 0.0, False

        cx, cy, yaw = self.rover.get_pose_2d()
        tx, ty = self.path[self.idx]
        dx, dy = tx - cx, ty - cy
        dist = float(np.hypot(dx, dy))

        # 도착 판정 (마지막 점은 엄격하게)
        is_last = (self.idx == len(self.path) - 1)
        tol = self.final_tol if is_last else self.waypoint_tol
        if dist < tol:
            self.idx += 1
            return 0.0, 0.0, True

        target_yaw = float(np.arctan2(dy, dx))
        err = target_yaw - yaw
        err = (err + np.pi) % (2.0 * np.pi) - np.pi

        ang_vel = float(np.clip(self.kp_ang * err, -self.max_ang, self.max_ang))
        if abs(err) > self.point_turn_th:
            lin_vel = 0.0
            # 제자리 회전: 남은 각도 안에서 멈출 수 있는 속도까지만 낸다.
            # 시뮬레이터는 각가속도를 제한하고(화성 0.99 rad/s²) 명령보다 빨리 돌기
            # 때문에, 오차에 비례한 명령만 주면 목표 방향을 지나쳤다가 되돌아오기를
            # 끝없이 반복한다 (제자리에서 좌우로만 도는 현상).
            if self.turn_decel > 0.0:
                cap = np.sqrt(2.0 * self.turn_decel * abs(err)) / max(self.spin_gain, 1e-3)
                ang_vel = float(np.clip(ang_vel, -cap, cap))
        else:
            lin_vel = self.max_lin * np.cos(err)
        return lin_vel, ang_vel, False
