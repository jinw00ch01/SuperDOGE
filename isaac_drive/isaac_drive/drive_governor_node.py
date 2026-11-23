"""drive_governor_node — 학습한 구동 조절기를 주행 명령 경로에 끼운다.

    supervisor ──▶ /<ns>/cmd_vel_raw ──▶ [이 노드] ──▶ /<ns>/cmd_vel ──▶ 시뮬레이터 바퀴

상위 노드는 지형을 모른 채 "3 m/s 로 가라" 를 보낸다. 이 노드는 차체가 느끼는
것(기울기, 흔들림, 바퀴 속도와 실제 속도의 차이)을 보고 그 명령을 얼마나 깎을지
정한다. 정책은 isaac_sim/rl/ 에서 PPO 로 학습한 작은 신경망이고, 추론은 numpy
행렬 곱 세 번이다 (torch 불필요).

입력 토픽
  cmd_in_topic   geometry_msgs/Twist   상위 노드의 명령
  odom_topic     nav_msgs/Odometry     자세 (→ 중력 방향, 차체 속도)
  joint_topic    sensor_msgs/JointState 바퀴 6 개의 각속도

vehicle_v3 의 /ground_truth/odom 은 자세만 채우고 속도(twist)는 0 이다. 그래서
차체 속도는 연속한 두 odom 의 자세 차이로 구한다 (velocity_source:=pose_diff).
twist 를 채워 주는 odom 이라면 velocity_source:=twist 로 바꿀 수 있다.

제어 주기는 odom 의 시각(= 시뮬레이션 시각)으로 맞춘다. 시뮬레이터가 실시간보다
느리게 돌아도 학습 때와 같은 15 Hz(시뮬레이션 기준)로 동작한다.

주의: 시뮬레이터의 고정 구동 제한과 같이 쓰면 그쪽이 먼저 깎는다 (화성 기본값
0.92 m/s). 조절기에 맡기려면 시뮬레이터를 `--drive-limits off` 로 띄울 것.
센서가 끊기면(odom 0.5 s 이상 없음) 정지 명령을 낸다.
"""
from __future__ import annotations

import os

import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState

from isaac_drive.navigation import governor_policy as gp


def default_policy_path() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory("isaac_drive"),
                            "policies", "drive_governor.npz")
    except Exception:
        return ""


class DriveGovernor(Node):
    def __init__(self) -> None:
        super().__init__("drive_governor_node")
        self.declare_parameter("policy_path", "")
        self.declare_parameter("cmd_in_topic", "cmd_vel_raw")
        self.declare_parameter("cmd_out_topic", "cmd_vel")
        self.declare_parameter("odom_topic", "ground_truth/odom")
        self.declare_parameter("joint_topic", "joint_states_raw")
        self.declare_parameter("control_hz", 15.0)
        self.declare_parameter("velocity_source", "pose_diff")   # pose_diff | twist
        self.declare_parameter("twist_in_body_frame", True)      # twist 를 쓸 때만
        self.declare_parameter("sensor_timeout_sec", 0.5)

        path = str(self.get_parameter("policy_path").value) or default_policy_path()
        self.policy = gp.GovernorPolicy(path)
        self.period = 1.0 / float(self.get_parameter("control_hz").value)
        self.twist_in_body = bool(self.get_parameter("twist_in_body_frame").value)
        self.use_twist = str(self.get_parameter("velocity_source").value) == "twist"
        self._prev_pose = None         # (시각, 위치, 자세 xyzw)
        self.v_b = np.zeros(3)
        self.w_b = np.zeros(3)
        self.timeout = float(self.get_parameter("sensor_timeout_sec").value)

        self.v_des = 0.0
        self.w_des = 0.0
        self.wheel = np.zeros(6)
        self._wheel_idx = None
        self.v_out = 0.0
        self.w_out = 0.0
        self.steer_fix = 0.0
        self._last_run = None          # 마지막으로 정책을 돌린 odom 시각 (s)
        self._last_odom_wall = None
        self._stopped = False
        self._n = 0
        self._scale_sum = 0.0

        self.pub = self.create_publisher(
            Twist, str(self.get_parameter("cmd_out_topic").value), 10)
        self.create_subscription(
            Twist, str(self.get_parameter("cmd_in_topic").value), self._on_cmd, 10)
        self.create_subscription(
            Odometry, str(self.get_parameter("odom_topic").value), self._on_odom, 10)
        self.create_subscription(
            JointState, str(self.get_parameter("joint_topic").value), self._on_joint, 10)
        self.create_timer(0.1, self._watchdog)
        self.create_timer(10.0, self._report)
        self.get_logger().info(
            f"drive governor ready — 정책 {path} (학습 샘플 "
            f"{int(self.policy.meta.get('samples', 0)):,}), 제어 {1.0 / self.period:.0f} Hz")

    def _on_cmd(self, msg: Twist) -> None:
        self.v_des = float(msg.linear.x)
        self.w_des = float(msg.angular.z)

    def _on_joint(self, msg: JointState) -> None:
        if self._wheel_idx is None:
            try:
                self._wheel_idx = [msg.name.index(n) for n in gp.DRIVE_JOINTS]
            except ValueError:
                return
        if len(msg.velocity) > max(self._wheel_idx):
            self.wheel = np.array([msg.velocity[i] for i in self._wheel_idx])

    def _on_odom(self, msg: Odometry) -> None:
        self._last_odom_wall = self.get_clock().now()
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        o = msg.pose.pose.orientation
        quat = (o.x, o.y, o.z, o.w)
        if self.use_twist:
            lin = msg.twist.twist.linear
            ang = msg.twist.twist.angular
            self.v_b = np.array([lin.x, lin.y, lin.z])
            self.w_b = np.array([ang.x, ang.y, ang.z])
            if not self.twist_in_body:
                self.v_b = gp.rotate_world_to_body(quat, self.v_b)
                self.w_b = gp.rotate_world_to_body(quat, self.w_b)
        else:
            self._velocity_from_pose(stamp, msg.pose.pose.position, quat)
        if self._last_run is not None and 0.0 <= stamp - self._last_run < self.period - 1e-3:
            return
        self._last_run = stamp
        obs = gp.build_obs([self.v_des], [self.w_des], gp.gravity_in_body(quat),
                           self.w_b, self.v_b, self.wheel,
                           [self.v_out], [self.w_out], [self.steer_fix])
        action = self.policy(obs)
        v_cmd, w_cmd = gp.action_to_command(action, [self.v_des], [self.w_des])
        self.v_out, self.w_out = float(v_cmd[0]), float(w_cmd[0])
        self.steer_fix = float(action[0, 2])
        self._publish(self.v_out, self.w_out)
        self._stopped = False
        if abs(self.v_des) > 0.05:
            self._n += 1
            self._scale_sum += abs(self.v_out) / min(abs(self.v_des), gp.V_CAP)

    def _velocity_from_pose(self, stamp: float, position, quat) -> None:
        """연속한 두 자세의 차이 → 차체 기준 선속도·각속도."""
        pos = np.array([position.x, position.y, position.z])
        prev = self._prev_pose
        if prev is not None and stamp - prev[0] < 1e-4:
            return                               # 같은 시각의 중복 메시지
        self._prev_pose = (stamp, pos, quat)
        if prev is None or stamp - prev[0] > 0.25:
            self.v_b = np.zeros(3)               # 처음이거나 끊겼다 이어진 경우
            self.w_b = np.zeros(3)
            return
        dt = stamp - prev[0]
        self.v_b = gp.rotate_world_to_body(quat, (pos - prev[1]) / dt)
        # q_prev⁻¹ ⊗ q_now = 차체 기준으로 본 회전. 작은 각이면 벡터부 × 2 가 회전각.
        x0, y0, z0, w0 = prev[2]
        x1, y1, z1, w1 = quat
        dw = w0 * w1 + x0 * x1 + y0 * y1 + z0 * z1
        dx = w0 * x1 - x0 * w1 - y0 * z1 + z0 * y1
        dy = w0 * y1 + x0 * z1 - y0 * w1 - z0 * x1
        dz = w0 * z1 - x0 * y1 + y0 * x1 - z0 * w1
        sign = 1.0 if dw >= 0.0 else -1.0
        self.w_b = 2.0 * sign * np.array([dx, dy, dz]) / dt

    def _publish(self, v: float, w: float) -> None:
        t = Twist()
        t.linear.x = float(v)
        t.angular.z = float(w)
        self.pub.publish(t)

    def _watchdog(self) -> None:
        if self._last_odom_wall is None:
            return
        age = (self.get_clock().now() - self._last_odom_wall).nanoseconds * 1e-9
        if age > self.timeout:
            if not self._stopped:
                self.get_logger().warn(f"odom 이 {age:.1f} s 동안 없음 → 정지 명령")
                self._stopped = True
            self.v_out = self.w_out = 0.0
            self._publish(0.0, 0.0)

    def _report(self) -> None:
        if self._n:
            self.get_logger().info(
                f"최근 10 s: 전진 명령의 평균 {100.0 * self._scale_sum / self._n:.0f}% 를 내보냄 "
                f"(지금 명령 {self.v_des:.2f} → {self.v_out:.2f} m/s)")
        self._n, self._scale_sum = 0, 0.0


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = DriveGovernor()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
