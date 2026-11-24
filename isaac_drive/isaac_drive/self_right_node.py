"""self_right_node — 넘어진 로버를 팔로 다시 세운다 (학습한 정책).

    odom(자세) + joint_states(팔 관절) ──▶ [이 노드] ──▶ <ns>/arm/joint_command
                                               └──▶ righting/active (Bool)

평소에는 아무것도 하지 않는다. 차체가 flipped_deg 넘게 기운 채 flipped_hold_sec
동안 멈춰 있으면 "넘어졌다" 로 보고, 팔 관절 목표를 15 Hz(시뮬레이션 시각 기준)로
내보내 차체를 밀어 세운다. 일어서면 팔을 천천히 접어 넣고 끝낸다.
righting/active 가 True 인 동안 supervisor 는 바퀴를 세워 둔다.

정책은 isaac_sim/rl/train_right.py 로 학습한다 (관측·행동 정의는
navigation/righting_policy.py). 속도는 odom 의 연속한 두 자세 차이로 구한다
(vehicle_v3 의 odom 은 twist 를 채우지 않는다).
"""
from __future__ import annotations

import os

import numpy as np
import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool

from isaac_drive.navigation import governor_policy as gp
from isaac_drive.navigation import righting_policy as rp


def default_policy_path() -> str:
    try:
        from ament_index_python.packages import get_package_share_directory
        return os.path.join(get_package_share_directory("isaac_drive"),
                            "policies", "self_right.npz")
    except Exception:
        return ""


class SelfRight(Node):
    def __init__(self) -> None:
        super().__init__("self_right_node")
        self.declare_parameter("policy_path", "")
        self.declare_parameter("odom_topic", "ground_truth/odom")
        self.declare_parameter("joint_topic", "joint_states_raw")
        self.declare_parameter("joint_command_topic", "arm/joint_command")
        self.declare_parameter("active_topic", "righting/active")
        self.declare_parameter("flipped_hold_sec", 2.0)
        self.declare_parameter("attempt_sec", 20.0)
        self.declare_parameter("max_attempts", 3)

        path = str(self.get_parameter("policy_path").value) or default_policy_path()
        self.policy = rp.GovernorPolicy(path)
        self.period = 1.0 / rp.CONTROL_HZ
        self.hold_sec = float(self.get_parameter("flipped_hold_sec").value)
        self.attempt_sec = float(self.get_parameter("attempt_sec").value)
        self.max_attempts = int(self.get_parameter("max_attempts").value)

        self.state = "IDLE"            # IDLE | RIGHTING | RETRACT | GAVE_UP
        self.q = None                  # 팔 관절 각도·속도 (joint_states)
        self.dq = np.zeros(6)
        self._arm_idx = None
        self.target = rp.ARM_HOME.copy()
        self._prev_pose = None
        self.v_b = np.zeros(3)
        self.w_b = np.zeros(3)
        self._last_run = None
        self._flipped_since = None
        self._started = 0.0
        self._upright_steps = 0
        self.attempts = 0

        self.cmd_pub = self.create_publisher(
            JointState, str(self.get_parameter("joint_command_topic").value), 10)
        self.active_pub = self.create_publisher(
            Bool, str(self.get_parameter("active_topic").value), 10)
        self.create_subscription(
            Odometry, str(self.get_parameter("odom_topic").value), self._on_odom, 10)
        self.create_subscription(
            JointState, str(self.get_parameter("joint_topic").value), self._on_joint, 10)
        self.create_timer(0.5, self._publish_active)
        self.get_logger().info(
            f"self-right ready — 정책 {path} (학습 샘플 "
            f"{int(self.policy.meta.get('samples', 0)):,})")

    # ── 입력 ─────────────────────────────────────────────────────────────
    def _on_joint(self, msg: JointState) -> None:
        if self._arm_idx is None:
            try:
                self._arm_idx = [msg.name.index(n) for n in rp.ARM_JOINTS]
            except ValueError:
                return
        if len(msg.position) > max(self._arm_idx):
            self.q = np.array([msg.position[i] for i in self._arm_idx])
        if len(msg.velocity) > max(self._arm_idx):
            self.dq = np.array([msg.velocity[i] for i in self._arm_idx])

    def _rates(self, stamp: float, position, quat) -> None:
        pos = np.array([position.x, position.y, position.z])
        prev = self._prev_pose
        if prev is not None and stamp - prev[0] < 1e-4:
            return
        self._prev_pose = (stamp, pos, quat)
        if prev is None or stamp - prev[0] > 0.25:
            self.v_b = np.zeros(3)
            self.w_b = np.zeros(3)
            return
        dt = stamp - prev[0]
        self.v_b = gp.rotate_world_to_body(quat, (pos - prev[1]) / dt)
        x0, y0, z0, w0 = prev[2]
        x1, y1, z1, w1 = quat
        dw = w0 * w1 + x0 * x1 + y0 * y1 + z0 * z1
        dx = w0 * x1 - x0 * w1 - y0 * z1 + z0 * y1
        dy = w0 * y1 + x0 * z1 - y0 * w1 - z0 * x1
        dz = w0 * z1 - x0 * y1 + y0 * x1 - z0 * w1
        self.w_b = 2.0 * (1.0 if dw >= 0.0 else -1.0) * np.array([dx, dy, dz]) / dt

    # ── 제어 ─────────────────────────────────────────────────────────────
    def _on_odom(self, msg: Odometry) -> None:
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        o = msg.pose.pose.orientation
        quat = (o.x, o.y, o.z, o.w)
        self._rates(stamp, msg.pose.pose.position, quat)
        if self._last_run is not None and 0.0 <= stamp - self._last_run < self.period - 1e-3:
            return
        self._last_run = stamp
        if self.q is None:
            return
        g_b = np.asarray(gp.gravity_in_body(quat), dtype=np.float64).reshape(-1)
        tilt = float(np.degrees(np.arccos(np.clip(-g_b[2], -1.0, 1.0))))
        spin = float(np.linalg.norm(self.w_b))
        speed = float(np.linalg.norm(self.v_b))

        if self.state == "IDLE":
            if tilt > rp.FLIPPED_DEG and spin < 0.5:
                if self._flipped_since is None:
                    self._flipped_since = stamp
                elif stamp - self._flipped_since >= self.hold_sec:
                    self.state = "RIGHTING"
                    self._started = stamp
                    self._upright_steps = 0
                    self.target = np.clip(rp.ARM_HOME, self.q - rp.MAX_GAP, self.q + rp.MAX_GAP)
                    self.attempts += 1
                    self.get_logger().warn(
                        f"넘어짐 감지 (기울기 {tilt:.0f}°) → 팔로 세우기 시작 "
                        f"({self.attempts}/{self.max_attempts}번째)")
                    self._publish_active()
            else:
                self._flipped_since = None
                if tilt < 30.0:
                    self.attempts = 0
        elif self.state == "RIGHTING":
            obs = rp.build_obs(g_b, self.w_b, self.q, self.dq, self.target)
            self.target = rp.next_target(self.target, self.q, self.policy(obs)[0])
            self._send(self.target)
            ok = tilt < rp.UPRIGHT_DEG and spin < 0.4 and speed < 0.3
            self._upright_steps = self._upright_steps + 1 if ok else 0
            if self._upright_steps >= int(rp.CONTROL_HZ):
                self.get_logger().info(
                    f"일어섬 ({stamp - self._started:.1f} s) → 팔 접는 중")
                self.state = "RETRACT"
            elif stamp - self._started > self.attempt_sec:
                self.get_logger().warn(
                    f"{self.attempt_sec:.0f} s 안에 못 일어섬 (기울기 {tilt:.0f}°) → 팔을 접고 다시 판단")
                self.state = "RETRACT"
        elif self.state == "RETRACT":
            self.target = self.target + np.clip(rp.ARM_HOME - self.target, -0.04, 0.04)
            self._send(self.target)
            if np.abs(self.target - rp.ARM_HOME).max() < 0.02 \
                    and np.abs(self.q - rp.ARM_HOME).max() < 0.15:
                self._flipped_since = None
                if tilt > rp.FLIPPED_DEG and self.attempts >= self.max_attempts:
                    self.state = "GAVE_UP"
                    self.get_logger().error(
                        f"{self.attempts}번 시도했지만 일어서지 못함 — 포기 (바퀴는 계속 정지)")
                else:
                    self.state = "IDLE"
                self._publish_active()

    def _send(self, target) -> None:
        m = JointState()
        m.header.stamp = self.get_clock().now().to_msg()
        m.name = list(rp.ARM_JOINTS)
        m.position = [float(v) for v in target]
        self.cmd_pub.publish(m)

    def _publish_active(self) -> None:
        self.active_pub.publish(Bool(data=self.state != "IDLE"))


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SelfRight()
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
