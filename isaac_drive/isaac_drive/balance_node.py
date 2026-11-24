"""balance_node — 비탈에서 균형을 지키는 정책을 주행 명령 경로에 끼운다.

    supervisor ──▶ /<ns>/cmd_vel_raw ──▶ [이 노드] ──▶ /<ns>/cmd_vel ──▶ 바퀴
                                            └──▶ /<ns>/arm/joint_command (균형추)

drive_governor_node 와 같은 자리에 들어가고(둘 중 하나만 켠다), 두 가지를 더 한다.
  1. 물러서기 — 전진 명령을 깎는 것뿐 아니라 뒤로 빼기도 한다 (명령의 −0.25 배까지).
  2. 팔 균형추 — 차체가 10° 넘게 기울면 팔을 기울어진 반대쪽으로 뻗어 무게중심을 옮긴다.

팔은 채취(PICK_READY)·넘어짐 복구(RECOVERY) 중에는 건드리지 않는다. 평지에서는
팔 명령을 내지 않는다 (home 에 돌아온 뒤에는 조용히 있는다).

주의: 팔이 home 을 벗어나 있는 동안 hazard_map_node 는 깊이 영상을 쓰지 않는다
(팔이 화면을 가린다). 즉 균형추를 쓰는 동안은 지도가 갱신되지 않는다.
"""
from __future__ import annotations

import numpy as np
import rclpy
from sensor_msgs.msg import JointState
from std_msgs.msg import String

from isaac_drive.drive_governor_node import DriveGovernor
from isaac_drive.navigation import balance_policy as bp
from isaac_drive.navigation import governor_policy as gp

ARM_BUSY_PHASES = ("PICK_READY", "RECOVERY")


class BalanceGovernor(DriveGovernor):
    NODE_NAME = "balance_node"
    POLICY_FILE = "balance.npz"
    DIMS = (bp.OBS_DIM, bp.ACT_DIM)

    def __init__(self) -> None:
        super().__init__()
        self.declare_parameter("joint_command_topic", "arm/joint_command")
        self.declare_parameter("phase_topic", "mission/phase")
        self.weight = np.zeros((1, 2))
        self.q_arm = bp.ARM_HOME.copy()
        self.arm_target = bp.ARM_HOME.copy()
        self._arm_idx = None
        self._arm_out = False          # 팔이 home 을 벗어나 있음 (이 노드가 움직이는 중)
        self.arm_free = True
        self._arm_steps = 0
        self.arm_pub = self.create_publisher(
            JointState, str(self.get_parameter("joint_command_topic").value), 10)
        self.create_subscription(
            String, str(self.get_parameter("phase_topic").value), self._on_phase, 10)

    def _on_phase(self, msg: String) -> None:
        self.arm_free = msg.data.split("/")[-1] not in ARM_BUSY_PHASES

    def _on_joint(self, msg: JointState) -> None:
        super()._on_joint(msg)
        if self._arm_idx is None:
            try:
                self._arm_idx = [msg.name.index(n) for n in bp.ARM_JOINTS]
            except ValueError:
                return
        if len(msg.position) > max(self._arm_idx):
            self.q_arm = np.array([msg.position[i] for i in self._arm_idx])

    def _decide(self, quat) -> tuple:
        g_b = gp.gravity_in_body(quat)
        obs = bp.build_obs([self.v_des], [self.w_des], g_b, self.w_b, self.v_b, self.wheel,
                           [self.v_out], [self.w_out], [self.steer_fix],
                           self.weight, self.q_arm)
        action = self.policy(obs)
        v_cmd, w_cmd = bp.action_to_command(action, [self.v_des], [self.w_des])
        if not self.arm_free:
            # 팔은 다른 노드가 쓰는 중 — 손대지 않고, 끝났을 때 지금 자세에서 이어 간다.
            self.weight = np.zeros((1, 2))
            self.arm_target = self.q_arm.copy()
            self._arm_out = False
        else:
            self.weight = bp.weight_vector(action)
            target = bp.arm_targets(self.arm_target, self.weight, bp.gate(g_b))[0]
            folded = abs(target[1]) < 0.02 and abs(target[2] - bp.ARM_HOME[2]) < 0.02
            if folded:                                   # 접힌 뒤에는 앞을 보게 되돌린다
                target[0] -= float(np.clip(target[0], -bp.ARM_RATE, bp.ARM_RATE))
            self.arm_target = target
            away = float(np.abs(target - bp.ARM_HOME).max()) > 0.02
            if away or self._arm_out:
                js = JointState()
                js.header.stamp = self.get_clock().now().to_msg()
                js.name = list(bp.ARM_JOINTS)
                js.position = [float(v) for v in target]
                self.arm_pub.publish(js)
                self._arm_steps += 1
            self._arm_out = away
        return float(v_cmd[0]), float(w_cmd[0]), float(action[0, 2])

    def _report(self) -> None:
        super()._report()
        if self._arm_steps:
            self.get_logger().info(
                f"최근 10 s: 팔 균형추 사용 {self._arm_steps / 15.0:.1f} s "
                f"(지금 눕힘 {self.q_arm[1]:.2f} rad, 방향 {np.degrees(self.q_arm[0]):.0f}°)")
        self._arm_steps = 0


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = BalanceGovernor()
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
