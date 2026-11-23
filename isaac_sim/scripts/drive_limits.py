"""로버 구동부(Ackermann ScriptNode)에 속도·가속 제한을 넣는다.

SimulationApp 기동 후에만 import 할 것.

왜 필요한가
-----------
vehicle_v3 의 바퀴 관절은 damping 4000 · maxForce 1e7 의 속도 구동이라 사실상
토크 제한이 없고, 원래 Ackermann 스크립트는 /cmd_vel 을 그대로 바퀴 목표
속도로 넘긴다. 상위 노드는 최대 3 m/s · 1.5 rad/s 를 계단 입력으로 준다.
지구 중력에서는 이게 버텨졌지만 (실측 최고 1.5 m/s, 가속 1.9 m/s², 최대
기울기 19°), 화성 중력에서는 같은 명령으로 출발 3 초 만에 차체가 뒤로
넘어갔다 (logs/physics/mars_run1: pitch −82° → roll 128° 로 전복).
복원 모멘트(무게 × 팔 길이)는 중력에 비례해 0.38 배로 줄었는데 차체가 가진
운동에너지(속도²)는 그대로이기 때문이다.

기준
----
같은 차체가 중력만 다른 곳에서 "같은 모양"으로 움직이려면 Froude 수
v²/(g·L) 가 같아야 한다 → 속도·각속도는 sqrt(g) 에, 가속도는 g 에 비례.
지구에서 버틴 실측값(1.5 m/s, 1.9 m/s²)에 이 비율을 곱한 것이 화성 기본값이다.

  ratio = g_mars / g_earth = 3.72 / 9.81 = 0.379,  sqrt = 0.616
  최고 속도   1.5 m/s   × 0.616 → 0.9 m/s
  선가속      1.9 m/s²  × 0.379 → 0.7 m/s²
  회전 속도   1.5 rad/s × 0.616 → 0.9 rad/s
  각가속                         → 1.0 rad/s²

실제 화성 로버(Curiosity 약 0.04 m/s)보다는 여전히 훨씬 빠르다. 여기서 맞춘
것은 "이 차체가 화성 중력에서 넘어지지 않는 한계"이지 실제 임무 속도가 아니다.
"""
from __future__ import annotations

from pxr import Usd

EARTH_G = 9.81
# 지구 중력에서 원래 스크립트로 주행했을 때의 실측 한계 (logs/percep_err/rec1).
EARTH_REFERENCE = {"max_lin": 1.5, "max_ang": 1.5, "acc_lin": 1.9, "acc_ang": 2.6}


def limits_for_gravity(g: float) -> dict:
    """Froude 상사로 지구 기준값을 중력 g 에 맞춘다."""
    ratio = float(g) / EARTH_G
    root = ratio ** 0.5
    return {
        "max_lin": round(EARTH_REFERENCE["max_lin"] * root, 2),
        "max_ang": round(EARTH_REFERENCE["max_ang"] * root, 2),
        "acc_lin": round(EARTH_REFERENCE["acc_lin"] * ratio, 2),
        "acc_ang": round(EARTH_REFERENCE["acc_ang"] * ratio, 2),
    }


def parse_limits(text: str, base: dict) -> dict:
    """'max_lin=0.8,acc_lin=0.5' 형식의 override 를 base 위에 덮는다."""
    out = dict(base)
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        key, _, val = part.partition("=")
        key = key.strip()
        if key not in out:
            raise ValueError(f"알 수 없는 구동 제한 항목: {key!r} "
                             f"(가능: {', '.join(out)})")
        out[key] = float(val)
    return out


# 모든 로버에 같은 내용으로 들어간다 (ScriptNode 는 전역 이름공간을 공유하므로
# 로버별로 내용이 다르면 마지막에 초기화된 값이 전체를 덮어쓴다). 로버별 상태는
# 노드 prim path 를 키로 하는 dict 에 둔다. 전역 이름은 전부 _ack_ / _ACK_ 접두어.
ACK_LIMITED_TPL = '''
import math
import omni.graph.core as og

_ACK_STEER_JOINT_NAMES = ['FL_Steer_Revolute', 'FR_Steer_Revolute', 'RL_Steer_Revolute', 'RR_Steer_Revolute']
_ACK_DRIVE_JOINT_NAMES = ['FL_Drive_Continuous', 'FR_Drive_Continuous', 'CL_Drive_Continuous', 'CR_Drive_Continuous', 'RL_Drive_Continuous', 'RR_Drive_Continuous']
_ACK_WHEELBASE_LENGTH = 0.849
_ACK_MIDDLE_WHEEL_DISTANCE = 0.894
_ACK_WHEEL_RADIUS = 0.1
_ACK_OFFSET = -0.0135
_ACK_STEERING_GAIN = 1.1
_ACK_WHEEL_VELOCITY_GAIN = 2.0
_ACK_LIMITS = __ACK_LIMITS__

try:
    _ACK_STATES
except NameError:
    _ACK_STATES = {}


def _ack_component(value, index):
    try:
        return float(value[index])
    except Exception:
        try:
            return float(getattr(value, ("x", "y", "z")[index]))
        except Exception:
            return 0.0


def _ack_now():
    try:
        from isaacsim.core.api import SimulationContext
        return float(SimulationContext.instance().current_time)
    except Exception:
        import omni.timeline
        return float(omni.timeline.get_timeline_interface().get_current_time())


def _ack_slew(current, target, step):
    if target > current + step:
        return current + step
    if target < current - step:
        return current - step
    return target


def _ack_limit(db, lin_cmd, ang_cmd):
    """명령을 최고 속도로 자르고, 직전 출력에서 가속 한계만큼만 움직인다."""
    try:
        key = str(db.node.get_prim_path())
    except Exception:
        key = "_"
    lim = _ACK_LIMITS
    lin_cmd = max(-lim["max_lin"], min(lim["max_lin"], lin_cmd))
    ang_cmd = max(-lim["max_ang"], min(lim["max_ang"], ang_cmd))
    now = _ack_now()
    st = _ACK_STATES.get(key)
    if st is None or now < st["t"]:
        st = {"t": now, "lin": 0.0, "ang": 0.0}
        _ACK_STATES[key] = st
    dt = now - st["t"]
    if dt <= 0.0:
        return st["lin"], st["ang"]
    dt = min(dt, 0.1)
    st["t"] = now
    st["lin"] = _ack_slew(st["lin"], lin_cmd, lim["acc_lin"] * dt)
    st["ang"] = _ack_slew(st["ang"], ang_cmd, lim["acc_ang"] * dt)
    return st["lin"], st["ang"]


def setup(db):
    pass


def compute(db):
    lin_vel, ang_vel = _ack_limit(
        db,
        _ack_component(db.inputs.linearVelocity, 0),
        _ack_component(db.inputs.angularVelocity, 2))

    direction = 1.0 if lin_vel >= 0.0 else -1.0
    if ang_vel > 0.0:
        turn_direction = 1.0
    elif ang_vel < 0.0:
        turn_direction = -1.0
    else:
        turn_direction = 0.0

    lin = abs(lin_vel)
    ang = abs(ang_vel)
    turning_radius = float("inf") if ang == 0.0 else lin / ang
    minimum_radius = _ACK_MIDDLE_WHEEL_DISTANCE * 0.8
    half_base = _ACK_WHEELBASE_LENGTH / 2.0

    r_ml = turning_radius - (_ACK_MIDDLE_WHEEL_DISTANCE / 2.0) * turn_direction
    r_mr = turning_radius + (_ACK_MIDDLE_WHEEL_DISTANCE / 2.0) * turn_direction
    r_fl = math.hypot(r_ml, half_base)
    r_fr = math.hypot(r_mr, half_base)
    r_rl = r_fl
    r_rr = r_fr

    if turning_radius < minimum_radius:
        vel_fl = -ang * turn_direction
        vel_fr = ang * turn_direction
        vel_rl = -ang * turn_direction
        vel_rr = ang * turn_direction
        vel_ml = -ang * turn_direction
        vel_mr = ang * turn_direction
        theta_fl = -math.pi / 4.0
        theta_fr = math.pi / 4.0
        theta_rl = math.pi / 4.0
        theta_rr = -math.pi / 4.0
    else:
        vel_fl = (r_fl * ang if ang > 0.0 else lin) * direction
        vel_fr = (r_fr * ang if ang > 0.0 else lin) * direction
        vel_rl = (r_rl * ang if ang > 0.0 else lin) * direction
        vel_rr = (r_rr * ang if ang > 0.0 else lin) * direction
        vel_ml = (r_ml * ang if ang > 0.0 else lin) * direction
        vel_mr = (r_mr * ang if ang > 0.0 else lin) * direction
        theta_fl = math.atan2(half_base - _ACK_OFFSET, r_fl) * turn_direction
        theta_fr = math.atan2(half_base - _ACK_OFFSET, r_fr) * turn_direction
        theta_rl = math.atan2(half_base + _ACK_OFFSET, r_rl) * -turn_direction
        theta_rr = math.atan2(half_base + _ACK_OFFSET, r_rr) * -turn_direction

    k = _ACK_WHEEL_VELOCITY_GAIN / (_ACK_WHEEL_RADIUS * 2.0)
    db.outputs.steerJointNames = _ACK_STEER_JOINT_NAMES
    db.outputs.driveJointNames = _ACK_DRIVE_JOINT_NAMES
    db.outputs.steeringAngles = [_ACK_STEERING_GAIN * a for a in
                                 (theta_fl, theta_fr, theta_rl, theta_rr)]
    db.outputs.wheelVelocities = [k * v for v in
                                  (vel_fl, vel_fr, vel_ml, vel_mr, vel_rl, vel_rr)]
    db.outputs.execOut = og.ExecutionAttributeState.ENABLED
    return True
'''


def apply_drive_limits(stage, rover_roots, limits: dict | None) -> int:
    """각 로버의 Ackermann ScriptNode 를 제한이 있는 버전으로 바꾼다.

    limits 가 None 이면 아무것도 바꾸지 않는다 (원래 스크립트 그대로).
    return: 바꾼 노드 수.
    """
    if not limits:
        print("[drive] 구동 제한 없음 — 원래 Ackermann 스크립트 그대로")
        return 0
    src = ACK_LIMITED_TPL.replace("__ACK_LIMITS__", repr(dict(limits)))
    n = 0
    for root in rover_roots:
        rp = stage.GetPrimAtPath(root)
        if not rp or not rp.IsValid():
            continue
        for prim in Usd.PrimRange(rp):
            if prim.GetTypeName() != "OmniGraphNode" or prim.GetName() != "Ackermann":
                continue
            attr = prim.GetAttribute("inputs:script")
            if attr and attr.IsValid():
                attr.Set(src)
                n += 1
    print(f"[drive] 구동 제한 적용 (Ackermann 노드 {n}개): "
          f"최고 {limits['max_lin']} m/s · {limits['max_ang']} rad/s, "
          f"가속 {limits['acc_lin']} m/s² · {limits['acc_ang']} rad/s²")
    return n
