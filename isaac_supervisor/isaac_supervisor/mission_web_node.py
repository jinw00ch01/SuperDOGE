"""Mission web node — Flask + Socket.IO dashboard for Mars rover ops.

화성↔지구 통신 비유: 브라우저(지구)가 ROS2 토픽(화성 rover)을 web 으로
구독한다. ROS2 콜백 -> Socket.IO emit -> 브라우저가 SC2 풍 HUD 갱신.

접속 주소
---------
포트 하나(WEB_PORT, 기본 8088)로 화면·상태·카메라 영상을 모두 내보낸다. 카메라는
web_video_server(127.0.0.1:WEB_VIDEO_PORT)의 MJPEG 를 이 서버가 /stream 으로 다시
내보내므로, 브라우저는 이 서버 주소 하나만 알면 된다 → localhost 뿐 아니라
  · 같은 네트워크의 다른 기기:  http://<이 PC 의 IP>:8088  (시작할 때 로그에 찍힌다)
  · 터널(포트 하나만 넘겨 주는 방식)로 만든 외부 주소
에서도 그대로 열린다. WEB_TOKEN(접속 키)을 주면 `?key=<키>` 로 한 번 들어온
브라우저만 볼 수 있다 — 외부 주소로 열 때는 반드시 줄 것 (ACTIONS 버튼이 로버를
움직인다). WEB_ACTIONS=0 이면 버튼이 꺼진 보기 전용이 된다.

ACTIONS 버튼 (Socket.IO 'action' → ROS 토픽)
--------------------------------------------
  mode    <ns>/mission/mode      AUTO ↔ MANUAL (MANUAL 이면 키보드 W/A/S/D 로 주행)
  estop   <ns>/mission/command   ESTOP ↔ RESUME — 모든 로버에 보낸다
  rtb     <ns>/mission/command   RTB ↔ RTB_CANCEL
  reset   <ns>/mission/command   RESET
  replan  <ns>/mission/command   REPLAN
  skip    <ns>/mission/command   SKIP
  mapclr  <ns>/hazard_map/reset  주행 중 만든 위험 지도를 비운다
  cam / speed                    화면 쪽 설정 (기록만 남긴다)
누른 것은 전부 ROS 로그와 action_log_path(JSON 한 줄씩)에 남고, mission_manager 가
돌려주는 결과(<ns>/mission/event)와 함께 화면의 EVENT LOG 에 나온다.

Endpoints:
  GET /         -> templates/index.html
  GET /static/* -> CSS/JS/이미지
  GET /stream   -> 카메라 MJPEG (web_video_server 중계)
  GET /api/info -> 접속 주소·설정
  Socket.IO     -> 'state' 'odom' 'minimap' 'cmd_vel' 'path' 'target' 'event'
                <- 'action' 'teleop'

Run:
  ros2 run isaac_supervisor mission_web_node
  → http://localhost:8088 접속.
"""
from __future__ import annotations

import hmac
import json
import math
import os
import re
import socket
import subprocess
import threading
import time
import urllib.request
from collections import deque
from typing import Dict, List, Optional

# Async mode: we use "threading" (Flask-SocketIO's built-in werkzeug + thread
# worker) rather than eventlet. eventlet.monkey_patch() rewires the standard
# threading/socket modules, which deadlocks rclpy's executor when we spin it
# in a daemon thread (the spinner stops cooperatively yielding to Flask). For
# our telemetry-only load (a handful of clients, low message rate) threading
# mode is plenty.
_ASYNC_MODE = "threading"

import rclpy
from ament_index_python.packages import get_package_share_directory
from flask import (Flask, Response, abort, jsonify, redirect, render_template,
                   request, send_file)
from flask_socketio import SocketIO
from geometry_msgs.msg import PointStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry, Path
from visualization_msgs.msg import MarkerArray
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from std_msgs.msg import Empty, String

from isaac_interfaces.msg import MissionState


SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
)


def _yaw_from_quat(q) -> float:
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def _attitude_from_quat(q) -> tuple:
    """(roll, pitch, tilt) [deg]. pitch + = 앞이 들림, roll + = 왼쪽이 들림(오른쪽으로 기움),
    tilt = 차체 위쪽 축이 수직에서 벗어난 각 (60° 를 넘으면 넘어진 것으로 본다)."""
    # 차체 x·y 축의 world z 성분 = 회전 행렬의 R[2,0], R[2,1]
    r20 = 2.0 * (q.x * q.z - q.w * q.y)
    r21 = 2.0 * (q.y * q.z + q.w * q.x)
    r22 = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
    pitch = math.degrees(math.asin(max(-1.0, min(1.0, r20))))
    roll = math.degrees(math.asin(max(-1.0, min(1.0, r21))))
    tilt = math.degrees(math.acos(max(-1.0, min(1.0, r22))))
    return roll, pitch, tilt


def lan_urls(port: int) -> List[str]:
    """같은 네트워크의 다른 기기에서 열 수 있는 주소들."""
    ips: List[str] = []
    try:
        out = subprocess.run(["hostname", "-I"], capture_output=True, text=True,
                             timeout=2.0).stdout
        ips = [a for a in out.split() if "." in a and not a.startswith(("127.", "172.17."))]
    except Exception:
        pass
    if not ips:
        try:                                   # 기본 경로로 나가는 주소 (패킷은 보내지 않는다)
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sk:
                sk.connect(("192.0.2.1", 9))
                ips = [sk.getsockname()[0]]
        except Exception:
            pass
    return [f"http://{ip}:{port}" for ip in ips]


# 운영자 명령: HUD 버튼 이름 → mission/command 에 보낼 문자열 (토글은 상태를 보고 고른다)
SIMPLE_COMMANDS = {"reset": "RESET", "replan": "REPLAN", "skip": "SKIP"}
TELEOP_MAX_LIN = 1.0        # m/s
TELEOP_MAX_ANG = 1.0        # rad/s


class MissionWebRosNode(Node):
    def __init__(self, sio: SocketIO) -> None:
        super().__init__("mission_web_node")
        self._sio = sio
        self.declare_parameter("body_cam_topic_for_url",
                               "/perception/image_annotated")
        self.declare_parameter("wrist_cam_topic_for_url",
                               "/perception/wrist_image_annotated")
        self.declare_parameter("overview_cam_topic_for_url",
                               "/camera/overview/image_raw")
        self.declare_parameter("chase_cam_topic_for_url",
                               "/camera/chase/image_raw")
        # Overview 슬롯을 정적 이미지로 대체할 때 사용. terrain_NNNNN/preview.png.
        self.declare_parameter(
            "terrain_preview_path",
            os.path.expanduser(
                "~/dev_ws/rover_ws/src/a2_isaac/isaac_sim/assets/"
                "generated_terrains/terrain_00004/preview.png"))
        # rover_namespaces — 멀티 rover 추적용. 빈 리스트 또는 [""] 면 단일
        # rover (absolute 토픽 sub). 예: ["rover_1", "rover_2"] 면 각 namespace
        # 의 토픽을 모두 sub 하고 payload 에 rover_id 필드를 박아 emit.
        self.declare_parameter("rover_namespaces", [""])
        # 버튼을 누른 기록 (JSON 한 줄씩). 빈 문자열이면 ~/.ros/log/hud_actions.jsonl
        self.declare_parameter("action_log_path", "")
        nss_raw = self.get_parameter("rover_namespaces").value
        # ROS2 string_array param 이 빈 list 면 None 으로 들어오는 케이스 방어
        self.rover_namespaces: List[str] = (
            [str(s) for s in nss_raw] if nss_raw else [""]
        )
        # Per-rover cache so a late-connecting browser can be primed.
        # 단일 모드는 rover_id="" 키로 저장됨.
        self._last_state: Dict[str, dict] = {}
        self._last_odom: Dict[str, dict] = {}
        self._last_minimap: Dict[str, dict] = {}
        self._last_path: Dict[str, dict] = {}
        # explore path (coverage) 와 supervisor path (A* APPROACH/RTB) 를
        # 분리 캐싱. 같은 slot 공유 시 두 source 가 번갈아 덮어쓰며 미니맵
        # 경로가 진동했음 (2026-05-27 디버깅).
        self._last_path_explore: Dict[str, dict] = {}
        self._last_path_supervisor: Dict[str, dict] = {}
        self._last_target: Dict[str, Optional[dict]] = {}
        # target slot 도 path 와 동일 패턴 — explore (coverage markers) vs
        # supervisor (APPROACH/RTB) 분리 라우팅. 같은 slot 공유 시 EXPLORE
        # 중에도 supervisor 의 옛 mineral 별이 stale 하게 보이거나 두 source
        # 가 번갈아 진동.
        self._last_target_explore: Dict[str, Optional[dict]] = {}
        self._last_target_supervisor: Dict[str, Optional[dict]] = {}

        minimap_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        self._mode_pub: Dict[str, object] = {}
        self._command_pub: Dict[str, object] = {}
        self._teleop_pub: Dict[str, object] = {}
        self._mapclr_pub: Dict[str, object] = {}
        self._teleop_last: Dict[str, float] = {}       # ns → 마지막 teleop 시각 (기록용)
        self._events: deque = deque(maxlen=400)          # 늦게 들어온 브라우저에게 줄 최근 기록
        path = str(self.get_parameter("action_log_path").value).strip()
        self.action_log_path = os.path.expanduser(
            path or os.path.join(os.environ.get("ROS_LOG_DIR", "~/.ros/log"),
                                 "hud_actions.jsonl"))
        try:
            os.makedirs(os.path.dirname(self.action_log_path), exist_ok=True)
        except OSError:
            pass
        self._log_lock = threading.Lock()
        for ns in self.rover_namespaces:
            self._subscribe_for_rover(ns, minimap_qos)
            self._mode_pub[ns] = self.create_publisher(
                String, self._topic(ns, "mission/mode"), 10)
            self._command_pub[ns] = self.create_publisher(
                String, self._topic(ns, "mission/command"), 10)
            self._teleop_pub[ns] = self.create_publisher(
                Twist, self._topic(ns, "teleop/cmd_vel"), 10)
            self._mapclr_pub[ns] = self.create_publisher(
                Empty, self._topic(ns, "hazard_map/reset"), 10)
        self.get_logger().info(
            f"mission_web_node ROS interface ready — "
            f"rover_namespaces={self.rover_namespaces}.")

    def _topic(self, ns: str, relative: str) -> str:
        """ns="" 이면 absolute /<relative>. 그 외엔 /<ns>/<relative>."""
        rel = relative.lstrip("/")
        if not ns:
            return "/" + rel
        return f"/{ns.strip('/')}/{rel}"

    def _subscribe_for_rover(self, ns: str, minimap_qos) -> None:
        """단일 rover (ns="" 또는 "rover_1" 등) 의 모든 토픽 sub."""
        from functools import partial
        self.create_subscription(
            MissionState, self._topic(ns, "mission/state"),
            partial(self._on_state, ns), 10)
        self.create_subscription(
            Odometry, self._topic(ns, "ground_truth/odom"),
            partial(self._on_odom, ns), SENSOR_QOS)
        self.create_subscription(
            Twist, self._topic(ns, "cmd_vel"),
            partial(self._on_cmd_vel, ns), 10)
        self.create_subscription(
            OccupancyGrid, self._topic(ns, "mission/minimap"),
            partial(self._on_minimap, ns), minimap_qos)
        self.create_subscription(
            Path, self._topic(ns, "mission/path"),
            partial(self._on_path, ns), 10)
        self.create_subscription(
            MarkerArray, self._topic(ns, "mission/markers"),
            partial(self._on_markers, ns), 10)
        self.create_subscription(
            Path, self._topic(ns, "supervisor/path"),
            partial(self._on_supervisor_path, ns), 10)
        self.create_subscription(
            PointStamped, self._topic(ns, "supervisor/target"),
            partial(self._on_supervisor_target, ns), 10)
        self.create_subscription(
            String, self._topic(ns, "mission/event"),
            partial(self._on_event, ns), 20)

    # ── 기록 ────────────────────────────────────────────────────────────
    def push_event(self, ns: str, kind: str, text: str, **extra) -> dict:
        """EVENT LOG 한 줄. kind: action(운영자가 누름) | rover(로버가 알림)."""
        ev = {"t": time.time(), "rover_id": ns, "kind": kind, "text": text}
        ev.update(extra)
        self._events.append(ev)
        self._sio.emit("event", ev)
        return ev

    def _on_event(self, ns: str, msg: String) -> None:
        self.push_event(ns, "rover", msg.data)

    def _record_action(self, ns: str, action: str, sent: str, origin: str) -> None:
        who = ns or "rover"
        self.get_logger().info(f"[HUD] action={action} rover={who} → {sent} (from {origin})")
        line = json.dumps({"time": time.strftime("%Y-%m-%dT%H:%M:%S"), "rover": ns,
                           "action": action, "sent": sent, "from": origin},
                          ensure_ascii=False)
        try:
            with self._log_lock, open(self.action_log_path, "a") as f:
                f.write(line + "\n")
        except OSError as e:
            self.get_logger().warn(f"action log 쓰기 실패: {e}", throttle_duration_sec=30.0)
        self.push_event(ns, "action", f"{action.upper()} → {sent}", origin=origin)

    # ── 운영자 명령 ─────────────────────────────────────────────────────
    def do_action(self, ns: str, action: str, value: str, origin: str) -> str:
        """HUD 버튼 하나를 처리하고, 무엇을 보냈는지 돌려준다."""
        action = (action or "").strip().lower()
        if ns not in self._command_pub:
            return f"모르는 로버 {ns!r}"
        st = self._last_state.get(ns, {})
        raw = str(st.get("state", ""))
        phase = raw.split("/")[-1]
        if action == "estop":
            # 하나라도 서 있지 않으면 전부 세운다. 전부 서 있으면 전부 푼다.
            phases = [str(self._last_state.get(n, {}).get("state", "")).split("/")[-1]
                      for n in self.rover_namespaces]
            cmd = "RESUME" if phases and all(p == "ESTOP" for p in phases) else "ESTOP"
            for n in self.rover_namespaces:
                self._command_pub[n].publish(String(data=cmd))
                self._record_action(n, action, cmd, origin)
            return cmd
        if action == "mode":
            cmd = "AUTO" if raw.startswith("MANUAL/") else "MANUAL"
            self._mode_pub[ns].publish(String(data=cmd))
        elif action == "rtb":
            cmd = "RTB_CANCEL" if phase == "RETURN_TO_BASE" else "RTB"
            self._command_pub[ns].publish(String(data=cmd))
        elif action in SIMPLE_COMMANDS:
            cmd = SIMPLE_COMMANDS[action]
            self._command_pub[ns].publish(String(data=cmd))
        elif action == "mapclr":
            cmd = "hazard_map/reset"
            self._mapclr_pub[ns].publish(Empty())
        elif action in ("cam", "speed"):
            cmd = re.sub(r"[^A-Za-z0-9 ._/+-]", "", str(value))[:40] or "?"   # 화면 쪽 설정
        else:
            return f"모르는 동작 {action!r}"
        self._record_action(ns, action, cmd, origin)
        return cmd

    def do_teleop(self, ns: str, lin: float, ang: float, origin: str) -> None:
        pub = self._teleop_pub.get(ns)
        if pub is None or not (math.isfinite(lin) and math.isfinite(ang)):
            return
        t = Twist()
        t.linear.x = max(-TELEOP_MAX_LIN, min(TELEOP_MAX_LIN, float(lin)))
        t.angular.z = max(-TELEOP_MAX_ANG, min(TELEOP_MAX_ANG, float(ang)))
        pub.publish(t)
        now = time.time()
        moving = abs(t.linear.x) > 1e-3 or abs(t.angular.z) > 1e-3
        if moving and now - self._teleop_last.get(ns, 0.0) > 2.0:
            # 매 메시지(10 Hz)를 적지 않고, 쉬었다가 다시 움직일 때만 한 줄.
            self._record_action(ns, "teleop", f"수동 주행 v={t.linear.x:+.1f} w={t.angular.z:+.1f}",
                                origin)
        if moving:
            self._teleop_last[ns] = now

    def topics_for_camera_urls(self) -> dict:
        return {
            "body":     str(self.get_parameter("body_cam_topic_for_url").value),
            "wrist":    str(self.get_parameter("wrist_cam_topic_for_url").value),
            "overview": str(self.get_parameter("overview_cam_topic_for_url").value),
            "chase":    str(self.get_parameter("chase_cam_topic_for_url").value),
        }

    def last_snapshot(self) -> dict:
        """Per-rover cache snapshot — dict-by-ns 형태.
        예: {"state": {"rover_1": {...}, "rover_2": {...}}, "odom": {...}, ...}
        client 가 connect 시 모든 rover 의 latest 값을 prime 받을 수 있다.
        """
        return {
            "rover_namespaces": list(self.rover_namespaces),
            "state":   dict(self._last_state),
            "odom":    dict(self._last_odom),
            "minimap": dict(self._last_minimap),
            "path":    dict(self._last_path),
            "target":  dict(self._last_target),
            "events":  list(self._events),
        }

    def _emit(self, event: str, ns: str, payload: dict) -> None:
        """ns 를 payload 의 rover_id 필드로 박아 emit. 단일 모드는 빈 문자열."""
        payload["rover_id"] = ns
        self._sio.emit(event, payload)

    def _on_state(self, ns: str, msg: MissionState) -> None:
        payload = {
            "state": msg.state,
            "previous_state": msg.previous_state,
            "battery_percent": float(msg.battery_percent),
            "low_battery": bool(msg.low_battery),
            "critical_battery": bool(msg.critical_battery),
            "cargo_count": int(msg.cargo_count),
            "cargo_capacity": int(msg.cargo_capacity),
            "collected_count": int(msg.collected_count),
            "collection_goal": int(msg.collection_goal),
            "collected_blue":   int(msg.collected_blue),
            "collected_yellow": int(msg.collected_yellow),
            "collected_green":  int(msg.collected_green),
            "active_task": msg.active_task,
            "last_error": msg.last_error,
        }
        prev_cached_state = (
            self._last_state[ns]["state"] if ns in self._last_state else "")
        self._last_state[ns] = payload
        self._emit("state", ns, payload)
        # MISSION_COMPLETE 진입 시 미니맵 잔재 즉시 청소.
        # coverage_node 가 mission state 를 모르고 sector path/markers 를
        # 계속 발행하므로 backend 에서 차단해야 minimap 의 옛 sector anchor
        # 별 + 점선 path 가 사라짐 (2026-05-27 시연 캡처 디버깅).
        if payload["state"] == "MISSION_COMPLETE":
            self._last_path_explore[ns] = {"pts": []}
            self._last_path_supervisor[ns] = {"pts": []}
            self._last_target_explore[ns] = None
            self._last_target_supervisor[ns] = None
            self._emit_active_path(ns)
            self._emit_active_target(ns)
        # EXPLORE 진입 시 explore + supervisor slot 모두 청소.
        # supervisor 도 같이 비우는 이유: state msg 가 mission_manager 의
        # supervisor NaN msg 보다 먼저 backend 에 도달하는 race 가 있으면
        # supervisor 의 옛 mineral target 이 fallback 으로 쓰여 별이 옛
        # APPROACH 위치를 고수함 (2026-05-27 사용자 보고: PICK 후 EXPLORE
        # 재시작 시 별이 안 바뀌고 path 만 바뀌는 현상).
        elif payload["state"] == "EXPLORE" and prev_cached_state != "EXPLORE":
            self._last_path_explore[ns] = {"pts": []}
            self._last_path_supervisor[ns] = {"pts": []}
            self._last_target_explore[ns] = None
            self._last_target_supervisor[ns] = None
            self._emit_active_path(ns)
            self._emit_active_target(ns)

    def _on_odom(self, ns: str, msg: Odometry) -> None:
        p = msg.pose.pose.position
        v = msg.twist.twist.linear
        w = msg.twist.twist.angular
        yaw = _yaw_from_quat(msg.pose.pose.orientation)
        # vehicle_v3 의 GT_SCRIPT 는 PubGtOdom 에 position/orientation 만 보내고
        # twist 는 채우지 않는다 → 수신측 v.x/v.y 가 항상 0. position 변화율로
        # numerical velocity 계산 (per-rover prev cache).
        now_ns = self.get_clock().now().nanoseconds
        speed = 0.0
        prev_map = getattr(self, "_prev_odom_for_speed", None)
        if prev_map is None:
            prev_map = {}
            self._prev_odom_for_speed = prev_map
        prev = prev_map.get(ns)
        if prev is not None:
            dt = (now_ns - prev["ns"]) * 1e-9
            if dt > 0.02:
                dx = float(p.x) - prev["x"]
                dy = float(p.y) - prev["y"]
                speed = math.hypot(dx, dy) / dt
        prev_map[ns] = {"x": float(p.x), "y": float(p.y), "ns": now_ns}
        roll, pitch, tilt = _attitude_from_quat(msg.pose.pose.orientation)
        payload = {
            "x": float(p.x), "y": float(p.y), "z": float(p.z),
            "yaw_deg": math.degrees(yaw),
            "roll_deg": roll, "pitch_deg": pitch, "tilt_deg": tilt,
            "vx": float(v.x), "vy": float(v.y),
            "speed": float(speed),
            "wz": float(w.z),
        }
        self._last_odom[ns] = payload
        self._emit("odom", ns, payload)

    def _on_cmd_vel(self, ns: str, msg: Twist) -> None:
        self._emit("cmd_vel", ns, {
            "lin": float(msg.linear.x),
            "ang": float(msg.angular.z),
        })

    def _emit_active_path(self, ns: str) -> None:
        """Phase-aware path 라우터.

        Supervisor (mission_manager) 가 APPROACH/PICK_READY/RTB 시 path 를
        채워 발행하고 EXPLORE 진입 시 빈 path 를 발행한다. 따라서 supervisor
        path 가 non-empty 면 그게 active, 빈 path 면 coverage(EXPLORE) path
        가 active. 두 source 가 같은 slot 을 덮어쓰지 않도록 라우팅.
        """
        sup = self._last_path_supervisor.get(ns, {"pts": []})
        active = sup if sup.get("pts") else self._last_path_explore.get(
            ns, {"pts": []})
        self._last_path[ns] = active
        self._emit("path", ns, active)

    def _on_path(self, ns: str, msg: Path) -> None:
        # MISSION_COMPLETE 후엔 coverage 가 stale path 계속 보내도 무시.
        if self._last_state.get(ns, {}).get("state") == "MISSION_COMPLETE":
            return
        pts = [(float(ps.pose.position.x), float(ps.pose.position.y))
               for ps in msg.poses]
        self._last_path_explore[ns] = {"pts": pts}
        # EXPLORE 별 위치 = path 의 끝점 (sector anchor) 로 결정.
        # coverage_node 의 current_target marker 는 다음 waypoint (가까운 점)
        # 라 rover 옆에 별이 보이고 path 만 멀리 그려졌음 (2026-05-27 사용자
        # 보고). path[-1] 이 진짜 anchor 라 APPROACH 와 동일 패턴 (별이 path
        # 끝점) 으로 통일.
        if pts:
            self._last_target_explore[ns] = {
                "x": pts[-1][0], "y": pts[-1][1]}
        self._emit_active_path(ns)
        self._emit_active_target(ns)

    def _on_supervisor_path(self, ns: str, msg: Path) -> None:
        pts = [(float(ps.pose.position.x), float(ps.pose.position.y))
               for ps in msg.poses]
        # supervisor 가 빈 path 발행하면 (EXPLORE 시) supervisor slot 비워서
        # _emit_active_path 가 coverage path 로 fallback.
        self._last_path_supervisor[ns] = {"pts": pts}
        self._emit_active_path(ns)

    def _emit_active_target(self, ns: str) -> None:
        """supervisor 가 채운 target 우선, 비어있으면 explore (coverage marker)
        target fallback. path 와 같은 분리 라우팅 패턴."""
        sup = self._last_target_supervisor.get(ns)
        active = sup if sup else self._last_target_explore.get(ns)
        self._last_target[ns] = active
        self._emit("target", ns, dict(active) if active else {})

    def _on_supervisor_target(self, ns: str, msg: PointStamped) -> None:
        import math as _math
        x, y = float(msg.point.x), float(msg.point.y)
        if not (_math.isfinite(x) and _math.isfinite(y)):
            # EXPLORE 진입 시 mission_manager 가 NaN 발행 → supervisor slot 비움.
            self._last_target_supervisor[ns] = None
        else:
            self._last_target_supervisor[ns] = {"x": x, "y": y}
        self._emit_active_target(ns)

    def _on_markers(self, ns: str, msg: MarkerArray) -> None:
        # MISSION_COMPLETE 후엔 coverage 의 stale anchor marker 무시.
        if self._last_state.get(ns, {}).get("state") == "MISSION_COMPLETE":
            return
        # _on_path 가 path 의 끝점을 explore_target 으로 set 하므로 보통은
        # marker 정보가 필요 없음. path 가 아직 한 번도 안 들어왔거나 비었을
        # 때만 marker 의 current_target 으로 fallback (그 외엔 무시 — marker
        # 의 current_target 은 다음 waypoint 라 별이 rover 옆에 그려져 사용자
        # 직관과 안 맞음).
        existing_path = self._last_path_explore.get(ns, {}).get("pts", [])
        if existing_path:
            return
        target = None
        for m in msg.markers:
            if m.ns == "target":
                if m.action == 2:
                    target = None
                else:
                    target = {"x": float(m.pose.position.x),
                              "y": float(m.pose.position.y)}
                break
        self._last_target_explore[ns] = target
        self._emit_active_target(ns)

    def _on_minimap(self, ns: str, msg: OccupancyGrid) -> None:
        # OccupancyGrid -> compact dict the browser can paint on a canvas.
        payload = {
            "w": int(msg.info.width),
            "h": int(msg.info.height),
            "res": float(msg.info.resolution),
            "ox": float(msg.info.origin.position.x),
            "oy": float(msg.info.origin.position.y),
            "data": list(msg.data),
        }
        self._last_minimap[ns] = payload
        self._emit("minimap", ns, payload)


def _spin_ros(ros_node: Node) -> None:
    try:
        rclpy.spin(ros_node)
    except Exception:
        pass


def _resolve_web_dir() -> str:
    """Locate share/isaac_supervisor/web (installed) or src/.../web (dev).
    Prefer installed share so colcon install --symlink-install picks up
    edits live."""
    try:
        share = get_package_share_directory("isaac_supervisor")
        cand = os.path.join(share, "web")
        if os.path.isdir(cand):
            return cand
    except Exception:
        pass
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(os.path.join(here, "..", "web"))


def main(args: list[str] | None = None) -> None:
    web_dir = _resolve_web_dir()
    app = Flask(
        __name__,
        template_folder=os.path.join(web_dir, "templates"),
        static_folder=os.path.join(web_dir, "static"),
        static_url_path="/static",
    )
    app.config["SECRET_KEY"] = "mars-rover-ops"
    # 같은 출처(이 서버가 내보낸 화면)에서만 Socket.IO 를 받는다 — 다른 사이트의
    # 스크립트가 사용자의 브라우저를 빌려 버튼을 누르지 못하게.
    sio = SocketIO(app, async_mode=_ASYNC_MODE)

    rclpy.init(args=args)
    ros_node = MissionWebRosNode(sio)

    web_port = int(os.environ.get("WEB_PORT", "8088"))
    video_host = os.environ.get("WEB_VIDEO_HOST", "127.0.0.1")
    video_port = int(os.environ.get("WEB_VIDEO_PORT", "8080"))
    token = os.environ.get("WEB_TOKEN", "").strip()
    actions_on = os.environ.get("WEB_ACTIONS", "1").strip().lower() not in (
        "0", "false", "no", "off")

    def authorized() -> bool:
        if not token:
            return True
        got = request.cookies.get("hud_key", "")
        return hmac.compare_digest(got.encode(), token.encode())

    def origin() -> str:
        # 터널을 거치면 remote_addr 은 터널 쪽 주소다. 그래도 누가 눌렀는지의 단서.
        fwd = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        return fwd or (request.remote_addr or "?")

    @app.before_request
    def _gate():
        if not token:
            return None
        key = request.args.get("key")
        if key is not None:
            if not hmac.compare_digest(key.encode(), token.encode()):
                abort(403)
            # 키를 주소창에 남기지 않도록 쿠키에 옮기고 키 없는 주소로 돌려보낸다.
            resp = redirect(request.path)
            resp.set_cookie("hud_key", token, max_age=7 * 24 * 3600, httponly=True,
                            samesite="Lax")
            return resp
        if not authorized():
            abort(403)
        return None

    @app.route("/")
    def index():
        topics = ros_node.topics_for_camera_urls()
        def mjpeg(topic: str) -> str:
            # 같은 서버의 /stream — 브라우저가 어느 주소로 들어왔든 그대로 통한다.
            return f"/stream?topic={topic}"
        # terrain_preview_path 가 .../generated_terrains/terrain_NNNNN/preview.png
        # 패턴이라 부모 디렉토리 이름이 곧 terrain id.
        terrain_path = str(ros_node.get_parameter("terrain_preview_path").value)
        terrain_name = os.path.basename(os.path.dirname(terrain_path)) or "—"
        return render_template(
            "index.html",
            # 시연 단계: wrist 슬롯의 stream 을 body(nav) YOLO 로 변경
            # (cam_wrist 변수명은 viewport-wrist class/css 호환 위해 유지).
            cam_wrist=mjpeg(topics["body"]),
            cam_chase=mjpeg(topics["chase"]),
            # 좌하단 슬롯 — 팔(M0609) 손목 카메라. 예전의 terrain preview 자리.
            cam_arm=mjpeg(topics["wrist"]),
            terrain_name=terrain_name,
            actions_on=actions_on,
        )

    @app.route("/stream")
    def stream():
        """web_video_server 의 MJPEG 를 그대로 흘려보낸다 (포트 하나로 모으기)."""
        topic = request.args.get("topic", "")
        if not re.fullmatch(r"/[A-Za-z0-9_/]{1,120}", topic):
            abort(400)
        url = (f"http://{video_host}:{video_port}/stream?topic={topic}"
               f"&type=mjpeg&quality=70")
        try:
            up = urllib.request.urlopen(url, timeout=6.0)
        except Exception:
            abort(502)

        def relay():
            try:
                while True:
                    chunk = up.read1(65536)
                    if not chunk:
                        break
                    yield chunk
            except Exception:
                pass                       # 영상이 끊김 — 브라우저가 3 초 뒤 다시 연결한다
            finally:
                up.close()

        return Response(relay(), headers={"Cache-Control": "no-store"},
                        content_type=up.headers.get(
                            "Content-Type", "multipart/x-mixed-replace"))

    @app.route("/api/info")
    def api_info():
        return jsonify({
            "local": f"http://localhost:{web_port}",
            "lan": lan_urls(web_port),
            "key_required": bool(token),
            "actions": actions_on,
            "rovers": list(ros_node.rover_namespaces),
            "action_log": ros_node.action_log_path,
        })

    @app.route("/terrain/preview")
    def terrain_preview():
        path = str(ros_node.get_parameter("terrain_preview_path").value)
        if not os.path.isfile(path):
            abort(404)
        return send_file(path, mimetype="image/png")

    @sio.on("action")
    def _on_action(msg):
        if not authorized():
            return {"ok": False, "text": "접속 키 없음"}
        if not actions_on:
            return {"ok": False, "text": "보기 전용 (WEB_ACTIONS=0)"}
        msg = msg if isinstance(msg, dict) else {}
        sent = ros_node.do_action(str(msg.get("rover_id", "")), str(msg.get("action", "")),
                                  str(msg.get("value", "")), origin())
        return {"ok": not sent.startswith("모르는"), "text": sent}

    @sio.on("teleop")
    def _on_teleop(msg):
        if not authorized() or not actions_on or not isinstance(msg, dict):
            return
        try:
            ros_node.do_teleop(str(msg.get("rover_id", "")), float(msg.get("lin", 0.0)),
                               float(msg.get("ang", 0.0)), origin())
        except (TypeError, ValueError):
            pass

    @sio.on("connect")
    def _on_connect():
        if not authorized():
            return False
        snap = ros_node.last_snapshot()
        # rover namespace 리스트 먼저 — JS 가 selector UI 구성 시 사용
        sio.emit("rovers", {"namespaces": snap["rover_namespaces"]}, to=request.sid)
        # Per-rover cache 를 ns 별로 re-emit. payload 마다 rover_id 박혀있음
        # (사실 캐시 시점 emit 의 payload 와 동일). 늦게 들어온 client 도
        # 모든 rover 의 latest state 받음.
        for event_key in ("state", "odom", "minimap", "path", "target"):
            for ns, payload in snap[event_key].items():
                if payload is None:
                    continue
                p = dict(payload)
                p["rover_id"] = ns
                sio.emit(event_key, p, to=request.sid)
        sio.emit("events", {"items": snap["events"]}, to=request.sid)

    spinner = threading.Thread(target=_spin_ros, args=(ros_node,), daemon=True)
    spinner.start()

    host = os.environ.get("WEB_HOST", "0.0.0.0")
    port = web_port
    ros_node.get_logger().info(
        f"web server listening on http://{host}:{port}  "
        f"(async={_ASYNC_MODE}, web_dir={web_dir})")
    suffix = "/?key=<WEB_TOKEN>" if token else ""
    ros_node.get_logger().info(
        f"HUD 주소 — 이 PC: http://localhost:{port}{suffix}   같은 네트워크: "
        + ("  ".join(u + suffix for u in lan_urls(port)) or "(주소를 못 찾음)")
        + f"   · 접속 키 {'필요' if token else '없음'} · 버튼 {'켜짐' if actions_on else '꺼짐(보기 전용)'}"
        + f" · 버튼 기록 {ros_node.action_log_path}")
    try:
        sio.run(app, host=host, port=port,
                debug=False, use_reloader=False,
                allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        pass
    finally:
        ros_node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
