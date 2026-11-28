"""Mission web node — Flask + Socket.IO dashboard for Mars rover ops.

화성↔지구 통신 비유: 브라우저(지구)가 ROS2 토픽(화성 rover)을 web 으로
구독한다. ROS2 콜백 -> Socket.IO emit -> 브라우저가 SC2 풍 HUD 갱신.

접속 주소
---------
포트 하나(WEB_PORT, 기본 8088)로 화면·상태·카메라 영상을 모두 내보낸다. 카메라는
web_video_server(127.0.0.1:WEB_VIDEO_PORT)의 MJPEG 를 이 서버가 /stream 으로 다시
내보내므로, 브라우저는 이 서버 주소 하나만 알면 된다 → localhost 뿐 아니라
  · 같은 네트워크의 다른 기기:  http://<이 PC 의 IP>:8088  (시작할 때 로그에 찍힌다)
  · 인터넷 주소: WEB_PUBLIC=1 이면 cloudflared 터널을 열어 https://….trycloudflare.com
    주소를 받는다 (PublicTunnel). 주소는 로그와 화면 위쪽에 나온다.
에서도 그대로 열린다. WEB_TOKEN(접속 키)을 주면 `?key=<키>` 로 한 번 들어온
브라우저만 볼 수 있다 (이 PC 에서 localhost 로 여는 것은 키 없이 된다). 인터넷 주소는
키 없이는 열지 않는다 — WEB_PUBLIC=1 인데 키가 없으면 하나 만들어 붙인다 (ACTIONS
버튼이 로버를 움직인다). WEB_ACTIONS=0 이면 버튼이 꺼진 보기 전용이 된다.

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
  GET /api/info -> 접속 주소·설정 (같은 네트워크 주소, hud_public 이면 인터넷 주소)
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
import secrets
import shutil
import socket
import subprocess
import threading
import time
import urllib.request
from collections import deque
from typing import Any, Dict, List, Optional

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


class PublicTunnel:
    """HUD 포트 하나를 터널로 내보내 인터넷 주소(https://….trycloudflare.com)를 받는다.

    WEB_PUBLIC=1 일 때만. 밖에서 들어오는 연결을 여는 것이 아니라 이 PC 가 Cloudflare
    로 나가는 연결을 만들고, 그쪽이 받은 요청을 이 연결로 돌려보낸다. 그래서 공유기
    설정이나 방화벽 변경 없이 어디서든 열린다 — 대신 영상과 상태가 Cloudflare 의
    서버를 거친다.

    계정 없이 쓰는 임시 주소는 cloudflared 프로세스가 새로 뜰 때마다 바뀐다. 스택을
    다시 띄울 때마다 주소가 바뀌면 이미 넘겨준 주소가 죽으므로, **터널 프로세스는
    HUD 노드와 따로 살려 둔다**: 떠 있는 터널이 있으면(STATE 파일의 pid) 그대로 다시
    쓰고, 없을 때만 새로 띄운다. HUD 가 내려가 있는 동안 그 주소는 502 를 내다가
    HUD 가 다시 뜨면 이어진다. 주소가 바뀌는 때는 재부팅했거나 터널을 직접 닫았을 때
    (tools/hud_tunnel.sh stop)뿐이다.

    **고정 주소 (Tailscale Funnel)**: tailscale 이 설치돼 있고 로그인돼 있으면 그쪽을 먼저
    쓴다. 주소가 https://<PC 이름>.<tailnet 이름>.ts.net 으로 정해져 있어 재부팅해도 같다
    (`tailscale funnel --bg <port>` — tailscaled 가 설정을 기억한다). 이때 떠 있던
    cloudflared 임시 터널은 닫는다 (공개 주소는 하나만). 로그인돼 있지 않으면 아래의
    임시 주소로 간다.

    cloudflared 실행 파일이 필요하다 (PATH 또는 ~/.local/bin). ssh 만으로 되는
    localhost.run 도 시험했지만 익명 터널이 2~20 kB/s 로 묶이고 주소가 도중에 바뀌어
    영상은커녕 화면도 제대로 뜨지 않았다 — 쓰지 않는다.
    """

    URL_RE = re.compile(r"https://[a-z0-9-]+\.trycloudflare\.com")
    INSTALL_HINT = ("cloudflared 가 없어 인터넷 주소를 만들지 못함 — "
                    "https://github.com/cloudflare/cloudflared/releases 의 "
                    "cloudflared-linux-amd64 를 ~/.local/bin/cloudflared 로 두면 된다")
    STATE = os.path.expanduser("~/.ros/hud_tunnel.json")      # {pid, url, port}
    LOG = os.path.expanduser("~/.ros/hud_tunnel.log")         # cloudflared 가 찍는 것

    def __init__(self, port: int, on_url, log) -> None:
        self.port = port
        self.on_url = on_url
        self.log = log
        self.url = ""
        self.provider = "cloudflared"
        self.pid = 0
        self.exe = shutil.which("cloudflared") or ""
        self.ts = shutil.which("tailscale") or ""
        local = os.path.expanduser("~/.local/bin/cloudflared")
        if not self.exe and os.access(local, os.X_OK):
            self.exe = local
        if not self.exe and not self.ts:
            log(self.INSTALL_HINT)
            return
        threading.Thread(target=self._run, daemon=True).start()

    # ── Tailscale Funnel (고정 주소) ────────────────────────────────────
    def _ts(self, *args, timeout: float = 8.0) -> str:
        try:
            r = subprocess.run([self.ts, *args], capture_output=True, text=True,
                               timeout=timeout, stdin=subprocess.DEVNULL)
            return (r.stdout or "") + (r.stderr or "")
        except (OSError, subprocess.SubprocessError) as e:
            return f"ERR {e}"

    def _ts_url(self) -> str:
        """Tailscale 에 로그인돼 있으면 이 PC 의 고정 주소, 아니면 ""."""
        if not self.ts:
            return ""
        try:
            st = json.loads(subprocess.run(
                [self.ts, "status", "--json"], capture_output=True, text=True, timeout=8.0,
                stdin=subprocess.DEVNULL).stdout)
        except (OSError, ValueError, subprocess.SubprocessError):
            return ""
        name = str((st.get("Self") or {}).get("DNSName") or "").rstrip(".")
        if st.get("BackendState") != "Running" or not name.endswith(".ts.net"):
            return ""
        return f"https://{name}"

    def _ts_funnel_on(self) -> bool:
        out = self._ts("funnel", "status")
        return "Funnel on" in out and f"127.0.0.1:{self.port}" in out

    def _ts_funnel_start(self) -> bool:
        out = self._ts("funnel", "--bg", str(self.port), timeout=20.0)
        if self._ts_funnel_on():
            return True
        self.log("Tailscale Funnel 을 켜지 못함 — " + " ".join(out.split())[:300])
        return False

    def _close_cloudflared(self) -> None:
        """고정 주소로 넘어갈 때 떠 있던 임시 터널을 닫는다."""
        try:
            with open(self.STATE) as f:
                st = json.load(f)
            pid = int(st.get("pid", 0))
        except (OSError, ValueError):
            return
        if st.get("provider") == "tailscale":
            return
        if self._alive(pid):
            try:
                os.kill(pid, 15)
                self.log(f"임시 터널을 닫음: {st.get('url')}")
            except OSError:
                pass

    def _run_tailscale(self) -> bool:
        """고정 주소로 내보내고, 내려갈 때까지 지킨다. 쓸 수 없으면 바로 False."""
        url = self._ts_url()
        if not url:
            return False
        reused = self._ts_funnel_on()
        if not reused and not self._ts_funnel_start():
            return False
        self._close_cloudflared()
        self.provider, self.url, self.pid = "tailscale", url, 0
        tmp = self.STATE + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"pid": 0, "url": url, "port": self.port, "provider": "tailscale",
                       "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, f)
        os.replace(tmp, self.STATE)
        self.on_url(url, "tailscale funnel · 고정 주소" + (" · 켜져 있던 것을 다시 씀" if reused else ""))
        misses = 0
        while misses < 3:                      # 한 번 실패로 끊겼다고 보지 않는다
            time.sleep(15.0)
            misses = 0 if self._ts_funnel_on() else misses + 1
        self.url = ""
        self.on_url("", self.provider)
        return True

    def _alive(self, pid: int) -> bool:
        """그 pid 가 이 포트를 내보내는 cloudflared 인가."""
        try:
            with open(f"/proc/{pid}/cmdline", "rb") as f:
                cmd = f.read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            return False
        return "cloudflared" in cmd and f"127.0.0.1:{self.port}" in cmd

    def _reuse(self) -> bool:
        try:
            with open(self.STATE) as f:
                st = json.load(f)
        except (OSError, ValueError):
            return False
        if int(st.get("port", 0)) != self.port or not st.get("url") \
                or not self._alive(int(st.get("pid", 0))):
            return False
        self.pid, self.url = int(st["pid"]), str(st["url"])
        return True

    def _spawn(self) -> bool:
        """터널을 새로 띄우고 주소가 찍힐 때까지 기다린다. HUD 노드가 죽어도 남는다."""
        os.makedirs(os.path.dirname(self.LOG), exist_ok=True)
        with open(self.LOG, "w") as out:
            proc = subprocess.Popen(
                [self.exe, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{self.port}"],
                stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT,
                start_new_session=True)
        for _ in range(120):                       # 최대 60 s
            time.sleep(0.5)
            if proc.poll() is not None:
                return False
            try:
                with open(self.LOG, errors="replace") as f:
                    m = self.URL_RE.search(f.read())
            except OSError:
                m = None
            if m:
                self.pid, self.url = proc.pid, m.group(0)
                tmp = self.STATE + ".tmp"
                with open(tmp, "w") as f:
                    json.dump({"pid": self.pid, "url": self.url, "port": self.port,
                               "started": time.strftime("%Y-%m-%dT%H:%M:%S")}, f)
                os.replace(tmp, self.STATE)
                return True
        proc.terminate()
        return False

    def _run(self) -> None:
        wait = 5.0
        while True:
            if self._run_tailscale():
                continue
            if not self.exe:
                time.sleep(30.0)               # tailscale 로그인을 기다린다
                continue
            self.provider = "cloudflared"
            reused = self._reuse()
            if reused or self._spawn():
                wait = 5.0
                self.on_url(self.url, self.provider + (" · 떠 있던 터널을 다시 씀" if reused else ""))
                while self._alive(self.pid):
                    time.sleep(5.0)
                self.url = ""
                self.on_url("", self.provider)
            else:
                wait = min(wait * 2.0, 120.0)
            self.log(f"터널({self.provider})이 없음 — {wait:.0f} s 뒤 새로 연다 (주소가 바뀐다)")
            time.sleep(wait)


def persistent_key(path: str = "~/.ros/hud_key") -> str:
    """hud_public 인데 hud_key 를 안 줬을 때 쓰는 접속 키. 한 번 만들어 파일에 두고 계속 쓴다
    (스택을 다시 띄워도 넘겨준 주소가 그대로 열리게). 바꾸려면 파일을 지우면 된다."""
    path = os.path.expanduser(path)
    try:
        with open(path) as f:
            key = f.read().strip()
        if len(key) >= 8:
            return key
    except OSError:
        pass
    key = secrets.token_urlsafe(9)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as f:
        f.write(key + "\n")
    return key


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
        self._minimap_sig: Dict[str, tuple] = {}     # 마지막으로 보낸 지도 (바뀌었는지 비교용)
        self._odom_emit_ns: Dict[str, int] = {}      # 로버별 마지막 odom 전송 시각
        self._cmd_emit_ns: Dict[str, int] = {}
        self._last_terrain: Dict[str, dict] = {}     # 주변 높낮이 (hazard_map_node 의 요약)
        self._last_minerals: Dict[str, dict] = {}    # 기억한 광물 (mission_manager)
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
        self._events: deque = deque(maxlen=1500)          # 늦게 들어온 브라우저에게 줄 최근 기록
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
            String, self._topic(ns, "mission/known_minerals"),
            partial(self._on_known_minerals, ns), 5)
        self.create_subscription(
            String, self._topic(ns, "mission/event"),
            partial(self._on_event, ns), 20)
        self.create_subscription(
            String, self._topic(ns, "elevation/summary"),
            partial(self._on_terrain, ns), 5)

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
            "terrain": dict(self._last_terrain),
            "minerals": dict(self._last_minerals),
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
        # 화면에는 초당 5 번이면 충분하다 — odom 은 초당 수십 번 오고, 그때마다 브라우저로
        # 보내면 보내는 쪽(파이썬 Socket.IO)과 받는 쪽(지도 다시 그리기)이 모두 바쁘다.
        if now_ns - self._odom_emit_ns.get(ns, 0) < int(0.2e9):
            return
        self._odom_emit_ns[ns] = now_ns
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
        now_ns = self.get_clock().now().nanoseconds
        if now_ns - self._cmd_emit_ns.get(ns, 0) < int(0.2e9):      # 화면에는 초당 5 번
            return
        self._cmd_emit_ns[ns] = now_ns
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

    def _on_known_minerals(self, ns: str, msg: String) -> None:
        """기억한 (아직 못 모은) 광물 목록 — 지도에 번호와 함께 그린다. JSON 그대로 넘긴다."""
        try:
            payload = json.loads(msg.data)
        except ValueError:
            return
        if self._last_minerals.get(ns) == payload:
            return
        self._last_minerals[ns] = payload
        self._emit("minerals", ns, payload)

    def _on_terrain(self, ns: str, msg: String) -> None:
        """로버 둘레의 높낮이 (지형 스캐너 → 높이 지도 요약). JSON 그대로 넘긴다."""
        try:
            payload = json.loads(msg.data)
        except ValueError:
            return
        self._last_terrain[ns] = payload
        self._emit("terrain", ns, payload)

    def _on_minimap(self, ns: str, msg: OccupancyGrid) -> None:
        """OccupancyGrid → 브라우저가 canvas 에 그릴 수 있는 작은 묶음.

        예전에는 칸 값을 숫자 목록 그대로 보냈다 (500×500 = 25 만 개, 로버마다 초당 3 번).
        Socket.IO 가 그 목록을 파이썬으로 한 칸씩 훑고 JSON 으로 바꾸느라 이 노드가 CPU 한
        코어의 76 % 를 썼고 그 80 % 가 여기였다 (로버 3 대, logs/prof1). 지도는 같은 값이 길게
        이어지므로 "값,개수" 를 이어 붙인 문자열 하나로 보내고, 바뀌지 않았으면 보내지 않는다.
        """
        import numpy as np
        raw = msg.data.tobytes() if hasattr(msg.data, "tobytes") else bytes(
            (v & 0xFF) for v in msg.data)
        meta = (int(msg.info.width), int(msg.info.height))
        if self._minimap_sig.get(ns) == (meta, raw):
            return
        self._minimap_sig[ns] = (meta, raw)
        a = np.frombuffer(raw, dtype=np.int8)
        if a.size:
            starts = np.concatenate(([0], np.flatnonzero(a[1:] != a[:-1]) + 1))
            lens = np.diff(np.concatenate((starts, [a.size])))
            rle = ",".join(map(str, np.column_stack(
                (a[starts].astype(np.int64), lens)).ravel().tolist()))
        else:
            rle = ""
        payload = {
            "w": meta[0],
            "h": meta[1],
            "res": float(msg.info.resolution),
            "ox": float(msg.info.origin.position.x),
            "oy": float(msg.info.origin.position.y),
            "rle": rle,
        }
        self._last_minimap[ns] = payload
        self._emit("minimap", ns, payload)


def _spin(node) -> None:
    """노드를 돌린다 — 되면 EventsExecutor(rclpy.experimental)로.

    기본 실행기는 콜백 하나를 처리할 때마다 구독·타이머 전체로 대기 목록을 다시 만든다.
    콜백 자체보다 그 비용이 크다 (mission_manager: 콜백 3 %, 그 밖 9 %). EventsExecutor 는
    준비된 것만 처리한다. ROVER_EXECUTOR=classic 이면 예전 실행기를 쓴다."""
    import os
    if os.environ.get("ROVER_EXECUTOR", "events") != "classic":
        try:
            from rclpy.experimental import EventsExecutor
        except ImportError:
            EventsExecutor = None
        if EventsExecutor is not None:
            executor = EventsExecutor()
            executor.add_node(node)
            node.get_logger().info("실행기: EventsExecutor")
            executor.spin()
            return
    rclpy.spin(node)


def _spin_ros(ros_node: Node) -> None:
    try:
        _spin(ros_node)
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
    public_on = os.environ.get("WEB_PUBLIC", "0").strip().lower() in ("1", "true", "yes", "on")
    # 인터넷 주소는 누구나 접속을 시도할 수 있다 — 접속 키 없이는 내보내지 않는다.
    token_made = public_on and not token
    if token_made:
        token = persistent_key()
    key_q = f"/?key={token}" if token else ""
    tunnel_box: Dict[str, Any] = {"t": None}

    TUNNEL_HEADERS = ("X-Forwarded-For", "X-Forwarded-Host", "Forwarded",
                      "Cf-Connecting-Ip", "Cf-Ray",
                      "Tailscale-Funnel-Request", "Tailscale-User-Login")

    def from_this_pc() -> bool:
        """이 PC 의 브라우저가 localhost 로 직접 연 요청인가.

        터널로 들어온 요청도 127.0.0.1 에서 오므로 주소만으로는 구분할 수 없다. 터널은
        Host 가 터널 도메인이고 전달 헤더(X-Forwarded-For, Cf-*)가 붙는다 — 둘 중
        하나라도 있으면 밖에서 온 것으로 본다.
        """
        if request.remote_addr not in ("127.0.0.1", "::1"):
            return False
        if via_tunnel():
            return False
        host = (request.host or "").rsplit(":", 1)[0] if not (request.host or "").startswith("[") \
            else (request.host or "").split("]")[0] + "]"
        return host in ("localhost", "127.0.0.1", "[::1]")

    def via_tunnel() -> bool:
        return any(h in request.headers for h in TUNNEL_HEADERS)

    def authorized() -> bool:
        if not token:
            # 키 없이 띄운 HUD 는 터널로 들어온 요청을 받지 않는다. 터널 프로세스는 HUD 와
            # 따로 살아 있으므로, hud_public 없이 다시 띄웠을 때 예전 주소로 열리면 안 된다.
            return not via_tunnel()
        if from_this_pc():                   # 이 PC 에서는 키 없이
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
        tunnel = tunnel_box["t"]
        public = tunnel.url if tunnel is not None else ""
        return jsonify({
            "local": f"http://localhost:{web_port}",
            "lan": lan_urls(web_port),
            # 인터넷 주소 (hud_public:=true 일 때. 터널이 열리기 전에는 빈 문자열)
            "public": public,
            "public_on": public_on,
            # 다른 기기에 그대로 넘겨줄 주소 — 접속 키가 붙어 있다. 이 응답은 이미
            # 키를 가진 브라우저만 받는다.
            "share_lan": [u + key_q for u in lan_urls(web_port)],
            "share_public": (public + key_q) if public else "",
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
        for event_key in ("state", "odom", "minimap", "terrain", "minerals", "path", "target"):
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
    # 직접 준 키는 로그에 다시 적지 않는다. 여기서 만든 키는 적어야 쓸 수 있다.
    suffix = key_q if token_made else ("/?key=<hud_key>" if token else "")
    local_note = " (이 PC 에서는 키 없이)" if token else ""

    def on_public_url(url: str, provider: str) -> None:
        if url:
            via = "Tailscale" if provider.startswith("tailscale") else "Cloudflare"
            keep = ("재부팅해도 이 주소는 그대로다" if via == "Tailscale"
                    else "스택을 다시 띄워도 이 주소는 그대로다 (재부팅하면 바뀐다)")
            ros_node.get_logger().info(
                f"HUD 주소 — 인터넷: {url}{suffix}   ({provider} 터널. 영상·상태가 {via} "
                f"서버를 거쳐 나간다. {keep}. 닫기: tools/hud_tunnel.sh stop)")
            ros_node.push_event("", "info", f"인터넷 주소 열림: {url}")
        else:
            ros_node.push_event("", "info", "인터넷 주소 닫힘 (터널이 끊김)")

    if public_on:
        tunnel_box["t"] = PublicTunnel(
            web_port, on_public_url, lambda m: ros_node.get_logger().warn(m))
    ros_node.get_logger().info(
        f"HUD 주소 — 이 PC: http://localhost:{port}{local_note}   같은 네트워크: "
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
