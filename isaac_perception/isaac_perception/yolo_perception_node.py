"""ROS2 perception node: YOLO mineral detection on rover (nav) camera and
optionally wrist camera. Publishes DetectionArray + annotated Image per
channel.

Two channels:
- nav (rover body cam): world-frame mineral coordinates (depth + odom +
  camera_offset). Used by mission_manager for APPROACH planning.
- wrist (gripper cam): camera optical frame mineral coordinates only.
  Used by arm_executor for precision visual servoing + IK target.

Contract: docs/interfaces/INTERFACE_CONTRACTS.md I2.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import rclpy
from geometry_msgs.msg import Point, Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Header

from isaac_interfaces.msg import Detection, DetectionArray


SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
)

# Annotated image 만 RELIABLE 로 발행 — web_video_server (RELIABLE subscribe)
# 와 QoS 호환을 위해. detection FSM 자체는 SENSOR_QOS 그대로 유지.
ANNOTATED_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)


VALUE_SCORE_BY_NAME = {
    "blue_mineral":   10.0,
    "yellow_mineral": 50.0,
    "green_gas":      25.0,
}


@dataclass
class CamIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


def _image_to_bgr(msg: Image) -> np.ndarray:
    if msg.encoding not in ("rgb8", "bgr8"):
        raise ValueError(f"unsupported image encoding: {msg.encoding}")
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    img = buf.reshape(msg.height, msg.width, 3)
    if msg.encoding == "rgb8":
        img = img[..., ::-1]
    return np.ascontiguousarray(img)


def _stamp_sec(header) -> float:
    return header.stamp.sec + header.stamp.nanosec * 1e-9


def _quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """xyzw quaternion → 3x3 회전 행렬."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _quat_yaw(q: np.ndarray) -> float:
    x, y, z, w = q
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _depth_to_meters(msg: Image) -> Optional[np.ndarray]:
    if msg.encoding == "32FC1":
        arr = np.frombuffer(bytes(msg.data), dtype=np.float32)
        return arr.reshape(msg.height, msg.width).copy()
    if msg.encoding == "16UC1":
        arr = np.frombuffer(bytes(msg.data), dtype=np.uint16)
        return (arr.reshape(msg.height, msg.width).astype(np.float32) * 1e-3).copy()
    return None


class CamChannel:
    """Per-camera processing: subscribes to RGB/Depth/Info(/Odom), runs YOLO
    on the latest RGB frame at the channel's tick, and publishes
    DetectionArray + (optional) annotated Image.

    mode="world":   compute world-frame XYZ for each detection using
                    depth + intrinsics + camera_offset + odom (full
                    orientation, sampled at the image time). Needs odom_topic.
    mode="optical": compute camera optical frame XYZ only (z forward,
                    x right, y down). Downstream node (e.g. arm IK) does
                    its own frame transform.

    Both modes report the estimated *centre* of the object: depth gives the
    visible surface, which is pushed back along the viewing ray by the
    object's radius (from the bbox size).

    world 모드 좌표 변환의 전제
      · odom 의 기준점은 팔 받침대(m0609/base_link) 다 — run_vehicle_v3 의
        GT odom 스크립트가 articulation root 의 자세를 낸다. camera_offset_xyz
        는 그 기준점에서 본 카메라 위치 (x 전방, y 좌측, z 위).
      · 카메라 광축은 기준점의 +x 와 나란하다 (vehicle_v3.usd 실측, 기울기 0).
      · 영상의 stamp 는 실제 찍힌 시각보다 image_latency_sec 만큼 늦다.
    """

    def __init__(self, *, node: Node, name: str, detector,
                 rgb_topic: str, depth_topic: str, info_topic: str,
                 det_topic: str, ann_topic: str, frame_id: str,
                 mode: str, publish_annotated: bool,
                 odom_topic: Optional[str] = None,
                 camera_offset_xyz: Optional[Tuple[float, float, float]] = None,
                 image_latency_sec: float = 0.0,
                 max_yaw_rate: float = 0.0,
                 object_radius_m: float = 0.13):
        assert mode in ("world", "optical")
        self.node = node
        self.name = name
        self.detector = detector
        self.frame_id = frame_id
        self.mode = mode
        self.publish_annotated_flag = publish_annotated
        self.camera_offset = np.array(camera_offset_xyz or (0.0, 0.0, 0.0), dtype=float)
        self.image_latency_sec = float(image_latency_sec)
        self.max_yaw_rate = float(max_yaw_rate)
        self.object_radius_m = float(object_radius_m)

        self.intr: Optional[CamIntrinsics] = None
        self.last_rgb: Optional[Image] = None
        self.last_depth_m: Optional[np.ndarray] = None
        self.last_odom: Optional[Odometry] = None
        # (stamp, depth[m]) — RGB 와 같은 stamp 의 depth 를 고르기 위해 몇 장 보관.
        self._depth_buf: deque = deque(maxlen=6)
        # (stamp, position xyz, orientation xyzw) — 영상 시각의 자세 보간용.
        self._odom_buf: deque = deque(maxlen=600)
        self.skipped_turning = 0
        self._warned_clock = False
        # world 모드: 온전히 보인 검출로 쌓은 물체별 위치 기억. 가까이 가서 bbox 가
        # 화면 밖으로 잘리면 bbox 중심이 물체 중심에서 벗어나므로 (실측 9~10 cm),
        # 그때는 잘리기 전에 본 위치를 대신 낸다. 광물은 스스로 움직이지 않는다.
        self._tracks: List[dict] = []
        self.track_gate_m = 0.5
        self.track_window = 9
        self.track_ttl_sec = 120.0
        self.used_memory = 0

        self.frames_seen = 0
        self.frames_published = 0
        self.detections_total = 0

        node.create_subscription(Image, rgb_topic, self._on_rgb, SENSOR_QOS)
        node.create_subscription(Image, depth_topic, self._on_depth, SENSOR_QOS)
        node.create_subscription(CameraInfo, info_topic, self._on_info, SENSOR_QOS)
        if mode == "world" and odom_topic:
            node.create_subscription(Odometry, odom_topic, self._on_odom, SENSOR_QOS)

        self.pub_det = node.create_publisher(DetectionArray, det_topic, 10)
        self.pub_ann = None
        if publish_annotated:
            self.pub_ann = node.create_publisher(Image, ann_topic, ANNOTATED_QOS)

    def _log(self, msg: str, level: str = "info") -> None:
        # rclpy logger caches severity per source line, so call distinct
        # methods from distinct lines.
        full = f"[{self.name}] {msg}"
        logger = self.node.get_logger()
        if level == "warn":
            logger.warn(full)
        elif level == "error":
            logger.error(full)
        else:
            logger.info(full)

    def _on_rgb(self, msg: Image) -> None:
        self.last_rgb = msg
        self.frames_seen += 1

    def _on_depth(self, msg: Image) -> None:
        first = self.last_depth_m is None
        if first:
            self._log(f"depth msg: encoding={msg.encoding} {msg.width}x{msg.height}")
        d = _depth_to_meters(msg)
        if d is None:
            if first:
                self._log(f"depth decode FAILED for encoding={msg.encoding}", "warn")
            return
        if first:
            finite = np.isfinite(d)
            n_finite = int(finite.sum())
            if n_finite > 0:
                mn = float(d[finite].min())
                mx = float(d[finite].max())
                med = float(np.median(d[finite]))
            else:
                mn = mx = med = float('nan')
            self._log(
                f"depth array: shape={d.shape} finite={n_finite}/{d.size} "
                f"min={mn:.3f} max={mx:.3f} med={med:.3f}")
        self.last_depth_m = d
        if d is not None:
            self._depth_buf.append((_stamp_sec(msg.header), d))

    def _on_info(self, msg: CameraInfo) -> None:
        first = self.intr is None
        k = msg.k
        self.intr = CamIntrinsics(
            fx=float(k[0]), fy=float(k[4]),
            cx=float(k[2]), cy=float(k[5]),
            width=int(msg.width), height=int(msg.height),
        )
        if first:
            self._log(
                f"camera_info: {msg.width}x{msg.height} fx={self.intr.fx:.1f} "
                f"cx={self.intr.cx:.1f} cy={self.intr.cy:.1f}")

    def _on_odom(self, msg: Odometry) -> None:
        first = self.last_odom is None
        self.last_odom = msg
        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self._odom_buf.append((_stamp_sec(msg.header),
                               np.array([p.x, p.y, p.z]),
                               np.array([q.x, q.y, q.z, q.w])))
        if first:
            self._log(f"odom first: pos=({p.x:.2f},{p.y:.2f},{p.z:.2f})")

    def _depth_for(self, t_img: float) -> Optional[np.ndarray]:
        """RGB 와 stamp 가 가장 가까운 depth 프레임."""
        if not self._depth_buf:
            return None
        return min(self._depth_buf, key=lambda e: abs(e[0] - t_img))[1]

    def _pose_at(self, t: float):
        """시각 t 의 (position, rotation matrix, yaw rate). 없으면 None.

        odom 과 영상의 stamp 가 같은 시계(sim time)일 때만 보간한다. 1 초 넘게
        어긋나면 시계가 다르다고 보고 가장 최근 odom 을 쓴다.
        """
        buf = self._odom_buf
        if not buf:
            return None
        if len(buf) < 2 or abs(t - buf[-1][0]) > 1.0:
            if len(buf) >= 2 and not self._warned_clock:
                self._warned_clock = True
                self._log(f"영상 stamp({t:.3f}) 와 odom stamp({buf[-1][0]:.3f}) 가 "
                          f"1 s 넘게 다름 — 시각 보간 없이 최신 odom 사용", "warn")
            _, p, q = buf[-1]
            return p, _quat_to_matrix(q), 0.0
        i = len(buf) - 1
        while i > 1 and buf[i - 1][0] > t:
            i -= 1
        t0, p0, q0 = buf[i - 1]
        t1, p1, q1 = buf[i]
        a = 0.0 if t1 <= t0 else float(np.clip((t - t0) / (t1 - t0), 0.0, 1.0))
        if float(np.dot(q0, q1)) < 0.0:
            q1 = -q1
        q = (1.0 - a) * q0 + a * q1
        p = (1.0 - a) * p0 + a * p1
        # yaw rate: 보간 구간 앞뒤로 조금 넓게 (잡음 완화).
        j0, j1 = max(0, i - 3), min(len(buf) - 1, i + 2)
        dt = buf[j1][0] - buf[j0][0]
        yaw_rate = 0.0
        if dt > 1e-6:
            dy = _quat_yaw(buf[j1][2]) - _quat_yaw(buf[j0][2])
            yaw_rate = abs(float(np.arctan2(np.sin(dy), np.cos(dy)))) / dt
        return p, _quat_to_matrix(q), yaw_rate

    def tick(self) -> None:
        msg = self.last_rgb
        if msg is None:
            return
        try:
            bgr = _image_to_bgr(msg)
        except Exception as e:
            self._log(f"RGB decode failed: {e}", "warn")
            return

        dets = self.detector.detect(bgr)
        t_img = _stamp_sec(msg.header)
        depth = self._depth_for(t_img)
        pose = None
        if self.mode == "world":
            pose = self._pose_at(t_img - self.image_latency_sec)
            # 회전 중에는 시각 오차가 방위 오차로 크게 번진다 → 좌표 무효 처리.
            # (0,0,0) 은 mission_manager 가 무시하는 기존 규약.
            if (pose is not None and self.max_yaw_rate > 0.0
                    and pose[2] > self.max_yaw_rate):
                pose = None
                self.skipped_turning += 1
        out = DetectionArray()
        out.header = Header()
        out.header.stamp = msg.header.stamp
        out.header.frame_id = self.frame_id

        for d in dets:
            vscore = VALUE_SCORE_BY_NAME.get(d.cls_name, 0.0)
            x1, y1, x2, y2 = [int(round(v)) for v in d.bbox]
            xyz, size_m, truncated = self._estimate_xyz(
                d.cx, d.cy, (x1, y1, x2, y2), depth, pose)
            if self.mode == "world" and any(xyz):
                xyz = self._remember(xyz, truncated, t_img)
            det_msg = Detection()
            det_msg.class_name = d.cls_name
            det_msg.world_position = Point(
                x=float(xyz[0]), y=float(xyz[1]), z=float(xyz[2]))
            det_msg.confidence = float(d.conf)
            det_msg.value_score = float(vscore)
            det_msg.mineral_id = -1
            det_msg.bbox_size_m = Vector3(
                x=float(size_m[0]), y=float(size_m[1]), z=float(size_m[2]))
            det_msg.bbox_xmin = x1
            det_msg.bbox_ymin = y1
            det_msg.bbox_xmax = x2
            det_msg.bbox_ymax = y2
            out.detections.append(det_msg)

        out.detections.sort(key=lambda d_: d_.confidence, reverse=True)
        self.pub_det.publish(out)
        self.frames_published += 1
        self.detections_total += len(out.detections)

        if self.pub_ann is not None:
            annotated = self.detector.draw_overlay(bgr, dets)
            img_msg = Image()
            img_msg.header.stamp = msg.header.stamp
            img_msg.header.frame_id = msg.header.frame_id
            img_msg.height = annotated.shape[0]
            img_msg.width = annotated.shape[1]
            img_msg.encoding = "bgr8"
            img_msg.is_bigendian = 0
            img_msg.step = annotated.shape[1] * 3
            img_msg.data = annotated.tobytes()
            self.pub_ann.publish(img_msg)

    def _estimate_xyz(self, px: float, py: float, bbox, depth, pose):
        """bbox 중심 화소 → 물체 중심의 3D 좌표.

        Returns (xyz, size_m, truncated).
        size_m = (bbox 폭[m], bbox 높이[m], 추정 지름[m]). 좌표를 못 구하면
        xyz = (0, 0, 0). truncated = bbox 가 화면 가장자리에 닿음.
        """
        zero = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), False)
        if self.intr is None or depth is None:
            return zero
        if self.mode == "world" and pose is None:
            return zero

        h, w = depth.shape
        x1, y1, x2, y2 = bbox
        bw, bh = max(1, x2 - x1), max(1, y2 - y1)
        # 표면 깊이: bbox 가운데 40% 영역의 유효 화소 중앙값 (한 화소는 구멍·
        # 경계에 약하다).
        u0 = int(np.clip(round(px - 0.2 * bw), 0, w - 1))
        u1 = int(np.clip(round(px + 0.2 * bw), u0, w - 1))
        v0 = int(np.clip(round(py - 0.2 * bh), 0, h - 1))
        v1 = int(np.clip(round(py + 0.2 * bh), v0, h - 1))
        patch = depth[v0:v1 + 1, u0:u1 + 1]
        valid = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 50.0)]
        if valid.size == 0:
            self._log(f"depth invalid in bbox centre px=({px:.0f},{py:.0f})", "warn")
            return zero
        z = float(np.median(valid))

        # Camera optical frame (z forward, x right, y down). depth 는 광축 방향 거리.
        pt = np.array([(px - self.intr.cx) * z / self.intr.fx,
                       (py - self.intr.cy) * z / self.intr.fy,
                       z])
        # 표면 → 중심: 시선 방향으로 물체 반지름만큼 민다. 반지름은 bbox 크기에서.
        w_m, h_m = bw * z / self.intr.fx, bh * z / self.intr.fy
        radius = float(np.clip(0.25 * (w_m + h_m), 0.02, 0.25))
        truncated = x1 <= 2 or y1 <= 2 or x2 >= w - 3 or y2 >= h - 3
        if truncated:
            radius = max(radius, self.object_radius_m)   # 잘린 bbox 는 작게 나옴
        pt = pt * (1.0 + radius / float(np.linalg.norm(pt)))
        size_m = (w_m, h_m, 2.0 * radius)

        if self.mode == "optical":
            return (float(pt[0]), float(pt[1]), float(pt[2])), size_m, truncated

        # Optical -> 기준점 축 (x forward, y left, z up) + 카메라 위치
        p_ref = np.array([pt[2], -pt[0], -pt[1]]) + self.camera_offset
        position, rot, _ = pose
        world = position + rot @ p_ref
        return (float(world[0]), float(world[1]), float(world[2])), size_m, truncated

    def _remember(self, xyz, truncated: bool, t_img: float):
        """온전한 검출은 기억에 쌓고, 잘린 검출은 기억된 위치로 바꿔 돌려준다."""
        p = np.array(xyz)
        self._tracks = [t for t in self._tracks
                        if abs(t_img - t["seen"]) < self.track_ttl_sec]
        track, best = None, self.track_gate_m
        for t in self._tracks:
            d = float(np.linalg.norm(t["pos"] - p))
            if d < best:
                track, best = t, d
        if truncated:
            if track is None or len(track["samples"]) < 3:
                return xyz
            track["seen"] = t_img
            self.used_memory += 1
            q = track["pos"]
            return (float(q[0]), float(q[1]), float(q[2]))
        if track is None:
            track = {"samples": deque(maxlen=self.track_window), "pos": p, "seen": t_img}
            self._tracks.append(track)
        track["samples"].append(p)
        track["pos"] = np.median(np.array(track["samples"]), axis=0)
        track["seen"] = t_img
        q = track["pos"]
        return (float(q[0]), float(q[1]), float(q[2]))

    def have_str(self) -> str:
        flags: List[str] = []
        if self.last_rgb is not None: flags.append("rgb")
        if self.last_depth_m is not None: flags.append("depth")
        if self.intr is not None: flags.append("info")
        if self.mode == "world" and self.last_odom is not None: flags.append("odom")
        return ",".join(flags) or "-"


class YoloPerceptionNode(Node):
    def __init__(self) -> None:
        super().__init__("yolo_perception_node")

        # Shared YOLO config
        self.declare_parameter("model_path", "")
        self.declare_parameter("conf_threshold", 0.5)
        self.declare_parameter("iou_threshold", 0.45)
        self.declare_parameter("publish_rate_hz", 10.0)
        self.declare_parameter("log_period_sec", 2.0)

        # Nav (rover body) cam — world frame output for mission_manager
        self.declare_parameter("nav_rgb_topic",   "/camera/rover/image_raw")
        self.declare_parameter("nav_depth_topic", "/camera/rover/depth")
        self.declare_parameter("nav_info_topic",  "/camera/rover/camera_info")
        self.declare_parameter("odom_topic",      "/ground_truth/odom")
        self.declare_parameter("nav_det_topic",   "/perception/detections")
        self.declare_parameter("nav_ann_topic",   "/perception/image_annotated")
        self.declare_parameter("nav_frame_id",    "world")
        # odom 기준점(m0609/base_link)에서 본 로버 카메라 위치. vehicle_v3.usd 실측.
        self.declare_parameter("nav_camera_offset_xyz", [0.147, 0.0, -0.312])
        # 영상 stamp 가 실제 장면보다 늦는 시간. `--render-every 2` 에서 주행
        # 데이터로 맞춘 값 (렌더 2 프레임). 0 이면 stamp 그대로.
        self.declare_parameter("nav_image_latency_sec", 0.067)
        # 이보다 빨리 도는 중의 검출은 world 좌표를 내지 않는다 (0 = 끔).
        self.declare_parameter("nav_max_yaw_rate", 0.25)
        # bbox 가 화면 밖으로 잘렸을 때 쓰는 물체 반지름 (광물 폭 0.26~0.30 m).
        self.declare_parameter("object_radius_m", 0.13)
        self.declare_parameter("nav_publish_annotated", True)

        # Wrist (gripper) cam — optical frame output for arm_executor
        self.declare_parameter("enable_wrist", True)
        self.declare_parameter("wrist_rgb_topic",   "/camera/wrist/image_raw")
        self.declare_parameter("wrist_depth_topic", "/camera/wrist/depth")
        self.declare_parameter("wrist_info_topic",  "/camera/wrist/camera_info")
        self.declare_parameter("wrist_det_topic",   "/perception/wrist_detections")
        self.declare_parameter("wrist_ann_topic",   "/perception/wrist_image_annotated")
        self.declare_parameter("wrist_frame_id",    "wrist_camera_optical")
        self.declare_parameter("wrist_publish_annotated", True)

        model_path = str(self.get_parameter("model_path").value).strip()
        if not model_path:
            from pathlib import Path
            here = Path(__file__).resolve().parent.parent
            model_path = str(here / "models" / "mineral_yolo_best.pt")

        from .yolo_mineral_detector import YoloMineralDetector
        self.detector = YoloMineralDetector(
            model_path,
            conf=float(self.get_parameter("conf_threshold").value),
            iou=float(self.get_parameter("iou_threshold").value),
        )
        self.get_logger().info(f"YOLO loaded: {model_path}")

        self.nav = CamChannel(
            node=self, name="nav", detector=self.detector,
            rgb_topic=str(self.get_parameter("nav_rgb_topic").value),
            depth_topic=str(self.get_parameter("nav_depth_topic").value),
            info_topic=str(self.get_parameter("nav_info_topic").value),
            odom_topic=str(self.get_parameter("odom_topic").value),
            det_topic=str(self.get_parameter("nav_det_topic").value),
            ann_topic=str(self.get_parameter("nav_ann_topic").value),
            frame_id=str(self.get_parameter("nav_frame_id").value),
            mode="world",
            publish_annotated=bool(self.get_parameter("nav_publish_annotated").value),
            camera_offset_xyz=tuple(
                float(v) for v in self.get_parameter("nav_camera_offset_xyz").value),
            image_latency_sec=float(self.get_parameter("nav_image_latency_sec").value),
            max_yaw_rate=float(self.get_parameter("nav_max_yaw_rate").value),
            object_radius_m=float(self.get_parameter("object_radius_m").value),
        )

        self.wrist: Optional[CamChannel] = None
        if bool(self.get_parameter("enable_wrist").value):
            self.wrist = CamChannel(
                node=self, name="wrist", detector=self.detector,
                rgb_topic=str(self.get_parameter("wrist_rgb_topic").value),
                depth_topic=str(self.get_parameter("wrist_depth_topic").value),
                info_topic=str(self.get_parameter("wrist_info_topic").value),
                odom_topic=None,
                det_topic=str(self.get_parameter("wrist_det_topic").value),
                ann_topic=str(self.get_parameter("wrist_ann_topic").value),
                frame_id=str(self.get_parameter("wrist_frame_id").value),
                mode="optical",
                publish_annotated=bool(self.get_parameter("wrist_publish_annotated").value),
                object_radius_m=float(self.get_parameter("object_radius_m").value),
            )

        rate_hz = max(0.5, float(self.get_parameter("publish_rate_hz").value))
        self.create_timer(1.0 / rate_hz, self._tick)
        log_period = max(0.5, float(self.get_parameter("log_period_sec").value))
        self.create_timer(log_period, self._log_status)
        self.get_logger().info(
            f"yolo_perception_node ready @ {rate_hz:.1f} Hz "
            f"(nav: world, wrist: {'optical' if self.wrist else 'disabled'})")

    def _tick(self) -> None:
        self.nav.tick()
        if self.wrist is not None:
            self.wrist.tick()

    def _log_status(self) -> None:
        line = (f"nav: seen={self.nav.frames_seen} pub={self.nav.frames_published} "
                f"dets={self.nav.detections_total} turn_skip={self.nav.skipped_turning} "
                f"mem={self.nav.used_memory} "
                f"have=[{self.nav.have_str()}]")
        if self.wrist is not None:
            line += (f" | wrist: seen={self.wrist.frames_seen} "
                     f"pub={self.wrist.frames_published} "
                     f"dets={self.wrist.detections_total} "
                     f"have=[{self.wrist.have_str()}]")
        self.get_logger().info(line)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = YoloPerceptionNode()
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
