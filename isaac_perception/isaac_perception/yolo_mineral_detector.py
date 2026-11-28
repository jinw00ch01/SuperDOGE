"""YOLO Mineral Detector — best.pt 로 RGB 이미지에서 mineral detect.

사용 예:
    from yolo_mineral_detector import YoloMineralDetector
    det = YoloMineralDetector("models/mineral_yolo_best.pt", conf=0.5)
    dets = det.detect(bgr_image)
    for d in dets:
        print(d.cls_name, d.conf, d.bbox)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np


CLASS_COLORS_BY_NAME = {
    "blue_mineral":   (255, 100, 100),   # BGR — blue/cyan tint
    "yellow_mineral": (50, 220, 255),    # BGR — yellow
    "green_gas":      (100, 255, 100),   # BGR — green
}
_DEFAULT_COLOR = (200, 200, 200)


@dataclass
class Detection:
    cls_id: int
    cls_name: str
    conf: float
    bbox: tuple        # (x1, y1, x2, y2) in pixels
    cx: float
    cy: float


def _cuda_wait_blocking() -> bool:
    """GPU 결과를 기다리는 동안 CPU 를 쉬게 한다 (cudaDeviceScheduleBlockingSync).

    CUDA 의 기본값은 기다리는 동안 CPU 코어 하나를 계속 돌리는 것이다. 시뮬레이터와 GPU 를
    나눠 쓰면 기다리는 시간이 길어서, 추론 한 번에 CPU 를 20~28 ms 썼다 (벽시계와 같음).
    이 설정으로 10 ms 가 된다 (걸리는 시간은 같다). CUDA 를 처음 쓰기 전에 불러야 한다.
    ROVER_YOLO_SPIN=1 이면 건너뛴다."""
    import os
    if os.environ.get("ROVER_YOLO_SPIN", "") == "1":
        return False
    try:
        import ctypes
        import torch  # noqa: F401
        import torchvision  # noqa: F401  (자기 몫의 CUDA runtime 을 따로 싣는다)
        ok = False
        libs = sorted({line.split()[-1] for line in open("/proc/self/maps")
                       if "libcudart" in line})
        for path in libs:
            ok |= ctypes.CDLL(path).cudaSetDeviceFlags(ctypes.c_uint(4)) == 0
        return ok
    except Exception:
        return False


class YoloMineralDetector:
    def __init__(self, model_path: str, conf: float = 0.5,
                 iou: float = 0.45, device: Optional[str] = None):
        self.blocking_wait = _cuda_wait_blocking()
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics 미설치. 'pip install ultralytics'\n"
                f"원본 에러: {e}"
            )
        mp = Path(model_path).expanduser().resolve()
        if not mp.exists():
            raise FileNotFoundError(f"YOLO model not found: {mp}")
        self.model = YOLO(str(mp))
        self.conf = conf
        self.iou = iou
        self.device = device
        self._fp16_kw = self._pick_fp16()
        raw_names = self.model.names
        if isinstance(raw_names, dict):
            self.names = [raw_names[i] for i in sorted(raw_names.keys())]
        else:
            self.names = list(raw_names)

    def _pick_fp16(self) -> dict:
        """GPU 에서는 반정밀도(FP16)로 추론한다 — GPU 시간이 절반이다 (16 → 8 ms).

        실제 카메라 영상 577 장으로 FP32 와 비교: 검출 수가 다른 것 1 장(conf 가 딱 0.50 인 것),
        검출 115 개의 conf 차이 최대 0.002, 상자 차이 최대 0.53 px, 종류가 바뀐 것 0.
        ultralytics 판에 따라 인자 이름이 다르다 (quantize / half) → 되는 것을 고른다.
        ROVER_YOLO_FP32=1 이면 쓰지 않는다."""
        import os
        if os.environ.get("ROVER_YOLO_FP32", "") == "1":
            return {}
        try:
            import torch
            if not torch.cuda.is_available():
                return {}
        except Exception:
            return {}
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        for kw in ({"quantize": "fp16"}, {"half": True}):
            try:
                self.model(dummy, conf=0.5, verbose=False, **kw)
                return kw
            except Exception:
                continue
        return {}

    def detect(self, bgr: np.ndarray) -> List[Detection]:
        """BGR 이미지 → Detection 리스트."""
        kwargs = {"conf": self.conf, "iou": self.iou, "verbose": False, **self._fp16_kw}
        if self.device is not None:
            kwargs["device"] = self.device
        results = self.model(bgr, **kwargs)
        out: List[Detection] = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = xyxy
                out.append(Detection(
                    cls_id=cls_id,
                    cls_name=self.names[cls_id] if cls_id < len(self.names) else f"cls{cls_id}",
                    conf=conf,
                    bbox=(x1, y1, x2, y2),
                    cx=(x1 + x2) / 2,
                    cy=(y1 + y2) / 2,
                ))
        return out

    @staticmethod
    def draw_overlay(bgr: np.ndarray, dets: List[Detection]) -> np.ndarray:
        """Detection 들을 이미지에 그려서 반환 (copy)."""
        import cv2
        out = bgr.copy()
        for d in dets:
            x1, y1, x2, y2 = [int(v) for v in d.bbox]
            color = CLASS_COLORS_BY_NAME.get(d.cls_name, _DEFAULT_COLOR)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = f"{d.cls_name} {d.conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, max(0, y1 - th - 6)),
                          (x1 + tw + 4, y1), color, -1)
            cv2.putText(out, label, (x1 + 2, max(12, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        return out


__all__ = ["YoloMineralDetector", "Detection", "CLASS_COLORS_BY_NAME"]
