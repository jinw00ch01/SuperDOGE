"""보는 쪽이 없는 카메라는 그리지 않는다 (`run_vehicle_v3.py --camera-gate on`, 기본).

왜 필요한가
-----------
로버 한 대에 렌더 대상이 5 개다: 몸체 카메라, 손목 컬러, 손목 깊이(같은 카메라를 따로 한 번 더),
태양 카메라, 추적 카메라. 로버 3 대면 15 개를 렌더할 때마다 전부 그렸다. 그런데
  · 태양 카메라는 localization 런치만 쓴다 (임무 스택에서는 구독자 0)
  · 추적 카메라는 HUD 에서 고른 로버 것만 본다 (나머지 2 대는 구독자 0)
  · 손목 카메라는 광물에 다가가 집는 동안에만 쓴다
그리지 않아도 되는 것을 그리느라 GPU 와 CPU(영상 발행)를 쓰고, 그 열이 시뮬레이터를 늦춘다.

어떻게
------
렌더 대상(IsaacCreateRenderProduct 노드)마다 거기서 나가는 영상 토픽을 찾아 두고, 0.5 초마다
그 토픽의 구독자 수를 센다. 하나도 없으면 그 렌더 대상의 갱신을 끈다 (`inputs:enabled` →
hydra_texture.set_updates_enabled). 구독자가 생기면 다시 켠다 — 화질은 그대로다.
camera_info 구독자는 세지 않는다 (영상을 안 보면서 info 만 붙들고 있는 노드가 있다).

구독하는 쪽이 필요할 때만 구독해야 효과가 난다:
  · yolo_perception_node — 손목 영상은 임무 단계가 APPROACH/PICK_READY 이거나 HUD 가 볼 때만
  · arm_executor_node   — 손목 깊이는 팔이 일하는 동안만
  · web_video_server     — 원래 보는 사람이 있을 때만 구독한다
"""
from __future__ import annotations

import time


class CameraGate:
    RP_TYPE = "isaacsim.core.nodes.IsaacCreateRenderProduct"
    CAM_TYPE = "isaacsim.ros2.bridge.ROS2CameraHelper"

    def __init__(self, stage, node, period: float = 0.5, hold_sec: float = 3.0):
        self._node = node
        self.period = float(period)
        self.hold = float(hold_sec)              # 구독자가 없어진 뒤 이만큼은 더 그린다
        self._next = 0.0
        self.items = []                          # {"path", "topics", "on", "seen"}
        self.switches = 0
        rps, helpers = {}, []
        for prim in stage.Traverse():
            a = prim.GetAttribute("node:type")
            if not a or not a.IsValid():
                continue
            t = a.Get()
            if t == self.RP_TYPE:
                rps[str(prim.GetPath())] = []
            elif t == self.CAM_TYPE:
                helpers.append(prim)
        for h in helpers:
            rp_attr = h.GetAttribute("inputs:renderProductPath")
            topic_attr = h.GetAttribute("inputs:topicName")
            if not rp_attr or not topic_attr:
                continue
            topic = str(topic_attr.Get() or "").strip()
            ns_attr = h.GetAttribute("inputs:nodeNamespace")
            ns = str(ns_attr.Get() or "").strip().strip("/") if ns_attr else ""
            if not topic:
                continue
            if not topic.startswith("/") or (ns and not topic.startswith(f"/{ns}/")):
                topic = (f"/{ns}/" if ns else "/") + topic.lstrip("/")
            for src in rp_attr.GetConnections():
                key = str(src.GetPrimPath())
                if key in rps:
                    rps[key].append(topic)
        for path, topics in rps.items():
            if not topics:
                continue                         # 무엇을 내보내는지 모르는 것은 건드리지 않는다
            self.items.append({"path": path, "topics": sorted(set(topics)), "on": True,
                               "seen": time.monotonic()})
        print(f"[camera-gate] 렌더 대상 {len(self.items)}개를 구독자 수에 따라 켜고 끈다 "
              f"(주기 {self.period:g} s, 구독자가 없어진 뒤 {self.hold:g} s 유지)")

    def _set(self, item, on: bool) -> None:
        import omni.graph.core as og
        try:
            og.Controller.set(og.Controller.attribute("inputs:enabled", item["path"]), bool(on))
            item["on"] = bool(on)
            self.switches += 1
        except Exception as exc:
            print(f"[camera-gate] {item['path']} 전환 실패 ({exc}) — 이 카메라는 늘 켜 둔다")
            item["topics"] = []

    def step(self) -> None:
        now = time.monotonic()
        if now < self._next:
            return
        self._next = now + self.period
        for item in self.items:
            if not item["topics"]:
                continue
            subs = 0
            for t in item["topics"]:
                try:
                    subs += self._node.count_subscribers(t)
                except Exception:
                    subs += 1                    # 셀 수 없으면 있는 것으로 친다
            if subs > 0:
                item["seen"] = now
                if not item["on"]:
                    self._set(item, True)
            elif item["on"] and now - item["seen"] > self.hold:
                self._set(item, False)

    def summary(self) -> str:
        on = [i for i in self.items if i["on"]]
        return (f"켜진 렌더 대상 {len(on)}/{len(self.items)} (전환 {self.switches}회): "
                + ", ".join(sorted(i["topics"][0] for i in on)))
