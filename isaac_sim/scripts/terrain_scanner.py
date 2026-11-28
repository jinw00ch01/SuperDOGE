"""지형 스캐너 — 로버 위 기둥에 단 360° 거리 센서 (광선 방식 3D LiDAR 의 단순화).

왜 필요한가
-----------
로버의 깊이 카메라는 지면 위 19 cm 높이에서 정면 60° 만 본다. 낮은 눈높이라
구덩이 안쪽·언덕 너머가 가려지고, 옆과 뒤는 전혀 모른다. 이 센서는 지면 위
약 1.1 m 의 기둥 끝에서 사방으로 광선을 쏴서 맞은 점까지의 거리를 잰다.

무엇을 흉내 내고 무엇을 안 하나
-------------------------------
  · 광선은 PhysX 충돌 형상에 대해 쏜다 (`raycast_closest`). 가려진 곳은 안 보이고,
    로버 자기 몸(팔·바퀴)에 맞은 광선은 버린다 — 실제 센서와 같다.
  · 광선은 차체에 고정이라 차체가 기울면 같이 기운다.
  · 거리 잡음은 정규분포 하나(기본 1 cm)뿐이다. 반사율·빔 퍼짐·먼지·회전 중
    왜곡(한 바퀴 도는 동안 움직임)은 없다 — 한 순간에 전부 잰 것으로 친다.
  · 렌더러를 쓰지 않으므로 GPU 부담이 없다 (RTX LiDAR 는 8 GB GPU 에서 카메라 3 대와
    함께 돌리기 버겁다).

출력
----
  /<ns>/terrain_scan   sensor_msgs/PointCloud2 (x, y, z float32)
      좌표계 = 로버 몸체(ground_truth/odom 이 가리키는 그 프레임), 시각 = 시뮬레이션 시각.
      받는 쪽(hazard_map_node)이 그 시각의 자세로 world 로 옮긴다.

고리(ring) 배치: 평지에서 맞는 점의 반경이 r_min…r_max 에 고르게 놓이도록
내려보는 각을 정하고(−atan(H/r)), 오르막 벽을 보도록 수평 근처·위쪽 고리를 더한다.
"""
from __future__ import annotations

import math

import numpy as np


class TerrainScanner:
    def __init__(self, stage, node, namespaces, rover_roots,
                 mast=(-0.25, 0.0, 0.60), nominal_height=1.10,
                 r_min=0.9, r_max=9.0, ring_step=0.35,
                 extra_elev_deg=(-4.0, -1.5, 1.0, 4.0, 8.0, 13.0),
                 az_step_deg=3.0, max_range=10.0, period=0.5, noise=0.01,
                 seed=0):
        import omni.physx
        from pxr import UsdGeom  # noqa: F401
        from sensor_msgs.msg import PointCloud2, PointField

        self._PointCloud2, self._PointField = PointCloud2, PointField
        self._stage = stage
        self._query = omni.physx.get_physx_scene_query_interface()
        self.mast = np.asarray(mast, dtype=np.float64)
        self.max_range = float(max_range)
        self.period = float(period)
        self.noise = float(noise)
        self._rng = np.random.default_rng(seed)
        radii = np.arange(float(r_min), float(r_max) + 1e-6, float(ring_step))
        elev = np.concatenate([-np.arctan2(float(nominal_height), radii),
                               np.radians(np.asarray(extra_elev_deg, dtype=np.float64))])
        az = np.radians(np.arange(0.0, 360.0, float(az_step_deg)))
        # 고리마다 방위를 조금씩 어긋나게 — 같은 칸만 되풀이해 맞지 않게.
        dirs = []
        for k, e in enumerate(elev):
            a = az + (k % 4) * math.radians(float(az_step_deg)) / 4.0
            ce = math.cos(e)
            dirs.append(np.stack([ce * np.cos(a), ce * np.sin(a),
                                  np.full_like(a, math.sin(e))], axis=1))
        self.dirs = np.concatenate(dirs, axis=0)          # 몸체 좌표의 단위 벡터 (N, 3)
        self._rovers = []
        for ns, root in zip(namespaces, rover_roots):
            ns_norm = (ns or "").strip("/").strip()
            body = self._find_body(root)
            if body is None:
                print(f"[scanner] {root}: 몸체 prim 을 못 찾음 — 이 로버는 스캐너 없음")
                continue
            topic = f"/{ns_norm}/terrain_scan" if ns_norm else "/terrain_scan"
            self._rovers.append({
                "ns": ns_norm, "root": str(root), "body": body,
                "pub": node.create_publisher(PointCloud2, topic, 5),
                "frame": f"{ns_norm}/base" if ns_norm else "base",
            })
        self._next = 0.0
        self._hit_log = {}            # 첫 스캔에서 무엇에 맞았는지 한 번 찍는다
        self.last_stats = {}
        print(f"[scanner] 지형 스캐너: 광선 {len(self.dirs)}개 (고리 {len(elev)} × 방위 {len(az)}), "
              f"기둥 {tuple(self.mast)} (몸체 좌표), 최대 {self.max_range:g} m, "
              f"{1.0 / self.period:g} Hz, 잡음 {self.noise * 100:g} cm, "
              f"로버 {[r['ns'] for r in self._rovers]}")

    def _find_body(self, root: str):
        """ground_truth/odom 과 같은 prim: 로버 아래 첫 articulation root (없으면 m0609 base_link)."""
        prim = self._stage.GetPrimAtPath(root)
        if not prim or not prim.IsValid():
            return None
        from pxr import Usd
        fallback = None
        for p in Usd.PrimRange(prim):
            if p.HasAPI("PhysicsArticulationRootAPI"):
                return p
            if fallback is None and p.GetName() == "base_link" and "m0609" in str(p.GetPath()):
                fallback = p
        return fallback

    def step(self, sim_time: float) -> None:
        if not self._rovers or sim_time < self._next:
            return
        self._next = sim_time + self.period
        from pxr import UsdGeom
        cache = UsdGeom.XformCache()
        for r in self._rovers:
            M = cache.GetLocalToWorldTransform(r["body"])
            t = M.ExtractTranslation()
            rot = M.ExtractRotationMatrix()            # 행 벡터 규약: v_world = v_body · rot
            R = np.array([[rot[i][j] for j in range(3)] for i in range(3)], dtype=np.float64)
            origin = self.mast @ R + np.array([t[0], t[1], t[2]])
            dw = self.dirs @ R
            o = (float(origin[0]), float(origin[1]), float(origin[2]))
            dist = np.full(len(dw), np.nan)
            own = 0
            root, q, rng_max = r["root"], self._query, self.max_range
            # numpy 원소를 광선마다 꺼내면 느리다 → 파이썬 tuple 목록으로 한 번에 바꿔 둔다.
            for k, d in enumerate([tuple(v) for v in dw.tolist()]):
                hit = q.raycast_closest(o, d, rng_max)
                if not hit["hit"]:
                    continue
                if str(hit["rigidBody"]).startswith(root) or str(hit["collision"]).startswith(root):
                    own += 1                       # 자기 몸에 가림
                    continue
                dist[k] = hit["distance"]
                if self._hit_log is not None:
                    key = "/".join(str(hit["collision"]).split("/")[:4])
                    self._hit_log[key] = self._hit_log.get(key, 0) + 1
            if self._hit_log is not None:
                print(f"[scanner] {r['ns']} 첫 스캔이 맞은 것: {self._hit_log}")
                self._hit_log = None
            ok = np.isfinite(dist)
            dd = dist[ok]
            if self.noise > 0.0:
                dd = dd + self._rng.normal(0.0, self.noise, size=dd.shape)
            pts = (self.dirs[ok] * dd[:, None] + self.mast).astype(np.float32)
            self.last_stats[r["ns"]] = (int(ok.sum()), own, len(dw))
            r["pub"].publish(self._cloud(pts, r["frame"], sim_time))

    def _cloud(self, pts: np.ndarray, frame: str, sim_time: float):
        msg = self._PointCloud2()
        msg.header.frame_id = frame
        msg.header.stamp.sec = int(sim_time)
        msg.header.stamp.nanosec = int(round((sim_time - int(sim_time)) * 1e9)) % 1000000000
        msg.height, msg.width = 1, int(pts.shape[0])
        PF = self._PointField
        msg.fields = [PF(name=n, offset=4 * i, datatype=PF.FLOAT32, count=1)
                      for i, n in enumerate("xyz")]
        msg.is_bigendian = False
        msg.point_step = 12
        msg.row_step = 12 * msg.width
        msg.is_dense = True
        msg.data = pts.tobytes()
        return msg
