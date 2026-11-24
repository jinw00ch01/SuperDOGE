# Isaac Sim 기반 화성 탐사 로버 자원 채취 시스템

> Mars Mineral Collection Rover Simulation — ROS2 + Isaac Sim + PPO

본 ROS2 workspace는 절차생성된 화성 지형에서 6륜 로버가 광물을 인식·접근·수집·복귀하는 미션을 자동 시뮬레이션합니다.

---

## 🐕 SuperDOGE — 원본(a-2_isaac)에서 달라진 점

이 저장소는 팀 프로젝트 [`sungyu-sung/a-2_isaac`](https://github.com/sungyu-sung/a-2_isaac) 를
Ubuntu 24.04 · ROS 2 Jazzy · Isaac Sim 5.1 로 옮긴 뒤, 시연용 임시방편을 하나씩 걷어내는 개선판입니다.
아래 표 밖의 내용(시연 명령, 팀 구성 등)은 원본 README 그대로입니다.

| 항목 | 원본 | SuperDOGE |
|---|---|---|
| 중력 | 지구 9.81 m/s² | **화성 3.72 m/s²** (`--physics mars`, 기본값) |
| 접촉 마찰 | PhysX 기본값 0.5 (지형 meta 의 구역 마찰은 미사용) | 레골리스 0.58/0.52, 바위 0.70/0.60, 지형 meta 의 sand·rocky 구역별 값 적용 |
| 구동 | `/cmd_vel` 을 그대로 바퀴 속도로 (3 m/s 계단 입력) | 중력에 맞춘 속도·가속 제한 (화성: 0.92 m/s, 0.72 m/s²) |
| 광물 위치 추정 (2.5 m 이내 중앙값) | 수평 오차 18.7 cm, 높이 +46 cm | 수평 3.7 cm (90% 4.7 cm), 높이 −3.6 cm |
| `ik_descend_dz=-0.40` | 인식 높이 편향을 상쇄하는 임시값 | 제거 (기본 0.05) |
| 광물 파지 | 팔 받침대 2.5 m 안의 광물을 손가락으로 순간이동해 붙임 (magic grasp) | **넓은 집게로 조여서 마찰만으로 듦** (`--grasp friction`, 기본값). 부착용 joint 없음 |
| 그리퍼 | OnRobot RG2-FT (최대 벌림 10 cm — 광물 17×19×23 cm 를 쥘 수 없음) | 27 cm 까지 벌어지는 평행 집게 (`--gripper wide`, 기본값) |
| 광물 질량 | 0.3 kg (스티로폼 수준) | 4 kg (암석 밀도 기준, 화성 무게 14.9 N) |
| 손목 카메라 | 장착 위치·자세가 추정값이라 미사용 | USD 실측값으로 교체, pick 직전 수평 위치 보정에 사용 (손끝 오차 평균 4.0 → 1.4 cm) |
| 장애물 지도 | 지형과 함께 만들어 둔 정답 지도 `obstacle_grid.npy` 로 A* | **지도 없이 출발** — 로버 깊이 카메라로 주행 중에 위험 지도를 만들고, 새 장애물이 경로를 막으면 다시 계획 (`map_source:=online`, 기본값) |
| 구동 조절 (선택) | — | 강화학습(PPO)으로 배운 **구동 조절기** — 기울기·흔들림·헛돎을 보고 속도 명령을 깎음 (`drive_governor:=true`, 기본은 꺼짐: 실제 임무 스택 검증 전) |
| 팔 기구학 (DH) | link_6 를 1.8 cm 높게, 6 mm 옆으로 계산 | USD 관절 프레임과 0.3 mm 이내 일치 |

### 화성 물리 (`isaac_sim/scripts/planet_physics.py`, `drive_limits.py`)

```bash
tools/isaac-pypi isaac_sim/scripts/run_vehicle_v3.py --terrain terrain_00026 --rovers rover_1            # 화성 (기본)
tools/isaac-pypi isaac_sim/scripts/run_vehicle_v3.py --terrain terrain_00026 --rovers rover_1 --physics earth   # 원본 동작
tools/isaac-pypi isaac_sim/scripts/check_planet_physics.py --terrain terrain_00026 --physics mars       # 중력·마찰 실측
```

- 맞춘 것: 중력, 지면·바위·바퀴의 마찰 계수(구역별), 반발 계수 0, 광물 질량(0.3 → 4 kg, `--mineral-mass` 로 변경).
  실측(`check_planet_physics.py`): 중력 3.720 m/s², 미끄럼 마찰 레골리스 0.518 · rocky 0.482 · sand 0.226.
- **맞추지 못한 것**: 바퀴 침하와 흙 밀림(terramechanics), 흙의 점착력, 대기. PhysX 는 강체 접촉만 풀기 때문입니다.
  그래서 마찰 계수는 흙의 내부마찰각에서 `tan(각)` 으로 잡았습니다 (근거는 `planet_physics.py` 머리말).
- 중력만 화성으로 바꾸면 로버는 출발 3 초 만에 뒤로 넘어갑니다. 복원 모멘트는 0.38 배가 됐는데 속도는 그대로이기 때문입니다.
  `drive_limits.py` 가 지구에서 버틴 실측 한계(1.5 m/s, 1.9 m/s²)를 Froude 상사(속도 ∝ √g, 가속 ∝ g)로 줄여 적용합니다.
  `--drive-limits off` 로 끄거나 `--drive-limits max_lin=0.6,acc_lin=0.5` 처럼 항목별로 바꿀 수 있습니다.

### 광물 위치 추정 (`isaac_perception/yolo_perception_node.py`)

오차의 원인은 bbox 가 아니라 좌표 변환이었습니다. 고친 것:
카메라 위치(`/ground_truth/odom` 의 기준점은 팔 받침대 `m0609/base_link` — 거기서 카메라는 (0.147, 0, −0.312)),
roll/pitch 반영, 영상이 찍힌 시각의 자세로 변환(odom 보간), bbox 가운데 영역의 깊이 중앙값,
표면→중심 보정, 회전 중 검출 제외, bbox 가 화면 밖으로 잘린 근접 구간에서는 잘리기 전에 본 위치 사용.

### 접촉 조건부 부착 (`isaac_sim/scripts/contact_grasp.py`, `arm_executor_node.py`)

- 팔: 공구 축을 수직 아래로 두고 손끝을 인식한 광물 중심 위 24 cm 로 가져간 뒤, 4 cm/s 로 내리면서 pickup 을 계속 보냅니다.
- 시뮬레이터: 손끝 주변 반지름 2.5 cm 구가 광물 collider 와 겹친 순간에만, 그때의 상대 자세 그대로 FixedJoint 를 겁니다.
- 닿지 않으면 pick 은 실패로 끝나고 수집으로 세지 않습니다. 인식 높이가 몇 cm 틀려도 하강 중에 닿으므로 흡수됩니다.
- 팔은 로버가 완전히 멈춘 뒤에 목표를 계산합니다 (정지 명령 뒤에도 4~7 cm 더 굴러갑니다). 팔을 뻗는 동안 차체가 밀린 만큼도 다시 계산합니다.
- 손목 카메라 보정: 손끝을 광물 위로 가져간 뒤 위에서 내려다본 검출로 수평 위치를 고칩니다 (최대 2 회).
  로버 카메라는 옆에서 보므로 표면→중심 방향이 불확실하지만, 위에서는 bbox 중심이 곧 수평 중심입니다.
  손끝 중심과 광물 중심의 수평 거리: 보정 없음 0.6~8.1 cm (평균 4.0), 보정 있음 0.7~2.3 cm (평균 1.4).
  `use_wrist_correction:=false` (arm_executor 파라미터) 로 끌 수 있습니다.
- 이 방식은 RG2-FT 를 그대로 둘 때 쓰는 중간 단계입니다 (`--grasp contact --gripper rg2` + 런치 `grasp_mode:=contact`).
  손가락이 조여서 드는 것이 아니라 닿은 뒤 FixedJoint 로 고정합니다. RG2-FT 는 10 cm 까지만 벌어지고 광물은 17×19×23 cm 라 쥘 수 없기 때문입니다.
- 원본 방식과 비교하려면 `--grasp magic --gripper rg2` + 런치 `grasp_mode:=snap`.

### 넓은 집게 + 마찰 파지 (`isaac_sim/scripts/wide_gripper.py`, `arm_executor_node.py`) — 기본값

| 시뮬레이터 | 런치 `grasp_mode` | 드는 방법 |
|---|---|---|
| `--grasp friction --gripper wide` (기본) | `friction` (기본) | 손가락이 조이는 힘과 마찰 |
| `--grasp contact --gripper rg2` | `contact` | 손끝이 닿으면 FixedJoint |
| `--grasp magic --gripper rg2` | `snap` | 순간이동 + FixedJoint (원본) |

- 집게: RG2-FT 의 본체·손가락을 비활성화하고 같은 자리에 평행 집게를 만듭니다 (손목 카메라와 브래킷은 그대로).
  벌림 3~27 cm, 손가락 길이 17 cm, 손가락 하나가 미는 힘 40 N, 패드 마찰 1.5/1.2.
  들 수 있는 한계는 2·μ·F = 96 N 이고 광물(4 kg)의 화성 무게는 14.9 N 입니다.
- 팔: 광물 위에서 손목 카메라로 수평 위치와 **윗면 높이**를 잰 뒤, 손가락을 벌린 채 집게 몸통이 윗면 3 cm 위에 올 때까지 내리고 닫습니다.
- "쥠" 판정은 물리 상태를 읽기만 합니다: 닫기 명령 뒤 손가락이 완전히 닫히지 않은 간격에서 멈춰 있고, 그 사이에 광물 collider 가 있을 때.
  들어올린 뒤와 적재함 위에서 한 번씩 다시 확인해서, 놓쳤으면 수집으로 세지 않습니다.
- 들고 있는 동안 광물 collider 는 켜져 있고 광물과 집게 사이에 joint 는 없습니다.
- 실측 (terrain_00026, 로버 1 대): 광물 0.3 kg 에서 3/3, 4 kg 에서 3/3 — 여섯 번 모두 첫 시도에 쥐었고 적재할 때까지 손가락 간격 변화는 0.1 cm 이내였습니다.
  쥔 간격 16.6~19.5 cm 는 광물 폭(17~19 cm)과 같습니다. 옆으로 누운 광물(3번)도 집었습니다.
- 남은 비현실: 집게는 치수만 정한 가상의 것이고 실제 제품이 아닙니다. 로버에 적재함 collider 가 없어서, 적재함 위에서 집게를 열면
  그 순간 손가락 사이에 있던 광물을 숨기는 것으로 "적재" 를 대신합니다 (원본과 같은 추상화). 로버 2 대 동시 실행은 아직 확인하지 않았습니다.

### 지도 없이 주행 (`isaac_drive/navigation/hazard_map.py`, `hazard_map_node.py`, `online_map.py`) — 기본값

원본은 지형 생성기가 같이 만든 정답 지도(`obstacle_grid.npy` — 경사 25° 이상 + 바위 + 구조물)를 읽어 A* 를 돌렸습니다.
지금은 같은 정보를 로버가 본 깊이 영상에서 직접 만듭니다.

- `hazard_map_node`: 깊이 영상 한 장마다 점을 0.1 m 격자에 모으고 주변 0.3 m 에 평면을 맞춰,
  기울기 25° 초과 · 한 칸 안 높이 차 12 cm 초과 · 평면에서 벗어남 4 cm 초과인 칸을 위험으로 판정합니다.
  판정은 영상 여러 장에 걸쳐 누적합니다. 결과는 `/<ns>/hazard_map` (`nav_msgs/OccupancyGrid`: 안 봄 / 안전 / 위험).
- `coverage_node`, `mission_manager_node`: 빈 격자로 시작해 지도가 올 때마다 로버 반경만큼 부풀려 갈아 끼우고,
  가던 경로가 새로 본 장애물에 막히면 다시 계획합니다. 안 본 칸은 지나갈 수 있다고 가정합니다.
  목표 방향이 25° 넘게 벗어나 있으면 제자리에서 먼저 돕니다 (카메라 시야 좌우 30° 안으로 넣은 뒤 전진).
- 팔이 HOME 이 아닐 때의 영상은 쓰지 않습니다 (팔·집게가 장애물로 찍힘). 광물은 장애물이 아니라 집을 대상이므로 4 m 안에서 검출된 광물 주변 0.4 m 는 위험에서 뺍니다.
- 사전에 아는 것으로 남긴 것: 운용 구역 크기(50×50 m)와 기지 위치·접근 금지 반경. `obstacle_grid.npy` 와 `heightmap.npy` 는 읽지 않습니다.
- 실측 (로버 1 대, 광물 3 개 수집 후 기지 복귀까지):

  | 지형 | 임무 | 정답 장애물과 최소 거리 | 지도 정밀도 | 재현율 | 경로 재계획 |
  |---|---|---|---|---|---|
  | terrain_00026 | 완료 (246 s) | 0.84 m | 99.8 % | 83.2 % | 6 회 |
  | terrain_00028 | 완료 (210 s) | 0.65 m | 99.5 % | 94.5 % | 8 회 |
  | (참고) terrain_00026, 정답 지도 사용 | 완료 (207 s) | 0.60 m | — | — | — |

  정밀도 = 위험으로 판정한 칸 중 정답 장애물 0.2 m 안에 있는 비율, 재현율 = 본 범위의 정답 장애물 칸 중 찾은 비율.
  놓친 칸은 주로 바위의 윗면과 뒷면입니다 (옆면이 막혀 있어 경로에는 영향 없음).
- **남은 비현실**: 로버 위치는 아직 시뮬레이터의 정답값(`/ground_truth/odom`)입니다 — 지도는 직접 만들지만 "내가 어디 있는지" 는 받아 씁니다.
  카메라보다 낮게 꺼진 곳(구덩이 안쪽)은 보지 못합니다. 탐사 진척(안개) 계산은 "로버 주변 5 m 를 본 것으로 친다" 는 원본 방식 그대로입니다.
- 원본 방식으로 돌리려면 런치에 `map_source:=prebuilt`.

### 접근·탐색 주행 수정 (2026-11-24, `mission_manager_node.py`, `online_map.py`, `navigator.py`, `yolo_perception_node.py`)

Web HUD 로 지켜보다 드러난 네 가지를 기록으로 원인을 확인하고 고쳤습니다.

| 증상 | 원인 (기록으로 확인) | 수정 |
|---|---|---|
| 광물 근처에서 좌우로만 돌다 포기 | 지도 없는 주행에서는 광물도 "위험 칸" 이라, 접근 경로가 광물 1 m 앞에서 끝남. 로버가 발밑의 마지막 경유점을 향해 조향하며 제자리에서 맴돎 (접근 9 회 중 6 회가 0.9~1.2 m 남기고 탈락) | 집으러 가는 광물 주변 0.5 m 의 위험 칸은 빼고 계획 (`grid_excluding`). 그래도 막혀 있으면 그 광물을 60 초 제외 |
| 탐색 중 한 자리에서 4 분간 좌우로 회전 | 제자리 회전을 "오차 × 이득" 으로만 명령. 시뮬레이터는 각가속도를 제한(0.99 rad/s²)하고 명령의 1.9 배로 돌아서 목표 방향을 지나쳤다 돌아오기를 무한 반복 (주기 약 4.5 s) | 남은 각도 안에서 멈출 수 있는 속도까지만 명령: `√(2·0.4·|오차|)/1.9`. 탐색·접근·복귀 모두 적용 |
| 노란 광물로 가다 말고 못 집음 | 먼 광물은 비스듬히 보여서 깊이 화소가 뒤쪽 배경이나 앞쪽 땅을 읽음 → 거리 추정이 2~7 m 틀림 → 엉뚱한 곳으로 접근하고, 목표 고정(lock)이 진짜 위치의 검출을 "다른 광물" 로 버림 | 깊이가 가장 촘촘히 모인 구간의 값 사용, 같은 자리에서 3 번 본 검출만 목표로, 같은 방향의 더 가까운 검출이 오면 고정 위치 수정 |
| 건물(배럭)을 타고 올라 뒤집힘, 골리앗 앞에서 5 분간 밀기만 함 | 원본 자산은 구조물마다 바닥 면적 전체를 덮는 **투명한 충돌 상자**를 씀. 골리앗은 다리 사이가 뚫려 있고 배럭은 밑동이 상자보다 좁아서, 깊이 카메라는 빈 땅을 보는데(그 자리가 지도에 100 % "안전") 로버는 벽에 막힘 | 시뮬레이터가 투명 상자를 끄고 **보이는 모델 그대로** 충돌 (`--structure-collision mesh`, 기본. 원본은 `box`). 위험 칸 옆 0.8 m 안의 못 본 칸도 막힌 것으로 침 |
| 막혀도 계속 밀기만 함 | 장애물은 깊이 카메라 + 위치로만 알고, 접촉을 느끼는 수단이 없음 | **막힘 감지** (`hazard_map_node`): 전진 명령을 4 초 받고도 15 cm 를 못 가면 진행 방향 0.9 m 앞에 반경 0.45 m 위험 원을 찍음 → 두 계획기가 돌아감 |
| 광물이 없는 허공을 집음 | 12 m 밖 노란 광물의 거리가 3 프레임 동안 2.7 m 로 잡혔고(깊이 화소가 가까운 땅에 걸림), "같은 방향의 더 가까운 검출로 목표 수정" 이 그것을 받아들여 로버 바로 앞을 집으러 감. 팔은 손목 카메라에 광물이 안 보여도 그대로 진행했음 | 인식: 추정 거리에서 계산한 크기가 8 cm 미만이면 좌표를 버림 (실제 광물은 0.17~0.33 m, 문제의 검출은 5 cm). 팔: 손목 카메라가 광물을 못 보면 **집지 않고 취소** (`abort_if_wrist_sees_nothing`) |

- 측정:
  - 제자리 회전 수정 — 학습 환경 평가(terrain_00026, 로버 16 대 × 6 분, 전복 0 회 유지): 목표점 도달이 고정 제한 1.58 → **1.96 개/분**, 구동 조절기 2.20 → **2.28 개/분**.
  - 거리 추정 — 6 m 넘게 떨어진 검출 중 거리 오차 1 m 초과 비율: 7.1 % → **3.4 %** (임무 1 회씩, 정답 광물 위치 기준).
  - 구조물 — terrain_00028, 구동 조절기 모드, 같은 설정으로 세 번: 투명 상자 그대로일 때 한 번은 배럭에서 전복, 한 번은 골리앗 앞에서 정지(1/3 수집).
    보이는 모델로 충돌을 바꾼 뒤 **3/3 수집 후 완주**, 장애물까지 최소 0.71 m, 전복 없음, 지도 재현율 96.3 %.
  - 임무(terrain_00026, 기본 설정): 3/3 수집(파랑·노랑·파랑), 접근 3 회 모두 한 번에 도착, 장애물까지 최소 1.00 m, 전복 없음.
- 한계: 임무 검증은 지형당 1~2 회입니다. 광물이 바위에 0.8 m 안으로 붙어 있으면 여전히 접근하지 못하고 제외합니다.
- 광물이 여럿 보일 때의 우선순위는 원본 그대로입니다: 종류별 가치(노랑 50 > 초록 25 > 파랑 10)가 먼저, 신뢰도는 동점 처리용(×0.1), 거리는 보지 않음. 한 번 접근을 시작하면 그 광물에 고정됩니다.

### 강화학습 구동 조절기 (`isaac_sim/rl/`, `isaac_drive/drive_governor_node.py`) — 선택 (기본 꺼짐)

상위 노드(coverage / supervisor)는 지형을 모른 채 "3 m/s 로 가라" 를 보냅니다. 화성 중력에서 이걸 그대로 바퀴에 넘기면 넘어지므로
지금 기본값은 고정된 속도·가속 제한(`drive_limits.py`)입니다 — 평지든 비탈이든 같은 값. 조절기는 그 자리에 들어가는 작은 신경망입니다.

```
supervisor ──▶ /<ns>/cmd_vel_raw ──▶ [drive_governor_node] ──▶ /<ns>/cmd_vel ──▶ 바퀴
                                         ▲ 자세(기울기) · 차체 속도 · 바퀴 6 개 속도
```

- **무엇을 정하나**: 선속도·각속도를 명령의 몇 배(0~1)로 낼지 + 전진 중 조향 보정(±0.3 rad/s). 명령보다 빠르게 가거나 반대로 돌 수 없습니다.
  카메라·지도는 안 보고 차체가 느끼는 것만 씁니다 (입력 20 개: 명령, 중력 방향, 각속도, 선속도, 바퀴 속도, 직전 출력).
- **학습**: 로버 16 대를 카메라·ROS 없이 한 장면에 올려 물리만 돌립니다 (초당 약 500 샘플, PhysX→USD 갱신을 끄면 4.5 배 빨라짐).
  PPO 는 torch 만으로 직접 썼습니다 (`isaac_sim/rl/ppo.py`, 새 패키지 설치 없음). 정책은 20 → 64 → 64 → 3, 추론은 numpy 행렬 곱 세 번.
  보상 = 명령 방향으로 실제 나아간 속도 − 기울기 12° 초과분 − 흔들림 − 헛돎(시킨 속도와 실제 속도의 차) − 옆 미끄러짐 − 전복(−30, 종료).
  지면 마찰은 학습 중 0.6~1.1 배로 계속 바꿉니다. 포함된 정책(`isaac_drive/policies/drive_governor.npz`)은 terrain_00026 에서 150 만 샘플(55 분) 학습한 것입니다.
- **평가** (`isaac_sim/rl/eval_drive.py`): 로버 16 대 × 6 분, 같은 출발점·같은 목표점 목록. 실제 스택의 Navigator 와 같은 식으로 지형을 모른 채 목표점을 향합니다.
  "조정한 고정 제한" 은 고정 제한을 손으로 푼 것 중 전복이 없던 가장 빠른 조합입니다 (최고 2.0 m/s, 1.5 m/s², 1.5 rad/s, 2.0 rad/s²).

  | 조건 | 지표 | 명령 그대로 (원본) | 고정 제한 (기본) | 조정한 고정 제한 | 학습한 조절기 |
  |---|---|---|---|---|---|
  | terrain_00026 (학습 지형) | 목표 도달 /분 | 0.89 | 1.58 | 1.99 | **2.20** |
  | | 전복 (96 분) | 260 | 0 | 0 | 0 |
  | | 헛돎 (m/s) | 2.01 | 0.11 | 0.58 | 0.19 |
  | terrain_00028 (처음 보는 지형) | 목표 도달 /분 | — | 1.46 | 1.95 | **2.16** |
  | | 전복 | — | 0 | 0 | 0 |
  | | 헛돎 | — | 0.09 | 0.54 | 0.16 |
  | terrain_00026, 마찰 ×0.6 | 목표 도달 /분 | — | 0.99 | 0.94 | **1.09** |
  | | 전복 | — | 0 | 0 | 0 |
  | | 헛돎 | — | 0.37 | 0.99 | 0.31 |
  | terrain_00015 (처음 보는 지형, 최대 경사 61°) | 목표 도달 /분 | 0.33 | 0.59 | 0.66 | **0.81** |
  | | 전복 | 255 | 8 | 30 | 7 |
  | | 헛돎 | 2.44 | 0.52 | 1.13 | 0.38 |

  헛돎 = 전진 명령 중 |바퀴에 시킨 속도 − 차체의 실제 전진 속도| 의 평균.
- **읽는 법 (과장 없이)**:
  - 완만한 지형(00026·00028, 평균 경사 5~6°)에서는 **전복은 고정 제한만으로도 0 회**입니다. 조절기의 이득은 "전복 없이 10 % (조정한 제한 대비) ~ 40 % (기본 제한 대비) 빠르고, 조정한 제한보다 헛돎이 1/3" 입니다.
  - 가파른 지형(00015)에서는 **어느 방식도 전복을 못 없앱니다.** 전복 직전을 기록해 보면 대부분 이미 30~45° 기울어진 채로 바퀴를 계속 굴리다 넘어갑니다.
    포함된 정책은 완만한 지형만 겪어서 이 상황에서 물러서지 않습니다 (오르막 35° 에서도 1.4 m/s 를 냄).
  - **가파른 지형에서 이어서 학습해 봤지만 채택하지 않았습니다.** 두 번(100 만 샘플씩) 모두 비탈에서 거의 멈추는 쪽으로 배웠고(10° 이상에서 0.06~0.17 m/s),
    terrain_00015 에서 목표 도달이 0.81 → 0.32 · 0.41 개/분으로 절반이 됐는데 전복은 7 → 3 · 10 회로 줄지 않았습니다. 속도를 깎기만 하는 조절기로는 이미 기운 차체를 되돌릴 수 없습니다.
  - 실제 임무 스택 주행 (`drive_governor:=true`, 1 회씩): terrain_00026 은 완료 — 3/3 수집, 장애물까지 최소 0.66 m, 최대 기울기 16.6°, 최고 1.46 m/s (고정 제한은 1.08 m/s), 조절기는 전진 명령의 평균 79 % 를 통과시켰습니다.
    terrain_00028 은 **건물을 들이받고 뒤집혔습니다** (위 "접근·탐색 주행 수정" 의 넷째 항목 — 지도 쪽 문제였지만, 조절기 모드는 더 빨라서 같은 실수가 더 크게 나옵니다). 그래서 기본값은 계속 꺼짐입니다.
- **켜는 법**: 시뮬레이터 `--drive-limits off` (안 그러면 고정 제한 0.92 m/s 가 먼저 깎음) + 런치 `drive_governor:=true`.
  시뮬레이터의 `/ground_truth/odom` 은 속도(twist)가 비어 있어서, 노드는 연속한 두 자세의 차이로 차체 속도를 구합니다. odom 이 0.5 s 끊기면 정지 명령을 냅니다.
- **다시 학습 / 이어서 학습 / 평가**:

  ```bash
  tools/isaac-pypi isaac_sim/rl/train_drive.py --out isaac_sim/rl/runs/gov_new                 # 처음부터 (150 만 샘플, 약 55 분)
  tools/isaac-pypi isaac_sim/rl/train_drive.py --out isaac_sim/rl/runs/gov_steep \
      --terrain terrain_00004 --resume isaac_sim/rl/checkpoints/gov_gentle.pt --steep-goal-frac 0.5 --samples 1e6   # 가파른 지형에서 이어서
  tools/isaac-pypi isaac_sim/rl/eval_drive.py --terrain terrain_00015 \
      --policy isaac_sim/rl/runs/gov_steep/policy.npz --controllers fixed,policy --minutes 6
  cp isaac_sim/rl/runs/gov_steep/policy.npz isaac_drive/policies/drive_governor.npz             # 채택 후 colcon build
  ```
- 학습 환경은 로버끼리의 충돌을 끄고 광물을 치운 상태입니다. 바위에 0.9 m 안으로 다가가면 벌점 없이 다른 곳에서 다시 시작합니다 (바위 회피는 지도·경로 계획의 몫).

### 넘어짐 복구 · 비탈 균형 (`isaac_sim/rl/righting_env.py`, `balance_env.py`, `isaac_drive/self_right_node.py`) — 학습 중, 기본 꺼짐

2026-11-24 에 시작했고 **아직 끝나지 않았습니다.** 세우기는 학습 도중의 정책(90 만 샘플, `isaac_drive/policies/self_right.npz`)을 넣어 두었고, 균형 정책은 아직 없습니다.

- **넘어진 로버를 팔로 다시 세우기**: 뒤집힌 자세로 떨어뜨린 로버가 팔 관절 6 개로 땅을 밀어 일어서도록 PPO 로 학습합니다.
  팔 구동력은 150/150/100/40/40/40 N·m 로 제한했고(시뮬레이터 기본 9600 N·m 는 사실상 무한 — 제한값은 실제 팔 사양을 확인한 것이 아니라 가정), 팔이 차체를 통과하지 못하게 자기 충돌을 켰습니다.
  중간 점검(약 35 만 샘플, 150 회): 넘어진 채 출발한 137 회 중 **67 %** 가 일어섰고(등으로 누운 경우 80 %, 옆으로 59 %), 일어선 뒤 팔을 접어도 모두 서 있었습니다. 걸린 시간 중앙값 4 초. 무작위 동작만으로도 23 % 는 일어섭니다.
  `self_right_node` 는 기울기 60° 초과로 2 초 멈춰 있으면 이 정책으로 팔을 움직이고, 그동안 supervisor 는 바퀴를 세우고 단계를 **RECOVERY** 로 바꿉니다 (웹 UI 의 PHASE 는 RECOVERY, TASK 는 `self-righting with arm (1/3)` → `folding arm after righting`; 일어서면 하던 단계로 복귀, 채취 중이었으면 EXPLORE 부터). 런치 `self_right:=true`.
  임무 스택 시험 (terrain_00026, 시뮬레이터 `--flip-at 45` 로 45 초에 강제로 뒤집음, 학습 도중의 정책 90 만 샘플): 두 번 모두 일어섰고(6.5 초, 3.1 초) 이후 3/3 수집과 귀환까지 끝냈습니다. 두 번뿐이라 성공률을 말할 수는 없습니다.
- **비탈에서 균형 유지**: 구동 조절기에 "뒤로 빼기"(전진 배율 −0.25~1)와 "팔 균형추" 를 더한 정책입니다. 팔(13.5 kg / 전체 62.6 kg)을 뻗으면 무게중심이 앞 5 cm · 뒤 13 cm · 좌우 8 cm 움직입니다(실측).
  보상에 "안정 여유"(무게중심에서 내린 중력선이 바퀴 접지 사각형 가장자리에서 얼마나 안쪽인지)를 넣었습니다. 가파른 지형(terrain_00015)에서 학습 중이고 **평가 전**입니다.

---

## 🎯 시연 명령어 세트 (D-day 기준)

> **구성**: 멀티 rover 2대 · `terrain_00026` · 로버당 광물 2개 채집 → RTB → MISSION_COMPLETE.
> 터미널 제로 상태 (env 변수 없음) 에서 그대로 복붙. launch default 가 이 값들 그대로라 args 생략 가능.

### T1 — Isaac Sim (멀티 rover spawn)
```bash
cd ~/dev_ws/rover_ws/src/a2_isaac
tools/isaac-pypi isaac_sim/scripts/run_vehicle_v3.py \
    --terrain terrain_00026 --rovers rover_1 rover_2
```
콘솔에서 다음 확인 후 T2:
```
[run_v3] ready — 2대 vehicle namespaces: /rover_1, /rover_2
```

### T2 — Mission stack (협조 + Web HUD 자동)
```bash
source /opt/ros/humble/setup.bash
source ~/dev_ws/rover_ws/install/setup.bash
cd ~/dev_ws/rover_ws
ros2 launch isaac_bringup mvp_multi.launch.py
```
default 그대로: `terrain_id=terrain_00026`, `collection_goal=2`, `rovers="rover_1 rover_2"`.

### T3 — 브라우저
```bash
xdg-open http://localhost:8088
```
상단 `rover_1` / `rover_2` chip 토글로 STATUS · 카메라 전환. Coverage Map 은 두 rover 동시 표시.

**종료**: T2 `Ctrl+C` → 전체 mission stack + Web HUD 정리.

---

## 🚀 팀원이 main pull 후 해야 할 것 (Quick Start)

### 1. 환경 셋업 (1회만)

**Isaac Sim 5.1 PyPI binary** — [docs/SETUP_ISAAC_PYPI.md](docs/SETUP_ISAAC_PYPI.md) 참조:
- 권장 venv 경로: `~/dev_ws/isaac_sim_pypi/venv`
- ROS2 humble 워크스페이스: `~/dev_ws/isaac_sim/IsaacSim-ros_workspaces/humble_ws`
- 다른 경로 사용 시 env var override: `ISAAC_PYPI_VENV`, `ISAAC_ROS2_WS`

**.bashrc** — [docs/SETUP_BASHRC.md](docs/SETUP_BASHRC.md) 참조 (ROS humble + 워크스페이스 source + ROS_DOMAIN_ID).

**ROS_DOMAIN_ID** (팀 합의 — 같은 LAN 충돌 방지):

| 트랙 | 담당자 | ID |
|:---:|:---:|:---:|
| T1 | 김현중 | **111** |
| T2 | 최진우 | **114** |
| T3 | 이찬휘 | **112** |
| T4 | 성선규 | **113** |
| T5 | 이지민 | **115** |

```bash
export ROS_DOMAIN_ID=<본인 값>   # ~/.bashrc 에 추가
```

### 2. 빌드 (main pull 후 또는 코드 변경 시)

```bash
cd ~/dev_ws/rover_ws
colcon build --symlink-install
source install/setup.bash
```

> ✅ **별도 build 불필요한 자산** (모두 main 에 baked):
> - `vehicle_v3.usd` (액션그래프 내장 로버) — `build_vehicle_v3.py` 재실행 불필요
> - `terrain_00001~00023` (heightmap + obstacle + meta + epic obstacle 4종) — `mars_terrain_generator_v3.py` 재실행 불필요
> - YOLO 모델 `mineral_yolo_best.pt` (5.3MB) — git tracked
> - Doosan M0609 + RG2 gripper USDs
> 빌드 스크립트는 USD/terrain 자체를 수정·재생성할 때만 실행.

### 3. 시연 실행 (3 터미널)

```bash
# T1 — Isaac Sim (source 없이! tools/isaac-pypi wrapper 가 자체 처리)
cd ~/dev_ws/rover_ws/src/a2_isaac
tools/isaac-pypi isaac_sim/scripts/run_vehicle_v3.py --terrain terrain_00004

# T2 — ROS2 노드 묶음 (perception + coverage + supervisor + arm + GT cheat)
source /opt/ros/humble/setup.bash && source ~/dev_ws/rover_ws/install/setup.bash
ros2 launch isaac_bringup mvp.launch.py

# T3 — 카메라 view 2개 (body + wrist)
source /opt/ros/humble/setup.bash && source ~/dev_ws/rover_ws/install/setup.bash
ros2 launch isaac_bringup rqt_views.launch.py
```

→ rover 가 자율적으로 EXPLORE → APPROACH → PICK 반복 후, `collection_goal` (기본 5개) 도달
또는 배터리 critical 시 **RETURN_TO_BASE** → 베이스캠프 (0,0) 도착 → **MISSION_COMPLETE**.

#### 3-bis. 화성 환경 실험 (시연 X, 거동 검증·튜닝용)

`run_vehicle_v4_test.py` — `run_vehicle_v3.py` 사본 + 화성 중력 3.72 m/s² 적용판.
시연 main path 와 격리되어 있으니 자유롭게 실험 가능. `vehicle_v3.usd` 자체는 안 건드림.

**Set A — gravity 만 override (Tier 0):**
```bash
# T1 — 9.81 → 3.72 만 변경, 다른 파라미터는 v3 그대로
cd ~/dev_ws/rover_ws/src/a2_isaac
tools/isaac-pypi isaac_sim/scripts/run_vehicle_v4_test.py \
    --terrain terrain_00026 --rovers rover_1 rover_2 --no-tier1

# T2 / T3 는 §3 의 mvp_multi / 브라우저 그대로
```
결과: WHEEL_VELOCITY_GAIN 2.0 이 지구 기준 튜닝값이라 중력 1/3 환경에서
**rover 가 통통 튀다 전복**. "화성 = 단순히 중력만 바꾸면 안 됨" 검증.

**Set B — gravity + Tier 1 파라미터 (안정 거동):**
```bash
# T1 — gravity + ACK_SCRIPT 화성용 교체 (WHEEL_VELOCITY_GAIN 0.7, STEERING_GAIN 0.8, cmd_vel LPF α=0.2)
cd ~/dev_ws/rover_ws/src/a2_isaac
tools/isaac-pypi isaac_sim/scripts/run_vehicle_v4_test.py \
    --terrain terrain_00026 --rovers rover_1 rover_2

# T2 — coverage max_lin / max_ang 도 화성 스케일로 다운
source /opt/ros/humble/setup.bash && source ~/dev_ws/rover_ws/install/setup.bash
cd ~/dev_ws/rover_ws
ros2 launch isaac_bringup mvp_multi.launch.py max_lin:=1.0 max_ang:=0.6

# T3 — 브라우저 그대로
xdg-open http://localhost:8088
```
결과: 통통 튐 사라지고 안정 주행. 단 속도가 1 m/s 라 시연 시간 ↑.
참고로 실제 NASA Curiosity 평균 ~1.5 cm/s — 우리 v4_test 도 여전히 ~30 배 빠름.

> **시연 D-day 는 §3 의 지구용 (`run_vehicle_v3.py`) 만 사용**. v4_test 는 발표
> talking point + 향후 "Mars-ready rover" 작업의 baseline.

### 4. Mission Control UI (Web HUD)

`mvp.launch.py` 가 띄우는 **mission_web_node** (Flask+SocketIO) + **web_video_server** 가
SC2 풍 HUD 를 제공합니다.

```bash
# 1회만: 의존성 설치
pip3 install --user flask-socketio eventlet
sudo apt install -y ros-humble-web-video-server

# 브라우저 접속
xdg-open http://localhost:8088     # 또는 LAN 내 다른 PC: http://<host-ip>:8088
```

- **좌하단**: coverage minimap (canvas, rover 위치/베이스캠프 표시)
- **중앙하단**: portrait + phase 배지 + battery/collected 바 + POS/YAW/SPEED/CMD/TASK/ERR
- **우상단**: 광물 종류별 카운터 + collected/goal
- **우하단**: 3×3 액션 그리드 (디자인 단계 — placeholder, 버튼 비활성)
- **중앙 메인**: Isaac Sim overview (현재 sun cam placeholder) / Body YOLO / Wrist YOLO

원격조종 (teleop_twist_keyboard) 과 AUTO/MANUAL/ESTOP 버튼 연동은 다음 단계.

### 5. T5 정공법 졸업 (선택, 정확도 검증 완료 시점에)

`mvp.launch.py` 의 `odom_to_estimated_pose` 노드를 빼고 별도 터미널에서:
```bash
ros2 launch isaac_bringup localization.launch.py
```
+ `arm_executor` / `mission_manager_node` 의 `odom_topic` 파라미터를 `/rover/estimated_odom` 으로 swap.

---

> ℹ️ 명칭 명료성·책임 분리를 위한 **8개 패키지 구조**가 `main`에 정착 (2026-05-26 cleanup 으로 isaac_rl stub 정리). 설계 의도: [docs/STUDY_AND_PLAN.md](docs/STUDY_AND_PLAN.md) Part XI.

## 트랙 ↔ 담당자

| 트랙 | 담당자 | 영역 | GPU |
|:---:|:----:|------|:---:|
| **T1** | **김현중** | Environment (절차생성 지형, basecamp, Mars physics) | 5060 |
| **T2** | **최진우** | Perception + M0609 (vision, manipulation) | 5080 |
| **T3** | **이찬휘** | Driving (mission FSM, A*, coverage, PPO wrapper) — **Critical Path** | 5080 |
| **T4** | **성선규** | Integration + PM (ROS2 wiring, UI, demo) — 사용자 본인 | 5070 Ti |
| **T5** | **이지민** | Localization + Infra (TRN, EKF, sensor fusion) | 5080 |

각 트랙의 onboarding: [docs/tracks/T*_BRIEF.md](docs/tracks/)

## 현재 상태 (2026-05-26 발표 직전)

🎉 **MVP 시연 모델 완성** — 화성 rover 자율 mineral 수집 mission 의 end-to-end loop:
`EXPLORE → 발견 → APPROACH → PICK (FixedJoint snap) → CARGO swing → RELEASE → EXPLORE` 반복 동작 검증 완료.

✅ **구현 완료 (시연 PASS)**:
- **I1 지형 자산** — v2 생성기로 `terrain_00001`~`terrain_00022` 생성 (장애물·22 mineral·basecamp baked)
- **vehicle_v3.usd** — 액션그래프 내장 자립 로버 (T1+T2+T3+T5 통합): ROS2 sensor 발행 + `/cmd_vel→Ackermann→휠` 구동 + `/arm/joint_command→m0609+finger` 제어 + `/grasp/command` FixedJoint snap + GT odom (dev cheat)
- **isaac_drive** — `coverage_node` (BCD anchor sweep + A* obstacle 회피 + replan), `odom_to_estimated_pose` (GT cheat 어댑터), `minimap_publisher`
- **isaac_perception** — `yolo_perception_node` (nav + wrist YOLO + depth backproject → mineral world XYZ)
- **isaac_supervisor** — `mission_manager_node` (EXPLORE/APPROACH/PICK phase + cmd_vel mux + arm action client + lock-on)
- **isaac_manipulation** — `arm_executor_node` (T2 DLS-IK 상태머신: HOME→APPROACH→DESCEND→pickup→HOME_MID→CARGO→release→HOME)
- **isaac_localization (T5 이지민)** — `ekf_fusion` (wheel+imu+sun+TRN EKF), 8개 sensor/fusion 노드 구현. **정확도 추가 튠 필요** (졸업 작업)
- **isaac_bringup** — `mvp.launch.py` (시연용 GT cheat 모드), `rqt_views.launch.py` (카메라 view 2개), `localization.launch.py` (T5 stack + EDL pose prior)
- **isaac_interfaces** — msg + action 계약 정착

⚠️ **시연용 cheat (졸업 시 청산)**:
- `/ground_truth/odom` — `odom_to_estimated_pose` 가 `/rover/estimated_pose` 로 forwarding. T5 EKF stack 정확도 검증 완료되면 `localization.launch.py` 가 대체.
- `ik_descend_dz=-0.40` — perception z bias (+47cm 평균) 보정. T2 depth 정확도 올리면 default 복귀.
- FixedJoint snap grip — 30cm 광물 vs 4cm finger gap 비대칭 (hardware 한계). 시연용 hack 박제.

⏳ **추가 작업 (시연 후)**:
- T5 EKF: sun_yaw + TRN 정확도 튠
- T2 perception: depth z bias 해결
- T3 RL inference: PPO 정책 통합 (`driving_policy_node`)
- 정리 대기 목록: [list_to_fix.md](list_to_fix.md)

## 빠른 시작 (Day 1 셋업, ~5분)

### 1. 워크스페이스 생성 + clone

```bash
# ROS2 워크스페이스 (~/dev_ws/rover_ws/) 만들고 src/ 안에 우리 repo clone
mkdir -p ~/dev_ws/rover_ws/src
cd ~/dev_ws/rover_ws/src

# 폴더명을 a2_isaac으로 명시 (repo는 a-2_isaac이지만 우리 환경은 a2_isaac로 통일)
git clone https://github.com/sungyu-sung/a-2_isaac.git a2_isaac

cd a2_isaac
```

> ℹ️ 위치는 `~/dev_ws/rover_ws/` 권장 (문서 명령어 그대로 copy-paste 가능). 다른 위치도 기능적으로 무관 — [docs/SETUP_BASHRC.md](docs/SETUP_BASHRC.md) §위치 무관성 참조.

### 2. 의존성 (한 번만)

```bash
# Python 의존성 (지형 생성기 + meta.json schema 검증 + PNG 오버뷰)
pip install --user usd-core jsonschema numpy matplotlib

# 트랙 owner별 추가 (필요한 사람만)
# T3 이찬휘 — A* 빠른 구현
pip install --user pyastar2d
```

### 3. ROS2 빌드 + source

```bash
cd ~/dev_ws/rover_ws

# 첫 빌드 (~30초)
colcon build --symlink-install

# source (매 터미널 또는 .bashrc에 등록)
source install/setup.bash

# 검증 — 8개 패키지 등록 확인
ros2 pkg list | grep isaac_
# 예상 출력: isaac_bringup, isaac_drive, isaac_interfaces, isaac_localization,
#           isaac_manipulation, isaac_perception, isaac_sim, isaac_supervisor
```

### 4. (선택) bashrc 설정

매 터미널마다 source / cd / build 명령 안 치려면 **`~/.bashrc`에 한 번 등록**해두면 편함. 권장 alias + 환경 변수 + ROS_DOMAIN_ID 충돌 방지:

→ **[docs/SETUP_BASHRC.md](docs/SETUP_BASHRC.md)** — 필수/권장/선택 단계별 + 트랙별 추가 alias

### 5. Isaac Sim 시각 확인

```bash
# I1 지형은 이미 생성되어 있음 (terrain_00001~00003)
# Isaac Sim에서 master scene 열기:
isaac ~/dev_ws/rover_ws/src/a2_isaac/isaac_sim/worlds/mars_exploration_world.usd

# 또는 다른 seed로 새 terrain 생성 (v3 = epic obstacle 4종 포함):
cd ~/dev_ws/rover_ws/src/a2_isaac
python3 isaac_sim/scripts/mars_terrain_generator_v3.py --seed 777 --terrain-id terrain_00004
```

→ I1 풀 가이드: [docs/interfaces/I1_TERRAIN_ASSETS.md](docs/interfaces/I1_TERRAIN_ASSETS.md)

### 6. 시연 실행 (3 터미널, MVP 통합)

> 📌 **사전 셋업 필수** — Isaac Sim 5.1 PyPI binary 환경: [docs/SETUP_ISAAC_PYPI.md](docs/SETUP_ISAAC_PYPI.md)
> .bashrc / ROS_DOMAIN_ID: [docs/SETUP_BASHRC.md](docs/SETUP_BASHRC.md)

```bash
# ───── T1: Isaac Sim vehicle_v3 (source 없이, tools/isaac-pypi wrapper) ─────
cd ~/dev_ws/rover_ws/src/a2_isaac
tools/isaac-pypi isaac_sim/scripts/run_vehicle_v3.py --terrain terrain_00004

# ───── T2: ROS2 노드 묶음 (perception + coverage + supervisor + arm + GT cheat) ─────
# 새 터미널, source 필요
source /opt/ros/humble/setup.bash && source ~/dev_ws/rover_ws/install/setup.bash
ros2 launch isaac_bringup mvp.launch.py

# ───── T3: 카메라 view 2개 (body + wrist) ─────
# 새 터미널, source 필요
source /opt/ros/humble/setup.bash && source ~/dev_ws/rover_ws/install/setup.bash
ros2 launch isaac_bringup rqt_views.launch.py
```

→ rover 가 자율적으로 EXPLORE → mineral 발견 → APPROACH → PICK → CARGO RELEASE 반복.

### 시연 시나리오별 launch — 자세히는 [docs/DEMO_COMMANDS.md](docs/DEMO_COMMANDS.md)

```bash
# 단일 rover GT cheat baseline
ros2 launch isaac_bringup mvp.launch.py collection_goal:=5

# 단일 rover T5 정공법 (GT 없이 EKF stack)
ros2 launch isaac_bringup integrated_localization.launch.py collection_goal:=1

# 멀티 rover (2대 동시 운용 + 협조 + Web HUD 멀티 추적)
ros2 launch isaac_bringup mvp_multi.launch.py collection_goal:=1

# T5 EKF stack 단독 (open-loop 검증)
ros2 launch isaac_bringup localization.launch.py
```

### 시연용 → 정공법 졸업 경로

| Cheat | mvp.launch.py | 졸업 launch |
|---|---|---|
| GT odom (`odom_to_estimated_pose`) | ✓ default | `localization.launch.py` 실행 (T5 EKF) |
| `ik_descend_dz=-0.40` (perception z bias 보정) | ✓ default | T2 depth 정확도 향상 시 launch param 제거 |
| FixedJoint snap grip | ✓ default | (hardware 한계 — 영구) |

## Workspace 구조 (2026-05-26 시연 시점)

⭐ = 시연용 (mvp.launch.py / T1~T3 명령) 핵심 파일.

```text
a2_isaac/
├─ README.md                              # ⭐ 이 문서 (Quick Start 포함)
├─ list_to_fix.md                         # 시연 후 작업 목록 + 졸업 경로
├─ .gitignore                             # build/cache/temp + .pt 예외 등록
│
├─ docs/                                  # 설계 문서 + 가이드 + 트랙 onboarding
│  ├─ STUDY_AND_PLAN.md                   # 전체 설계 의도
│  ├─ SETUP_BASHRC.md                     # ⭐ ~/.bashrc + ROS_DOMAIN_ID 가이드
│  ├─ SETUP_ISAAC_PYPI.md                 # ⭐ Isaac Sim PyPI venv 셋업
│  ├─ TREE_CLEANUP_PLAN.md                # 정리 작업 기록 (참조용 archive)
│  ├─ interfaces/                         # I1~I5 인터페이스 계약 (msg/action 규격)
│  ├─ tracks/                             # 트랙별 onboarding (T1~T5_BRIEF.md)
│  ├─ pm_tools/                           # PM 운영 (KICKOFF/DAILY/DECISIONS)
│  ├─ system_design/                      # 아키텍처 평가 (ARCHITECTURE_EVAL_*)
│  └─ troubleshooting/                    # 진단 기록 (camera regression 등)
│
├─ tools/
│  ├─ isaac-pypi                          # ⭐ Isaac Sim PyPI 환경 wrapper (T1 entry)
│  └─ README.md                           # tools/ 가이드
│
├─ isaac_bringup/                         # ① 진입점 — launch 파일 모음 (T4)
│  ├─ launch/
│  │  ├─ mvp.launch.py                    # ⭐ 단일 rover GT cheat MVP (Web HUD 자동)
│  │  ├─ mvp_multi.launch.py              # ⭐ 멀티 rover 통합 (협조 + Web HUD 멀티 추적)
│  │  ├─ integrated_localization.launch.py # 단일 rover T5 정공법 (GT 없이 EKF)
│  │  ├─ localization.launch.py           # T5 EKF stack 단독 (open-loop 검증)
│  │  ├─ rqt_views.launch.py              # body+wrist 카메라 (단일)
│  │  └─ rqt_views_multi.launch.py        # 4창 카메라 (멀티 rover_1/2 × body/wrist)
│  └─ rviz/localization_map.rviz
│
├─ isaac_sim/                             # ② Isaac Sim 환경 + 자산 (T1)
│  ├─ assets/
│  │  ├─ vehicle/
│  │  │  └─ vehicle_v3.usd                # ⭐ 액션그래프 내장 자립 로버 (시연 사용)
│  │  ├─ markers/tier2_mineral/           # ⭐ 광물 mesh + texture (T2 YOLO 학습 대상)
│  │  ├─ epic_obstacles/                  # SC2 풍 landmark obstacle 4종 (sun_yaw/TRN 특징점)
│  │  ├─ generated_terrains/
│  │  │  └─ terrain_00001~00023/          # 23개 v3 terrain (epic obstacle 통일)
│  │  ├─ doosan-robot2/, onrobot_rg2/, rover/  # arm + gripper + rover 자산
│  ├─ worlds/                             # terrain_NNNNN.usd + mars_exploration_world.usd
│  └─ scripts/
│     ├─ run_vehicle_v3.py                # ⭐ 단일/멀티 (--rovers) 런처
│     ├─ build_vehicle_v3.py              # vehicle_v3.usd 빌더 (USD 수정 시만)
│     ├─ mars_terrain_generator_v3.py     # ⭐ terrain 생성기 (epic obstacle 포함)
│     ├─ bake_m0609_home.py               # M0609 home pose 베이크 도구
│     └─ test_grip_unit.py                # grip 단위 테스트
│
├─ isaac_drive/                           # ③ 주행 (T3 — Critical Path)
│  ├─ isaac_drive/
│  │  ├─ coverage_node.py                 # ⭐ BCD anchor sweep + A* obstacle 회피
│  │  ├─ odom_to_estimated_pose.py        # GT cheat 어댑터 (T5 졸업 시 폐기)
│  │  ├─ minimap_publisher.py             # /mission/minimap 시각화
│  │  └─ navigation/                      # A*, BCD, fog_map, terrain_loader 모듈
│  └─ scripts/                            # state_writer + viewer (matplotlib 보조)
│
├─ isaac_perception/                      # ④ 인지 (T2)
│  ├─ isaac_perception/
│  │  ├─ yolo_perception_node.py          # ⭐ YOLO + depth backproject → mineral world XYZ
│  │  └─ yolo_mineral_detector.py         # YOLO wrapper class
│  └─ models/mineral_yolo_best.pt         # ⭐ 학습된 YOLO 모델 (5.3MB)
│
├─ isaac_supervisor/                      # ⑤ mission FSM + Web HUD (T4)
│  ├─ isaac_supervisor/
│  │  ├─ mission_manager_node.py          # ⭐ EXPLORE/APPROACH/PICK/RTB FSM + 멀티 rover 협조
│  │  ├─ mission_web_node.py              # ⭐ SC2 풍 Web HUD (Flask + SocketIO, 멀티 추적)
│  │  └─ battery_monitor_node.py
│  └─ web/                                # HTML/CSS/JS (rover selector + minimap canvas)
│
├─ isaac_manipulation/                    # ⑥ M0609 arm (T2)
│  └─ isaac_manipulation/
│     ├─ arm_executor_node.py             # ⭐ DLS-IK 상태머신 + /grasp/command FixedJoint
│     └─ kinematics.py                    # M0609 DH 파라미터 + DLS-IK
│
├─ isaac_localization/                    # ⑦ localization (T5, 정확도 졸업 대기)
│  └─ isaac_localization/
│     ├─ ekf_fusion.py                    # ⭐ EKF (wheel+imu+sun+TRN) + EDL prior
│     ├─ trn.py                           # Terrain Relative Navigation
│     ├─ localization_node.py
│     ├─ terrain_map_publisher.py
│     └─ sensors/                         # wheel_odom + imu_integrator + sun_yaw + splitter
│
├─ isaac_interfaces/                      # ⑧ msg/action 계약 (T4)
│  ├─ msg/                                # MissionState, Detection, BatteryState 등
│  └─ action/ExecuteArmTask.action
│
└─ temp/                                  # (gitignored) 개인 임시 작업
```

> ✅ = 동작, ⏳ = stub (트랙 owner가 채울 자리), 📦 = binary asset

## 패키지 매핑 (2026-05-26 cleanup 기준 — 8개)

| 패키지 | 트랙 | 담당자 | 주요 노드 | 상태 |
|--------|:----:|:-----:|----------|:----:|
| `isaac_bringup` | T4 | 성선규 | mvp / mvp_multi / integrated_localization / localization / rqt_views(_multi) | ✅ |
| `isaac_sim` | T1 | 김현중 | terrain v3 generator + vehicle_v3 빌더 + 23 terrain | ✅ |
| `isaac_perception` | T2 | 최진우 | `yolo_perception_node` (YOLO + depth backproject) | ✅ |
| `isaac_drive` | T3 | 이찬휘 | `coverage_node` (BCD + A*) + navigation 모듈 + `odom_to_estimated_pose` | ✅ |
| `isaac_supervisor` | T4 | 성선규 | `mission_manager_node` (FSM + 협조), `mission_web_node`, `battery_monitor_node` | ✅ |
| `isaac_manipulation` | T2 | 최진우 | `arm_executor_node` (DLS-IK + grasp snap) | ✅ |
| `isaac_localization` | T5 | 이지민 | `ekf_fusion_node`, `trn_node`, `sun_yaw_node`, `wheel_odom_node` 등 9 노드 | ⏳ 정확도 졸업 |
| `isaac_interfaces` | T4 | 성선규 | MissionState, Detection, BatteryState, ExecuteArmTask | ✅ |

> 2026-05-26 cleanup: `isaac_rl` 패키지 (PPO stub) + 6 outdated launch + `sim_bridge_node` (mock) + 옛 v2 빌더 도구 모두 정리.

## 핵심 문서 인덱스

| 누가 보나 | 무엇 | 위치 |
|----------|------|------|
| 신규 합류자 (5명) | 본인 트랙 onboarding | `docs/tracks/T{1-5}_BRIEF.md` |
| 모든 개발자 (Day 1) | **bashrc 셋업 가이드** | [docs/SETUP_BASHRC.md](docs/SETUP_BASHRC.md) |
| Claude Code 사용 시 | 자동 로드 context | `docs/tracks/T{1-5}_CLAUDE.md` |
| 모든 개발자 | 5개 인터페이스 계약 | [docs/interfaces/INTERFACE_CONTRACTS.md](docs/interfaces/INTERFACE_CONTRACTS.md) |
| 김현중, 이지민 (좌표계 합의) | I1 풀 가이드 | [docs/interfaces/I1_TERRAIN_ASSETS.md](docs/interfaces/I1_TERRAIN_ASSETS.md) |
| meta.json 필드 참고 | 라인별 주석 | [docs/interfaces/META_JSON_FIELDS.md](docs/interfaces/META_JSON_FIELDS.md) |
| 성선규 (PM) | 매일/주간 운영 | `docs/pm_tools/` |
| 모든 개발자 | 전체 설계 의도 | [docs/STUDY_AND_PLAN.md](docs/STUDY_AND_PLAN.md) |

## Architecture

- 전체 시스템 아키텍처: [system_architecture_full.svg](docs/flowcharts/system_architecture_full.svg)
- 미션 시나리오: [project_overview_flowchart.svg](docs/flowcharts/project_overview_flowchart.svg) · [PNG](docs/flowcharts/project_overview_flowchart_landscape.png)
- 개발 일정 timeline: [development_timeline.svg](docs/flowcharts/development_timeline.svg) · [PNG](docs/flowcharts/development_timeline.png)
