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
| 바퀴 충돌 형상 | 바퀴 mesh 의 convex hull (다각형 + 축에서 1.7~4.8 mm 편심) — 구동 접지력이 설정 마찰의 60~65 % | **축에 중심을 맞춘 캡슐** (`--wheel-collider round`, 기본값) — 설정 마찰 그대로. 짧은 급경사 띠를 넘는 한계가 20° → 23~25° |
| link 속도 상한 | 로버 모델이 차체·바퀴 link 의 선속도를 1.5 m/s 에서 잘라 냄 (주행도, 넘어질 때 떨어지는 속도도) | **해제** (`--link-speed-cap off`, 기본값). 모델 파일은 그대로 두고 시뮬레이터가 뜰 때 덮어씀 |
| 광물 위치 추정 (2.5 m 이내 중앙값) | 수평 오차 18.7 cm, 높이 +46 cm | 수평 3.7 cm (90% 4.7 cm), 높이 −3.6 cm |
| `ik_descend_dz=-0.40` | 인식 높이 편향을 상쇄하는 임시값 | 제거 (기본 0.05) |
| 광물 파지 | 팔 받침대 2.5 m 안의 광물을 손가락으로 순간이동해 붙임 (magic grasp) | **넓은 집게로 조여서 마찰만으로 듦** (`--grasp friction`, 기본값). 부착용 joint 없음 |
| 그리퍼 | OnRobot RG2-FT (최대 벌림 10 cm — 광물 17×19×23 cm 를 쥘 수 없음) | 27 cm 까지 벌어지는 평행 집게 (`--gripper wide`, 기본값) |
| 광물 질량 | 0.3 kg (스티로폼 수준) | 4 kg (암석 밀도 기준, 화성 무게 14.9 N) |
| 손목 카메라 | 장착 위치·자세가 추정값이라 미사용 | USD 실측값으로 교체, pick 직전 수평 위치 보정에 사용 (손끝 오차 평균 4.0 → 1.4 cm) |
| 장애물 지도 | 지형과 함께 만들어 둔 정답 지도 `obstacle_grid.npy` 로 A* | **지도 없이 출발** — 로버 깊이 카메라로 주행 중에 위험 지도를 만들고, 새 장애물이 경로를 막으면 다시 계획 (`map_source:=online`, 기본값) |
| 구동 조절 (선택) | — | 강화학습(PPO)으로 배운 **구동 조절기** — 기울기·흔들림·헛돎을 보고 속도 명령을 깎음 (`drive_governor:=true`, 기본은 꺼짐: 실제 임무 스택 검증 전) |
| Web HUD | `localhost:8088` 로만 사용, 영상은 다른 포트(8080)에서 직접. ACTIONS 버튼 9 개는 자리만 있고 동작 안 함. 초상화는 예전 그리퍼 그림 | **포트 하나로 다른 기기에서도 열림** — 같은 공유기 안의 주소, `hud_public:=true` 면 인터넷 주소(Cloudflare 터널)까지 (영상 중계 + 접속 키 + 보기 전용 스위치), **버튼 9 개 동작** (E-STOP·수동 주행·기지 복귀 등), 누른 것은 EVENT LOG·ROS 로그·파일에 기록, 초상화는 지금 로버(넓은 집게)를 시뮬레이터에서 찍은 것 + 자세계 |
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

### 바퀴 접지력 교정 (2026-11-24, `isaac_sim/scripts/wheel_colliders.py`) — 기본값

"오르막에서 0.02 m/s 밖에 안 난다" 를 쫓다가 찾은 것입니다. 마찰 한계(레골리스 0.52 → 27.5°)보다 훨씬 낮은 20° 근처에서 이미 못 올라갔습니다.

- **원인**: 로버 모델의 바퀴 충돌 형상은 바퀴 mesh 의 convex hull, 즉 다각형입니다. 바퀴가 돌면 꼭짓점이 땅을 치며 떠서, 헛도는 동안 **물리 스텝의 82 % 에서 바퀴가 땅에 닿아 있지 않았습니다** (접촉 기록으로 확인). 제동(바퀴 잠김)은 형상이 돌지 않으니 정상입니다.
  USD 원기둥으로 바꿔도 PhysX 가 다시 다각형 convex 로 근사해서 같습니다. 굴러가는 방향으로 완전히 둥근 형상(캡슐·구)이어야 합니다.
- **평지에서 잰 접지력** (`probe_traction.py`: 바퀴를 "차체 속도 + 일정 값" 으로 헛돌리며 차체 가속도 / 중력. 로버 13 대, 설정 동마찰 0.52):

  | 바퀴 충돌 형상 | 구동 (헛돎 0.15~2 m/s) | 제동 (바퀴 잠김) |
  |---|---|---|
  | mesh convex hull (원본) | 0.31 ~ 0.35 | 0.50 ~ 0.54 |
  | 원기둥 (다각형 근사) | 0.31 ~ 0.39 | 0.48 ~ 0.54 |
  | **캡슐** (적용) | **0.50 ~ 0.52** | 0.48 ~ 0.49 |
  | 구 | 0.50 ~ 0.52 | 0.48 ~ 0.49 |

  물리 주기를 60 → 240 Hz 로 올려도, mesh 의 편심(1.7~4.8 mm)만 없애도 달라지지 않았습니다. 형상이 둥근지가 전부였습니다.
- **비탈에서** (`probe_slope.py`, terrain_00015. 이 지형의 23° 넘는 급경사는 길이 중앙 1.0 m · 최대 2 m 의 짧은 띠라서 "띠를 넘어 1 m 더 가는가" 를 봅니다. 같은 자리끼리 비교, 명령 0.92 m/s = 기본 제한):

  | 띠의 경사 | 띠 아래 끝에서 정지 출발 (원본 → 캡슐) | 2.5 m 앞에서 달려서 진입 (원본 → 캡슐) |
  |---|---|---|
  | 15° | 13/14 → 11/14 (0.71 → 0.89 m/s) | 4/5 → 4/5 |
  | 20° | 3/9 → **6/9** (0.17 → 0.82 m/s) | 1/2 → 2/2 |
  | 23° | 0/6 → 2/6 | 0/3 → **3/3** |
  | 25° | 0/3 → 1/3 | 0/1 → 1/1 |
  | 27° · 29° | 0/3 → 0/3 | 자리 없음 |

  27° 부터는 진짜 마찰 한계입니다 (`tan θ ≤ μ`). 가파른 구간은 자리가 1~6 곳뿐이라 경향만 믿을 수 있습니다. 전복은 같은 시험 전체(149 회 시도, 명령 0.3~2.0 m/s)에서 원본 20 회 → 캡슐 6 회였는데, 줄어든 것은 전부 1.2 m/s 이상의 빠른 명령에서입니다. 기본 제한(0.92 m/s) 이하에서는 양쪽 다 81 회 중 1 회였습니다.
- **HUD 에서 0.02 m/s 로 멈춰 있던 바로 그 자리** (terrain_00015 의 (−11.1, 11.5), 길이 1.5 m · 경사 23.6~26.5° 의 띠. 로버 16 대를 0.3~3 m 앞에 놓고 기본 제한 0.92 m/s 로 출발): 원본 바퀴는 **1/16** 만 넘었고(전복 2), 캡슐은 **16/16** 이 4~7 초에 넘었습니다(전복 0, 최대 기울기 28~47°).
- **임무 스택** (terrain_00015, 기본 모드 + `self_right:=true`, 1 회): 3/3 수집, 551 s 동안 272 m, 최대 기울기 25°, 전복 없음, 막힘 감지 2 회(23° 띠 앞과 모래 구역 — 둘 다 그 뒤 임무가 이어짐). 같은 지형의 직전 네 번(원본 바퀴, 420 s 제한)은 2/3 · 0/3 · 1/3 · 0/3 이었고 그중 두 번은 넘어졌습니다 — 다만 시간 제한(600 s vs 420 s)과 광물을 찾는 운이 달라서 이 차이를 전부 바퀴 덕으로 볼 수는 없습니다.
  완만한 지형 회귀 시험도 1 회씩 통과: terrain_00026 은 60 초에 강제로 뒤집은 것을 3.8 초에 세우고 3/3 수집 + 기지 복귀(225 s), terrain_00028 은 3/3 + 복귀(245 s, 전복 없음).
- 캡슐은 반지름 0.105 m (mesh 의 바깥 반지름 평균), 몸통 길이는 접지 폭의 절반입니다. 양 끝이 반구라 축 높이에서 mesh 보다 양옆으로 7 cm 씩 넓습니다 (땅에 닿는 폭은 mesh 의 절반). 보이는 모양은 그대로입니다. `--wheel-collider mesh` 로 원본 동작이 됩니다.
- 같이 알게 된 것: 로버 모델의 link 에 `maxLinearVelocity = 1.5` 가 걸려 있어 차체 속도가 1.5 m/s 근처에서 물리적 이유 없이 막혔습니다 → 아래 "link 속도 상한 해제" 에서 풀었습니다.
  아직 고치지 않은 것: 구동 관절에 위치 강성(100 N·m/deg, 목표 0)이 남아 있어 계산상 바퀴 속도가 ±1.6 cm/s 로 출렁입니다. 그리고 주행 기구학은 바퀴 반지름을 0.10 m 로 계산하는데 mesh 는 0.105 m 라, 헛돌지 않으면 명령보다 5 % 빠릅니다.

### link 속도 상한 해제 (2026-11-24, `wheel_colliders.lift_link_speed_cap`) — 기본값

로버 모델(`vehicle_v3.usd`)은 차체·바퀴·조향 link 16 개에 `physxRigidBody:maxLinearVelocity = 1.5` 를 두고 있습니다. PhysX 는 이 값을 넘는 link 속도를 힘이 아니라 **속도를 직접 깎아서** 막습니다. 그래서

- 평지에서 바퀴를 2.4 m/s 로 돌려도 차체는 1.57 m/s 에서 멈췄고 (접지력은 남아 있는데도),
- **떨어지는 속도도 1.5 m/s 에서 잘렸습니다.** 화성 중력에서는 0.30 m 만 떨어져도 1.5 m/s 인데, 넘어지는 로버의 link 는 그보다 빨라질 수 없었습니다 — 전복이 실제보다 느리고 약하게 계산된 셈입니다.

모델 파일은 건드리지 않고, 바퀴 형상처럼 시뮬레이터가 뜰 때 값을 100 m/s 로 덮어씁니다 (`run_vehicle_v3.py --link-speed-cap off`, 기본값. `model` 을 주면 원래대로). 학습·평가 환경도 같습니다 (`eval_drive.py`·`train_*.py` 의 `--speed-cap`).

풀고 나서 다시 잰 것:

| | 상한 있음 (모델 그대로) | 해제 |
|---|---|---|
| 평지에서 바퀴 2.4 m/s 일 때 차체 속도 | 1.57 m/s 에서 멈춤 | 2.5 m/s 까지 올라감 |
| 넘어짐 복구 (학습한 정책 그대로, 넘어진 채 출발 190 회) | 96 % | **99 %** (옆으로 100 %, 등으로 97 %. 걸린 시간 중앙 3.2 s) |
| 급경사 지형 눈 가리고 주행 (terrain_00015, 16 대 × 15 분, 고정 제한 0.92 m/s) 전복 | 9 회 / 2719 m | 8 회 / 2670 m |
| 차체가 30° 넘게 기운 시간 | 1.4 % | 3.1 % |

복구 정책은 상한이 걸린 채로 학습한 것인데 풀어도 성공률이 떨어지지 않았습니다. 주행 중 전복 횟수는 사실상 같고(같은 자리에서 같은 식으로 넘어집니다 — 아래 전복 분석), 크게 기운 시간은 늘었습니다: 상한이 기우는 움직임을 늦춰 주고 있었다는 뜻입니다. 임무 회귀 시험(terrain_00026, 60 초에 강제로 뒤집기): 10 초 만에 일어나 3/3 수집 + 기지 복귀.

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
| 막혀도 계속 밀기만 함, 비탈에서 바퀴만 헛돎 | 장애물은 깊이 카메라 + 위치로만 알고, 접촉이나 헛돎을 느끼는 수단이 없음 | **막힘 감지** (`hazard_map_node`): 4 초 구간마다 명령한 거리와 실제로 간 거리를 비교해, 15 % 도 못 갔으면(방향을 40° 넘게 트는 중은 제외) 진행 방향 0.9 m 앞에 반경 0.45 m 위험 원을 찍음 → 두 계획기가 돌아감 |
| 광물이 없는 허공을 집음 | 12 m 밖 노란 광물의 거리가 3 프레임 동안 2.7 m 로 잡혔고(깊이 화소가 가까운 땅에 걸림), "같은 방향의 더 가까운 검출로 목표 수정" 이 그것을 받아들여 로버 바로 앞을 집으러 감. 팔은 손목 카메라에 광물이 안 보여도 그대로 진행했음 | 인식: 추정 거리에서 계산한 크기가 8 cm 미만이면 좌표를 버림 (실제 광물은 0.17~0.33 m, 문제의 검출은 5 cm). 팔: 손목 카메라가 목표 0.5 m 안에서 광물을 하나도 못 보면 **집지 않고 취소** (`abort_if_wrist_sees_nothing`). 파랑·노랑 광물에만 적용 — 초록 가스는 실제로 있어도 손목 카메라에 안 잡히거나 19 cm 어긋나게 잡혀서(3 번 중 3 번) 처음 넣은 규칙이 있는 가스를 거절했음. 가스는 예전처럼 로버 카메라 추정값으로 진행 |

- 측정:
  - 제자리 회전 수정 — 학습 환경 평가(terrain_00026, 로버 16 대 × 6 분, 전복 0 회 유지): 목표점 도달이 고정 제한 1.58 → **1.96 개/분**, 구동 조절기 2.20 → **2.28 개/분**.
  - 거리 추정 — 6 m 넘게 떨어진 검출 중 거리 오차 1 m 초과 비율: 7.1 % → **3.4 %** (임무 1 회씩, 정답 광물 위치 기준).
  - 구조물 — terrain_00028, 구동 조절기 모드, 같은 설정으로 세 번: 투명 상자 그대로일 때 한 번은 배럭에서 전복, 한 번은 골리앗 앞에서 정지(1/3 수집).
    보이는 모델로 충돌을 바꾼 뒤 **3/3 수집 후 완주**, 장애물까지 최소 0.71 m, 전복 없음, 지도 재현율 96.3 %.
  - 임무(terrain_00026, 기본 설정): 3/3 수집(파랑·노랑·파랑), 접근 3 회 모두 한 번에 도착, 장애물까지 최소 1.00 m, 전복 없음.
- 한계: 임무 검증은 지형당 1~2 회입니다. 광물이 바위에 0.8 m 안으로 붙어 있으면 여전히 접근하지 못하고 제외합니다.
- 광물이 여럿 보일 때의 우선순위는 원본 그대로입니다: 종류별 가치(노랑 50 > 초록 25 > 파랑 10)가 먼저, 신뢰도는 동점 처리용(×0.1), 거리는 보지 않음. 한 번 접근을 시작하면 그 광물에 고정됩니다.

### Web HUD — 주소로 열기 · 버튼 · 기록 (2026-11-24, `mission_web_node.py`, `isaac_supervisor/web/`, `mission_manager_node.py`)

`mvp_multi.launch.py` 로 띄웠을 때의 HUD 입니다. `mvp.launch.py`(단일 로버, namespace 없음)도 같은 HUD 노드를 쓰지만, 버튼·접속 키용 launch 인자와 토픽 연결은 `mvp_multi.launch.py` 에만 넣었고 시험도 그쪽으로만 했습니다.

**화면 구성**

| 줄 | 왼쪽 → 오른쪽 |
|---|---|
| 1 행 (화면 높이의 44 %) | ROVER CHASE · BODY — YOLO · ARM CAM |
| 2 행 | COVERAGE MAP · 상태 수치 (PHASE, BATTERY, COLLECTED, POS, YAW, SPEED, TILT, CMD, TASK, ERR) · EVENT LOG · 초상화 + 자세계 · ACTIONS |

- **EVENT LOG** 는 최신이 위입니다. 줄이 칸보다 많아지면 세로, 문장이 칸보다 길면 가로 스크롤 막대가 생깁니다 (문장은 자르지 않습니다). 예전 줄을 보려고 내려가 있는 동안에는 새 줄이 와도 보던 자리가 밀리지 않습니다. 제목 줄 오른쪽 **COPY** 를 누르면 지금까지의 기록(최대 400 줄)이 시간순 텍스트로 복사됩니다.
- **초상화** 는 `isaac_sim/scripts/render_portrait.py` 가 시뮬레이터에서 찍은 지금 로버입니다 (아래). 그 위에 선택한 로버 이름과 상태가 겹쳐 나오고, 테두리 색이 상태를 따릅니다 (수동 = 노랑, 복구 = 빨강, E-STOP = 빨강 깜빡임). 아래 **자세계** 는 odom 의 자세에서 구한 roll·pitch·기울기이고, 기울기는 25° 부터 노랑, 40° 부터 빨강입니다 (60° 를 넘으면 넘어진 것).

**다른 기기에서 열기 (localhost 가 아닌 주소)**

HUD 는 포트 하나(기본 8088)만 씁니다. 카메라 영상도 HUD 서버가 중계하므로 (`/stream?topic=…`) 그 포트만 닿으면 전부 보입니다. `web_video_server` 는 이 PC 안(127.0.0.1)에서만 듣습니다.

```bash
ros2 launch isaac_bringup mvp_multi.launch.py rovers:=rover_1 ... \
     hud_public:=true        # 인터넷 주소(https://….trycloudflare.com)를 만든다 (기본: false)
     hud_key:=<접속 키>      # 주면 키 없이는 아무것도 안 열림 (기본: 없음. hud_public 이면 자동으로 만듦)
     hud_actions:=false      # 주면 버튼이 꺼진 보기 전용 (기본: true)
     web_port:=8088
```

주소는 HUD 시작 로그(`HUD 주소 — …`)와 화면 왼쪽 위 알약 두 개에 나옵니다. 알약을 누르면 접속 키가 붙은 주소가 복사되니 그대로 다른 기기에 넘기면 됩니다.

| 어디서 | 주소 | 필요한 것 |
|---|---|---|
| 이 PC | `http://localhost:8088` | 없음 (접속 키가 걸려 있어도 이 PC 에서는 키 없이 열림) |
| 같은 공유기 안의 기기 | `http://<이 PC 의 IP>:8088` | 방화벽(ufw)이 켜져 있으면 그 포트를 열어야 함 — 기본값이 "들어오는 연결 차단" 이라 안 열면 접속이 그냥 멈춥니다. `sudo ufw allow from 192.168.0.0/24 to any port 8088 proto tcp` (대역은 자기 공유기에 맞게) |
| 인터넷 어디서나 | `https://<임시 이름>.trycloudflare.com` | `hud_public:=true` + `cloudflared` 실행 파일 (아래) |

- **인터넷 주소** 는 HUD 노드가 `cloudflared` 터널을 직접 열어서 만듭니다 (`mission_web_node.PublicTunnel`). 밖에서 들어오는 연결을 여는 것이 아니라 이 PC 가 Cloudflare 로 나가는 연결을 만드는 방식이라 공유기·방화벽 설정이 필요 없습니다. 계정 없이 쓰는 임시 주소라 **띄울 때마다 바뀌고**, HUD 를 내리면 사라집니다. **영상과 상태가 Cloudflare 의 서버를 거쳐 나간다**는 점은 알고 쓰십시오.
  `cloudflared` 설치: [공식 릴리스](https://github.com/cloudflare/cloudflared/releases)의 `cloudflared-linux-amd64` (약 40 MB) 를 `~/.local/bin/cloudflared` 로 두고 `chmod +x` (root 불필요). 없으면 HUD 는 그대로 뜨고 로그에 안내만 남깁니다.
- **접속 키**: 키가 걸려 있으면 `…/?key=<키>` 로 한 번 들어온 브라우저만 쿠키로 통과하고, 나머지는 화면·영상·상태·버튼 전부 403 입니다. 인터넷 주소는 누구나 접속을 시도할 수 있으므로 키 없이는 내보내지 않습니다 — `hud_public:=true` 인데 `hud_key` 를 안 줬으면 HUD 가 키를 만들어 로그와 알약 주소에 붙입니다. 터널을 거쳐 온 요청은 127.0.0.1 에서 오지만 전달 헤더(`X-Forwarded-For`, `Cf-*`)와 Host 로 구분해서 키를 요구합니다. 버튼이 로버를 움직이므로, 보기만 시킬 거면 `hud_actions:=false` 를 같이 주십시오.
- 이미 떠 있는 HUD(`hud_key` 를 준 것)를 내리지 않고 인터넷 주소만 만들려면 `tools/hud_tunnel.sh [포트]`.
- 화면의 Socket.IO 스크립트는 CDN(`cdn.socket.io`)에서 받습니다. HUD 를 여는 기기가 인터넷에 닿지 않으면 화면은 뜨지만 상태가 갱신되지 않습니다.

확인한 것 (2026-11-24, terrain_00026 로버 1 대, `hud_public:=true`):

| | 결과 |
|---|---|
| 인터넷 주소로 브라우저에서 열기 | 0.5 초에 화면, 카메라 셋 모두 재생 (chase 0.8 MB/s · body 0.1 MB/s · arm 0.2 MB/s), 상태·지도·EVENT LOG 갱신, 40 초 동안 끊김 없음 |
| 인터넷 주소에서 버튼 | REPLAN, E-STOP → PHASE·TASK 가 `E-STOP` → RESUME 후 임무 계속 |
| 키 없이 / 틀린 키로 (인터넷 주소, 같은 네트워크 주소) | 403 |
| 터널을 거쳐 `Host: localhost` 로 속여서 | 403 |
| 이 PC 에서 `localhost` 로 | 키 없이 열림 |

같은 공유기 안의 다른 기기에서 실제로 여는 것은 방화벽 규칙을 넣은 것까지만 했고, 다른 기기로 접속해 보지는 않았습니다. 참고로 ssh 만으로 되는 `localhost.run` 익명 터널도 시험했는데 속도가 20 → 2 kB/s 로 묶이고 (화면이 뜨는 데 8 초, 영상은 사실상 정지) 주소가 40 초 만에 바뀌어 쓰지 않습니다.

**ACTIONS** — 위쪽 탭에서 고른 로버에게 보냅니다 (E-STOP 만 전부에게).

| 버튼 | 하는 일 | 로버에게 가는 것 |
|---|---|---|
| AUTO ⇄ MANUAL | 자율 ↔ 수동. 수동이면 W/A/S/D 또는 방향키로 주행 (10 Hz, 키를 떼면 정지) | `<ns>/mission/mode`, 주행은 `<ns>/teleop/cmd_vel` |
| E-STOP ⇄ RESUME | **모든 로버 정지.** 주행 명령 0, 하던 팔 동작은 그 자리에서 멈춤. PHASE 가 `E-STOP`, TASK 가 `E-STOP`(팔을 세웠으면 `E-STOP · 팔 정지`)으로 바뀌고 해제할 때까지 그대로. RESUME: 팔을 HOME 으로 되돌린 뒤 (그동안 차체는 정지) 기지 복귀 중이었으면 복귀를, 아니면 탐색부터 다시 | `<ns>/mission/command` `ESTOP` / `RESUME` |
| RTB | 지금 기지로 복귀. 복귀 중에 누르면 취소 (목표 달성·배터리 때문에 하는 복귀는 취소 안 됨) | `RTB` / `RTB_CANCEL` |
| RESET (두 번 눌러야 실행) | 채집 수를 0 으로. 임무가 끝났으면 다시 시작 | `RESET` |
| REPLAN | 지금 경로를 버리고 다시 계획 | `REPLAN` |
| YOLO ON/OFF | 카메라 영상을 인식 표시가 있는 것 ↔ 원본으로 (이 브라우저만) | — |
| SKIP | 접근 중인 목표를 한동안 건너뜀 | `SKIP` |
| MAP CLR (두 번 눌러야 실행) | 주행 중 만든 위험 지도를 비움 | `<ns>/hazard_map/reset` |
| SPD | 수동 주행 속도 0.3 → 0.6 → 0.9 m/s | — |

로버가 거절한 명령은 이유가 EVENT LOG 에 나옵니다 (예: 채취 동작 중의 RTB, E-STOP 중의 RTB).

**기록** — 버튼을 누르면 세 군데에 남습니다 (브라우저에서만 바뀌는 YOLO·SPD 도 포함).

| 어디 | 형식 |
|---|---|
| HUD 의 EVENT LOG | `20:03:25 rover_1 ESTOP → ESTOP` (노란 줄 = 누른 것, 그 아래 로버의 응답) |
| ROS 로그 (`mission_web_node`) | `[HUD] action=estop rover=rover_1 → ESTOP (from 172.30.1.20)` |
| `~/.ros/log/hud_actions.jsonl` (`ROS_LOG_DIR` 가 있으면 그 아래. `action_log_path` 로 변경) | `{"time": "2026-11-24T19:47:57", "rover": "rover_1", "action": "estop", "sent": "ESTOP", "from": "127.0.0.1"}` |

로버 쪽에서는 `mission_manager` 가 받은 명령과 결과를 자기 로그(`operator command ESTOP: …`)에 남기고 `<ns>/mission/event` (std_msgs/String) 로 알립니다. phase 전환도 같은 토픽으로 나가서 EVENT LOG 에 같이 보입니다.

확인한 것 (terrain_00026, 로버 1 대): 채취 동작 중 E-STOP → 팔이 그 자리에서 멈춤 → RESUME → 5 초에 HOME 복귀 → 다음 광물로 접근해 임무 계속. 수동 주행 명령(0.6 m/s, 0.7 rad/s)이 바퀴까지 전달. 접속 키가 있을 때 키 없는 요청은 화면·정적 파일·영상·상태·Socket.IO 모두 거절.

### HUD 초상화 다시 찍기 (`isaac_sim/scripts/render_portrait.py`)

```bash
tools/isaac-pypi isaac_sim/scripts/render_portrait.py     # → isaac_supervisor/web/static/img/rover_portrait.png
```

시뮬레이터가 쓰는 것과 같은 조립을 거친 로버를 찍습니다: `vehicle_v3.usd` + 넓은 집게 + 캡슐 바퀴를 화성 중력에서 지면에 내려앉히고, 팔을 `--pose` 자세로 든 모습. 손으로 그리거나 합성한 그림이 아니므로 로버 모델이나 집게가 바뀌면 다시 돌리면 됩니다. 자리는 평평하고 바위·구조물에서 5 m 넘게 떨어진 곳을 고릅니다 (`--at X,Y` 로 지정 가능). 카메라 위치·보는 점·초점거리·크기는 옵션으로 바꿉니다 (파일 머리말).

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

### 넘어짐 복구 (`isaac_sim/rl/righting_env.py`, `isaac_drive/self_right_node.py`) — 선택 기능, 기본 꺼짐

넘어진 로버가 팔 관절 6 개로 땅을 밀어 다시 일어섭니다 (PPO, 200 만 샘플, 약 100 분).

- 학습 조건: 뒤집힌 자세로 떨어뜨린 뒤 14 초 안에 기울기 15° 아래로 1 초 유지하면 성공.
  팔 구동력은 150/150/100/40/40/40 N·m 로 제한했고(시뮬레이터 기본 9600 N·m 는 사실상 무한 — **제한값은 실제 팔 사양을 확인한 것이 아니라 가정**), 팔이 차체를 통과하지 못하게 자기 충돌을 켰습니다.
- 평가 (200 회, `train_right.py --eval`): 넘어진 채 출발한 191 회 중 **98 %** 가 일어섰습니다 (옆으로 누운 124 회 중 98 %, 등으로 누운 67 회 중 99 %). 걸린 시간 중앙값 3.7 초, 90 % 가 4.7 초 안. 일어선 뒤 팔을 접어도 모두 서 있었습니다. 무작위 동작만으로는 23 %.
  바퀴 충돌 형상을 캡슐로 바꾼 뒤 같은 정책을 다시 평가하면 190 회 중 **96 %** (옆 118 회 중 97 %, 등 72 회 중 96 %), 중앙값 3.3 초입니다. 정책은 원본 바퀴로 학습한 그대로이고 다시 학습하지 않았습니다 (`--wheels mesh` 로 예전 조건 재현).
- `self_right_node` 는 기울기 60° 초과로 2 초 멈춰 있으면 이 정책으로 팔을 움직이고, 그동안 supervisor 는 바퀴를 세우고 단계를 **RECOVERY** 로 바꿉니다 (웹 UI 의 PHASE 는 RECOVERY, TASK 는 `self-righting with arm (1/3)` → `folding arm after righting`; 일어서면 하던 단계로 복귀, 채취 중이었으면 EXPLORE 부터). 런치 `self_right:=true`.
- 임무 스택 시험 (시뮬레이터 `--flip-at <초>` 로 강제로 뒤집음): 네 번 모두 일어섰고(2.9~6.5 초) 이후 3/3 수집과 귀환을 마쳤습니다 (terrain_00026 세 번은 학습 도중의 정책, terrain_00028 한 번은 최종 정책).
  강제가 아닌 실제 전복도 두 번 있었습니다 (terrain_00015, 구조물 둘레의 파인 턱 — 아래 "남은 문제"): 두 번 모두 약 5 초 만에 일어나 임무를 이어 갔습니다.
- **남은 문제 — 위에서 안 보이는 턱**: terrain_00015 의 구조물은 땅을 0.55 m 파낸 자리에 놓여 있고 그 둘레가 26~35° 의 턱입니다. 위쪽에서 다가가면 깊이 카메라에 턱 아래가 보이지 않아 지도에 "못 본 칸" 으로 남고(위험으로 치지 않음), 로버가 그대로 내려가다 넘어집니다. 같은 자리에서 두 번 재현됐습니다. 지도의 경사 한계를 20° 로 낮춰도(`hazard_slope_deg:=20`) 막지 못했습니다 — 보이지 않는 것이 문제이기 때문입니다.
- 한계: 평지와 완만한 곳에서 뒤집힌 경우만 학습했습니다. 가파른 비탈이나 바위 틈에 낀 경우는 시험하지 않았습니다.
- 추적 카메라(chase)는 차체 기울기를 따라가지 않게 바꿔서, 넘어지고 일어서는 모습이 웹 UI 에서 보입니다.

### 비탈 균형 정책 (`isaac_sim/rl/balance_env.py`, `isaac_drive/balance_node.py`) — 실험, **채택 안 함**

구동 조절기에 "뒤로 빼기"(전진 배율 −0.25~1)와 "팔 균형추"(팔 13.5 kg / 전체 62.6 kg 을 뻗어 무게중심을 앞 5 cm · 뒤 13 cm · 좌우 8 cm 옮김)를 더해, 같은 중력·마찰의 가파른 지형(terrain_00015, 목표의 절반을 15° 이상 비탈에 둠)에서 150 만 샘플 학습했습니다. 결과는 쓰지 않는 쪽입니다.

| terrain_00015, 로버 16 대 × 6 분 | 고정 제한 | 구동 조절기 | 균형 정책 |
|---|---|---|---|
| 목표 도달 (개/분) | 0.44 | 0.73 | 0.20 |
| 전복 | 7 | 9 | **0** |
| 기울기 30° 넘는 시간 | 1.5 % | 1.1 % | 9.7 % |
| 25~28° 비탈에서의 속도 (m/s) | 0.12 | 0.09 | 0.05 |
| 25~28° 비탈에서 헛도는 비율 | 87 % | 90 % | 79 % |

- 전복 0 은 "거의 안 움직여서" 얻은 것입니다. 임무 스택 시험에서는 20° 비탈에서 앞이 45° 들린 채 70 초 동안 버티는 동작도 나왔습니다.
- 이 표에서는 25° 이상의 비탈을 어떤 제어기로도 못 올랐습니다 (바퀴가 80~90 % 헛돎). 등판 한계는 `tan θ ≤ μ` 로 정해지고(중력은 양쪽에서 상쇄), 레골리스 마찰 0.58/0.52 면 헛돌기 시작한 뒤의 한계가 27.5° 입니다. 6 륜을 모두 구동하므로 팔로 무게를 옮겨도 접지력의 합은 같습니다.
  **다만 이 표는 원본 바퀴 형상으로 잰 것입니다.** 그때는 구동 접지력이 설정의 60~65 % 만 나와서 실제 한계가 20° 근처였습니다 (위 "바퀴 접지력 교정"). 제어기를 학습시켜서가 아니라 바퀴 충돌 형상을 고쳐서 한계가 올라갔습니다.
  같은 평가를 캡슐 바퀴로 다시 하면 (균형 정책은 다시 재지 않음): 고정 제한은 목표 도달 0.44 → **0.68** 개/분 · 주행 1586 → 2111 m · 전복 7 → 7, 구동 조절기는 0.73 → 0.81 개/분 · 전복 9 → 8 입니다. 더 멀리 가지만 **전복 수는 줄지 않았습니다.**
  고정 제한 기준으로 10~20° 비탈에서의 속도는 0.18 → 0.37 m/s 로 올랐고, 20~25° 비탈에 머문 시간은 4 % → 19 % 로 늘었습니다 (더 위까지 올라가서 거기서 막힘 — 그 구간 속도는 0.11 m/s).
- 전복은 지형 경사 19~23° 인 곳에서 차체가 30~50° 까지 들린 직후에 났습니다.
- 그래서 넣은 것: 헛돎 감지(아래 "막힘 감지" — 명령한 거리의 15 % 도 못 가면 앞쪽을 위험으로 표시), 지도의 경사 한계를 바꾸는 런치 인자 `hazard_slope_deg`, 그리고 넘어졌을 때의 복구(위).
- 실행하려면 `drive_governor:=balance` + 시뮬레이터 `--drive-limits off`. 비교 표는 `eval_drive.py --balance` 로 다시 만들 수 있습니다 (고정 제한 열은 같은 환경에서 팔을 home 에 둔 것, 구동 조절기 열은 팔 제어가 없는 원래 환경).

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

> SuperDOGE: `mvp_multi.launch.py` 의 HUD 는 화면 구성·버튼·접속 방법이 달라졌습니다 — 위 "Web HUD — 주소로 열기 · 버튼 · 기록" 을 보십시오. 아래는 원본 설명입니다.

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
