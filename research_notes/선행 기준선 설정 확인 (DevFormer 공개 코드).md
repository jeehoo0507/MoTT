# 선행 기준선 설정 확인 — DevFormer 공개 코드 기준 (2026-11-02, 재개 라운드 4)

QUEUE 11/01 15:10 할 일 4 · 23:45 "원 논문 설정 확인"의 일부. 무인 라운드에서 웹 검색·alphaXiv가 막혀 원문 PDF는 못 읽음 (PROGRESS STUCK 줄). 대신 10/29에 받아 둔 **DevFormer 공개 저장소** (`github.com/ai4co/devformer` v1.0.1 shallow clone, `/home/coder/ext_devformer/devformer/`, 코드는 실행하지 않고 읽기만)에서 확인할 수 있는 것만 적는다. 이 저장소에는 DevFormer 자신과, 그 논문이 비교한 attention-model 기준선 (`--model attention`, 스크립트 이름 `am-pg`·`am-il`)의 학습 설정이 들어 있다.

**확인 범위:** 아래 "원 설정" 칸은 코드·스크립트에서 읽은 것이다. 논문 본문의 표·서술 (예: `am-pg`를 논문에서 Park 2022로 부르는지, 보고 지표의 정확한 정의)은 **확인 필요**. Park TMTT 2022 원문과 Manoharan TSIPI 2026 원문은 여전히 미확인.

## 근거 파일 (저장소 안 경로)
- `src/options.py`: `--K 20` ("number of decap"), `--embedding_dim 128`, `--n_encode_layers 3`, `--normalization batch`, `--tanh_clipping 10`, `--baseline` (rollout/critic/exponential), `--bl_alpha 0.05` (rollout 기준선 t-test), `--N_aug` (DevFormer 대칭 증강).
- `src/problems/dpp/problem_decap.py` `get_costs`: 보상 = `reward_gen(probe, pi, 5)` → `reward_function_serial.model_5`.
- `src/problems/dpp/reward_function_serial.py`: 판 = `10x10_pkg_chip.npy` 하나 (n = m = 10 고정), 주파수 201점 (`freq_201.npy`). `model_5` = Σ_f (|Z_init(f)| − |Z_final(f)|) · 1e9/f / 10 — 목표 임피던스 없이 저주파 쪽에 가중을 둔 임피던스 감소량. (목표 Zt를 쓰는 `model_1`·`model_6`도 있으나 `get_costs`는 5번을 씀.)
- `src/models/attention_model.py`: 노드 입력 `init_embed = Linear(node_dim, 128)`; 입력 3채널 중 세 번째가 종류 표시 (2 = 탐침, 1 = keep-out, 0 = 빈칸) → 나머지 두 채널은 좌표.
- `scripts/dpp/train/am-pg.sh`: `--model attention --training_mode RL --baseline rollout --epoch_size 50000 --n_epochs 10` (학습 문제 5만 × 10 epoch).
- `scripts/dpp/train/am-il.sh`, `ours_2000.sh`: `--training_mode IL`, 전문가 해 2000개 (`guiding_2000_new.pkl`); DevFormer는 `--N_aug 4 --EE --SE` (CSE).

## 원 설정 vs 우리 재구현 (`bp_park_*`, `baselines_ml/park_am.py`)

| 항목 | DevFormer 저장소의 AM 기준선 (`am-pg`) | 우리 `bp_park_rl` / `bp_park_rlonly` |
|---|---|---|
| 판 | 10×10 pkg-chip **한 장** (같은 PDN, 문제마다 탐침 1개·keep-out 위치만 다름) | 세 종류 (pcb·pkg·ip) × 크기 8·10·12, 판 파라미터가 문제마다 다름 |
| decap | 한 종류 (`01nF_decap.npy`), **개수 K = 20 고정** | 종류 3개, 개수는 정책이 정함 (멈춤 행동) |
| 목표 | 개수 고정에서 임피던스 감소 최대 | 700점 정확 검사로 목표 Zt 만족 + 개수 최소 |
| 보상 | Σ_f (|Z_init|−|Z_final|)·1e9/f / 10, 201점, 목표 없음 | −개수/N_ref − λ·(실패 + 위반), λ 라그랑주 |
| 입력 | 칸마다 (x, y, 탐침/keep-out/빈칸) | 0단 좌표 입력 (`--feat coord`: 표시·놓인 종류·8점 위반·(x, y)·탐침까지 거리) |
| 구조 | Kool AM, encode-once, BatchNorm, 3층 128 | Kool식 포인터, 매 수 재인코딩, LayerNorm, 4층 128 |
| RL 기준선 | greedy rollout, paired t-test (α 0.05)로 교체 | greedy rollout, val 최고 갱신 때 교체 |
| 학습 예산 | 5만 문제 × 10 epoch (시간 기준 아님) | `bp_park_rl`: distill 0.75 h + RL 1.25 h / `bp_park_rlonly`: RL만 2.0 h (λ0 20) |
| 평가 | 개수 고정에서 보상 (임피던스 감소) | 성공률 (정확 검사) + 개수÷g+p·E0+ |

**의미:** 원 설정에는 "개수를 줄인다"와 "목표를 만족한다"가 없다 (K = 20 고정). 그래서 원 논문의 숫자와 우리 표의 숫자는 직접 비교할 수 없고, 우리 비교는 "같은 구조·입력 (좌표 + 포인터)을 우리 문제 정의에 옮겼을 때"다. 원 설정에서의 경향 재현 (QUEUE 23:45 sanity check)은 DevFormer 판 형식 (201점 표 Z)과 우리 정확 검사가 맞지 않아 하지 않음 (PROPOSALS (e), 10/29 03:40).

## DevFormer 자체 (기준선으로 돌리지 않음)
- 오프라인 모방 학습 (전문가 해 2000개) + 행동 순열 대칭 증강 (CSE: EE·SE, N_aug 4). 우리 증류 단계와 같은 계열 (선생님 해 모방); 우리는 그 뒤 RL을 더 함.
- 재구현하지 않은 이유: 구조의 핵심 (순열 대칭 인코더)은 개수 고정 문제를 전제하고, 우리 행동 공간 (종류·멈춤·제거)으로 옮기면 B-Park와 거의 같아짐. 공개 체크포인트는 Drive 다운로드가 필요해 쓰지 않음 (PROPOSALS (e)).

## Manoharan TSIPI 2026
저장소에 전문 없음. `reports/최종 모델 설계.md`·`baselines_ml/manoharan.py` docstring 기준 설명만 있음 → RESULTS의 차이 줄은 그 설명 기준이고 원문 대조는 **확인 필요**.
