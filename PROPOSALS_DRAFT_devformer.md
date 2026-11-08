# PROPOSALS 초안 — B-DevFormer (선행 기준선, 사람 결정 전 실행 금지)

상태 (2026-11-08 작성): 코드·smoke만 `b-devformer` 브랜치에 있음. **학습·큐 추가 안 함.** main에 합치기, 드라이버 재시작, 큐 추가는 모두 사람 결정.
원문: Kim et al., "DevFormer: A Symmetric Transformer for Context-Aware Device Placement", ICML 2023 (arXiv 2205.13225v3). 아래 쪽 번호 = PDF 쪽 (본문 쪽 번호와 같음). 공개 코드 메모: `research_notes/선행 기준선 설정 확인 (DevFormer 공개 코드).md`.

## 왜
- 가장 가까운 선행: 같은 decap 배치 문제, transformer 정책, **학습 없이 10×10 → 15×15·HBM으로 옮김** (p. 7 4.5·4.6절, Table 3). 우리 주장 (처음 보는 판 크기·종류로의 일반화)과 정면으로 겹쳐서 심사자가 "DevFormer와 비교했나"를 물을 것.
- 지금 표의 B-Park (`baselines_ml/park_am.py`)는 Kool AM 포인터 + 우리 distill → RL (또는 RL만). DevFormer가 AM에 더한 것 — 탐침 거리 임베딩 (PPE), 탐침·맥락 쿼리 (PCN·RCN), 순열 대칭 모방 학습 (CSE), RL 없음 — 은 B-Park에 없다. 10/29 메모 ("B-Park와 거의 같아져서 재구현 안 함")는 구조만 봤고 학습법 (CSE, RL 없음)이 다른 점을 놓쳤다.
- 노벨티 2줄: 기준선이라 노벨티 아님. 다른 점 (보고용): "오프라인 모방 + 순열 대칭"만으로 우리 문제 (목표 Zt + 최소 개수, 종류 3개, 제거 행동)를 얼마나 푸는지 = 우리 distill → RL의 RL 단계가 무엇을 더하는지의 외부 기준.

## 무엇 (코드)
- `baselines_ml/devformer.py` — `DevFormer` 정책, `permute_adds` (행동 순열), `distill_step` (EE + SE).
- `train.py`: `--model devformer` (→ `--feat coord`), `--df_perm P`, `--df_self λ`, `--df_layers` (기본 3). 기본값 (0, 0)이면 기존 모델·기존 distill과 같은 코드 경로 (smoke 4b: P = 0, λ = 0에서 기존 distill과 loss·기울기 완전히 같음). `--model devformer --stage rl`은 거부 (RL 없음 고정).
- `scripts/run_rung.py`: `--skip_rl` 추가 (distill만, 결과 = `<name>_distill/best.pt`). **`--rl_hours 0`은 distill만이 아님**: `train.py --hours 0`은 "시간 제한 없음"이라 RL이 iters 끝 (10⁶)까지 돈다.
- `scripts/pipeline_day2.py`: train 항목에 `"distill_only": true` (+ 선택 `"distill_hours"`) — 완료 판정·side-10 eval이 `_distill`을 봄. 다른 항목 동작은 그대로.
- `scripts/smoke_devformer.py` (CPU, 합성 판): 아래 결과.

## 원 논문 vs 우리 재구현

| 항목 | DevFormer 원문 (쪽) | B-DevFormer | 충실/변경 |
|---|---|---|---|
| 칸 입력 | x_i = (x, y, c), c ∈ {0 허용, 1 keep-out, 2 탐침} (p. 14 A.3) | (x, y) + c one-hot + 전원 입구 표시 + **놓인 종류 one-hot (T = 3)** — `--feat coord`에서 잘라 씀 | 충실 + 추가 (우리 상태·판 정보) |
| 위치 순서 | 위치 인코딩 없음, 좌표는 값으로만 (p. 3-4 Fig. 3) | 같음 (smoke 2: 칸 섞어도 출력이 같게 섞임, 오차 ~1e-6) | 충실 |
| PPE | h0 = φ_node(x) + φ_PPE(‖x_i − x_probe‖), 선형 (p. 4) | 같음; 탐침 여러 개 (1~3) → 가장 가까운 탐침까지 거리 | 충실 (다중 탐침만 변경) |
| 인코더 | L = 3, MHA 128, FF 512, post-norm BatchNorm (p. 4 식, p. 5 Implementation) | L = 3, 128/512, post-norm **LayerNorm** | 변경 (크기 섞인 패딩 판, B-Park와 같은 이유) |
| 인코딩 횟수 | f_enc(x) 한 번, 디코더가 자기회귀 (p. 3) | **매 수 다시 인코딩** (놓인 종류·위반이 입력 → 상태마다 바뀜; 정책 인터페이스가 상태만 받음) | 변경 |
| PCN | MLP(h_probe) (p. 4) | 같음 (탐침 칸 평균) | 충실 |
| RCN | MLP(h_{a_{t−1}}) 직전 행동 (p. 4) | MLP(놓인 칸 평균 임베딩) — 인터페이스에 직전 행동이 없고 상태는 집합 | 변경 |
| 목표·위반 맥락 | 없음 (판 하나 고정, 목표 없음) | SCN = Linear(전역 상태: 8점 위반, log f/f1, 목표 충족, 남은 수, 채움 비율), Q에 더함 | 추가 (없으면 문제마다 다른 목표를 모름) |
| 디코더 | Q = Linear(PCN + RCN), MHA 1층, Linear, softmax (p. 4); tanh clip 10 (공개 코드 `options.py`) | MHA glimpse 1번 → B-Park 포인터 (채널마다 쿼리: 종류 t 추가 / 제거), 10·tanh, 멈춤 logit | 변경 (행동 공간) |
| 행동·끝 | 빈 칸 하나 고르기, K 고정 (p. 3 2.3절; 학습 K = 20, p. 7) | (칸, 종류) 추가 / 제거 / 종류 변경 / 멈춤, 개수는 정책이 정함 | 변경 (우리 문제) |
| 목적 | K개에서 임피던스 감소 점수 최대 (p. 5) | 정확 검사로 목표 Zt 만족 + 개수 최소 | 변경 (우리 문제) |
| decap | 한 종류 | 세 종류 | 변경 |
| 학습 방식 | 오프라인 모방만, RL 없음 (p. 5-6 3.3절, Table 1 "Offline Contextual Pretraining") | `--stage distill`만, RL 없음 (`--stage rl` 거부) | 충실 |
| 전문가 해 | GA (M = 100 시뮬레이션), N = 2000 (p. 5, p. 23 E.1) | 우리 선생님 해 (g+p → LS, `data/teacher`), 다른 모든 모델과 같은 학습 풀 | 변경 (공정성: 같은 선생님) |
| EE (식 3) | 전문가 해마다 무작위 순열 P = 4개 + 원래 순서 (p. 5, p. 18 표 P = 4) | 선생님 궤적의 **추가 단계** 순서만 섞음 (제거·멈춤 꼬리는 같은 상태열이라 그대로), P = 4 + 원래 순서. 섞은 순서의 앞부분이 먼저 목표를 채우면 우리 규칙에서 다음 추가가 금지 → 그 사본은 버림 (`aug_ok`로 기록) | 충실 (꼬리·버림만 변경) |
| SE (식 4) | ‖π_θ̃(a′) − π_θ(t(a′))‖₁, a′ ~ 현재 정책 사본, 확률은 순서 전체; λ에 1e32를 곱해 λ·L ≈ 0.1~1 (p. 5, p. 24 E.2) | a′ ~ 현재 정책 (no-grad = θ̃), 추가 단계 순열 1개, **log 공간** \|log π_θ(t(a′)) − log π_θ̃(a′)\| / k (k = 추가 수). 0 ⇔ 그 표본에서 순서 대칭 (Theorem 1과 같은 조건). 1e32 스케일은 우리 긴 궤적에서 확률이 underflow·폭발해 안 씀 | 변경 (수치) |
| λ | 5 (p. 23 E.1·p. 25 E.3 본문) / 8 (p. 23 Table 5 캡션, p. 24 E.2 "최선") — 원문 안에서 엇갈림 | λ = 5 (ablation 대부분이 쓴 값); 스케일이 달라 숫자 자체는 원문과 비교 불가 → 원문의 보정 규칙 "λ·L_self ≈ 0.1~1"을 로그로 확인만 | 변경 (보고만) |
| lr·배치 | lr 1e-5, B = 100 (p. 18 C.1) | 우리 공통 `COMMON` (lr 3e-4, B 16, d 128, heads 8) | 변경 (모든 모델 같은 공통 인자) |
| 디코딩 | zero-shot 추론, 정책의 greedy 해 1개 (M = 1) (p. 18, Kool식) | greedy (모든 모델과 같은 평가) | 충실 |
| 매개변수 수 | — | 0.863 M (B-Park 1.012 M) | — |

## smoke (`python scripts/smoke_devformer.py`, CPU, 합성 7×7 판, 20초 미만)
```
1. forward ok: x (4, 49, 30) -> logits (4, 49, 4), stop (4,) | params: smoke (d 64) 0.219M, run size (d 128, L 3, FF 512) 0.863M (B-Park d 128 L 4: 1.012M)
2. permutation invariance: shuffled cells -> embeddings max |diff| 9.5e-07, logits 1.6e-06, stop 3.0e-08 (valid cells)
3. AP augmentation: add-phase lengths k [2, 4, 3, 3], P = 4 distinct permuted orders per instance (k >= 3), same set and suffix; replay valid 16/16, each valid one ends in the teacher solution
4a. distill_step EE P=4 + SE lambda=5: loss 3.324, |grad| 8406.186, stats {'aug_ok': 1.0, 'self_loss': 0.0315, 'self_ok': 1.0}
4b. P=0, lambda=0 equals the plain distill step: loss 3.2930 vs 3.2930, max |grad diff| 0.0e+00
5a. greedy rollout ok: count [4, 17, 12, 22] feasible [True, True, True, True]
5b. train.py --stage distill --model devformer --df_perm 4 --df_self 5 (3 iters, CPU) ok
5c. train.py --stage rl --model devformer is refused (distill-only)
OK
```
- 한 iter 비용 (CPU, 합성 10×10, B 16, d 128, 3회 중앙값, 계획용 추정): 기존 distill 1.25 s, EE (P = 4) 3.91 s (×3.1), EE + SE 7.65 s (×6.1). 학습 안 된 정책에서는 SE 표본의 순열 재생이 12.5%만 성공 (`self_ok`; 무작위로 오래 놓은 궤적은 섞으면 앞부분이 먼저 목표를 채움) — 학습이 진행돼 궤적이 짧아지면 오를 것으로 보지만 측정 아님. GPU 비율은 서버에서 다를 수 있음.

## 실행 계획 (사람 승인 후)
- **arm:** `--model devformer --df_perm 4 --df_self 5` (CSE = EE + SE, 원 논문의 방법 그대로). seed 0·1·2.
- **예산 — 권고: distill만 2.0 h** (`distill_only`, `distill_hours` 기본 = 0.75 + 1.25).
  - 근거 1 (같은 계산): 다른 모든 모델은 학습에 GPU 2.0 h (distill 0.75 + RL 1.25)를 쓰고, RL만 기준선 (`bp_park_rlonly`, `bm_manoharan_rlonly`)도 "같은 총 시간" 2.0 h를 받았다. DevFormer는 단계가 하나뿐이라 전부를 그 단계에 주는 것이 같은 규칙.
  - 근거 2: CSE 한 iter가 기존 distill의 ~6배라 0.75 h면 다른 distill 단계의 ~1/6 갱신만 받음 → 일부러 약하게 만든 기준선이 됨.
  - 0.75 h 쪽 읽기는 같은 run의 `log.csv` val 곡선 (0.75 h 시점, 인증 아닌 val)으로 따로 적음 — 추가 run 없음.
- **GPU 시간:** 본 실험 3 seed × 2.0 h = **6.0 GPU-h**. 선택: 뺀-PCB (학습 pkg·ip, 평가 pcb) 3 seed × 2.0 h = +6.0 → 합 12.0 GPU-h. seed 0만이면 +2.0 → 8.0. CPU: 다른 train 항목과 같은 side-10 인증 eval (드라이버 CPU 칸이 자동, run당 1번).
- **실행 전에 꼭 (사람):**
  1. `b-devformer`를 main에 합치고 서버에 pull.
  2. **`pipeline_day2.py` 드라이버 재시작** — 드라이버는 큐 JSON은 매번 다시 읽지만 코드는 시작할 때만 읽는다. 옛 드라이버는 `distill_only`를 모르고 distill 0.75 h → RL로 돌리며, RL은 `train.py`가 거부해 실패하지만 0.75 h짜리 `_distill/metrics.json`이 남아 새 드라이버가 "끝남"으로 본다 (그러면 그 폴더를 옮기고 다시).
  3. 서버에서 `.venv/bin/python scripts/smoke_devformer.py` → `runs/analysis/smoke_devformer.ok` (큐 항목의 `after`).
- 손으로 한 run 돌릴 때 (드라이버 밖, 같은 결과 경로):
  `.venv/bin/python scripts/run_rung.py --name bdf --arm "--model devformer --df_perm 4 --df_self 5" --day 2026-10-29 --pool runs/pool_v1 --seed 0 --distill_hours 2.0 --skip_rl --skip_eval`
  (드라이버는 `CERTIFY_WORKERS=2`로 부름. distill 단계는 정확 검사를 val에서만 씀.)

## day2_queue.json "train"에 넣을 항목 (넣지 않았음)
```json
{"name": "bdf", "arm": "--model devformer --df_perm 4 --df_self 5", "seed": 0, "distill_only": true, "after": "runs/analysis/smoke_devformer.ok"},
{"name": "bdf_s1", "arm": "--model devformer --df_perm 4 --df_self 5", "seed": 1, "distill_only": true, "after": "runs/analysis/smoke_devformer.ok"},
{"name": "bdf_s2", "arm": "--model devformer --df_perm 4 --df_self 5", "seed": 2, "distill_only": true, "after": "runs/analysis/smoke_devformer.ok"}
```
선택 (뺀-PCB, 여유 있으면; 기존 `lo_pcb_*`와 같은 arm 꼬리):
```json
{"name": "lo_pcb_bdf", "arm": "--model devformer --df_perm 4 --df_self 5 --pool_glob train_pkg*.pkl,train_ip*.pkl --val_pools val_pkg10_0,val_ip10_0", "seed": 0, "distill_only": true, "after": "runs/analysis/smoke_devformer.ok"},
{"name": "lo_pcb_bdf_s1", "arm": "--model devformer --df_perm 4 --df_self 5 --pool_glob train_pkg*.pkl,train_ip*.pkl --val_pools val_pkg10_0,val_ip10_0", "seed": 1, "distill_only": true, "after": "runs/analysis/smoke_devformer.ok"},
{"name": "lo_pcb_bdf_s2", "arm": "--model devformer --df_perm 4 --df_self 5 --pool_glob train_pkg*.pkl,train_ip*.pkl --val_pools val_pkg10_0,val_ip10_0", "seed": 2, "distill_only": true, "after": "runs/analysis/smoke_devformer.ok"}
```
결과 경로: `runs/2026-10-29/bdf{,_s1,_s2}_distill/` (best.pt, log.csv, eval/eval_*10_*__bdf_distill.json).

## 미리 적는 읽기 (결과 전 고정) — 보고만, 채택 판정 없음
- B-DevFormer는 기준선이므로 이 결과로 우리 모델의 어떤 결정도 바꾸지 않는다 (채택·기각 규칙 없음).
- 보고할 것 (10칸 eval, 종류마다 50문제, 정확 검사): seed마다 성공률, 둘 다 성공한 문제의 개수 ÷ g+p, seed 평균. 비교 줄: `bp_park_rl` (같은 좌표 입력, AM + distill → RL), `bp_park_rlonly_rl`, g+p, E0+, 최종 모델 — 짝지은 bootstrap 95% CI (seed 0, 기존 표와 같은 방식).
- 뺀-PCB를 돌리면: `lo_pcb_bdf`의 eval_pcb10 성공·개수를 기존 `lo_pcb_*`와 같은 표에.
- 학습 로그에서 함께 적을 것: `aug_ok` (EE 사본 중 재생 성공 비율), `self_ok`, λ·L_self가 원문 보정 범위 0.1~1 안이었는지, best iter, 0.75 h 시점 val.
- 해석 문장 틀 (숫자만 채움): "원 논문 방식 (오프라인 모방 + 순열 대칭, RL 없음)을 같은 선생님·같은 2 h로 옮기면 10칸 성공률 x/y/z". 결과를 보고 arm·λ·예산을 바꾸지 않는다. SE가 사실상 꺼져 있었으면 (`self_ok` 낮음) 그 사실을 같이 적는다.

## 남은 위험
- SE 재생 성공률이 낮으면 SE 항이 거의 0 → 사실상 EE만 (비용만 2배). 대안 (EE만, `--df_self 0`)은 이 초안에 넣지 않음 — 결과 보고 고르기를 피하려고 arm 하나만 미리 정함.
- 매 수 재인코딩·SCN·LayerNorm·포인터 채널은 우리 행동 공간 때문에 바꾼 것이라, "DevFormer 원본 수치 재현"이 아니라 "DevFormer 방법을 우리 문제에 옮김"으로만 쓸 것 (B-Park와 같은 단서).
- `collect_results.py`의 표들은 `*_rl` 이름 위주 — `bdf_distill` 줄은 결과가 나온 뒤 따로 넣어야 할 수 있음 (확인 필요).
