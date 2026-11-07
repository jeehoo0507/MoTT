# PROPOSALS 초안 — B-Manoharan v2 (`--model manoharan2`, 원문에 맞춘 재구현) — 에이전트, 2026-11-08 02:55 KST, 어떤 bm2 run도 돌기 전에 적음

**실행 금지 — 사람 승인 뒤 큐로.** 이 파일은 브랜치 `bm-faithful`에만 있다 (main에 합치지 않음). 승인되면 이 절을 PROPOSALS.md에 옮기고, 아래 "큐 항목"을 `agent/day2_queue.json`의 `train`에 넣는다 (이 초안에서는 넣지 않았다).

## 왜 (원문 감사)
기존 `bm_manoharan*`은 원문 PDF 없이 사람 메모로 만들었다 (`research_notes/선행 기준선 재구현 차이 표.md`: "원문 미확인" 여러 칸). 원문 (박사논문 Paper I, 쪽수는 **인쇄 쪽**, PDF 쪽 = 인쇄 + 14)을 읽고 맞춰 보니 코드가 원문과 **어긋나는** 곳이 여섯 군데다:

| # | 원문 | 기존 `manoharan` | 근거 쪽 |
|---|---|---|---|
| 1 | 포트 순서 = **반복** greedy: loop L 가장 낮은 포트를 단락시켜 고정 → Z 갱신 (Schur) → 나머지 다시 순위, 모두 순위 매길 때까지 | 맨판에서 **한 번** 순위 | pp.14–15, 식 (1) |
| 2 | f0 = "중간 대역, 보통 유도성이 지배하는 곳" | 맨판 **최악 위반** 주파수 | p.14 |
| 3 | IC 포트 여럿: L을 IC 포트들에 대해 **평균** | 최악 probe 하나 | 식 (7)–(8), p.22 |
| 4 | 디코더 한 층: 이번 에피소드의 **지난 디코더 입력 기록**에 self-attention → decap 토큰에 cross-attention → 내적 + 온도 softmax | 기록 없음 (현재 상태 query만) | 그림 1 (p.13), 식 (3)–(5) (pp.19–20) |
| 5 | ΔZ = Z_current − Z_target **선형**, 201점 (Linear 201→64) | 특징 주파수 8점의 log 비 | 식 (2) p.18, 그림 1 |
| 6 | decap 토큰 4특징 (C, ESL, ESR, 패키지) → 2층 MLP + LayerNorm; 라이브러리에 "no decap" 포함 | 3특징 → Linear 한 층 + self-attention | pp.16–17, 그림 1 |

기존 run의 B-Manoharan 결과 (RESULTS "선행 연구 재구현 seed 0·1·2": `bm_manoharan` pcb 0.00·pkg 0.01, `bm_manoharan_rlonly_l20` pcb 0.00·pkg 0.32)가 "Manoharan 방식의 한계"인지 "우리 재구현의 어긋남"인지 지금은 구분이 안 된다. 심사에서 "선행 연구를 약하게 만들었다"는 지적을 막으려면 원문대로 맞춘 판이 하나 있어야 한다.

## 무엇이 바뀌었나 (브랜치 `bm-faithful`)
- 새 파일 `baselines_ml/manoharan2.py` (`ManoharanEnv2`, `ManoharanNet2`, `manoharan2_teacher`, `manoharan_reward`), 새 이름 `--model manoharan2` / `--feat manoharan2`. **기존 `--model manoharan`은 그대로** (예전 run 재현 가능; 스모크 6번에서 확인). `pdn.py`·`sim.py`·`verify.py`·`bank.py`는 건드리지 않았다 (sim의 `_probe_diag`·`probe_impedance`를 import만).
- 1 포트 순서: 맨판 Z0(f0)에서 반복 greedy, 단락은 `sim.Env._rank1`과 같은 rank-1 Schur 갱신의 dy→∞ 극한 (계수 1/Z_jj). 판(instance)마다 한 번 계산해 batch 객체에 캐시 (작업 격자 기본 32점은 정밀 검사가 점을 붙여도 안 움직임). 브루트포스 (단락 노드를 직접 푼 Z)와 처음 두 순위 일치 (스모크 1).
- 2 f0 (**원문이 정하지 않은 f0에 대한 우리 선택**): 맨판에서, 작업 격자 기본 32점 중 log 대역 가운데 절반 (25 %–75 %) 안에서 IC 포트 임피던스 위상 (IC 포트 평균 |φ−90°|)이 +90°에 가장 가까운 점; 그 안에 유도성 점 (위상 > 0)이 없으면 대역 기하 중심. 합성 판 세 종류에서 0.8–2.8 GHz, 위상 +87.6~+88.4° (첫 평면 반공진 바로 아래 유도성 평탄부), 규칙이 항상 적용됨.
- 3 IC 포트 여럿: L_j = IC 포트 평균 Im Z_ii(f0 | j 단락)/(2πf0) (식 8). ΔZ도 IC 포트 평균 ("aggregated", p.22 — 연산자는 원문 미명시, 평균은 우리 선택).
- 4 디코더: q_t = e_port + e_ΔZ (식 3); q_t가 이번 에피소드의 q_0..q_t에 causal self-attention (residual + LayerNorm) → decap 토큰 T+1개 (no decap 포함)에 cross-attention (residual + LayerNorm) → u = (h W_Q)·(e_c W_K)/√d, softmax 온도 τ = 1 (원문 값 미기재). 기록은 env가 특징 텐서 [E, n, D]의 "단계 행"으로 들고 다녀서 `rollout.py`·`backward_logp` 인터페이스가 그대로이고, 기울기 계산 때 지난 query를 저장된 입력에서 다시 계산한다 (원문 "past decoder inputs"). 과거 행을 바꾸면 logit이 바뀌고 미래 행은 영향 0 (스모크 5).
- 5 ΔZ: (|Z_IC| − Zt)/Zt **선형**, 작업 격자 기본 **32점** (원문 201점; 우리 env의 기본 격자가 32점이고 붙는 점은 판마다 달라 쓰지 않음). Zt로 나눈 것은 우리 선택 (목표가 판마다 자릿수가 달라서), [−1, 100]로 자름.
- 6 decap 토큰: (C, ESL, ESR) log 정규화 → Linear(3→64) ReLU LN → Linear(64→128) ReLU LN (그림 1). 패키지 크기는 우리 판에 없음. "no decap"은 같은 3특징 입력 공간의 학습 행.
- 인코더 셋 모두 그림 1대로 Linear(→64) ReLU LN, Linear(64→128) ReLU LN. 파라미터 **193,987** (v1 152,449).
- `--reward manoharan` (train.py, 기본 `ours`): 원문 끝 보상 식 (9)–(15): 성공 R = N_ports − N_caps·d, 실패 R = −N_caps·d − Σ_f max(|Z|−Zt, 0)², d = 1/(mean Zt + ε), ÷N_ports. 실패 판정은 우리 것 (작업 격자 + 정밀 검사), |Z|는 작업 격자, IC 포트 여럿이면 합. 그룹 기준선·rollout 기준선 둘 다 연결 (λ는 계속 갱신되지만 안 쓰임).
  - 주의 (숫자, 합성 10칸 판 세 종류 4판씩): mean Zt ≈ 0.4–6.5 Ω → d ≈ 0.15–2.2, 부품 하나 벌점 d/N_ports ≈ 0.002–0.03 — 성공이면 R ≈ +1, 개수 압력은 약함. 빈 판 실패 R ≈ −1 ~ −2600 (pkg가 큼). 그룹 정규화 아래에서는 **M개가 모두 성공한 그룹에서 우리 보상과 같은 advantage** (둘 다 개수의 1차식) — 차이는 성공·실패가 섞인 그룹과 전부 실패한 그룹에서만 난다. 원문 p.26 "더 빡빡한 목표일수록 부품을 더 허용" 문장은 식 (12)–(13) (Zt가 작을수록 d가 커져 부품 벌점이 커짐)과 반대 — 식대로 구현했다.
- 학습 알고리즘: train.py의 RL 루프는 B×M 그룹 rollout → 에피소드 advantage → `backward_logp` 구조라, critic + replay를 넣으려면 트레이너를 새로 짜야 한다 → **하지 않음** (아래 "남은 차이"). 그룹 기준선 REINFORCE 유지.

## 스모크 (`python scripts/smoke_manoharan2.py`, CPU, 합성 8×8 판 6장, P = 3, 통과 — 기록 `runs/analysis/smoke_manoharan2.ok`가 서버에서 생김)
1. 반복 순서 = 놓을 수 있는 칸의 순열 (6/6), 한 번 순위와 6/6 판에서 다름 (위치 39–46개 다름), 캐시 = 새 계산, 브루트포스 처음 두 순위 일치.
2. f0 대역 안 (0.83–1.35 GHz, 대역 위치 0.58–0.65), 유도성 규칙 6/6, 위상 +87.6~+88.4°.
3. P = 3에서 L = IC 포트 평균 (포트별 다시 계산과 일치).
4. 학습 안 된 정책 greedy rollout 끝까지; 샘플 rollout + backward_logp 기울기 유한; 원문 보상 유한; 선생님 해를 반복 순서로 다시 걸으면 같은 해.
5. 과거 기록 행 변경 → logit 2.2e-2 변화, 미래 행 → 0.
6. 파라미터 193,987; v1 `manoharan` 빌드·rollout 그대로.
추가: `train.py --model manoharan2 --stage rl` 합성 판 (풀 없음) 1–2 iter, `--reward ours/manoharan`, `--baseline rollout` 모두 끝까지 돎. **풀 (`runs/pool_v1`)로는 아직 안 돌려봄** — 노트북에 풀 파일이 없음. 첫 run의 처음 25 iter 로그를 사람이 확인할 것.

## 실행 계획 (추천 하나)
- **추천: `bm2_rlonly` — RL만, 원문 보상, 무작위 초기화, seed 0·1·2, run당 2 h (우리 표준 예산), 합 6 GPU h** (서버 하나에서 순서대로 ~6 h + 평가 CPU).
  - arm: `--model manoharan2 --imit 0 --reward manoharan`, `rl_only: true`.
  - 왜: 원문은 RL만 (REINFORCE, p.24), 원문 보상. 우리 증류+RL (`bm2`)을 쓰면 "원문대로"가 아니라 "원문 구조 + 우리 학습"이 되어 이번 작업의 목적 (원문 충실도)과 어긋남. 예전 `bm_manoharan_rlonly` (λ0 = 1)는 "아무것도 안 놓고 실패"로 붕괴했는데, 원문 보상은 실패에 위반 제곱합 벌점이 커서 (위 숫자) 같은 붕괴가 덜할 것으로 예상 — 예상일 뿐이다.
  - 붕괴 감시 (결과 전에 고정): it 100까지 val 세 종류 모두 fail 1.00이고 loss가 0이면 기존 rlonly 선례대로 멈추고 `--lam` 대신 손댈 것이 없으므로 (원문 보상엔 λ가 없음) **보고만 하고 다른 변형으로 갈아끼우지 않는다** (사람 결정).
- 대안: `bm2` — 우리 증류 0.75 h + RL 1.25 h (`--model manoharan2`, 보상 ours, 모방 포함). 기존 `bm_manoharan`과 같은 파이프라인이라 "구조만 바꾼 효과"를 읽기 좋다. 같이 돌리면 +6 GPU h (합 12 h). 추천은 `bm2_rlonly`만.

## 큐 항목 (승인되면 `agent/day2_queue.json` `train`에 추가 — 이 초안에서는 넣지 않음)
```json
{"name": "bm2_rlonly",    "arm": "--model manoharan2 --imit 0 --reward manoharan", "seed": 0, "rl_only": true, "after": "runs/analysis/smoke_manoharan2.ok"},
{"name": "bm2_rlonly_s1", "arm": "--model manoharan2 --imit 0 --reward manoharan", "seed": 1, "rl_only": true, "after": "runs/analysis/smoke_manoharan2.ok"},
{"name": "bm2_rlonly_s2", "arm": "--model manoharan2 --imit 0 --reward manoharan", "seed": 2, "rl_only": true, "after": "runs/analysis/smoke_manoharan2.ok"}
```
대안을 같이 할 때만:
```json
{"name": "bm2",    "arm": "--model manoharan2", "seed": 0, "after": "runs/analysis/smoke_manoharan2.ok"},
{"name": "bm2_s1", "arm": "--model manoharan2", "seed": 1, "after": "runs/analysis/smoke_manoharan2.ok"},
{"name": "bm2_s2", "arm": "--model manoharan2", "seed": 2, "after": "runs/analysis/smoke_manoharan2.ok"}
```
`after` 파일은 서버에서 `python scripts/smoke_manoharan2.py`가 통과하면 생긴다 (먼저 브랜치를 서버에 받아야 함). 평가는 드라이버의 기존 10칸 정확 검사 평가가 자동으로 함 (`eval_*10_*`).

## 읽기 (결과 전에 고정, **보고만 — 채택 판정 없음**)
- 표: 10칸 pcb·pkg·ip 각 50문제, seed 0·1·2, 정책 단독 정확 검사 성공률·개수÷g+p (둘 다 성공) — 기존 B1 표와 같은 형식으로 `bm2_rlonly` (와 `bm2`) 행 추가.
- 비교 상대: **MoTT+P1 (`r2pk`, 정책 단독)** 같은 seed끼리 같은 50문제 짝지은 성공률 차, 3 seed 묶음 bootstrap 95 % CI. 종류마다 CI가 0 밖이면 "r2pk 높음/낮음", 아니면 "구분 안 됨". 기존 `bm_manoharan*`과의 차는 숫자만 (재구현 충실도 효과 — 판정 아님).
- 이 run으로 기존 판정·표 문구·최종 방법 (PROPOSALS 11/08 F)은 바꾸지 않는다. 보고서 문장은 "같은 2 h 예산에서, 원문 구조·보상, critic·replay 없이"를 붙여서만 쓴다.
- 되먹임 (`--repair 4`) 포함 평가는 이 계획에 없다 (원하면 사람이 F처럼 따로 등록).

## 남은 차이 (원문 vs `manoharan2`)
| 항목 | 원문 | `manoharan2` |
|---|---|---|
| 학습 알고리즘 | REINFORCE + 단계별 critic V(s_t) (식 16–19) + replay buffer 10k, mini-batch 64, γ = 0.99 (Table 1, pp.27–28) | 그룹 평균 기준선 REINFORCE (M 샘플), critic·replay·할인 없음 (끝 보상을 모든 단계에 같은 가중으로) |
| 예산 | 5만 에피소드, RTX 4090 ~24 h | 2 h/seed (≈ 1/12) |
| 데이터 | 실제 PCB BEM Z (25–75 포트), Murata ~1000종 중 에피소드마다 5–25종, 목표 1 mΩ–1 Ω 평탄/RL | 우리 세 종류 합성 판 (8–12칸), 종류마다 고정 3종 |
| ΔZ 점 수 | 201 | 32 (작업 격자 기본점) |
| decap 특징 | 4 (패키지 포함), 패키지 맞지 않는 종류 마스크 | 3, 마스크 없음 |
| 포트 특징 | L + 패드 크기 one-hot | L만 (log(L/중앙값)) |
| dropout | 후보 인코더에 "약한" dropout | 없음 (rollout과 기울기 재계산의 표본이 달라짐) |
| τ, d_k, head 수 | 미기재 | τ = 1, d = 128, 8 head |
| ΔZ 정규화·IC 집계 | 원 옴, 집계 연산자 미기재 | ÷Zt, IC 평균 |
| 보상 실패 판정 | 주파수 격자에서 Z > Zt | 작업 격자 + 정밀 검사 (우리 규칙) |
| 종료 | 목표 만족까지 | 목표 만족 (멈춤만 허용) 또는 순서 끝 |
| 디코딩 | greedy / top-K (K = 2) / 샘플 N = 100 | greedy (평가 공통 규칙) |

critic + replay를 넣으려면 필요한 것 (하지 않음, 기록만): (a) 정책에 value head V(s_t) 추가 (q_t 또는 h에서 MLP), (b) rollout이 단계별 보상 r_t (끝 단계만 R)와 할인 리턴 R_t = Σ γ^{k−t} r_k를 저장 — 지금 `backward_logp`는 에피소드당 가중 하나라 **단계별 가중**을 받도록 바꿔야 함 (`ep` 대신 단계 인덱스), (c) 손실 L_actor + λ·L_critic (식 19), (d) (특징 x, mask, action, R_t) 단계 튜플 10k replay buffer와 64 mini-batch 추가 갱신 루프 — 저장된 x로 logit을 다시 계산하므로 off-policy 보정 없이 원문처럼 그대로 쓰면 됨, (e) train.py RL 분기의 advantage·imitation·expert iteration 경로와 분리된 새 `--algo manoharan` 분기, resume.pt에 buffer 저장. 대략 200줄 + 별도 스모크.
