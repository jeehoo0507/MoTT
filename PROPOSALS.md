# PROPOSALS (에이전트 제안 — 실행 금지, 사람 승인 후 큐로)

## 확인 필요 (서지)
- Manoharan, Wang, Hwang, "RL for Decap Optimization in Postlayout PDNs Using Transformer Architecture", IEEE TSIPI 2026 / Scholars' Mine 3459 — 전문·권호 미확인.
- Park & Sengupta, "Neural Modal Decomposition", arXiv 2609.14402 — 학회 게재 여부 미확인 (설계서에는 NeurIPS 2026 포스터로 적혀 있음).
- Fan et al., "Looped Transformers for Length Generalization", ICLR 2025 — 저자·학회 확인 필요.
- `reports/최종 모델 설계.md`가 인용한 "Kuo 2026" (반복 횟수 랜덤 학습) — 출처 불명.
- **(11/02) 실제 논문 기준선 확인 (QUEUE 11/01 할 일 4)에 원문이 필요:** 무인 라운드 허용 목록 (`agent/loop.sh` ALLOWED)에 WebSearch·WebFetch·alphaXiv가 없어 원 논문 설정 (입력·상태·보상·학습 예산·평가 지표)을 확인할 수 없음. 선택지: (i) Park TMTT 2022, Manoharan TSIPI 2026, DevFormer ICML 2023 PDF (또는 해당 절 발췌)를 `research_notes/papers/`에 넣기, (ii) ALLOWED에 `WebSearch,WebFetch` 추가. 그 전까지 재구현 차이 표는 저장소 안 기록 (`reports/`, 코드 docstring) 기준으로만 씀.

## 환경 파일 수정이 필요한 문제 (사람 결정)
- `verify._probe_z_exact`의 배치 복소 solve가 이 서버(torch 2.14+cu130, MKL)에서 n ≥ 200 셀·스레드 ≥ 2이면 멈춘다 (`scripts/mkl_repro.py`). 지금은 `decap_loop/safe_verify.py`로 프로세스당 1스레드로 우회. 근본 수정은 `verify.py`에서 `torch.linalg.solve`를 주파수별 루프 또는 `scipy.linalg.lu_solve`로 바꾸는 것.

## 사람 결정이 필요한 것 (QUEUE 한도·평가 세트)
- **16·20칸 eval 풀 생성 시간 (2026-10-28 18:30 측정):** 풀 1개(25문제)를 1스레드로 만드는 시간이 판 변의 ~6제곱으로 늘어남 (풀 json `t_total`: pcb 학습 풀 8칸 ~12 s/문제, 10칸 ~54, 12칸 ~150; eval 10칸 25~104 s/문제). 16칸 6개는 18:20 기준 3시간째 첫 2개가 안 끝남. 이 비율이면 16칸 1개 ~6~9 h → 6개를 2 workers로 ~18~27 h, 20칸 1개 ~30 h 이상 → 6개 ~90 h. 지금 드라이버는 10·16칸 풀이 다 끝나야 E0·단 평가를 시작하므로 10칸 단 평가도 10/29 낮까지 밀린다. 선택지 (에이전트는 QUEUE 한도 4를 지키려고 아무것도 바꾸지 않음):
  (a) CPU 프로세스 한도를 6으로 올려 10칸 평가를 16칸 풀과 동시에 (지금 pod 메모리 ~10~21 GiB / 60, 16칸 풀 worker 하나 3~9 GB),
  (b) 16칸 풀 1개를 멈추고 그 자리에서 10칸 평가 먼저 (멈춘 풀의 진행분은 버려짐),
  (c) 20칸 eval을 파일당 25 → 8문제로 줄이기 (평가 세트 변경이라 사람 승인 필요; 아직 아무 방법도 20칸에서 평가되지 않았으므로 결과를 보고 고르는 문제는 없음).
  - **갱신 (18:55 측정):** 위 추정은 너무 비관적이었음. 실제 `eval_pcb16_0` 9941 s, `eval_pcb16_1` 10392 s (398·416 s/문제, pcb 10칸의 ~7배 → 변 길이의 ~4제곱). 남은 것: pkg16 ×2 (18:00·18:08 시작), ip16 ×2, ip10_1. ip 10칸이 pcb 10칸보다 문제당 ~2배 느려서 ip16이 가장 늦음 → 10·16칸 풀 전체는 10/29 새벽 끝 예상 (계획용 추정, 보고서 숫자 아님). 20칸은 같은 4제곱이면 pcb 풀 1개 ~7 h, 6개 2 workers로 ~1일. (a)(b)(c)는 여전히 사람 결정 사항이지만 급하지는 않음.

- **(d) 기준선 강화: "종류 하나만 쓰는 greedy+prune 중 최선"을 E0에 추가할지 (10/29 02:45).** ip 10칸에서 g+p는 성공 0.40인데, 같은 규칙을 100 pF 종류만 쓰게 하면 50/50 성공·개수가 3단 rl과 같음 (비 1.00~1.03, RESULTS.md "한 종류만" 표, `scripts/analyze_gp_one_type.py`). 지금의 "정책이 ip에서 g+p를 크게 이긴다"는 기준선 규칙의 약점에 기대고 있음. 선택지: (i) 지금 기준선 유지 + 고찰에 이 분석을 적기 (에이전트가 지금 하는 것), (ii) E0에 "종류별 g+p 3번 중 최선(성공 → 개수)"을 새 기준선으로 추가 (10·16칸, 정책 평가는 다시 안 해도 됨 — 같은 풀·같은 검사라 표만 늘어남; 16칸은 1회 수 시간 × 3종류 × 6풀). 평가 세트·검사는 안 바뀜.
  - **(d') 추가 메모 (10/29 04:25): E0+는 pcb에서 섞어 쓰는 g+p보다 약하다.** 첫 E0+ 기록 pcb10_0 (25문제): 성공 0.72 vs g+p 0.90 (짝지은 차 −0.28 [−0.44, −0.12]), 둘 다 성공한 문제의 개수 비 E0+/g+p **5.00** [3.79, 6.30] (종류 1만으로 평균 48개; 종류 0 0.04, 종류 2 0.00 성공). 즉 pcb에서는 종류를 섞는 것이 필요하고, "종류 하나 고정"은 강한 기준선이 아님 → "정책 vs E0+" 개수 비 (pcb 0.17~0.22)는 정책을 과대평가함. 제안: 정식 비교 기준을 **문제마다 {g+p, E0+} 중 최선** (성공 → 개수; 기록 재조합만, 새 계산 없음)으로 하는 것. 에이전트는 QUEUE 03:15 4번의 정의를 그대로 두고, 대신 RESULTS에 g+p 대비와 E0+ 대비를 둘 다 계속 보고함 (사람 결정).

- **(e) 외부 데이터 검증 (QUEUE 10/29 03:30 B) — 포기 (10/29 03:40, ~15분 사용).** `github.com/ai4co/devformer` (v1.0.1) shallow clone → `/home/coder/ext_devformer/` (저장소 밖, 코드는 실행 안 함; `.npy`만 `allow_pickle=False`로 읽음, `scripts/ext_devformer_inspect.py`). 확인한 것:
  1. 데이터 = **고정 Z 행렬 하나** `10x10_pkg_chip.npy` [201 주파수 × 100 × 100] complex (100 MHz~20 GHz 표, 대칭), 문제 100개 (`test.npz`)는 탐침 1개 + keep-out mask만 다름. **목표 임피던스 없음** (보상 = 정해진 K개로 Σ(|Z0|−|Z|)·1e9/f).
  2. **공개 체크포인트가 저장소에 없음** (README의 `data/dpp/pretrained/CSE_2000_epoch-50.pt`는 Google Drive zip; 쓰려면 그 저장소 코드와 pickle `torch.load`를 실행해야 함).
  3. decap = **한 종류, R+C만** (0.1436 Ω, 0.1 nF, ESL 0 → SRF 없음), 그리고 시뮬레이터가 `np.abs(decap)` (실수)를 decap 임피던스로 씀 → 우리 물리(복소 R+L+C)와 다름.
  - 안 맞는 이유: (i) 우리 성공 판정은 `verify.py` 700점 정확 검사 (격자 Laplacian으로 임의 f에서 Y(f)를 다시 풂) — 201점 표 Z에서는 돌릴 수 없음 → 새 판정이 필요하고 그건 규칙 2 밖. (ii) 정책 입력 차원이 T=3 종류에 묶여 있음 (`sim.feat_dim`, techs.py "All kinds keep T = 3") → 종류 1개 판은 가짜 종류 채우기 필요. (iii) ESL 0이면 `--fc_bias`의 log SRF가 정의되지 않음. (iv) 목표 Zt를 우리가 정해야 함 (Manoharan 4.2절은 0.5 Ω).
  - 사람이 원하면 필요한 것: 표 Z 어댑터 (bank를 201점 Z에서 직접) + 201점 판정 (보고서에 "DevFormer 판정 기준"으로 따로 표기) + Zt 선택 + 종류 패딩 규칙 + Drive 체크포인트 다운로드·실행 허가. 2일 이상 걸릴 것으로 보고, 마감(11/8 동결) 안에서 사다리·아키텍처 실험보다 뒤.

## 2일차 아키텍처 개선 — 노벨티 2줄 메모 (QUEUE 10/29 03:45, 판정은 사람)
- **(a) `--fc_bias` (r3a 10/29 03:10, 이어서 r2a):** 가장 가까운 선행: MatNet 2021 (두 집합 사이 행렬을 attention 편향으로), Manoharan 2026 (decap ESL·ESR·C 토큰). / 다른 점: 편향이 **공진 모드 k와 decap 종류 t 사이의 주파수 궁합** −|log f_k − log SRF_t|/τ_h이고 모드는 매 수 갱신되는 loaded 판의 피크 → 종류 선택을 규칙 없이 공진 상태에 연결.
- (b)는 QUEUE 03:15 수정으로 삭제 (이미 MoTT add head [u, c, u⊙c]에 있음).
- B-Park (`baselines_ml/park_am.py`)는 기준선이라 노벨티 아님.
- **(c) 검산 루프 — 코드 확인 메모 (10/29 04:10, 아직 구현 안 함):** 1수 깊이의 "놓기 전 반공진 검산"은 이미 1~3단에 있다: 모든 (포트, 종류) add와 remove마다 작업 격자 전체에서 정확히 계산한 V−V′와 최악 여유 `mn`이 포트 특징 끝 2(T+1)채널로 들어가고, MoTT·LoopMoTT가 `prior = w_dv·dV − w_mn·mn`으로 logit에 더한다 (`decap_loop/modal_env.py`, `decap_loop/model.py` forward). 그래서 (c)를 1수 Woodbury 평가로 만들면 기존 입력과 같다. 새것이 되려면 **여러 수 초안**(상위 m개로 d수 놓은 뒤 피크 재추출 → 모드 토큰 교체)이어야 하고, 그것은 DACT/NeuRewriter식 개선 루프에 가깝다 (설계 문서 §7 H4도 "기여 아님"). 결정: (a) r3a·r2a 판정 전에는 (c)를 만들지 않음 — 한 번에 하나만 바꾸기, GPU 큐가 이미 ~20 h.
- **(c) 여러 수 초안 검산 루프 — 설계 (10/29 08:00, 구현 안 함):** 루프 반복 r마다 (1) 현재 logit 상위 m=4개 (포트, 종류)를 각각 d=1~2수 놓은 loaded 판을 Woodbury로 만들고, (2) 그 판에서 피크를 다시 뽑아 (AAA, `modal_env`) 모드 토큰을 "초안 뒤 모드"로 교체, (3) 다음 반복이 그 토큰으로 logit을 고친다. 비용: 수마다 m·d번의 Woodbury + 피크 재추출이 CPU에서 → 지금 rollout 1수의 ~5~10배 (계획용 추정, 측정 아님) — 학습 예산 2 h 안에서 RL iter가 크게 줄어 같은 예산 비교가 불리해짐. 가장 가까운 선행: DACT/NeuRewriter (개선 루프), MCTS식 lookahead / 다른 점: 초안의 결과를 **모드 토큰**으로 다시 넣음. 결정: 비용 때문에 뒤로 미루고, 같은 방향의 싼 변형 (c')를 먼저 함.
- **(c') 모드 경유 add 점수 `--mode_head` (r2am, 10/29 08:00 GPU 큐 bm_manoharan 뒤):** add logit에 Σ_k α_ik·φ(m_k, c_t)를 더함 — α_ik = 포트 i가 모드 k를 겨누는 정도 (q(u_i)·k(m_k) + 포트↔모드 edge 편향, 모드 softmax), φ = 종류 t가 모드 k에 맞는 정도 (MLP[m, c, m⊙c]); 마지막 층 0 초기화라 학습 전에는 r2a와 같은 정책 (`scripts/test_mode_head.py`). 부모 r2a (2단 + fc 편향) 대비 이것 하나만 바꿈, 같은 예산. 가장 가까운 선행: US 6789241 (공진 피크 위치에 SRF 맞는 decap — 비학습 규칙), MatNet 2021 (행렬 편향 attention). / 다른 점: "어느 모드를 겨누나 × 그 모드에 맞는 종류"로 행동 점수를 **모드를 거쳐 분해**해 규칙 없이 학습 — 지금 add head [u, c, u⊙c]는 포트와 종류를 직접 짝지어 모드는 u 안에 섞여서만 들어감. 판정: r2a 대비 짝지은 CI, E0+ 대비 개수·성공률.

## 해보고 싶은 것 (오늘은 안 함)
- 선생님 LS를 cutting-plane 안에서 실행 (`make_teacher.py`가 지금은 g+p용으로 인증된 격자에서 LS 1회 → LS 해의 정확 검사 통과율 0.00~0.81, 선생님이 대부분 g+p로 떨어짐; `agent/logs/make_teacher.log`). eval의 `local_search`처럼 격자를 키우며 재실행하면 "LS 선생님"이 실제로 LS가 된다. 오늘 0·1·2단은 지금 선생님으로 같은 조건이라 비교는 유효; 바꾸면 사다리 전체 재학습 필요.
- 1단 modal 특징에 잔여 인덕턴스(rational fit의 s·e 항) 추가 — 데이터 분석에서 전이는 되지만 약함(AUC 0.60→0.65)이라 첫날은 생략.
- `--feat modal_only`(좌표 제거) ablation — "좌표가 해로운가"를 분리해서 보는 비교. 코드는 있음.
- 종류별 검사 대역(PCB 10 MHz~5 GHz, IP 200 MHz~20 GHz) — 지금은 셋 다 기본 대역. 대역을 바꾸면 PCB 10칸 공진이 대역 밖으로 나가므로 보류.

## 11/02 MoTT 개선 — 노벨티 2줄 메모 (QUEUE 11/01 15:10 할 일 3, 진단 H2 근거)
- **H2 수정 `--certify 2` (r2_mott seed 0 distill best.pt에서 RL 1.25 h만 다시, `r2c_rl` / `lo_pcb_r2c_rl`):** RL 보상의 성공 판정을 "인스턴스마다 최선 샘플 하나만 정확 검사" → "작업 격자에서 성공한 **서로 다른 샘플 모두** 정확 검사"로. 그룹 안 다른 샘플이 격자 틈(근소 실패)으로 성공 보상을 받는 것을 막는다. 가장 가까운 선행: 보상을 정확 시뮬레이터로 매기는 PI RL 전반 (Park 2022, Manoharan 2026 — 둘 다 고정 주파수 점에서 판정) / 다른 점: 학습 격자 + cutting-plane 정확 검사를 그룹 전체 보상에 쓰는 것 — 방법론 수정이지 기여 주장 아님 (사다리 공통 약점을 고치는 것).
- **(11/02 라운드 6 후보 → 라운드 7 구현, δ = log10 1.05, GPU 큐 `r2m`·`lo_pcb_r2m`) H2 싼 수정 `--feas_margin δ`:** RL 보상의 "성공"을 작업 격자 최악 log10(Z/Zt) ≤ 0 대신 ≤ −δ로 (예: δ = log10 1.05). 정확 검사 추가 없음 → r2c의 iter 1/3 문제 없음; 에피소드는 정책의 stop으로 끝나므로 (`sim.py` 16~17줄) 환경 파일 그대로, `train.py` 보상만. 근거: 평가 실패가 근소 실패 (최악 7~15% 초과, 라운드 1), r2c는 iter가 1/3로 줄어 비교가 흐려짐, 오늘 best-of-8 샘플링이 "격자 성공 중 개수 최소"를 고르자 성공이 떨어짐 (0.96 → 0.90). 가장 가까운 선행: 안전 여유를 둔 제약 RL 일반 (기여 아님, 사다리 공통 학습법 수정) / 다른 점: 없음 — 방법 수정. 설계: r2c와 같은 짝 (r2_mott seed 0 distill에서 RL 1.25 h만, vs `r2_mott_rl@last`, + lo_pcb 짝), 판정은 라운드 2 규칙. GPU가 비는 ~22:10 뒤, lo_*_r2a 판정 다음 순서로 검토.
- **`lo_<t>_r2a` (r2a를 한 종류 빼고 학습, 11/02 라운드 5):** 새 코드 없음 (기존 `--fc_bias` + 기존 leave-out arm). 가장 가까운 선행: MatNet 2021 (행렬 편향 attention), Manoharan 2026 (decap 토큰, 처음 보는 PCB·라이브러리 시험은 PCB 안에서만) / 다른 점: 종류 선택을 "모드 주파수 ↔ decap SRF" 궁합에 묶은 편향이 **처음 보는 판 기술 종류**로 전이되는지 — 라이브러리·대역이 바뀌어도 궁합 규칙 자체는 물리적으로 같다는 가설.

## 11/03 보너스: 학습에 없던 네 번째 종류 `chip` zero-shot (AGENT_PROMPT 보너스, DAILY 11/03 추천 5) — 결과 보기 전 고정
- 노벨티 2줄: 가장 가까운 선행: DevFormer 2023 (10×10 → 15×15·HBM zero-shot, 같은 기술 안), Manoharan 2026 (처음 보는 PCB·라이브러리, PCB 안) / 다른 점: 학습에 **없던 판 기술 종류** (판 물리·decap 라이브러리 둘 다 새것)로 학습 없이 옮겨 가는지, 사다리 단마다.
- 판 정의 (`decap_loop/techs.py` `CHIP`, `ZS_TECHS` — `TECHS`에 넣지 않아 학습·val·드라이버 glob에 안 들어감): 셀 100 µm, 유전체 0.2 µm (ε 3.9), 금속 0.5 µm (R_edge 69 mΩ, IP의 2배·PCB의 4배 — 가장 손실 큼), 전원 경로 2 mΩ + 30 pH, decap (1 nF, 3 pH, 50 mΩ) / (300 pF, 1.5 pH, 0.1 Ω) / (50 pF, 0.5 pH, 0.3 Ω) → SRF 2.9 / 7.5 / 31.8 GHz. 목표 범위 HARD (0.02, 0.2) — `scripts/probe_chip.py` (작업 격자 4문제): HARD에서 g+p 4/4, 개수 3~14, 종류 0·1을 섞어 씀 (EASY는 1~2개로 너무 쉬움). 대역·n_freq·T=3은 다른 종류와 같음.
- eval 판: `runs/pool_zs/eval_chip10_{0,1}.pkl` (10칸, 25문제씩, seed 309100·309101, cutting-plane — 다른 eval 판과 같은 `build_pools.py`, `--techs chip`). 기록: `runs/zs_chip/<seed|baselines>/eval/`.
- 평가 (학습 없음): 사다리 0~3단 `r0_coord_rl`·`r1_modal_rl`·`r2_mott_rl`·`r3_loop_rl` × seed 0·1·2 (세 종류 전체 학습 run 그대로), 최종 후보 `r2a_rl` seed 0·1·2 (같은 라운드 00:40에 추가, 판 생성 전), g+p, modal_rule, E0+ (종류 0·1·2 g+p). 4번째 CPU 칸 filler로만 (CPU 한도 4).
- **미리 적는 판정 (10/31 11:30 기준과 같은 꼴):** chip 50문제에서 (b) 2단 − 0단 성공률이 seed 3개 모두 > 0이면 "모달 표현이 처음 보는 종류 chip으로 0단보다 잘 옮겨 감"이라고 쓴다 (고찰용, 보너스 — 주장 아님). 하나라도 ≤ 0이면 그렇게 쓰지 않는다. 같이 보고: 각 단 vs g+p·modal_rule·E0+ 짝지은 bootstrap CI (seed 0), 단마다 seed 평균. 결과 보고 판 정의·목표 범위를 바꾸지 않는다.

## 11/03 MoTT 개선 계속 (QUEUE 11/03 11:40 KST) — T1: 규칙으로 보강한 선생님 `--teacher_dir data/teacher_rcp` (r2t·lo_pcb_r2t, 학습 결과 전에 고정)
- **진단 (라운드 27, 학습 결과 전):**
  - 기각: "M개 샘플이 모두 실패한 그룹에서 정규화 advantage가 개수만 줄이는 쪽으로 민다" — r2_mott_rl best.pt를 학습 풀 432문제의 저장된 작업 격자에서 M=8 샘플링하면 샘플 실패 0.2%, 모두 실패한 그룹 0개 (`scripts/diag_allfail.py`, `runs/analysis/diag_allfail_r2_mott_rl_best.json`). 학습 중 실패는 격자 위가 아니라 정확 검사에서 생김 (H2와 같음).
  - 지지: **지금 선생님이 학습 풀에서 약하다.** 정확 검사 (worst ≤ 1%)를 통과하는 선생님 해 (LS가 인증되면 LS, 아니면 g+p) 비율 ip 0.58 / pcb 0.88 / pkg 0.69 (`scripts/make_rule_teacher.py`, `data/teacher_rbest/summary.json`) → 증류 (선생님 있는 문제만 모방)와 RL 모방 항 (imit 1.0)이 어려운 문제의 30~40%에서 시연 없이 돈다 (QUEUE 관찰 2 "학습은 어려운 문제에서 짐"과 같은 방향). 같은 학습 풀에서 `modal_rule`을 평가와 같은 4-round cutting-plane으로 돌리면 정확 검사 통과가 ip 0.94~1.00, pcb 1.00 (pkg와 전체 요약은 `data/teacher_rcp/summary.json`, 라운드 27 PROGRESS에 숫자).
- **변경 하나:** 학습 문제마다 선생님 해 = {지금 선생님, cutting-plane 안 modal_rule 해} 중 정확 검사를 통과한 것 가운데 개수가 적은 것 (같으면 지금 선생님). `gp_count` (RL 개수 정규화 n_ref)는 그대로. 정책·보상·val·평가·환경 파일은 그대로. 짝: r2_mott seed 0 distill best.pt에서 RL 1.25 h만 (`r2t` vs `r2_mott_rl`·`r2_mott_rl@last`), `lo_pcb_r2_distill` best.pt에서 RL만 (`lo_pcb_r2t` vs `lo_pcb_r2_rl`), 같은 seed. 뺀 pcb run은 pkg·ip 학습 풀의 선생님만 읽음 (pcb 판 정보 없음; 규칙은 학습 없는 물리 규칙).
- 가장 가까운 선행: ExIt / DevFormer (오프라인 전문가 해 모방), US 6789241 (공진 피크 SRF 규칙 — 여기서는 정책 안이 아니라 선생님으로만) / 다른 점: 없음 — 학습법 수정 (선생님 범위), 기여로 쓰지 않음. 정책에는 규칙이 들어가지 않음 (추론 때 규칙 호출 없음).
- **미리 적는 판정 (라운드 7·14 규칙 그대로):** seed 0, 판마다 50문제 짝지은 bootstrap. (i) 같은 종류 pcb·pkg 중 하나 이상에서 r2t − r2_mott_rl 성공률 CI 하한 > 0이 day 1 기록과 `r2_mott_rl@last` **둘 다에서** 성립하거나, lo_pcb_r2t − lo_pcb_r2_rl (뺀 pcb) CI 하한 > 0, (ii) 그리고 다른 판에서 점추정이 −0.05 아래가 아니면 "이김" → seed 1·2 확장 + QUEUE 11/03 공정성: 같은 선생님으로 0단 좌표 (`r0t`, r0_coord distill에서 RL만)도 짝지어 봄. 두 기준이 갈리면 "평가 잡음 범위 — 판정 보류", 그 밖은 "효과 없음"/"해로움". 같이 보고 (판정 아님): 개수비, 그리고 규칙을 선생님으로 쓴 만큼 **r2t vs `modal_rule`** 짝지은 성공률·개수 (QUEUE 관찰 3 "규칙보다 나은가").
- 주의 (결과 전): RL만 다시 돌리므로 증류 단계의 선생님은 그대로 (지금 선생님) — 선생님 효과의 일부만 보는 비교.
- **T1' 같은 선생님으로 증류+RL 전체 (`r2tf`·`lo_pcb_r2tf`, 각 증류 0.75 h + RL 1.25 h, seed 0; r2t·lo_pcb_r2t 결과 전에 같이 큐에 넣음 — 선생님이 증류에서 가장 크게 쓰이고, 이기면 최종 방법은 이 전체 버전이라 r2t 결과와 무관하게 필요):** 짝 `r2tf` vs `r2_mott_rl` (같은 arm·같은 예산·같은 seed, 선생님만 다름), `lo_pcb_r2tf` vs `lo_pcb_r2_rl`. 판정 규칙은 T1과 같음 ((i)의 `@last` 갈래 포함). T1과 T1' 중 하나라도 "이김"이면 seed 1·2와 0단 짝은 **전체 버전 T1'** 으로 함 (T1만 이기고 T1'이 "효과 없음"이면 "판정 보류"로 쓰고 확장하지 않음 — 두 번 시험한 만큼 한쪽만 이긴 것은 약한 근거).
