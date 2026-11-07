# 사람 지시 (최우선)

## 2026-11-07 19:28 KST — 사람: M1 (`--mm_bias`)은 서버 2에서 사람이 돌린다. 서버 1은 건드리지 마라
- PROPOSALS "11/07 M1" 그대로. 코드 `decap_loop/model.py` (MoTT.mm_bias, 블록 mode_self에 bias), `train.py --mm_bias`, `scripts/smoke_mm_bias.py` (CPU 통과), `scripts/server2_m1.sh`. **서버 1에서는 M1 run을 돌리지 마라** — G1 확인 7 run과 "G1 보충 2" 평가가 먼저. 사람이 서버 2 기록(`runs/2026-11-07s2/{lo_pkg_r2pkfcm1,r2pkfcm1}_rl/eval`)을 push하면 pull해서 `scripts/m1_screen.py`로 거르기 판정만 RESULTS에 적어라 (확인 run 큐는 사람 지시 뒤).
- C1 판정 (서버 2 기록 push됨, 11/07 19:24): `r2pk_vsi` − `r2pk_ctl` pkg −0.02 [−0.16, +0.12], pcb +0.02, ip +0.02, 개수비 pcb 1.071 → **기준 미달**. RESULTS C1 표에 적어라. 참고: 대조군(+1 h RL만) pkg 0.86 vs `r2pk_rl` 0.78 (잡음 범위, 보고만).

## 2026-11-07 19:14 KST — 사람: G1 확인 run `r2pkfc_s1`·`r2pkfc_s2` 16·20칸 평가 추가 (CPU, 보고만)
- PROPOSALS "11/07 G1 보충 2" 그대로. `r2pkfc_s1_rl`의 학습이 끝나고 10칸 평가(판정용)가 끝나면 바로, 같은 방식으로 `r2pkfc_s2_rl`도.
- 명령 (run마다, **한 번에 한 파일씩**, 규모 곡선 평가와 같음): `eval_policy.py --pool runs/pool_v1 --sets <set> --policy <run>=runs/2026-10-29/<run>/best.pt --workers 1 --inst_chunk 5 --out runs/2026-10-29/<run>` — set 순서 `eval_pcb16_0`, `eval_pcb16_1`, `eval_pcb20_0`, `eval_pcb20_1`, `eval_ip16_0`, `eval_ip16_1`, `eval_ip20_0`, `eval_ip20_1`, `eval_pkg16_0`, `eval_pkg16_1`, `eval_pkg20_0`, `eval_pkg20_1` (pcb 먼저: 이것 때문에 하는 평가).
- 우선순위: G1 확인 run의 **10칸 평가(판정용)가 항상 먼저**. 다음 학습(GPU)은 그대로 진행 — 이 평가는 CPU 칸만 쓴다. 메모리: 기존 `mem_guard` 그대로, 45 GiB 넘으면 LS16 쪽을 멈춘다 (13:40 결정과 같음). 이 평가는 죽이지 마라.
- 끝나면 RESULTS에 PROPOSALS "G1 보충 2" 읽는 규칙대로 표 하나 (seed 0·1·2 묶음, 종류×크기, "큰 판 비용 있음/없음"). 판정은 하지 말고 사람에게 남긴다.

## 2026-11-07 14:30 KST — 사람: 두 번째 후보 C1' `--peak_gate` (서버 1, G1 탈락일 때 바로)
- PROPOSALS "11/07 C1'" 그대로. 코드: `decap_loop/pkgate_env.py` (서브클래스, sim.py 안 건드림), `train.py --peak_gate`, `scripts/eval_policy.py --env_feat`, `scripts/smoke_peak_gate.py` (CPU 통과). 네 할 일 순서:
  1. **지금 (CPU 4번째 칸, G1 판정 전이라도):** G-inf = `eval_policy.py --pool runs/pool_v1 --sets 'eval_*10_*' --policy r2pk_rl=runs/2026-10-29/r2pk_rl/best.pt --env_feat mott_pkgate --workers 1 --inst_chunk 5 --out runs/2026-10-29/r2pk_rl` (기록 `*__r2pk_rl@mott_pkgate.json`). r2pkfc 16·20칸 평가보다 **먼저**.
  2. GPU smoke: `scripts/smoke_peak_refine.py`의 3번(증류 30 + RL 3 iter)과 같은 것을 `--model mott --peak_gate`로 (`runs/analysis/smoke_peak_gate_gpu.ok`). 실패하면 원인을 PROGRESS에 적고 G-tr은 하지 마라.
  3. **G1 탈락이면** GPU: `r2pk_gate` = `"init": "runs/2026-10-29/r2pk_rl/best.pt"`, `"rl_hours": 1.0`, arm `--model mott --peak_gate`, seed 0 → side_slot 10칸 평가 (체크포인트 feat 그대로 = 게이트 포함). G1 통과면 G1 확인이 먼저, G-tr은 사람 지시 대기.
  4. 판정은 PROPOSALS C1' 규칙으로 (짝 `r2pk_ctl`은 서버 2 기록 `runs/2026-11-07s2/r2pk_ctl_rl/eval` — 사람이 push하면 pull). RESULTS에 C1·C1' 표 (G-inf, G-tr, C1 각각 vs 짝).
- 11/08 12:00 KST 뒤 새 학습 시작 금지.

## 2026-11-07 14:15 KST — 사람: 최종 RL 방식 C1 (PROPOSALS "11/07 C1 재개")
- **변경 (사람, 14:25 KST):** 대조군 `r2pk_ctl`과 C1 `r2pk_vsi`는 **사람이 서버 2에서 돌린다** (`scripts/server2_c1.sh`, 같은 A5000, 서버 1 `r2pk_rl/best.pt`에서 1.0 h). **서버 1에서는 이 둘을 돌리지 마라.** G1 탈락이면 서버 1 GPU는 비워 두고 사람 지시를 기다려라 (두 번째 후보 C1'이 올 수 있음). G1 통과면 G1 확인 7 run 그대로.
- (아래 원래 문장은 위 변경으로 대체) G1 판정이 "탈락"이면: ~~서버 1 GPU에 바로 `r2pk_ctl` → `r2pk_vsi` (둘 다 `"init": "runs/2026-10-29/r2pk_rl/best.pt"`, `"rl_hours": 1.0`, seed 0; arm은 PROPOSALS 그대로). 각 10칸 평가는 side_slot. 끝나면 거르기 판정 (PROPOSALS 규칙) → 통과면 확인 4 run (seed 1·2) 바로 큐.
- **G1이 "통과"면:** G1 확인 7 run이 먼저 (서버 1). C1은 서버 1에 넣지 말고 PROGRESS에 "C1 대기 (서버 2로)"라고만 적어라 — 사람이 서버 2에서 돌린다.
- `--self_repair` 경로는 11/07 smoke (`runs/analysis/smoke_self_repair.ok`) 그대로. 학습 조건·플래그는 바꾸지 않는다. 같은 1 h에 iter가 적은 것은 그대로 기록.
- 11/08 12:00 KST 뒤 새 학습 시작 금지.

## 2026-11-07 13:40 KST — 사람: r2pkfc 16·20칸 평가는 LS16과 동시에 허용 (바로 시작)
- 13:25 절의 r2pkfc 평가를 **LS16이 살아 있어도 지금 시작**한다 ("LS16 도는 동안 scale-eval 금지" 규칙의 예외, 이 평가만). 순서 `eval_pkg16_0` → `eval_pkg16_1` → `eval_pkg20_0` → `eval_pkg20_1`, **한 번에 하나씩**, `--workers 1 --inst_chunk 5`.
- 메모리: 기존 `mem_guard`를 그대로 쓴다. pod가 45 GiB를 넘으면 **LS16 쪽을** 멈추고 `.memskip`으로 기록한다 (사람 결정: r2pkfc 평가가 우선, LS16은 부록 기준선). r2pkfc 평가는 죽이지 마라.
- 우선순위: G1 `lo_pkg_r2pkfc` 10칸 평가가 시작되면 그것이 먼저 (판정용).

## 2026-11-07 13:25 KST — 사람: r2pkfc 16·20칸 평가 (보고만, CPU, 바로)
- PROPOSALS "11/07 G1 보충" 그대로. `r2pkfc_rl`을 `eval_pkg16_0`, `eval_pkg16_1`, `eval_pkg20_0`, `eval_pkg20_1` 순서로 평가 (규모 곡선 평가와 같은 명령, `--workers 1 --inst_chunk 5`, `--out runs/2026-10-29/r2pkfc_rl`). 그다음 시간이 되면 pcb·ip 16·20칸.
- 우선순위: `lo_pkg_r2pkfc` 10칸 평가 (G1 판정)가 먼저. 그 외 4번째 칸이 비는 대로. 판정에는 쓰지 않는다; RESULTS에는 "보고만"으로.

## 2026-11-07 10:50 KST — 사람: 범용화 후보 G1 (최우선, 바로 아래 10:45 "마지막 방법 탐색" 절을 대체)
- 10:45 절의 대조군·C1·C2·C3(루프)·C4는 **하지 않는다** (RL 절차 수정은 지금까지 모두 효과 없음). 대신 PROPOSALS **11/07 G1** 그대로 (수정 반영): 기본 MoTT 개선 후보 `r2pkfc`(P1+주파수 궁합 편향)와 그 pkg 빼고 학습 run.
- **사람이 `agent/day2_queue.json` train 끝에 2개를 넣었다:** `r2pkfc` → `lo_pkg_r2pkfc` (seed 0). 루프 후보 `r3pk`·`lo_pkg_r3pk`는 결과 전에 철회했다 (PROPOSALS G1 수정) — 루프는 더 학습하지 않는다. 드라이버가 순서대로 GPU에 올린다. 너는 큐 순서를 바꾸지 마라.
- 할 일:
  1. 첫 학습이 바로 실패하면 (새 플래그 조합: mott+peak_refine+fc_bias) 원인을 고치고 작은 smoke를 통과시킨 뒤 같은 이름으로 다시 큐. 모델 구조·학습 조건은 바꾸지 않는다.
  2. 네 run의 10칸 평가가 밀리지 않게 side_slot 대상에 넣어라 (드라이버 CPU 줄은 LS16이 막고 있음). LS16은 G1 뒤로 미룬다.
  3. 거르기 판정 (PROPOSALS G1 규칙 그대로) → 통과하면 (X = r2pkfc) **확인 7 run을 PROPOSALS 순서대로 바로 큐** (`lo_pkg_X_s1`, `lo_pkg_X_s2`, `X_s1`, `X_s2`, `lo_pcb_X`, `lo_pcb_X_s1`, `lo_pcb_X_s2`; arm은 seed만 다르게). 없으면 RESULTS ABLATIONS에 "G1 효과 없음"으로.
  4. 11/08 12:00 KST 뒤 새 학습 시작 금지. 판정 칸은 결과가 나오는 대로 RESULTS에.
- 서버 2는 사람이 비교 수치(S2)를 돌린다. 서버 2 일은 하지 않는다.

## 2026-11-07 10:45 KST — 사람: 마지막 방법 탐색 (서버 1 GPU 전부, 최우선; 위 10:20 절보다 우선)
**목표:** 정책 단독(repair 없음)에서 pkg 성공률을 올리는 학습 방법을 찾는다. pcb·ip는 떨어지지 않고, 개수는 늘지 않게. 최종 구조 r2pk (MoTT+P1)는 그대로 두고 학습 방법만 바꾼다. 서버 2는 사람이 비교 수치(PROPOSALS 11/07 S2)를 따로 돌린다 — 너는 서버 2 일을 하지 않는다.

**새 시도를 다시 연다 (11/06 12:00 UTC 마감을 사람이 연장).** 새 학습 시작 마감 **11/08 12:00 KST**, 실험 동결 11/10.

**순서:**
1. **CPU 정리:** 드라이버 CPU 줄의 남은 LS16 (`ls16r`·`ls16c`) 항목은 방법 탐색 뒤로 미룬다 (지금 도는 ip16_1은 끝까지 둔다). 10칸 평가가 밀리지 않게.
2. **대조군 먼저:** `r2pk_ctl` = r2pk seed 0 best.pt에서 기존 RL 그대로 같은 예산(1 h) 이어 학습. 모든 후보의 짝.
3. **후보 (seed 0, r2pk seed 0에서 1 h 이어 학습, 각각 결과 전에 PROPOSALS에 판정 규칙):** 순서는 근거로 네가 정한다.
   - C1 **검증된 해 따라 하기 (verified self-imitation):** 학습 판마다 S=8 샘플 → 정확 검사 → 실패 해는 `eval_policy.py`의 repair와 같은 방식(실패 주파수를 격자에 넣고 정책이 이어 풀기, 최대 4번)으로 고침 → 통과한 해 중 가장 적은 것을 판별 기록(archive)에 (더 적을 때만 교체) → 손실 = RL + λ·(archive 해 궤적 cross-entropy). 기존 expert iteration(teacher 교체)·imitation 코드를 재사용.
   - C2 **pkg 비중:** 학습 풀 샘플링에서 pkg 판 비중 2배 (다른 것은 그대로).
   - C3 **루프+P1 `r3pk`:** `--model loop --peak_refine`, 처음부터 (증류 0.75 h + RL 1.25 h), 짝 = `r2pk_rl` seed 0 (이미 있음).
   - C4 네 진단에서 나온 근거 있는 후보 (pkg 근소 실패 분석 등). 근거 없는 구조 변경은 하지 않는다.
4. **거르기 판정 (seed 0, 10칸 세 종류 50문제, 정책 단독):** 후보 − `r2pk_ctl` (C3은 − `r2pk_rl`): pkg **≥ +0.10**, pcb ≥ 0, ip ≥ −0.02, 둘 다 성공 개수비 ≤ 1.05. 통과한 것 중 pkg 차가 가장 큰 하나를 고른다. 11/08 09:00 KST까지 통과가 없으면 r2pk 그대로 확정.
5. **확인:** 고른 후보와 대조군을 seed 1·2로 (r2pk_s1·_s2에서 이어 학습). 3 seed 묶음 짝지은 CI: pkg CI 하한 > 0이고 다른 종류 해로움 없음 → 채택. 아니면 "seed 0에서만"으로 기록. 채택되면 16·20칸 평가도.
- 정직성: 결과를 보고 판정·seed·평가 세트를 바꾸지 않는다. 실패도 RESULTS ABLATIONS에 그대로.
- 참고 (Codex 기간, 근거 약함, 반복하지 마라): 작업 목표를 0.79 Zt로 보수화(pkg guard)는 pkg10 8문제 중 2개 더 풀었지만 decap 약 3.6배; 3단계 post-training(feasibility→robustness→sparsity)은 확장 규칙 미달 (seed 0 검증만).
- GPU는 학습만 한 번에 하나 (OOM). 평가는 side_slot.


## 2026-10-28 14:20 UTC — 파드 재시작 대응 (메모리)
- 서버 파드가 판 생성 중 두 번 재시작됨 (06:40, 13:45 UTC). Autostop은 꺼져 있음. 파드 메모리 한도는 **60 GiB** (`/sys/fs/cgroup/memory.max`), 메모리 초과(OOM)로 파드가 죽었다고 의심한다.
- 판 생성·정확 검사·선생님 해 생성의 **동시 프로세스를 최대 4개**로 줄여라 (`CERTIFY_WORKERS`, build_pools 병렬 수 등 전부).
- **20칸 평가 판은 맨 마지막에 프로세스 2개로** 만든다. 학습을 먼저 시작하는 것이 우선이다.
- 각 단계 시작·끝에 `cat /sys/fs/cgroup/memory.current`와 `grep oom_kill /sys/fs/cgroup/memory.events`를 로그에 남겨라. 메모리가 45 GiB를 넘으면 새 프로세스를 띄우지 마라.
- 학습은 체크포인트를 **20분마다** 저장하고, 재시작되면 마지막 체크포인트에서 이어서 학습하도록 하라 (재시작 시 처음부터 다시 하지 않게).
- 지금 우선순위: 학습용 판 → 선생님 해 → **0단·1단 학습 시작 (GPU)** → 나머지.

## 2026-10-29 03:10 UTC — 2일차 방향: 아키텍처로 해결하라
1일차 판단: 긍정적. 정책이 세 종류 10칸에서 g+p보다 개수 12~26% 적고 2~9배 빠르다. 그러나 ip 승리는 "100 pF 한 종류만 쓰는 greedy"로도 재현된다 (analyze_gp_one_type: 50/50, 개수 비율 0.98~1.00). 지금 표로는 "아키텍처가 이겼다"고 말할 수 없다.

**원칙: 개선은 모델 아키텍처·입력 표현·학습 방법론으로만 만든다.** 손으로 만든 규칙(종류 고정, 후처리 휴리스틱, 판 종류별 분기 코드)을 정책에 넣지 마라. 규칙은 기준선으로만 쓴다.

순서:
1. **강한 기준선 E0+ 추가:** "종류 인식 greedy" = 종류를 하나씩 고정한 greedy+prune을 모두 돌려 가장 적게 성공한 해를 고른다 (`analyze_gp_one_type.py` 확장, 정식 기준선으로 RESULTS에). 10칸 세 종류, 그다음 16칸. 앞으로 모든 비교는 이 기준선 대비 개수·성공률도 같이 보고.
2. **아키텍처 개선 (사다리 2·3단 위에서만):** 정책이 종류 선택을 스스로 배우게 하는 구조로 바꿔라. 후보 (하나씩, 한 번에 하나만 바꾸기):
   a. decap 종류를 **토큰**으로 (C·ESL·ESR·SRF 정규화), 모드↔decap cross-attention에 **주파수 궁합 편향** −|log f_k − log SRF_t| (reports/최종 모델 설계.md §3)
   b. 행동 점수를 (포트 × 종류) 쌍선형으로 — 포트와 종류를 함께 고른다
   c. 루프 반복이 "놓기 전 반공진 검산"을 하도록: 반복마다 상위 후보를 Woodbury로 평가해 모드 토큰 갱신 (설계 문서 §3 선택 모듈)
   각 변경은 같은 학습 예산(증류 0.75h + RL 1.25h)으로 학습하고 10칸 세 종류에서 평가. **종류 인식 greedy를 개수·성공률에서 넘는지**가 판정 기준.
3. **시드:** 최종 후보 1개 + 사다리 0~3단을 **시드 3개**로 다시 학습·평가 (10칸). 판별 짝지은 bootstrap 95% CI를 RESULTS에 자동 생성.
4. **16칸 평가** (g+p, 종류 인식 greedy, 사다리 rl, 최종 후보) → 그림 2.
5. **20칸:** 파일당 8문제로 줄여 만든다 (PROPOSALS (c) 채택). CPU 한도 4 유지.

정직성: 아키텍처 개선이 종류 인식 greedy를 못 넘으면 그대로 보고한다. 넘기려고 평가 세트·목표·검사·기준선을 바꾸지 마라.

## 2026-10-29 03:30 UTC — 선행 논문 기준선 추가 (위 1번 "종류 인식 greedy" 다음 순서)
목적: "우리 정책이 선행 연구 방식보다 낫다"를 같은 조건에서 보인다. 지금은 g+p 대비 숫자뿐이고, 0단(좌표)이 2·3단과 큰 차이가 없어 "좌표로도 된다"로 읽힐 위험이 있다.

A. **우리 판에서 재구현 (같은 학습 예산: 증류 0.75h + RL 1.25h, 같은 평가)**
   - B-Park: Park 2022 / DevFormer식 — 좌표 입력, 인코더-디코더 attention 포인터 정책 (Kool 2019 구조), REINFORCE + 롤아웃 기준선. 종류 선택은 우리 행동 공간 그대로. 새 파일 `baselines_ml/park_am.py`.
   - B-Manoharan: Manoharan 2026식 — 포트 순서를 loop inductance(주파수 한 점 f0) greedy 순위로 고정, 상태 = ΔZ(고정 주파수 격자) MLP + 포트 L 임베딩, decap 토큰(ESL·ESR·C) attention으로 **종류만** 선택 (빈칸 포함), 매 단계 Schur 갱신. 박사논문 Paper I 2장·3장 설명을 따른다. 새 파일 `baselines_ml/manoharan.py`.
   - 10칸 세 종류 → 16칸. RESULTS에 "선행 연구 재구현" 절로.
B. **외부 데이터 검증 (가능하면)**
   - DevFormer 공개 저장소(github.com/ai4co/devformer) 또는 RL4CO의 DPP 데이터(10×10, probe 1개/4개)를 받아 우리 환경 형식으로 변환. Manoharan 2026 4.2절의 방식: 우리 정책이 목표를 만족하는 최소 개수를 구하고, **같은 개수**를 DevFormer(공개 체크포인트가 있으면)에 주어 목표 만족 여부·임피던스 곡선을 비교.
   - 데이터·체크포인트를 못 받거나 형식이 안 맞으면 1시간 안에 포기하고 PROPOSALS.md에 이유를 적어라.
C. 재구현이 원 논문과 다를 수 있는 점(학습 예산, 행동 공간, 보상)을 RESULTS의 각 표 아래 한 줄씩 적어라. 보고서에는 "재구현 기준"으로 쓴다.

## 2026-10-29 03:45 UTC — 노벨티: 이미 있는 것을 우리 기여로 만들지 마라 (가벼운 안내)
아키텍처 개선(위 2번 a/b/c)을 고를 때, 아래는 **이미 있는 것**이라 그 자체로는 기여가 아니다. 써도 되지만 "새로운 점"으로 쓰지 말고, 이걸 **넘어서는** 쪽을 우선 시도하라.
- 매 수 Z(Woodbury/Schur) 갱신, Z 행렬 입력, decap 특징(ESL·ESR·C) 토큰 — Manoharan 2026
- 좌표 입력 attention 포인터 정책, 문맥 임베딩, 가중치 공유 디코더 — Park 2022, DevFormer 2023
- 공유 pole + 포트 residue 표현 자체 — Neural Modal Decomposition (대리모델)
- 공진 피크에 SRF 맞는 decap을 놓는 규칙 — US 6789241 (비학습)
- 행렬을 attention 편향으로 — MatNet 2021
- 가중치 공유 반복·입력 재주입 — Universal / Looped Transformer, Deep Thinking
- 그룹 기준선 REINFORCE, expert iteration — POMO, GRPO, ExIt

우리가 노리는 새로움 (이 방향으로 기울여라):
- **공진(모드) 정보를 decap 배치 정책의 상태로 쓰는 것**, 특히 모드↔decap 주파수 궁합으로 **종류 선택을 정책이 스스로 배우는 것**
- 판 종류(PCB·PKG·IP)를 섞어 학습한 정책 하나와 그 **체계적 평가** (기존은 PCB 안 또는 시연 2건)
- greedy의 근시안적 종류 선택이 반공진을 만든다는 분석과, 그것을 규칙 없이 아키텍처로 피하는 것

새 아이디어를 시작하기 전에 PROPOSALS.md에 2줄만 적어라: "가장 가까운 선행: … / 다른 점: …". 확신이 없으면 그냥 진행하고, 판정은 사람이 나중에 한다. 이 확인 때문에 실험을 멈추지는 마라.

## 2026-10-29 03:15 UTC — 수정 (위 2일차 지시보다 우선)
근거: PROGRESS 라운드 17~19, RESULTS.md, `decap_loop/model.py`·`loop_model.py` 코드 확인.

1. **GPU가 비어 있다 (라운드 17 기록).** 16칸 평가·기준선은 CPU만 쓴다. **아키텍처 실험 학습을 지금 GPU에서 바로 시작하라.** 학습의 정확 검사 worker는 CPU 한도 4 안에서 맞춰라 (`CERTIFY_WORKERS` 2 또는 1). 16칸 평가는 1 worker로 계속.
2. **후보 b는 삭제.** (포트 × 종류) 쌍 점수는 이미 2·3단에 있다 (`add_head([u, c, u⊙c])`). 다시 만들지 마라.
3. **후보 a를 구체화 (1순위, 한 줄 변경):** `MoTTBlock.dec_from_mode`는 지금 **편향 없이** attention만 한다. 여기에 공진↔decap 주파수 궁합 편향 `b_kt = −|log f_k − log SRF_t| / τ` (τ는 head마다 학습)를 넣어라. 모드 토큰의 f_k와 decap 토큰의 SRF는 이미 입력에 있다. 이것만 바꾼 3단(`r3a`)과 2단(`r2a`)을 같은 예산으로 학습 → 10칸 세 종류 평가. 판정: 같은 단의 편향 없는 버전 대비, 그리고 종류 인식 greedy 대비 개수·성공률.
4. **종류 인식 greedy (정식 기준선 E0+)**: 분석용 `analyze_gp_one_type`을 정식으로 올려라. 문제마다 종류 0·1·2만 쓰는 g+p를 각각 돌리고 **성공한 것 중 개수 최소**를 고른다 (셋 다 실패면 실패). 10칸 세 종류 → RESULTS 요약표에 한 행. 이미 돌린 종류별 기록은 재사용.
   - 이미 보인 것: ip는 종류 0만으로 50/50, pcb는 종류 0만으로 0.04 → **종류마다 맞는 소자가 다르다.** 이 대비(규칙은 판 종류마다 다시 골라야 하고, 정책은 하나로 고른다)를 RESULTS에 한 줄로 적어라. 보고서 핵심 문장 후보다.
5. 순서: (GPU) r3a 학습 → r2a 학습 / (CPU) E0+ 10칸 → 16칸 평가 계속 → 선행 논문 재구현은 r3a 결과 뒤.

## 2026-10-29 08:40 UTC — 루프 종료 예정 (최우선)
사람이 다른 연구에 집중하기로 했다. **이 에이전트 루프는 2026-10-29 15:00 UTC (KST 24:00)에 끝낸다.**

1. 지금부터 15:00 UTC 전까지는 하던 일을 그대로 한다. 다만 새 아키텍처 변경이나 새 코드는 시작하지 마라. 결과 기록과 정리를 우선한다.
2. **15:00 UTC 이후 처음 시작하는 라운드에서** (`date -u`로 확인) 다음을 하고 끝낸다.
   - PROGRESS.md 맨 위에 "루프 종료 요약"을 쓴다: 지금까지 확정된 결과 (run 경로와 함께), 진행 중인 작업과 끝나는 예상 시각, 아직 판정 안 된 것.
   - RESULTS.md를 마지막으로 재생성한다.
   - **드라이버(GPU·CPU 큐)와 지금 도는 학습·평가는 멈추지 마라.** 끝까지 돌게 둔다. 사람이 나중에 결과를 회수한다. 큐 순서는 지금 그대로 둔다.
   - `agent/STOP` 파일을 만든다 (loop.sh가 다음 라운드에서 종료한다).
   - commit, push.

## 2026-10-29 14:20 UTC — 종료 전 마지막 실험: 한 종류 빼고 학습 (위 08:40보다 우선)
사람 결정: 보고서 주장을 "처음 보는 기술 종류로의 범용화"로 올린다. 같은 종류 안에서는 coord도 잘해서 표현 차이가 안 보였다. 종류를 넘어갈 때 coord는 무너지고 modal은 버티는지가 질문이다 (근거: research_notes/토큰 데이터 분석/결과.md, coord·loop-L은 종류 사이에서 뒤집히고 modal 참여비·1-step 효과는 전이됨).

08:40의 "새 코드·실험 금지"는 이 실험에 한해 풀린다. 새 모델·새 학습 코드는 여전히 금지 (기존 `train.py` 옵션만 쓴다).

1. **큐 추가.** `agent/day2_queue.json`의 `train` 맨 앞에 아래 9개를 이 순서로 넣는다 (지금 도는 `r0_coord_s1`은 그대로 끝까지 돌게 두고, 그다음부터 이것들이 먼저 돈다). 기존 seed 항목은 뒤로 밀린다.
   - 학습 풀 이름 규칙 `train_{tech}{size}_{k}.pkl` (`scripts/build_pools.py`), tech = pcb·pkg·ip.
   - **뺀 종류의 val도 빼야 한다** (best.pt 고르기에 뺀 종류가 새면 안 된다).
   ```
   {"name": "lo_ip_r0",  "arm": "--feat coord  --pool_glob train_pcb*.pkl,train_pkg*.pkl --val_pools val_pcb10_0,val_pkg10_0", "seed": 0}
   {"name": "lo_ip_r1",  "arm": "--feat modal  --pool_glob train_pcb*.pkl,train_pkg*.pkl --val_pools val_pcb10_0,val_pkg10_0", "seed": 0}
   {"name": "lo_ip_r2",  "arm": "--model mott  --pool_glob train_pcb*.pkl,train_pkg*.pkl --val_pools val_pcb10_0,val_pkg10_0", "seed": 0}
   {"name": "lo_pcb_r0", "arm": "--feat coord  --pool_glob train_pkg*.pkl,train_ip*.pkl --val_pools val_pkg10_0,val_ip10_0", "seed": 0}
   {"name": "lo_pcb_r1", "arm": "--feat modal  --pool_glob train_pkg*.pkl,train_ip*.pkl --val_pools val_pkg10_0,val_ip10_0", "seed": 0}
   {"name": "lo_pcb_r2", "arm": "--model mott  --pool_glob train_pkg*.pkl,train_ip*.pkl --val_pools val_pkg10_0,val_ip10_0", "seed": 0}
   {"name": "lo_pkg_r0", "arm": "--feat coord  --pool_glob train_pcb*.pkl,train_ip*.pkl --val_pools val_pcb10_0,val_ip10_0", "seed": 0}
   {"name": "lo_pkg_r1", "arm": "--feat modal  --pool_glob train_pcb*.pkl,train_ip*.pkl --val_pools val_pcb10_0,val_ip10_0", "seed": 0}
   {"name": "lo_pkg_r2", "arm": "--model mott  --pool_glob train_pcb*.pkl,train_ip*.pkl --val_pools val_pcb10_0,val_ip10_0", "seed": 0}
   ```
   arm의 다른 옵션은 사다리 0·1·2단과 같아야 한다 (`r0_coord`, `r1_modal`, `r2_mott`의 arm 그대로 + 풀·val 두 옵션만 추가). 같지 않으면 맞춰라.
2. **확인 (GPU 쓰지 말 것, 지금 r0_coord_s1이 돌고 있다).** 각 glob이 실제로 고르는 학습 풀 파일 목록을 python으로 찍어 PROGRESS에 적는다 (빼려는 종류 파일이 0개, 남는 두 종류가 각각 기존과 같은 개수인지). `train.py`가 val_pools 2개를 받는지 코드로 확인. 문제가 있으면 큐에 넣지 말고 PROGRESS에 이유를 적는다.
3. **평가는 드라이버가 자동으로 한다** (끝난 rung마다 10칸 eval 풀 전부). 판정에 쓰는 것은 **뺀 종류의 eval 50문제**다.
4. **결과 표.** `scripts/collect_results.py`에 절 하나 추가: "한 종류 빼고 학습 (처음 보는 기술 종류)". 뺀 종류마다 lo_*_r0 / r1 / r2와 같은 단 전체 학습 run (r0_coord_rl, r1_modal_rl, r2_mott_rl), 종류 인식 greedy, modal_rule의 성공률 / 개수, 그리고 짝지은 bootstrap CI: (a) r1 vs r0, (b) r2 vs r0, (c) 각 lo run vs 같은 단 전체 학습 run (얼마나 떨어지나). 없는 결과는 "–".
5. **자동 보고.** 루프가 끝난 뒤에도 결과가 올라오게, 백그라운드 감시 스크립트 하나 (`scripts/report_watch.py`, `scripts/bg.py`나 nohup로 띄움): 30분마다 새 eval json이 생겼으면 `collect_results.py` → `git add RESULTS.md runs/**/eval/*.json` 등 결과 파일만 → commit "results: auto (lo_*)" → push. 9개 rung 평가가 모두 끝나면 마지막으로 한 번 더 push하고 스스로 끝난다. 실패해도 드라이버는 건드리지 않는다. 띄운 뒤 첫 사이클이 도는지 (push 안 해도 되는 상태면 "변화 없음" 로그) 확인한다.
6. **그다음 08:40 종료 절차를 한다** (루프 종료 요약에 이 실험의 예상 일정 포함: 학습 rung당 ~2 h, 9개 ~18 h + 평가). `agent/STOP`, commit, push.

**늦어도 2026-10-29 16:30 UTC 전에 STOP까지 끝낸다.** 1~2번이 막히면 5·6번만 하고 끝낸다 (그때 감시 스크립트는 기존 run 결과만 올린다).

## 2026-10-31 11:30 UTC — 한 종류 빼고 학습: seed 1·2 (사람이 직접 넣음, 에이전트 루프는 계속 STOP)
seed 0 결과 (RESULTS "한 종류 빼고 학습"): 뺀 종류에서 2단 MoTT는 전체 학습만큼 성공하고 (ip +0.00, pcb +0.04), 0단 좌표는 무너진다 (ip −0.44, pcb −0.36). pkg는 0단도 안 떨어져 판정에 못 쓴다. seed 하나라 운과 구분이 안 돼서 같은 9개를 seed 1·2로 다시 돈다.
- `agent/day2_queue.json` `train` 끝에 `lo_<t>_<r>_s1`, `lo_<t>_<r>_s2` 18개 (arm은 seed 0 항목 그대로). 순서: ip·pcb s1 → ip·pcb s2 → pkg s1·s2 (판정에 쓰는 종류 먼저). GPU ~36 h + 평가.
- `scripts/collect_results.py`: "seed별" 표 추가 — 뺀 학습 / 전체 학습 성공률 seed 0·1·2와 평균, 뺀 − 전체 (같은 seed), (a)(b) 단 차이의 seed별 값과 0보다 큰 seed 수. 기존 CI 표는 seed 0 run끼리.
- `scripts/report_watch.py`: 27개 rung (seed 0~2)을 기다림, 최대 96 h. git은 인증을 묻지 않고 300 s에서 끊음 (10/30 재시작 뒤 push가 보이지 않는 인증 대기로 32 h 멈췄다).
- 계속 써지는 로그 5개는 git에서 뺌 (`.gitignore`): `pull --autostash`가 추적 중인 로그를 새 파일로 바꿔, 드라이버가 지워진 파일에 쓰고 있었다.
판정 기준 (미리 적음): 뺀 종류 ip·pcb에서 (b) 2단 − 0단이 seed 3개 모두 0보다 크고, (c) 2단 뺀 − 전체의 seed 평균이 −0.05 이상이면 "처음 보는 종류로 범용화"를 seed 3개로 주장한다. 하나라도 어긋나면 그 종류는 주장에서 뺀다.

## 2026-10-31 11:40 UTC (11:55 고침) — 루프 재개 때 첫 실험: 루프 트랜스포머 vs 기본 트랜스포머
seed 1·2 큐가 끝나면 `scripts/resume_agent.py`가 `agent/STOP`을 지우고 루프를 자동으로 켠다 (~11/2 03:00 UTC). 루프의 첫 GPU run 하나를 이 비교에 쓰고, 그다음부터 평소처럼 탐색한다.
- 비교 대상: 기본 = `r2_mott` (MoTT 블록 6개), 루프 = `r3_loop` (같은 블록 1개를 R회 반복, 학습 R~U{2..8}, 평가 R=8). 같은 예산.
- 전체 종류 학습은 seed 0·1·2가 이미 있다 → RESULTS "루프 트랜스포머" ABLATIONS 표 (seed 0)와 seed별 표 (`loop_vs_plain_table`)를 먼저 읽는다. seed 평균으로는 pcb (0.83 → 0.91)·pkg (0.73 → 0.83)에서 루프가 높다.
- 그 run은 범용화 검증 `lo_pcb_r3` (pcb 빼고 학습, 루프, seed 0, arm `--model loop --pool_glob train_pkg*.pkl,train_ip*.pkl --val_pools val_pkg10_0,val_ip10_0`)를 기본으로 하되, 더 나은 설계가 있으면 이유를 적고 바꿔도 된다. pcb인 이유: ip는 기본이 이미 1.00.
- 판정 (미리 적음): 같은 종류 학습에서 루프 − 기본 성공률이 seed 3개 중 2개 이상 양수이고 seed 평균이 +0.05 이상인 판, 그리고 lo_pcb_r3 vs lo_pcb_r2의 CI가 0 밖이면 "루프가 낫다"고 쓴다. 범용화 쪽은 seed 1개라 보조 근거.

## 2026-11-01 15:10 UTC — 루프 재개 뒤 주 작업: MoTT 개선 + 실제 논문 기준선 + 꼼꼼한 검증 (위 11:40 첫 실험 다음)
위 11:40의 첫 실험(루프 vs 기본)은 그대로 먼저 한다. 그다음부터 이 절이 주 작업이다. 특정 구조를 강제하지 않는다: 진단을 먼저 하고, 그 근거로 무엇을 바꿀지 네가 정한다.

### 지금 결과에 대한 사람의 비판적 판독 (RESULTS "seed별" 표, ip·pcb는 seed 3개 완료, pkg 진행 중)
- 강한 것: 0단 좌표는 처음 보는 종류에서 무너진다 (뺀 − 전체 seed 평균 ip −0.34, pcb −0.32). pcb에서 1단·2단은 0단보다 seed 3개 모두 높다 (+0.20, +0.26).
- 10/31 11:30에 미리 적은 판정 기준을 그대로 적용하면: **pcb는 통과**, **ip는 실패**다 ((b) 2단 − 0단이 seed 1에서 −0.32, (c) 2단 뺀 − 전체 평균 −0.11 < −0.05). 기준을 바꾸지 말고 이대로 기록하라. pkg는 결과가 나오면 같은 기준으로 판정한다.
- 약점 1 (가장 큼): 학습 없는 `modal_rule`과 비교가 애매하다. 10칸 seed 0에서 pcb는 규칙 1.00 / 개수비 0.21 vs MoTT 0.88 / 0.21, pkg는 규칙 0.88 vs MoTT 0.72. MoTT가 확실히 이기는 곳은 ip의 개수(1.02 vs 1.56)뿐이다. "물리 규칙으로 충분한데 왜 학습하나"에 답해야 한다.
- 약점 2: MoTT의 pcb 성능이 이상하다. pcb를 보고 학습한 2단 0.83 < pcb를 빼고 학습한 2단 0.90 < pcb를 보고 학습한 0단 0.96. "처음 봐도 손해 없음"이 일반화가 아니라 pcb 학습에서 원래 약해서일 수 있다.
- 약점 3: MoTT가 불안정한 seed가 있다 (lo_ip_r2_s1 0.66). 종류가 3개뿐이고 문제 수가 종류당 50개라 0.07 같은 차이는 구분되지 않는다.

### 할 일 (순서대로)
1. **학습 vs 규칙 (진단, CPU):** 처음 보는 종류의 같은 eval 문제에서 lo_*_r2 (seed 0·1·2) vs `modal_rule` vs 종류 인식 greedy를 짝지어 비교한다 (성공률, 개수, 문제당 시간, 짝지은 bootstrap CI). 학습이 이기는 문제의 공통점(판 크기·공진 개수·목표 임피던스 여유 등)을 찾아 RESULTS에 절로 남긴다. 이기는 곳이 없으면 그것도 그대로 적는다.
2. **MoTT pcb 이상 진단 (분석 먼저, GPU는 그다음):** r2_mott_rl seed 0·1·2의 pcb 학습 곡선, best.pt 선택(val 풀 구성), 종류별 학습 풀 크기 균형, pcb에서 실패하는 문제 유형을 본다. 원인 가설을 PROGRESS에 미리 적고(무엇이 맞으면 무엇을 고친다) 확인한다.
3. **MoTT 개선:** 1·2의 근거로 고친다. 한 번에 한 가지만 바꾸고, 같은 예산·같은 arm 나머지로 학습해 바꾸기 전과 짝지어 비교한다 (먼저 seed 0, 이기면 seed 1·2). 판정 대상은 같은 종류 학습과 한 종류 빼고 학습 둘 다다. 새 모델 코드는 `decap_loop/` 환경 파일을 건드리지 않는 범위에서 허용한다.
4. **실제 논문 기준선 (꼼꼼하게):** 이 분야 논문들이 실제로 쓰는 기준선과 지표를 찾아 정리한다 (EXPLORE_PROMPT의 웹 검색 허용 범위에서). 최소: Park et al. TMTT 2022 (`bp_park_rl`), Manoharan et al. TSIPI 2026 (`bm_manoharan_rl`), DevFormer (ICML 2023), 그리고 PI 논문에서 흔한 GA·LS 휴리스틱, US 6789241식 규칙(`modal_rule`). 각 재구현이 원 논문 설정과 어디가 다른지 표로 남긴다. **`bm_manoharan_rl`의 성공률 0.00은 심사에서 바로 지적될 숫자다**: 재구현이 맞는지(포트 순서 규칙, 상태, 보상, 학습 예산) 먼저 점검하고, 설정 차이 때문이면 그 이유를 쓰고, 버그면 고친 뒤 다시 돌린다.
5. **검증 규칙 (모든 주장에):** seed 3개 평균 + 같은 문제 짝지은 bootstrap CI, 같은 학습 예산, 처음 보는 종류는 val에서도 빠졌는지 확인, 표는 `collect_results.py`로만. 미리 적은 판정 기준은 결과를 보고 바꾸지 않는다.
6. **나중 (범용화 주장이 서면):** "어떤 관찰로 왜 이렇게 설계했는가"를 RESULTS 한 절로 정리한다 (예: Park은 매 스텝 Z를 다시 계산하지 않음 → 정확한 1스텝 갱신 / 학습 없는 토큰 분석에서 좌표·루프 인덕턴스는 종류 사이에서 뒤집히고 모달 참여비·1스텝 효과는 전이됨 → 모달 토큰 / 사다리 결과). 주장이 서지 않은 부분은 쓰지 않는다.

일정: AGENT_PROMPT의 실험 동결(11/8)을 지킨다. 11/8 이후에는 평가·그림·표만.

## 2026-11-01 23:45 UTC — 위 15:10 절 보강: pkg 반전 + 기준선을 먼저 확실하게
위 15:10 절의 순서와 규칙은 그대로다. 아래 두 가지를 더한다.

### pkg 반전 (사람 판독, pkg seed 0·1 기준, s2는 아직)
pkg를 빼고 학습하면 pkg에서: 0단 좌표 0.78·0.84·0.78 (평균 0.80, 안 떨어짐), 1단 modal 0.74·0.74, 2단 MoTT 0.80·0.56 (평균 0.68). pkg를 보고 학습한 run들의 pkg 성공률은 대략 0단 0.76, 1단 0.76, 2단 0.83이다. 즉 pkg에서는 좌표가 버티고 MoTT가 떨어진다 — ip·pcb와 반대 방향이다.
- s2가 나오면 10/31 11:30 판정 기준을 그대로 적용해 pkg를 판정한다. 기준을 바꾸지 않는다.
- 15:10 할 일 2(진단)에 pkg를 넣는다: (a) pkg가 학습한 두 종류와 얼마나 닮았는지 (판 크기·공진 분포·포트 수 등 입력 분포 거리) — 좌표가 안 떨어지는 이유가 "쉬운 전이"인지, (b) lo_pkg_r2_s1 0.56이 lo_ip_r2_s1 0.66·pcb 이상과 같은 MoTT 불안정인지 (학습 곡선, best.pt 시점, 실패 문제 유형). 가설을 먼저 PROGRESS에 적는다.
- 결과와 무관하게 RESULTS에는 "범용화는 ip·pcb에서 성립/불성립, pkg에서는 ~"처럼 종류별로 정직하게 쓴다.

### 기준선이 먼저, 그다음 MoTT 강화
- 15:10 할 일 4(실제 논문 기준선)를 MoTT 개선(할 일 3)의 GPU run보다 **먼저** 끝낸다. MoTT가 무엇을 이겨야 하는지 기준이 먼저 서야 개선 판정이 의미 있다.
- 각 기준선마다: 원 논문의 설정(입력·상태·보상·학습 예산·평가 지표)을 논문 원문에서 확인 → 우리 재구현과의 차이 표 → 재구현이 원 논문 결과의 경향을 원 논문과 비슷한 설정에서 재현하는지 작은 sanity check (가능한 것만). 재현이 안 되면 이유를 적고, 그 기준선과의 비교에는 caveat를 붙인다.
- 그 기준선들과 같은 문제·같은 예산에서 MoTT 강화를 한다 (할 일 3 규칙: 한 번에 하나, 짝지은 비교, 같은 종류 학습 + 한 종류 빼고 학습 둘 다). 강화 방향은 진단(할 일 1·2 + 위 pkg 진단)의 근거로 네가 정한다. 근거 없는 구조 변경은 하지 않는다.

## 2026-11-02 21:30 KST — STUCK "원 논문 설정 확인"에 대한 사람 조치
`research_notes/선행 기준선 원문 설정 (Park 2022, Manoharan 2026).md`를 넣었다. Park는 arXiv v2 전문에서 쪽수와 함께 확인했다. Manoharan은 사람이 10/19에 박사논문 Paper I 전문을 읽은 메모이고, 보상 식·건너뜀 행동·종료 조건·n_f·네트워크 크기는 **미확인**이다. STUCK을 풀고 이 파일로 할 일 4(재구현 차이 표)를 마쳐라. 웹 검색·alphaXiv는 다시 시도하지 마라.
- 표에는 특히 (1) Park 원 문제가 "개수 고정 + 임피던스 감소"라 우리 "목표 만족 + 개수 최소"와 다르다는 것, (2) Manoharan 원 학습 예산(RTX 4090 ~24 h, 5만 에피소드) 대비 우리 2 h라는 것을 적는다. B-Manoharan pcb 0.00은 "같은 2 h 예산에서"라는 조건을 붙여서만 쓴다.
- Manoharan 미확인 항목은 우리 재구현의 선택으로 적고 "원문 미확인"을 붙인다. 원문 PDF는 사람이 나중에 넣을 수 있다.

## 2026-11-03 11:40 KST — 사람 결정: 아키텍처 확정을 미루고 MoTT 개선을 계속한다 (11/6까지)
r2g 판정 뒤 "근거 있는 후보가 생길 때까지 GPU를 비워 둔다"(라운드 24 결정)는 바꾼다. GPU를 비우지 말고 **새 진단 → 근거 → 한 가지 수정 → 짝 비교**를 계속 돌려라. 규칙은 15:10·23:45 절 그대로다: 결과 전에 판정 규칙을 PROPOSALS에 적고, 한 번에 하나만 바꾸고, 같은 distill에서 RL만 다시 돌려 같은 종류(r2*)와 뺀 pcb(lo_pcb_r2*)를 짝지어 본다.

### 일정
- 새 개선 시도는 **11/6 12:00 UTC까지만 시작**한다. 이긴 수정의 seed 1·2와 뺀 ip·pkg 확인을 11/8 동결 전에 끝내야 한다. 시도 하나는 GPU로 약 2.5 h(같은 종류 + 뺀 pcb)다.
- 11/6 12:00 UTC까지 이긴 것이 없으면 r2_mott를 최종 구조로 확정한다. 그때 "시도한 수정과 효과 없음"을 RESULTS ABLATIONS에 정직하게 남긴다. 이것도 보고서 내용이다.

### 지금 결과에서 보이는 근거 (후보를 고를 때 참고, 강제 아님)
네가 더 나은 근거를 찾으면 그쪽을 먼저 한다. 아래는 사람이 결과를 보고 적은 관찰이다.
1. **실패가 근소 실패다** (라운드 1: pcb MoTT 실패 해의 최악 log10(Z/Zt) 중앙값 0.03~0.06). 학습 중 보상을 고치는 시도(r2c·r2m·r2g)는 다 효과가 없었다. 남은 질문은 "끝에서 고치기"다. 예를 들어 정확 검사가 실패하면 정책이나 규칙으로 하나를 더 놓고 다시 검사한다. 다만 모든 방법(g+p·규칙·0단)에 같은 후처리를 줘야 공정하다. 개수가 늘어나는 것도 같이 보고한다.
2. **학습이 지는 곳은 어려운 문제에 몰려 있다** (라운드 1: pcb·pkg 맨판 위반·g+p 개수 상위 1/3에서 −0.44~−0.63). 학습 풀에서 어려운 문제의 비율이나 가중치, 또는 문제 난이도 분포를 진단하고 그 근거로 고친다.
3. **MoTT 해가 규칙 해와 닮았다** (H3: 종류 비율 0.09/0.17/0.74 ≈ modal_rule). 규칙이 pcb에서 1.00이다. 규칙을 선생님이나 시작 해로 쓰는 쪽(증류 선생님 교체, 또는 규칙 해에서 출발해 정책이 add/remove로 다듬기)이 근거가 있다. 이 경우 "규칙보다 나은가"를 같은 문제로 짝지어 본다.
4. **16칸 pkg에서 모든 학습 단이 g+p보다 낮다** (0.46~0.64 vs 0.82). 크기 일반화 쪽 실패다. 원인 진단(학습 판 크기 분포, 16칸에서의 실패 유형)이 먼저다.

### 공정성 (이긴 수정이 나오면)
학습법을 바꾼 수정(구조가 아니라)이 이기면, 같은 수정을 **0단 좌표에도** 적용해 짝지어 본다. 그래야 그 이득이 "MoTT라서"인지 "학습법이 좋아져서"인지 구분된다. 0단도 같이 오르면 보고서에는 학습법 개선으로 쓴다.

## 2026-11-04 18:20 KST — 사람: 보고서용 규모 곡선 (CPU 줄만, GPU 일정 그대로)
목적: 보고서 마지막 그림 "보드 크기 10 → 16 → 20에서 개수·성공률·시간". 모드 토큰을 여러 개 둔 근거("큰 판에서 대역 안 공진이 여러 개", 10/22 토큰 분석 한계 1)를 결과로 보이려면 16·20칸 비교가 필요하다. 새 학습은 없다. 평가만 한다.
1. **순서 바꾸기 (드라이버 CPU 줄 `agent/day2_queue.json`):** 남은 `base20 … modal_rule` 6개를 남은 `eval20 eval_ip20_1 …` 정책 평가 4개보다 **앞으로** 옮긴다. 그 뒤 `eval20 eval_ip20_1 …`, 그 뒤 `base20 … greedy+prune` 6개. g+p 20칸은 16칸 rss(19~34 GB)로 보아 45 GiB 규칙을 넘을 수 있으니 하나씩, 넘으면 그 파일은 건너뛰고 RESULTS에 "메모리로 생략"이라 적는다.
2. **P1 정책을 16·20칸에 추가 평가:** `r2pk_rl`과 `r2pk_s1_rl`을 `eval_{pcb,pkg,ip}16_{0,1}`과 `eval_{pcb,pkg,ip}20_{0,1}`에서 (명령은 기존 eval16/eval20과 같게, `--workers 1 --inst_chunk 5`). 위 1번 다음 순서. P2가 규칙대로 "이김"이면 `r2dd_rl`도 같은 목록으로 그 뒤에 (판정은 PROPOSALS P2 규칙 그대로, 이 항목 때문에 바꾸지 않는다).
3. **RESULTS 자동 표 하나 추가 ("규모 곡선"):** 종류별로 행 = 방법 (greedy+prune, gp_type_aware, modal_rule, r0_coord_rl, r1_modal_rl, r2_mott_rl, r2pk_rl, r2pk_s1_rl), 열 = 10 / 16 / 20칸의 (성공률, 둘 다 성공한 문제의 개수비 vs modal_rule, 문제당 초, 문제 수). 20칸은 문제 수가 8~16이라 CI를 같이 적고 "작은 표본"을 표에 표시한다.
4. 기록만 (판정 없음): 20칸에서 r1_modal(토큰 없음) vs r2_mott(모드 토큰) 짝 차이 — 토큰 구조 자체의 효과를 보는 유일한 비교다.
- 하지 않는 것: 새 크기(24칸 이상), 새 학습, 풀 확대. 11/8 동결 전에 끝난다는 보장이 없다.
- 정직성: 20칸에서 MoTT가 규칙에 지면 그대로 적는다. 평가 세트·목표·검사를 바꾸지 마라.

## 2026-11-04 18:15 KST — 사람: 루프 반복 수 R 스윕을 16·20칸에서 (CPU 줄, 평가만)
- 10칸 R2/4/8/16 스윕은 "늘려도 좋아지지 않음" (10칸은 대역 안 공진이 판당 1개라 반복할 게 없음). 큰 판에서 반복이 쓰이는지 본다.
- **제출:** `r3_loop_rl`을 `eval_{pcb,pkg,ip}16_{0,1}`과 `eval_{pcb,pkg,ip}20_{0,1}`에서 `@R2`·`@R4`·`@R16` (R8은 기존 평가). 명령은 10칸 `@R` 평가와 같게. 18:20 항목(규모 곡선) 뒤 순서.
- RESULTS 자동 표 "루프 반복 수 vs 판 크기": 행 = 판 크기·종류, 열 = R 2/4/8/16의 (성공률, 개수, 문제당 초). 기록만.
- 미리 적는 읽기 (결과 전): 16·20칸에서 R이 클수록 성공률이 오르거나 개수가 줄면 "큰 판에서 반복이 쓰인다"; 평평하면 "반복의 이득은 관찰되지 않았다"로 적는다.

## 2026-11-05 08:05 KST — 사람: 보고서 그림 "같은 판, 공진 규칙 vs MoTT" (CPU 4번째 칸, 그림만)
목적: 보고서 정성 예시 그림. `reports/fig_ip_gp_fail_ip10_0_13.png`(g+p vs r3_loop)와 같은 꼴로, 같은 판에서 `modal_rule`과 `r2pk_rl`(MoTT + P1)의 배치와 |Z|/Zt 곡선을 위아래로 놓는다. 새 학습·새 평가 없음. 이미 있는 평가 기록 JSON(occ)과 `scripts/fig_board.py`만 쓴다.
1. **ip16** (`eval_ip16_0` 또는 `_1`): 그림 1장 → `reports/fig_rule_vs_mott_ip16.png`.
2. **pcb20** (`eval_pcb20_0` 또는 `_1`): 그림 1장 → `reports/fig_rule_vs_mott_pcb20.png`. 기록이 없거나 둘 다 성공한 문제가 없으면 건너뛰고 이유만 적는다.
- **판 고르는 규칙 (결과 보기 전에 고정, 체리피킹 방지):** 두 방법 모두 정확 검사를 통과한 문제 중에서, 개수 차 (modal_rule − r2pk_rl)가 그 파일의 중앙값에 가장 가까운 문제. 동률이면 가장 작은 인덱스. 고른 인덱스와 두 방법의 개수를 그림 제목과 PROGRESS에 적는다.
- 그림 제목은 영어 (서버에 한글 글꼴 없음), 예: "ip-like 16x16 #k, same instance: modal_rule N decaps vs MoTT+P1 M decaps".
- 순서: 지금 CPU 4번째 칸이 R 스윕 20칸을 돌리고 있으면 그 파일이 끝난 뒤. GPU 일정(P3)은 건드리지 않는다. 동시 4개 한도 그대로.
- 끝나면 PNG를 커밋한다.

## 2026-11-05 19:50 KST — 사람: 서버 2에서 루프 확인 (L1·L2, 서버 1 일정 그대로)
- 사람이 서버 2에서 `scripts/server2_loop.sh`를 직접 돌린다 (PROPOSALS 11/05 L1·L2, 결과 전에 고정). **너는 이 run을 큐에 넣거나 다시 돌리지 마라.** 서버 1의 F1 → B1 순서도 바꾸지 마라.
- push되는 파일: `runs/2026-10-29/{r2_mott,r3_loop}_s{1,2}_rl/eval/eval_*{16,20}_*.json`, `runs/2026-11-05s2/…`. `collect_results`가 이 파일을 읽어 기존 표에 자동으로 넣는 것은 괜찮다. 문제당 초는 서버 2 CPU 값이므로 "규모 곡선"·시간 표에서 이 파일의 시간은 쓰지 마라 (가능하면 표에 "서버 2" 표시).
- L1·L2 판정 줄은 사람이 쓴다.

## 2026-11-05 20:15 KST — 사용자: 서버 2 무인 실행기로 L1·L2 및 보충 L3·L4 수행
- 서버 2 전용 `scripts/server2_once.sh` / `scripts/server2_once.py`를 준비한다. 사용자 실행 전이며 서버 1 드라이버에는 등록하지 않는다.
- 이 실행기는 L1·L2를 포함하므로 기존 `server2_loop.sh cpu/gpu`와 중복 실행하지 않는다. 서버 1 F1→B1은 그대로 둔다.
- 같은 GPU를 사용한다는 사용자 확인에 따라 보정 학습은 제외한다. 추가 등록 L3/L4는 PROPOSALS의 20:15 항목 그대로이며, 기존 L1/L2 판정 변경이나 결과에 따른 시드·크기 선택이 아니다.
- 기록이 올라오면 전체 종류·시드의 결과를 읽는다. 서버 2 시간은 서버 1 시간 비교표에 합치지 않는다. main MoTT 보고서에 루프는 확장/ablation으로 설명하며, 결과보다 강한 기전 주장을 하지 않는다.

## 2026-11-05 — 사용자 변경: 서버 1 파일 복사 없는 새 서버 단독 FRESH
- `server2_fresh.sh`로 clone 후 풀 생성·MoTT+P1/루프+P1 각 3 seed 학습·평가를 새 서버에서만 실행한다. 상세는 PROPOSALS FRESH.
- 서버 1 에이전트는 이 작업을 자신의 GPU/CPU 큐에 추가하지 않는다. F1→B1 및 진행 중인 작업은 유지한다.
- FRESH는 앞서 제안한 전송형 서버 2 실행을 대체한다. 서버 1 checkpoint를 평가하는 L1의 완료로 집계하지 않으며, 기존 L1/L2 최종 모델 판정을 변경하지 않는다.


## 2026-11-07 10:20 KST — 사람: Codex 관리 기간 되돌림 (최우선)
- 11/06 19:06 ~ 11/07 09:14 KST에 Codex가 넣은 커밋 21개 (관리자 교체 `agent/PROVIDER`·`agent/codex_loop.sh`, `train.py` hook, `day2_queue.json` peakmask 항목, delta/pkg_guard/curriculum/post-training/reserve/report_validation 연구 스크립트·문서·테스트 57개)를 **되돌렸다.** 코드 상태 = 11/06 12:37 KST (라운드 70, f687b21). 원본은 브랜치 `archive/codex-20261107`에 보존.
- 이 기간 Codex 연구 (delta memory, pkg guard, curriculum, post-training, reserve, report validation, peak mask 진단)는 **보고서 근거로 쓰지 않는다.** 결과 파일이 서버에 남아 있어도 RESULTS에 넣지 마라. 사람이 따로 판단한다.
- 서버의 Codex 프로세스는 사람이 `scripts/stop_codex.sh`로 멈춘다. 에이전트 관리자는 Claude (`agent/loop.sh`) 그대로.
- 11/05 서버 2 실행기 (`server2_once`·`server2_fresh`, PROPOSALS L3·L4·FRESH)는 코드만 남겨 둔다. 결과가 올라와도 판정은 사람이 한다.
- 다음 할 일 (에이전트): `train.py`가 바뀌었으므로 진행 중인 학습이 있으면 PROGRESS에 어떤 코드로 돌았는지 (git hash) 적어라. 서버 1 드라이버 큐 (F1·B1 남은 것)는 그대로 이어 간다. 새 시도 시작은 하지 않는다 (11/06 12:00 UTC 마감 지남).
