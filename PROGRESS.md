STUCK: 원 논문 설정 확인 (QUEUE 11/01 15:10 할 일 4·23:45) — 무인 라운드에서 WebSearch·alphaXiv가 두 라운드 연속 거부됨 (11/02 04:xx, 06:10). 사람: Park 2022 (arXiv 2203.15722)·Manoharan TSIPI 2026·DevFormer 원문 PDF를 `research_notes/`에 넣거나 허용 목록에 alphaXiv 추가. 그동안 재구현 차이는 저장소 자료 기준 + RL만 재학습으로 진행.

# PROGRESS (갱신 2026-11-02 06:15 UTC, 재개 라운드 2)

## 재개 라운드 2 (2026-11-02 05:58~ UTC)
- 상태: GPU `lo_pcb_r3` RL it 250 (끝 ~06:25), CPU 줄 `e0plus16 eval_pkg16_1 type1` (05:50~, 4~7 h), pod 15 GiB / 60, oom_kill 0. H1 마지막 확인 (`agent/logs/diag_last2.log`: pkg·ip last.pt) 끝 — 라운드 1 판정 그대로 (H1 pcb에서 지지).
- 결정: **`scripts/side_eval_wait.py` (4번째 CPU 칸, bg)** — lo_pcb_r3 → bm_manoharan_rlonly → bp_park_rlonly의 RL이 끝나는 대로 드라이버와 같은 10칸 평가 명령을 바로 돌림. 이유: 드라이버 CPU 줄은 16칸 E0+ (4~7 h)에 묶여 있어 루프 vs 기본 판정이 몇 시간 밀림. 칸 수: 학습 검사 2 + 드라이버 1 + 이것 1 = 4 (side16_gp 없음).
- **MoTT 개선 준비 (QUEUE 할 일 3, 기준선 두 개 뒤 GPU 순서 그대로):**
  - **H2 수정 `--certify 2`** (`train.py`): RL 보상에서 작업 격자로 성공한 **서로 다른 샘플 모두** 정확 검사 (지금은 인스턴스마다 최선 하나만). smoke (`scripts/smoke_certall.py`, CPU 2 iter, pkg8 B2 M4) 통과 — 샘플 8개 중 6개가 정확 검사 실패 (H2와 맞음) → `runs/analysis/smoke_certall.ok`.
  - 짝 설계: 같은 distill best.pt에서 **RL 1.25 h만** 다시 (`run_rung.py --init_from`, 큐 키 `"init"`) → `r2c` vs `r2_mott_rl` (seed 0, 같은 종류), `lo_pcb_r2c` vs `lo_pcb_r2_rl` (pcb 빼고). 결정: distill 재사용 — distill은 certify와 무관하므로 바뀐 것은 RL 보상 하나뿐이고 init 잡음이 빠짐. RESULTS ABLATIONS 행 추가 (결과 오면 자동).
  - **H1 수정 (그다음):** val을 종류당 16 → 48문제 (같은 생성 규칙, 새 파일 `val_*10_1`·`_2`, seed +5001·+5002, eval·train과 겹치지 않음 확인). `build_pools.py --val_k`. CPU 큐 `val pools k3 (H1)`는 지금 도는 E0+ 뒤; GPU `r2v`·`lo_pcb_r2v` (같은 distill에서 RL만)는 그 파일이 생기면 시작. val 시간이 3배라 같은 1.25 h 안 RL iter가 줄어듦 — 그것까지 포함한 비교.
  - 미리 적는 판정 (결과 보기 전): 각 수정은 seed 0에서 짝지은 성공률 차의 CI가 0 밖이고 (같은 종류 pcb·pkg 중 하나 이상 또는 뺀 pcb) 다른 판에서 −0.05 아래로 떨어지지 않으면 "이김" → seed 1·2로 확장. 아니면 "효과 없음"으로 적고 확장하지 않음.
- 드라이버 재시작 (06:03, 새 "init" 키) → `ATTACH train lo_pcb_r3`. 결정: 재시작 직후 CPU 줄이 맨 앞에 넣은 val 풀 작업을 E0+와 **동시에** 시작 → 06:25부터 CPU 5개가 될 뻔해서 즉시 멈추고 (rc −15) 이름을 바꿔 E0+ 뒤로 옮김 (메모리 메모 갱신). 잃은 것 40 s.
- GPU 큐 순서: lo_pcb_r3 (~06:25) → bm_manoharan_rlonly (~08:25) → bp_park_rlonly (~10:25) → r2c (~11:45) → lo_pcb_r2c (~13:05) → r2v·lo_pcb_r2v (val 파일 뒤).
- 다음 라운드: lo_pcb_r3 평가 (side_eval_wait) → 루프 vs 기본 판정 (QUEUE 10/31 11:40 기준) + RESULTS; bm_manoharan_rlonly 초기 val (빈칸 쏠림이 RL만으로 사라지는지).

## 재개 라운드 1 (2026-11-02 04:21~ UTC) — 진단 가설 (확인 전에 적음, QUEUE 11/01 15:10 할 일 2 + 23:45)
MoTT pcb 이상 (pcb를 보고 학습한 2단 0.83 < 뺀 2단 0.90 < 0단 0.96)과 pkg·ip seed 불안정 (lo_pkg_r2_s1 0.56, lo_ip_r2_s1 0.66):
- **H1 best.pt 선택 잡음:** RL val은 100 iter마다 3번뿐 (판마다 16문제), score = 평균(개수비 + 2·fail). → 맞으면 (같은 run의 last.pt가 pcb에서 best.pt와 크게 다름) val을 늘리거나 val 두 파일을 쓰는 쪽을 고친다. 확인: `runs/analysis/ckpt_last/` (r2_mott seed 0·1·2 last.pt를 pcb10 두 파일에서 평가, 백그라운드).
- **H2 학습 격자 부족 → 보상 오지정 (격자 틈 이용):** RL 보상은 작업 격자에서 계산되고 정확 검사는 인스턴스마다 가장 좋은 샘플 하나만; 격자는 풀마다 처음 F의 1.5배에서 막힘. → 맞으면 (실패가 목표 바로 위의 근소 실패이고, 학습 중 cert-bad가 pcb·pkg에서 높고, 격자 상한에 일찍 닿음) 학습 쪽 검사를 고친다 (예: 격자 상한 완화 또는 M개 샘플 전부 검사) — 학습법 변경이므로 r2_mott 하나만 바꿔 짝지어 비교.
- **H3 pcb 선생님 (g+p) 모방이 MoTT를 g+p식 배치로 끌어당김:** → 맞으면 pcb 실패 해의 종류 비율이 g+p (종류 0 많음)와 닮음. 그러면 선생님을 고친다.
- **H4 (pkg 반전) pkg가 학습한 두 종류와 가까워 좌표가 쉽게 전이:** → 맞으면 pkg 입력 분포 (판 모양·맨판 최악·위반 대역)가 pcb·ip 중 하나와 겹침. 판정에는 영향 없음 (기준대로 판정), 해석에만 씀.

### 이 라운드에서 한 것 (05:10 UTC)
- 상태: GPU 비어 있었음 (큐 다 끝남), pod 4 GiB / 60, oom_kill 0. `ensure_running.py` 정상.
- **GPU: `lo_pcb_r3` 시작 (04:23, QUEUE 10/31 11:40 첫 실험 기본안 그대로)** — arm `--model loop --pool_glob train_pkg*.pkl,train_ip*.pkl --val_pools val_pkg10_0,val_ip10_0`, seed 0, distill 0.75 h + RL 1.25 h (끝 ~06:25). 결정: 기본안 유지 — 더 나은 설계의 근거 없음. 평가·표 (`loop_vs_plain_table`)는 자동.
- 결정: CPU 큐에서 `eval16 eval_pkg16_1 r0~r3` 4개를 `e0plus16 eval_pkg16_1 type1·2` 앞으로 — 서로 독립, 16칸 E0+ 1개가 4~7 h라 lo_pcb_r3 평가 (06:25 뒤)가 그 뒤에 막히지 않게 (메모 day2-driver-cpu-lane).
- **할 일 1 (학습 vs 규칙):** `scripts/problem_features.py` (맨판 특징 → `runs/analysis/problem_features.json`), `collect_results.py` 새 절 "학습 vs 규칙". 결과: 처음 보는 종류에서 MoTT가 modal_rule을 이기는 곳은 **ip뿐** (lo_ip_r2 seed 0·2: 성공 +0.06 [+0.00, +0.14], 개수 0.714 [0.653, 0.775]·0.746; seed 1: −0.28). pcb: −0.06~−0.16, 개수 1.10~1.13배; pkg: −0.08·−0.32 (s2 평가 중). 학습이 지는 것은 어려운 문제에 몰림 (pcb·pkg 맨판 위반·g+p 개수 상위 1/3 점수 −0.44~−0.63, 하위 1/3 −0.02~−0.24). 정책 시간 0.3~0.9 s/문제 < 규칙 0.5~2.2 s ≪ E0+ 9~25 s.
- **할 일 2 (진단) 결과:**
  - H3 **기각**: pcb에서 MoTT (seed 0·1·2) 해의 종류 비율 0.09/0.17/0.74 ≈ modal_rule 0.09/0.18/0.73, g+p (0.52/0.11/0.37)와 다름; 실패 해도 8~10개로 적음.
  - H2 **지지**: 평가 실패는 근소 실패 (pcb MoTT 실패 해 최악 log10(Z/Zt) 중앙값 0.03~0.06, 모든 방법이 cutting-plane 4라운드 끝까지 감). RL 로그: 격자 상한에 닿은 iter 114~276 / ~350, 최선 샘플의 정확 검사 실패 pkg 7.5~13 / 16, pcb 2.7~7.2, ip 0~3 (RESULTS "진단: 학습 격자"). 모든 학습 방법의 pkg 성공률이 낮은 것 (0.56~0.90)과 맞음. MoTT만의 문제는 아님 (사다리 공통) — 그래서 고치면 사다리 전체에 해당하는 학습법 수정.
  - H1 **지지 (pcb)**: last.pt − best.pt 성공률 (pcb 10, 50문제): seed 0 +0.04 [+0.00, +0.10], seed 1 **+0.14 [+0.02, +0.26]**, seed 2 **+0.12 [+0.04, +0.22]** (RESULTS "진단: last.pt vs best.pt"). pkg·ip에서는 last.pt가 잃지 않음 (ip +0.00 세 seed, pkg +0.00 / +0.02 / +0.12 CI 0 포함) → **판 사이 교환이 아니라 선택 잡음**: best.pt (seed 1 iter 100, seed 2 iter 200)가 val 16문제 × 3번 잡음으로 일찍 골림. 비교 기준은 best.pt를 같은 평가 코드로 다시 평가한 `@best` (06:20 끝).
  - **평가 재현성 (새 발견):** 같은 설정 재평가는 비트 단위 같음 (s1·s2 pcb·pkg 4파일 0/50 다름). 그러나 seed 0 r2_mott_rl은 best = last (같은 가중치)인데 day 1 기록 (25문제 한 번에 rollout)과 지금 (`--inst_chunk 5`)이 pcb 3/50 문제에서 다른 해 (하나는 최악 0.24 → −0.003, 근소 아님) → 배치 크기 부동소수 차이로 동점 근처 argmax가 바뀌어 궤적이 갈림. day 1 seed 0 사다리 기록 vs 나중 run 비교의 ±0.04~0.06은 이 잡음 안일 수 있음 (RESULTS 진단 절에 한 줄). 결정: 기존 기록은 그대로 둠 (평가 바꾸지 않음); 판정에 쓰는 짝은 가능하면 같은 평가 코드끼리.
  - H4 **부분 지지**: pkg의 decap 라이브러리 (1 nF/220 pF/47 pF)와 맨판 위반 수준이 pcb (10 nF/1 nF/100 pF)와 ip (100/20/5 pF) **사이** (W1/sd 맨판 최악 pcb-pkg 0.56 vs ip-pcb 1.44). pkg를 빼는 것은 내삽, ip·pcb는 외삽 → 좌표가 pkg에서 안 무너지는 것과 맞음. 다만 pkg의 최악 주파수는 셋 중 가장 높음 (외삽). `scripts/type_distance.py`.
- **범용화 판정 (10/31 11:30 기준 그대로):** pcb 통과, ip 실패, **pkg 실패** ((b) seed 1 −0.28 → "seed 3개 모두 > 0" 불충족, s2와 무관).
- **할 일 4 (기준선) — B-Manoharan 0.00 점검:** 버그 아님. 증류가 g+p 선생님 해를 고정 포트 순서로 다시 걸은 궤적을 따라 하는데, 선생님이 어디에 놓을지는 Manoharan 상태 (전역 ΔZ 8점 + 현재 포트 L, 좌표 없음)로 정해지지 않음 → 다수 클래스 "빈칸"으로 수렴 (라운드 51 진단과 같음), RL이 imit 1.0으로 그 궤적에 계속 묶임. 원 논문은 RL만 (모방 없음). 결정: **`bm_manoharan_rlonly` (RL만 2.0 h = 같은 총 예산, `--imit 0`, 무작위 초기화)** 를 GPU 큐에 lo_pcb_r3 뒤로; Park도 원 논문이 RL만이라 같은 규칙으로 **`bp_park_rlonly`**. 기존 bm/bp 결과는 지우지 않고 둘 다 보고. 코드: `run_rung.py --skip_distill` (init 없음), 드라이버 항목 `"rl_only": true`, `scripts/smoke_rlonly.py` (CPU 3 iter 두 모델 통과 → `runs/analysis/smoke_rlonly.ok`, 항목의 "after").
  - 결정: 드라이버 재시작 (04:52) — 새 "rl_only" 키를 읽게. `scripts/driver_dryrun.py`로 두 줄 모두 ATTACH 확인 후 kill → `ensure_running` → `ATTACH train lo_pcb_r3`, `ATTACH eval10 lo_pkg_r2_s2_rl` 확인.
  - 원 논문 설정 확인 (QUEUE 23:45: 원문에서 입력·상태·보상·예산·지표)은 **못 함**: 무인 라운드의 허용 목록에 WebSearch·alphaXiv가 없음 (둘 다 거부), 저장소에도 TSIPI 2026 전문 없음 (`reports/최종 모델 설계.md` "전문 미확인"). 사람에게: 원문 PDF를 `research_notes/`에 넣어 주거나 허용 목록에 웹 검색 추가 필요 (PROPOSALS에 적음).
- 다음 라운드: diag_last2 (pkg·ip last.pt) 결과로 H1 판정; lo_pcb_r3 끝 (~06:25) → 평가 → 루프 vs 기본 판정; bm_manoharan_rlonly 시작 확인 (GPU, 초기 val). MoTT 개선 GPU run은 두 기준선 뒤 (QUEUE 23:45 순서) — 1순위 후보 H2 수정 (설계는 다음 라운드에 PROPOSALS에 2줄 + 코드).

## 루프 종료 요약 (2026-10-29 14:25 UTC, QUEUE 08:40 + 14:20)
에이전트 루프는 이 라운드로 끝남 (`agent/STOP`). **드라이버 (`scripts/pipeline_day2.py`, GPU·CPU 큐)·side16_gp·지금 도는 학습/평가는 멈추지 않음.** 결과는 `scripts/report_watch.py`가 30분마다 RESULTS.md + 새 eval json을 commit·push ("results: auto (lo_*)", 로그 `agent/logs/report_watch.log`). 파드가 재시작되면 `.venv/bin/python scripts/ensure_running.py` 한 번이면 드라이버·side16_gp·report_watch가 다시 뜨고 학습은 resume.pt에서 이어짐.

**확정된 결과** (10×10, 판마다 50문제, 정확 검사; 숫자는 `RESULTS.md` 자동 표, seed 0):
- 사다리 0~3단 distill·rl (`runs/2026-10-28/r{0_coord,1_modal,2_mott,3_loop}_{distill,rl}/eval/`): 모든 rl 단이 ip 성공 0.96~1.00, 개수÷g+p ip 0.74~1.05. 모드 토큰이 있는 2·3단만 ip 개수가 E0+ 수준 (÷E0+ 1.01~1.02); 0·1단은 1.23~1.45. pcb·pkg는 단 사이 차가 CI 안 (seed 1개).
- 증류 → RL (RESULTS "증류 → RL" 표): 0·1단은 RL이 개수를 크게 줄이고 pkg 성공률을 떨어뜨림; 2·3단은 개수 그대로 성공률 올림.
- 종류 인식 greedy E0+ (`runs/2026-10-29/baselines_e0plus/eval/`): ip 1.00, pcb 0.66, pkg 0.82. 우리 최선 r2a_rl (`runs/2026-10-29/r2a_rl/eval/`) vs E0+: ip 성공 같음, 개수 비 1.03 (못 넘음) · pcb 성공 +0.30, 개수 0.20 · pkg 성공 같은 수준, 개수 0.58.
- 비학습 모달 규칙 modal_rule (`runs/2026-10-28/baselines/eval/*modal_rule*`): pcb 1.00 / pkg 0.88 / ip 0.94. r2a_rl은 pcb·pkg에서 규칙과 CI 안 같음, ip에서만 개수 0.72 [0.66, 0.78]. → "학습이 규칙보다 낫다"는 지금 ip 10칸 개수에서만 말할 수 있음.
- 아키텍처 변경 (같은 예산, RESULTS ablation 표): r2a (2단 + 주파수 궁합 편향) vs r2_mott_rl — 성공률 pcb +0.08 [+0.02, +0.16], pkg +0.12 [+0.02, +0.24] (CI 0 밖, seed 1개), 개수 같음 (비 0.99~1.01); r3a vs r3_loop_rl — 모든 CI가 0·1 포함; r2am (`--mode_head`) vs r2a — 모든 CI가 0·1 포함, 효과 없음. 최종 후보 = r2a (seed 1·2는 GPU 큐에 있음).
- r3 루프 반복 수 R2/4/8/16 (`runs/2026-10-28/r3_loop_rl/eval/*@R*`): 평가 때 반복을 늘려도 좋아지지 않음.
- 선행 연구 재구현: B-Park (`runs/2026-10-29/bp_park_rl/eval/`) ip 0.72 / pcb 0.46 / pkg 0.16 — 모두 E0+·우리 사다리보다 낮음. B-Manoharan 학습 끝 (`runs/2026-10-29/bm_manoharan_rl/`, val greedy 거의 0개 성공 — 라운드 51 진단: 빈칸 클래스 쏠림), **10칸 평가는 아직 없음**.

**진행 중 / 예상 일정** (UTC):
- GPU: r0_coord_s1 RL (14:12~, 끝 ~15:30) → **한 종류 빼고 학습 9개** `lo_{ip,pcb,pkg}_r{0,1,2}` (rung당 ~2 h → ~10/30 09:30 끝) → seed 항목 r1_modal_s1 … r2a_s2 (10개, ~20 h 더).
- CPU 줄 (1 worker): e0plus16 pcb16_0 종류 0 retry2 (11:38~, 끝 시각 모름) → 끝난 학습의 10칸 평가 (bm_manoharan, r0_coord_s1, lo_* 순, rung당 ~12~60분) → 남은 16칸 E0+·사다리 평가 → 10칸 local search.
- side16_gp: 16칸 g+p (pcb16·pkg16 일부 끝, 나머지 진행).
- report_watch: 9개 lo_* 평가가 모두 끝나 push하면 스스로 끝남 (최대 72 h).

**아직 판정 안 된 것**: (1) **처음 보는 판 종류로의 범용화** — RESULTS "한 종류 빼고 학습" 절 (뺀 종류의 eval 50문제, (a) 1단 vs 0단, (b) 2단 vs 0단, (c) 뺀 학습 vs 전체 학습 CI), lo_* 평가 뒤 자동 생성. (2) B-Manoharan 10칸 표. (3) 사다리·r2a seed 1·2 → seed 3개 CI. (4) 16칸 표 (그림 2). (5) 20칸 평가 판은 아직 안 만듦.

## 라운드 59 (14:13~ UTC)
- 상태 정상: `ensure_running.py` → day2 드라이버·side16_gp 살아 있음, pod 23.4 GiB / 60, oom_kill 0.
- **r0_coord_s1 distill 끝** (~14:12, `runs/2026-10-29/r0_coord_s1/driver.log`): 2215 iter / 0.77 h, best it 1800 score 2.122, gpu peak 12.1 GB, OOM 0 (다른 run과 같은 예산). RL 시작 (`runs/2026-10-29/r0_coord_s1_rl/log.txt`, init = distill best it 1800, seed 1, 1.25 h).
- **QUEUE 14:20 (한 종류 빼고 학습) 처리** (push 전 pull에서 받음):
  1. 큐: `agent/day2_queue.json` `train`에 lo_ip_r0 … lo_pkg_r2 9개를 지시 순서대로 **r0_coord_s1 바로 뒤**에 넣음. 결정: 리스트 맨 앞 대신 r0_coord_s1 뒤 — 그 앞 항목은 모두 끝나서 도는 순서는 같고, 드라이버만 재시작될 때 r0_coord_s1이 살아 있는데 lo_ip_r0을 GPU에 같이 띄우는 일 (train.py 2개 겹침)을 막음. arm = 사다리 arm 그대로 (`--feat coord` / `--feat modal` / `--model mott`; day1 `r*_rl/config.json`과 r0_coord_s1 config 대조: seed·init·out 말고 다른 것 없음) + `--pool_glob`·`--val_pools`. COMMON (`run_rung.py`)에 풀·val 옵션 없음 → 덮어쓰기 없음.
  2. 확인 (CPU만): glob이 고르는 학습 풀 (`runs/pool_v1`, 전체 27 = 종류마다 9):
     - lo_ip `train_pcb*.pkl,train_pkg*.pkl` → 18 (pcb 9, pkg 9, **ip 0**): train_pcb{8,10,12}_{0,1,2}, train_pkg{8,10,12}_{0,1,2}
     - lo_pcb `train_pkg*.pkl,train_ip*.pkl` → 18 (pkg 9, ip 9, **pcb 0**)
     - lo_pkg `train_pcb*.pkl,train_ip*.pkl` → 18 (pcb 9, ip 9, **pkg 0**)
     - `train.py`: `--pool_glob`은 쉼표로 나눠 glob 합집합 (246줄), `--val_pools`는 쉼표 목록을 모두 읽음 (253줄), best 점수는 val 수와 무관한 평균 (473줄) → val 2개 OK. 9개 arm을 `train.parse()`로 파싱해 feat/model/pool_glob/val_pools 확인 (r2의 feat=phys는 `main()` 208줄에서 mott로 바뀜 — day1 r2_mott와 같음). 선생님 파일은 풀 이름별이라 그대로.
  3. 평가: 드라이버 `eval_jobs`가 끝난 rung마다 `eval_*10_*` 6파일 전부 (뺀 종류 포함) — 코드 변경 없음.
  4. `scripts/collect_results.py`에 `leave_out_table` 절 "한 종류 빼고 학습 (처음 보는 기술 종류)": 뺀 종류마다 lo_*_r0/r1/r2_rl · r0_coord_rl · r1_modal_rl · r2_mott_rl · gp_type_aware · modal_rule의 성공률/개수/÷E0+, CI (a) r1 vs r0 (b) r2 vs r0 (c) lo vs 전체 학습; 없으면 "–". lo 결과가 하나도 없으면 절을 숨김. 가짜 레코드 (r0/r1 ip 기록 이름만 바꿈)로 표 모양 확인.
  5. `scripts/report_watch.py` (bg.py로 띄움, pid 221167, `agent/logs/report_watch.log`): 새/바뀐 `runs/**/eval/*.json` (runs/는 gitignore라 `git add -f`) → collect_results → 그 파일+RESULTS.md만 commit → `pull --rebase --autostash` → push. 첫 사이클 14:17: 기존 eval json 114개 + RESULTS.md commit `77dfc1b`, push 됨. `ensure_running.py` JOBS에도 넣음 (파드 재시작 대비).
- QUEUE 14:20의 "새 모델·새 학습 코드 금지" 지킴: train.py·모델 변경 없음 (결과 표·감시 스크립트·큐만).
- 결정: 종료 절차 (QUEUE 14:20 6번)를 15:00 전인 이 라운드에서 함 — 14:20 지시가 08:40보다 우선이고 "1~5번 다음에 종료 절차"를 순서로 줌, 16:30 기한 안.

## 라운드 58 (13:57~ UTC)
- 상태 정상: `ensure_running.py` → day2 드라이버·side16_gp 살아 있음, pod 36.6 GiB / 60 (45 미만; side16_gp worker rss 25.8 GB가 가장 큼), oom_kill 0, load 3.6.
- r0_coord_s1 distill it 1525 (`runs/2026-10-29/r0_coord_s1/driver.log`): it 1500 val pcb 2.07/fail 0.62 · pkg 1.04/0.56 · ip 0.72/0.38 (16문제, 판정용 아님), gpu peak 12.1 GB, OOM 0.
- 라운드 57 뒤 새 평가 json 없음 → RESULTS 재생성 없음. 15:00 UTC 전 → QUEUE 08:40대로 새 코드·실험 없음. DAILY.md 01:50 UTC 작성 → 24시간 안 지남.
- 결정: 아무것도 바꾸지 않음 (라운드 50·51 결정 유지, 큐 순서 그대로).
- 다음 라운드: 15:00 UTC 뒤 첫 라운드에서 QUEUE 08:40 종료 절차 (루프 종료 요약 + RESULTS 재생성 + `agent/STOP`, 드라이버·큐 그대로).

## 라운드 57 (13:41~ UTC)
- 상태 정상: `ensure_running.py` → day2 드라이버·side16_gp 살아 있음, pod 26.0 GiB / 60, oom_kill 0, load 3.6.
- r0_coord_s1 distill it 750 (`runs/2026-10-29/r0_coord_s1/driver.log`): it 700 val pcb 1.85/fail 0.38 · pkg 1.00/0.88 · ip 0.74/0.19 (16문제, 판정용 아님), gpu peak 12.1 GB, OOM 0.
- 라운드 56 뒤 새 평가 json 없음 → RESULTS 재생성 없음. 15:00 UTC 전 → QUEUE 08:40대로 새 코드·실험 없음. DAILY.md 01:50 UTC 작성 → 24시간 안 지남.
- 결정: 아무것도 바꾸지 않음 (라운드 50·51 결정 유지, 큐 순서 그대로).
- 다음 라운드: 15:00 UTC 뒤 첫 라운드에서 QUEUE 08:40 종료 절차 (루프 종료 요약 + RESULTS 재생성 + `agent/STOP`, 드라이버·큐 그대로).

## 라운드 56 (13:20~ UTC)
- 상태 정상: `ensure_running.py` → day2 드라이버·side16_gp 살아 있음, pod 29.5 GiB / 60, oom_kill 0.
- **bm_manoharan 학습 끝** (13:25, `runs/2026-10-29/bm_manoharan_rl/log.txt`): RL 321 iter / 1.27 h, best it 300 score 8.458, replaced 10, gpu peak 12.9 GB, OOM 0. 다른 run과 같은 예산. it 300 val (16문제, 판정용 아님): pcb fail 1.00 · pkg fail 0.94 (count/g+p 1.50) · ip fail 1.00 → RL 뒤에도 greedy가 거의 0개 성공 (라운드 51의 빈칸 쏠림이 남음). 판정은 10칸 평가 (50문제) 결과로.
- 드라이버가 GPU 큐 다음 항목 **r0_coord_s1** (QUEUE 03:10 3번 seed 재학습) 시작 (13:25, `runs/2026-10-29/r0_coord_s1/rung_stdout.log`).
- 새 평가 결과 json 없음 → RESULTS 재생성 없음. 15:00 UTC 전 → QUEUE 08:40대로 새 코드·실험 없음. DAILY.md 01:50 UTC 작성 → 24시간 안 지남.
- 결정: 아무것도 바꾸지 않음 (e0plus16 16칸 작업 멈추지 않음 — 라운드 50 결정; 큐 순서 그대로).
- 다음 라운드: 15:00 UTC 뒤 첫 라운드에서 QUEUE 08:40 종료 절차 (루프 종료 요약 + RESULTS 재생성 + `agent/STOP`, 드라이버·큐 그대로).

## 라운드 55 (13:05~ UTC)
- 상태 정상: `ensure_running.py` → day2 드라이버·side16_gp 살아 있음, pod 34.6 GiB / 60 (45 미만), oom_kill 0.
- bm_manoharan RL it 200 val도 세 종류 fail 1.00, it 225 샘플링 fail 1.00, λ 10.7 → 11.9로 계속 오름 (`runs/2026-10-29/bm_manoharan_rl/log.txt`). OOM 0, gpu peak 12.6 GB.
- 라운드 54 뒤 새로 끝난 실험 없음 (runs 아래 새 json 없음) → RESULTS 재생성 없음. 15:00 UTC 전 → QUEUE 08:40대로 새 코드·실험 없음. DAILY.md는 01:50 UTC 작성 → 24시간 안 지남.
- 결정: 아무것도 바꾸지 않음 (라운드 51·53 결정 유지).
- 다음 라운드: bm_manoharan RL 끝 (~13:25)·OOM 확인; 15:00 UTC 뒤 첫 라운드에서 종료 절차.

## 라운드 54 (12:49~ UTC)
- 상태 정상: `ensure_running.py` → day2 드라이버·side16_gp 살아 있음, pod 24.8 GiB / 60, oom_kill 0.
- bm_manoharan RL it 125·150도 샘플링 fail 1.00 (it 100 val 세 종류 fail 1.00, `runs/2026-10-29/bm_manoharan_rl/log.txt`). OOM 0, gpu peak 12.3 GB.
- 라운드 53 뒤 새로 끝난 실험 없음 (runs 아래 새 json 없음) → RESULTS 재생성 없음. 15:00 UTC 전이라 QUEUE 08:40대로 새 코드·실험 없음. DAILY.md는 01:50 UTC 작성 → 24시간 안 지남.
- 결정: 아무것도 바꾸지 않음 (라운드 51·53 결정 유지).
- 다음 라운드: bm_manoharan RL 끝 (~13:25)·OOM 확인; 15:00 UTC 뒤 첫 라운드에서 종료 절차 (루프 종료 요약 + RESULTS 재생성 + `agent/STOP`, 드라이버·큐 그대로).

## 라운드 53 (12:30~ UTC)
- 상태 정상: `ensure_running.py` → day2 드라이버·side16_gp 살아 있음, pod 27.1 → 22.5 GiB / 60, oom_kill 0, load 3.5.
- **bm_manoharan RL it 100 val도 세 종류 모두 fail 1.00** (`runs/2026-10-29/bm_manoharan_rl/log.txt`). RL 학습 중 샘플링 fail도 it 25~100에서 0.98~1.00 (count/ref 대부분 nan = 성공 0). → 샘플링 REINFORCE가 distill의 "빈칸 쏠림"을 아직 풀지 못함. it 100 resume checkpoint 정상, OOM 0.
- 결정: 계속 아무것도 바꾸지 않음 (라운드 51 결정 유지 — QUEUE 08:40 새 코드 금지, 기준선 재구현을 결과 보고 고치지 않음). RL 끝(~13:25) 뒤 드라이버가 10칸 평가를 돌리면 그 결과 그대로 RESULTS "선행 연구 재구현"에 넣고, 계속 0개면 라운드 51에 적은 차이 줄을 붙인다.
- 새로 끝난 실험 없음 → RESULTS 재생성 없음.
- 다음 라운드: bm_manoharan RL it 200+ val·OOM; 13:25 뒤 10칸 평가 시작 확인; 15:00 UTC 뒤 첫 라운드에서 종료 절차.

## 라운드 52 (12:14~ UTC)
- 상태 정상: `ensure_running.py` → day2 드라이버·side16_gp 살아 있음, pod 29.7 GiB / 60, oom_kill 0, load 3.5~3.8 (CPU 4칸: e0plus16 1 + side16_gp 1 + bm RL certify 2). bm_manoharan distill → RL 넘어감 (`runs/2026-10-29/bm_manoharan_rl/log.txt`, 6분째, 아직 첫 val 없음).
- 새로 끝난 실험 없음 → RESULTS 재생성 없음. 15:00 UTC 전이라 QUEUE 08:40대로 새 코드·실험 없음.
- 다음 라운드: bm_manoharan RL val (fail 1.00에서 내려오는지)·OOM; 15:00 UTC 뒤 첫 라운드에서 종료 절차 (루프 종료 요약 + RESULTS 재생성 + `agent/STOP`, 드라이버·큐 그대로).

## 라운드 51 (11:56~ UTC)
- 상태 정상: `ensure_running.py` → day2 드라이버·side16_gp 살아 있음, pod 22.8 GiB / 60, oom_kill 0. 15:00 전이라 QUEUE 08:40대로 새 코드·실험 없음.
- **bm_manoharan distill val이 처음부터 세 종류 모두 fail 1.00** (`runs/2026-10-29/bm_manoharan/driver.log`, it 100~1100; 다른 run은 같은 시점 ip fail ~0.2). 원인 확인 (`scripts/diag_bm_val.py`, 읽기 전용, CPU 1 thread 1분; `last.pt` it 1100, `val_ip10_0` 16문제):
  - greedy: 1195 step 중 skip 1179, **놓은 decap 0개**, 성공 0.00 · 샘플링 (seed 0): 놓은 82개 (문제당 5.12), env 성공 0.25.
  - → 버그 아님. 고정 순서 + 종류/빈칸 행동에서 선생님 궤적의 대부분이 "빈칸"이라 (스텝 ~95%) argmax가 늘 빈칸을 고름. stop은 목표 만족 또는 순서 끝에서만 합법이라 빈칸만 고르면 끝까지 0개.
  - 결정: **아무것도 바꾸지 않음** — 평가는 모든 방법이 greedy (`eval_policy.py --samples 1`)로 같은 규약이고, 재구현을 고쳐 기준선을 좋게 만드는 것은 QUEUE 08:40 (새 코드 금지)·정직성 규칙에 어긋남. RL (샘플링 REINFORCE)이 이 쏠림을 풀지는 13:25 이후 10칸 평가로 판정. 계속 0개면 RESULTS 재구현 차이 줄에 "distill 뒤 greedy가 전부 빈칸 (클래스 쏠림), 원 논문은 샘플링/ε 탐색일 수 있음 — 확인 필요"로 적을 것.
- 다음 라운드: bm_manoharan RL val (fail이 1.00에서 내려오는지)·OOM 확인; 15:00 UTC 뒤 첫 라운드에서 QUEUE 08:40 종료 절차 (루프 종료 요약 + RESULTS 재생성 + `agent/STOP`, 드라이버·큐 그대로).

## 라운드 50 (10:58~ UTC)
- 상태 정상: `ensure_running.py` → day2 드라이버·side16_gp 살아 있음, pod 28.9 → 15.7 GiB / 60, oom_kill 0.
- **r2am (arch c' `--mode_head`) 학습 끝** (11:22, `runs/2026-10-29/r2am_rl/log.txt`): RL 356 iter / 1.27 h, best it 200 score 1.644, replaced 291, grid capped 139, gpu peak 16.6 GB, OOM 0. 다른 run과 같은 예산 (r2a: 361 iter, best it 300 score 1.778; val 16문제라 판정용 아님). 판정은 10칸 평가로.
- **bm_manoharan 다시 시작 (11:22) → distill it 25 통과** (loss 0.535, gpu peak 12.1 GB, OOM 0, `runs/2026-10-29/bm_manoharan/driver.log`): 라운드 49의 mul-sum 고침으로 triton backward 문제 해결 확인.
- 결정: **e0plus16 pcb16_0 종류 0 retry (10:34~, 36분째)를 멈추고 10칸 modal_rule 기준선을 그 자리에** (`kill_pat.py`; 큐에서 modal_rule 항목을 맨 끝에서 e0plus16 앞으로, e0plus16 이름은 "retry2"로 — 드라이버 FAILED 집합 우회). 이유: 이 16칸 작업은 진행 출력이 없어 끝 시각을 모름 (16칸 g+p 1 파일이 1.8 h; 종류 0만이면 pcb에서 더 많이 채워 더 느림), r2am이 11:22에 끝나면 그 평가가 그 뒤에 1 h 넘게 막힘 (라운드 49에 적은 규칙). modal_rule 10칸은 파일당 ~7분 (`eval_pkg10_1__modal_rule.json` 417 s)이라 r2am 끝까지의 틈을 채움. 버린 것: CPU 36분. CPU 수 4 그대로.
  - 주의: 이 16칸 E0+ 작업은 세 번째로 멈춤 (09:30, 11:10 포함). 15:00 루프 종료 뒤에는 아무도 멈추지 않으므로 밤새 이어서 돎. 13:30 bm_manoharan 평가 때는 다시 멈추지 않을 생각 (그대로면 bm 평가는 e0plus16 뒤 — 판정은 사람이 회수).
- **10칸 modal_rule 기준선 끝** (11:22, 11분 — 예상보다 빠름, `runs/2026-10-28/baselines/eval/eval_*10_*__modal_rule.json`, `agent/logs/base10.log`): US 6789241식 비학습 규칙 (공진 피크에 SRF 맞는 decap, 판 종류 분기 없음). 성공률 / 개수÷g+p / 개수÷E0+: ip 0.94 / 1.10 / 1.56 · pcb **1.00** / 0.84 / 0.21 · pkg 0.88 / 0.86 / 0.61; 문제당 0.42~0.82 s.
  - RESULTS에 **modal_rule 대비 짝지은 CI 표** 추가 (`collect_results.py`의 기존 `ci_table`을 ref="modal_rule"로 한 번 더 부름; 결과 표 정리일 뿐 새 실험·모델 코드 아님). 우리 최선 r2a_rl vs modal_rule (성공률 차 [CI] / 개수 비 [CI]): ip +0.06 [+0.00, +0.14] / **0.719 [0.659, 0.780]** · pcb −0.04 [−0.10, +0.00] / 0.983 [0.941, 1.022] · pkg −0.04 [−0.14, +0.06] / 1.005 [0.950, 1.058].
  - → **pcb·pkg에서는 비학습 모달 규칙이 우리 정책과 같은 수준** (성공률·개수 차 CI가 0·1을 포함 또는 경계), **ip에서만 정책이 개수 ~28% 적음**. 다른 단도 같음 (pcb에서 r2_mott_rl −0.12, r3_loop_rl −0.16은 규칙보다 낮음, CI 0 밖). bp_park_rl은 세 종류 모두 규칙보다 크게 낮음.
  - 해석 주의 (보고서 문장에 영향): QUEUE 03:15의 "규칙은 판 종류마다 다시 골라야 하고 정책은 하나로 고른다"는 **E0+ (종류 고정 greedy)에 대해서만** 맞음. 공진·SRF를 보는 규칙 하나는 판 종류 분기 없이 pcb·pkg에서 정책만큼 함 (종류 비율: pcb 종류 2가 0.73, ip 종류 0이 1.00 — 규칙도 종류를 판마다 다르게 고름). 즉 "공진 정보가 종류 선택에 핵심"이라는 주장은 강해지고, "학습이 규칙보다 낫다"는 ip 10칸 개수에서만 지금 숫자로 말할 수 있음. 16칸 modal_rule은 큐에 없음 (사람이 판단).
- **r2am 10칸 평가 끝 → arch c' 판정: 효과 없음** (11:36, 12분, `runs/2026-10-29/r2am_rl/eval/`, 각 판 50문제). 성공률 / 개수÷g+p / 개수÷E0+: ip 1.00 / 0.74 / 1.01 · pcb 0.94 / 0.83 / 0.19 · pkg 0.86 / 0.85 / 0.61.
  - vs 부모 r2a_rl (RESULTS ablation 표, 성공률 차 / 개수 비): ip +0.00 [+0.00, +0.00] / 0.991 [0.972, 1.004] · pcb −0.02 [−0.08, +0.04] / 1.022 [0.971, 1.079] · pkg +0.02 [−0.08, +0.12] / 1.023 [0.972, 1.074] → **모든 CI가 0·1을 포함, 차 없음.**
  - vs E0+: ip 성공 +0.00, 개수 비 1.013 [0.998, 1.035] (못 넘음, r2a와 같음) · pcb +0.28 [+0.12, +0.42] / 0.195 · pkg +0.04 [−0.08, +0.18] / 0.606.
  - vs modal_rule: ip 개수 0.711 [0.654, 0.771], pcb −0.06 [−0.14, +0.00], pkg −0.02 [−0.14, +0.10] (r2a와 같은 그림).
  - 결정: **최종 후보는 r2a 그대로** (r2am은 하나 더 바꾼 것이 이득 없음 → 더 단순한 쪽). GPU 큐의 r2a seed 1·2도 그대로.
- 지금 CPU 줄: e0plus16 pcb16_0 종류 0 **retry2** (11:38~, 끝 시각 모름). GPU: bm_manoharan (끝 ~13:30) → 드라이버가 10칸 평가를 e0plus16 다음에 함.
- 다음 라운드: bm_manoharan 학습 상태·OOM 확인; 15:00 UTC 뒤 첫 라운드에서 QUEUE 08:40 종료 절차 (루프 종료 요약 + RESULTS 재생성 + `agent/STOP`, 드라이버·큐는 그대로).

## 라운드 49 (09:14~ UTC)
- **bp_park 학습 끝** (09:18, `runs/2026-10-29/bp_park_rl/log.txt`): RL 199 iter / 1.27 h, best it 100 score 3.219, replaced 0, gpu peak 16.6 GB, OOM 0 (distill 2210 iter / 0.77 h, 라운드 46). 다른 run과 같은 예산.
- **bm_manoharan distill이 79초 만에 실패** (09:19, `runs/2026-10-29/bm_manoharan/driver.log`): 첫 backward에서 triton이 드라이버 모듈을 gcc로 빌드하다 `Python.h: No such file or directory` (이 이미지에 python3.14 헤더 없음). 드라이버는 실패 이름을 기억하고 다음 r2am으로 넘어감.
  - 원인 찾기 (`scripts/diag_triton.py`, GPU 작은 텐서): `ManoharanNet` 점수 `einsum("ed,etd->et")`의 CUDA backward가 triton을 부름 (`bmm(k, q[:,:,None])`도 같음; MHA·LayerNorm·index_put은 괜찮음). 3D×3D einsum인 park_am·model.py `mode_add`는 GPU에서 이미 돌았음 (bp_park, r2am distill 지금 진행 중).
  - 고침: `baselines_ml/manoharan.py` 142줄을 같은 값의 `(q[:,None]*k).sum(-1)`로 (모델·행동 공간 그대로, CPU smoke와 같은 값). diag의 전체 망 backward 통과.
  - 결정: **드라이버 재시작** (실패 이름을 메모리에서 지우려면 필요; 이름을 바꾸면 run 경로·RESULTS 표 이름이 달라짐). 재시작이 지금 도는 작업을 다시 띄우지 않게: `pipeline_day2.py` CPU 줄도 GPU 줄처럼 같은 명령이 살아 있으면 ATTACH (기다리기만 함); `agent/day2_queue.json`에서 bm_manoharan을 r2am **바로 뒤**로 (원래 r2am 앞이었음 — 그대로면 새 드라이버가 r2am 학습 중에 bm_manoharan을 GPU에 띄워 두 train.py가 겹침). 그 밖의 큐 순서는 그대로. 새 드라이버: `ATTACH train r2am` 확인.
- **r3@R2/4/16 10칸 평가 끝** (09:14, `runs/2026-10-28/r3_loop_rl/eval/*@R*.json`, 각 판 50문제; 학습은 루프 2~8회, 기본 평가 8회). 성공률 (ip / pcb / pkg): R2 1.00 / 0.90 / 0.74 · R4 1.00 / 0.88 / 0.88 · **R8 0.98 / 0.84 / 0.88** · R16 1.00 / 0.78 / 0.78; 개수÷g+p는 0.74 / 0.82~0.89 / 0.82~0.84로 거의 같음. → **평가 때 반복을 늘려도 좋아지지 않음** (학습 범위 밖 R16은 pcb·pkg가 낮은 쪽, 50문제라 차 ±0.1은 잡음 범위). "생각을 더 하면 낫다"는 주장은 이 결과로 못 함 — 그대로 보고.
- 결정: **e0plus16 pcb16_0 종류 0 retry (09:15~, 12분째)를 멈추고 bp_park 10칸 평가를 먼저** (`kill_pat.py`). 이유: r3@R이 09:14에 끝나고 bp_park 학습은 09:18에 끝나서, 그 4분 사이 드라이버가 16칸 작업(~3~5 h)을 골라 선행 연구 판정(QUEUE 03:30)이 오후까지 막힐 상황 (라운드 43과 같은 문제). 버린 것: CPU 12분. 재시작한 드라이버는 bp_park 평가 (09:30~, ~25분) 다음에 같은 16칸 작업을 다시 돌림 (출력 없음 → 자동). CPU 수 4 그대로.
- QUEUE 08:40 확인: 이번 변경은 실패한 기준선(QUEUE 03:30 B-Manoharan)을 돌게 하는 버그 고침과 드라이버 재시작 안전장치 — 새 아키텍처·실험 아님.
- **bp_park 10칸 평가 끝** (10:32, `runs/2026-10-29/bp_park_rl/eval/`, 각 판 50문제, 63분 — 해가 커서 cutting-plane이 느림, 1.8~9.0 s/문제) → RESULTS "선행 연구 재구현" 절 B-Park 표 (드라이버가 재생성). 성공률 / 개수÷g+p / 개수÷E0+ (둘 다 성공한 문제):
  - ip 0.72 / 0.97 / 1.38 · pcb 0.46 / 4.86 / 1.13 · pkg 0.16 / 3.39 / 3.04.
  - 같은 판의 우리 정책: r2a_rl 1.00 / 0.96 / 0.84 (성공률 ip / pcb / pkg), **좌표 입력 0단 r0_coord_rl도 1.00 / 0.96 / 0.76**. E0+ 대비 짝지은 CI (RESULTS): bp_park 성공률 차 ip −0.28 [−0.42, −0.16] · pcb −0.20 [−0.36, −0.02] · pkg −0.66 [−0.78, −0.52].
  - → **선행 연구식 B-Park (재구현 기준)은 세 종류 모두 E0+·g+p·우리 사다리보다 성공률이 낮고 해가 큼.** 같은 좌표 입력인 r0_coord_rl과의 차는 구조(Kool식 포인터)와 학습(rollout 기준선, 모방 항 없음)에서 옴 — "좌표 입력이면 다 된다"가 아니라 "우리 학습 방법이면 좌표로도 꽤 된다"로 읽어야 함 (0단과 2·3단 차는 사다리 seed로 판정).
  - 주의 (재구현 차이): RL이 199 iter뿐 (rollout 기준선 greedy 계산 때문에 25 iter당 ~580 s; r2a는 361 iter), best는 it 100. 같은 시간 예산이라 그대로 보고.
- 다음 라운드: **r2am 끝 (~11:25)** → bm_manoharan GPU 시작·첫 backward 통과·OOM 확인. CPU 줄은 10:34부터 e0plus16 pcb16_0 종류 0 retry (16칸, ~2~5 h) → r2am 10칸 평가가 그 뒤에 막힘. 11:25에 그 작업이 1 h 넘게 남았으면 메모 규칙대로 멈추고 이름을 "retry2"로 바꿔 r2am 평가를 먼저 (지금 멈추면 r2am 끝까지 CPU 줄에 넣을 짧은 작업이 없어 똑같이 ~50분을 잃음 → 지금은 그대로 둠).

## 라운드 48 (08:58~ UTC, 라운드 47 직후라 짧음)
- 상태 정상: `ensure_running.py` → day2 드라이버·side16_gp 살아 있음, pod 15.7 GiB / 60, oom_kill 0. bp_park RL it 150 (count/ref 1.80, fail 0.16, grid capped 29, OOM 0; best 그대로 it 100 score 3.219). CPU 줄 = r3_loop_rl@R2 10칸 평가 (08:42~), side16_gp 16칸 g+p 1 worker, RL certify 2.
- 새로 끝난 실험 없음 → RESULTS 재생성 없음. QUEUE 08:40: 15:00 UTC 전이라 하던 일 그대로, 새 코드 없음.
- 결정: 새 실험·코드 변경 없음 — CPU 4칸 꽉 참, 다음 판정(선행 연구 재구현 표)은 bp_park 10칸 평가가 있어야 가능.
- 다음 라운드: 라운드 47과 같음 (bp_park 끝 ~09:20 → 드라이버가 r3@R 작업 사이에 10칸 평가 → "선행 연구 재구현" 표; bm_manoharan 시작·OOM 확인).

## 라운드 47 (08:30~ UTC)
- 상태 정상: `ensure_running.py` → day2 드라이버·side16_gp 살아 있음, pod 14.3 GiB / 60, oom_kill 0. bp_park RL it 100 val (16문제, 보고서용 아님): pcb 3.10/fail 0.88 · pkg 1.67/0.94 · ip 0.65/0.31, new best score 3.219 (`runs/2026-10-29/bp_park_rl/log.txt`).
- **day1 eval10 r0_coord_distill 끝** (08:41, `runs/2026-10-28/r0_coord_distill/eval/`, 각 판 50문제) → 이제 사다리 0~3단 distill·rl 10칸 평가가 다 있음. 성공률 / 개수÷g+p / 개수÷E0+: ip 0.96 / 1.17 / 2.04 · pcb 0.76 / 1.78 / 0.48 · pkg 0.90 / 1.59 / 1.16.
- **RESULTS에 "증류 → RL" 표 추가** (`scripts/collect_results.py` `distill_rl_table`: 같은 단 `*_rl` vs `*_distill`, 짝지은 bootstrap 95% CI, 같은 문제). 방법론의 핵심 질문(RL이 증류를 넘었나)에 대한 표:
  - **0·1단:** RL이 개수를 크게 줄임 (둘 다 성공한 문제 개수 비 pcb 0.67·0.79, pkg 0.66·0.72, CI가 1 밖) **대신 pkg 성공률이 떨어짐** (0단 −0.14 [−0.28, −0.02], 1단 −0.22 [−0.38, −0.08]); pcb 성공률은 올라감 (+0.20·+0.10, CI 0 밖).
  - **2·3단:** RL이 개수는 거의 그대로 (비 0.94~1.01) 두고 성공률을 올림 (ip 2단 +0.08 [+0.02, +0.16], 3단 +0.16 [+0.08, +0.26]; pcb·pkg는 +0.00~+0.10, CI가 0 포함).
  - 해석 (seed 1개라 잠정): 모드 토큰이 있는 단은 증류만으로 개수가 이미 g+p보다 적고 (개수÷g+p 0.74~0.97), RL은 실패를 줄이는 쪽으로 감; 모드 토큰이 없는 단은 증류 해가 커서 (÷g+p 1.08~1.78) RL이 개수를 줄이다가 pkg에서 실패를 늘림. 사다리 seed 1·2 (GPU 큐에 있음)로 확인.
- 같은 수정: ablation 표 제목에서 r2am vs r2a_rl을 "편향 없는"이라고 잘못 부르던 문제 고침 (부모 설명을 ABLATIONS 항목마다 따로).
- 결정: 새 실험 없음 — CPU 4칸 꽉 참 (certify 2 + 드라이버 CPU 줄 1 + side16_gp 1), GPU는 bp_park. 다음 판정(선행 연구 재구현 표)은 bp_park 10칸 평가가 있어야 가능.
- **QUEUE 08:40 UTC 확인 (push 전 pull에서 받음): 루프는 15:00 UTC에 끝남.** 그 전까지 새 아키텍처·새 코드 없이 결과 기록·정리만; 15:00 뒤 첫 라운드에서 "루프 종료 요약" + RESULTS 재생성 + `agent/STOP` + push, 드라이버·학습·평가·큐 순서는 그대로 둠.
  - 결정: 이번 라운드의 `collect_results.py` "증류 → RL" 표는 그대로 둠 — 기존 평가 기록을 다시 묶는 결과 표일 뿐 (새 실험·모델 코드 아님, QUEUE의 "결과 기록과 정리"에 해당), 지시를 받기 전에 작성함. GPU 큐의 r2am (arch c', 라운드 45에 이미 구현·큐에 넣음)도 "큐 순서는 지금 그대로"에 따라 그대로 둠.
- 다음 라운드: bp_park 끝(~09:20) → 드라이버가 10칸 평가를 r3@R 작업 다음에 고름 → "선행 연구 재구현" 표 (bp_park vs E0+·r2a). bm_manoharan 시작·OOM 확인.

## 라운드 46 (08:10~ UTC)
- 상태 정상: `ensure_running.py` → day2 드라이버·side16_gp 살아 있음, pod 14.6 GiB / 60, oom_kill 0.
- **bp_park distill 끝** (08:01, `runs/2026-10-29/bp_park_distill/log.txt`): 2210 iter / 0.77 h (다른 run과 같은 시간 예산; 모델이 작아 iter가 더 많음), best it 1300 score 2.921, gpu peak 12.1 GB. 마지막 val (16문제, 보고서용 아님): pcb 7.99/fail 0.62 · pkg 5.67/0.50 · ip 0.75/0.25 → 좌표 포인터 정책은 증류 뒤 pcb·pkg에서 개수가 매우 많음 (r2a distill val pcb 1.04 · pkg 1.07). RL 시작 (rollout 기준선, `--imit 0`), it 25: count/ref 2.19, fail 0.20, OOM 0, 25 iter당 577 s (r2a 403 s보다 느림 — rollout 기준선 greedy 해 계산 때문으로 보임; 시간 예산 1.25 h는 같음).
- **day1 eval10 r1_modal_distill 끝** (08:12, `runs/2026-10-28/r1_modal_distill/eval/`, 각 판 50문제) → `collect_results.py`로 RESULTS 재생성. 성공률 / 개수÷g+p / 개수÷E0+: ip 1.00 / 0.91 / 1.59 · pcb 0.88 / 1.08 / 0.24 · pkg 0.90 / 1.13 / 0.82. → 1단에서 RL은 개수를 크게 줄이지만 (ip ÷E0+ 1.59→1.23, pcb ÷g+p 1.08→0.75, pkg 1.13→0.75) **pkg 성공률은 떨어뜨림 (0.90→0.68)**; pcb 성공률은 올림 (0.88→0.98). 2·3단은 RL이 성공률을 올리거나 유지 (라운드 44·45) — 1단만 개수↔성공률 맞바꿈. seed 1개라 사다리 seed 1·2로 확인.
- 결정: 새 실험·코드 변경 없음 — CPU 4칸 꽉 참 (certify 2 + 드라이버 CPU 줄 1 + side16_gp 1), GPU는 bp_park, 다음 판정(선행 연구 재구현 표)은 bp_park 10칸 평가가 있어야 가능.
- 다음 라운드: bp_park RL 진행·OOM 확인, r0_coord_distill 10칸 평가 기록 (~08:35). bp_park 끝(~09:20) → 10칸 평가 → "선행 연구 재구현" 표 (bp_park vs E0+·r2a).

## 라운드 45 (07:47~ UTC)
- 상태 정상: `ensure_running.py` → day2 드라이버·side16_gp 살아 있음, pod 17.7 GiB / 60, oom_kill 0. bp_park distill it 1550 val (16문제, 보고서용 아님): pcb 3.74/fail 0.62 · pkg 1.43/0.69 · ip 0.74/0.25.
- **day1 eval10 r2_mott_distill 끝** (07:52, `runs/2026-10-28/r2_mott_distill/eval/`, 각 판 50문제) → RESULTS 재생성. 성공률 / 개수÷g+p / 개수÷E0+: ip 0.92 / 0.74 / 1.02 · pcb 0.78 / 0.94 / 0.23 · pkg 0.70 / 0.97 / 0.69. → 2단도 RL이 성공률을 올림 (ip 0.92→1.00, pcb 0.78→0.88, pkg 0.70→0.72) 그리고 개수÷g+p를 줄임 (pcb 0.94→0.88, pkg 0.97→0.88).
- 결정: **드라이버 CPU 줄 순서를 바꿈** — 짧은 day1 10칸 평가 3개 (r1_modal_distill·r0_coord_distill·r3@R, 각 ~22~25분 / r3@R ~70분, 같은 명령·같은 기록 경로)를 "e0plus16 pcb16_0 종류 0 retry" 앞으로 (`agent/day2_queue.json`). 이유: 그대로면 16칸 작업(~3~5 h)이 07:52에 시작해 bp_park 10칸 평가(RL 끝 ~09:20)를 2 h 넘게 막음 (라운드 43과 같은 문제); 앞으로 옮긴 작업들이 ~10:00까지 줄을 채우고, 그 사이 끝난 bp_park 평가는 작업 사이에서 먼저 골라짐 (최대 지연 ~40분). 아무것도 멈추지 않음, CPU 수 4 그대로. 드라이버가 r1_modal_distill을 고른 것 확인.
- **arch (c') `--mode_head` 구현 + GPU 큐에 r2am 추가** (bm_manoharan 뒤, 사다리 seed 앞): add logit에 Σ_k α_ik·φ(m_k, c_t) (포트가 겨누는 모드 × 그 모드에 맞는 종류), 마지막 층 0 초기화 → 학습 전 정책 = 부모 r2a. 부모 r2a 대비 이것 하나만 바꿈, 같은 예산 (`decap_loop/model.py` `mode_add`, `train.py --mode_head`, 기본 꺼짐 → 기존 run·평가 영향 없음). `scripts/test_mode_head.py` (CPU) 통과: 초기 greedy 해 = 부모, 새 헤드에 기울기 흐름 (mott·loop). RESULTS ablation 표에 r2am_rl vs r2a_rl 행 예약 (`collect_results.py`). 노벨티 2줄·(c) 다수 수 초안 설계는 PROPOSALS.
  - 결정: (c) 다수 수 초안 루프는 미룸 — 수마다 m·d번 Woodbury + 피크 재추출 (CPU)이라 같은 2 h 예산에서 RL iter가 크게 줄어 비교가 불공정해짐. 같은 방향("모드를 거쳐 종류를 고른다")의 싼 변형 (c')를 먼저. 큐 위치: QUEUE 03:10 순서 (2. 아키텍처 → 3. 시드)를 따라 시드 앞; 선행 연구 기준선(QUEUE 03:30) 순서는 그대로.
- 다음 라운드: bp_park distill→RL 전환·OOM 확인 (~08:00), day1 distill 평가 기록 (r1·r0 → RESULTS). bp_park 끝(~09:20) → 10칸 평가 → "선행 연구 재구현" 표.

## 라운드 44 (07:09~ UTC)
- 상태 정상: `ensure_running.py` → day2 드라이버·side16_gp 살아 있음, pod 16.1 GiB / 60, oom_kill 0.
- **r2a 학습 끝** (07:15): RL 361 iter / 1.27 h, best it 300 score 1.778, replaced 310, gpu peak 16.6 GB, OOM 0 (`runs/2026-10-29/r2a_rl/log.txt`). distill (1603 iter / 0.77 h) + RL 모두 r2·r3a와 같은 예산. 드라이버가 bp_park 시작 (distill 단계 OOM 0).
- **r2a 10칸 평가 끝** (07:28, `runs/2026-10-29/r2a_rl/eval/`, 각 판 50문제, 0.30~1.12 s/문제) → RESULTS 재생성 (드라이버). 성공률 / 개수÷g+p / 개수÷E0+: ip 1.00 / 0.74 / 1.03 · pcb 0.96 / 0.82 / 0.20 · pkg 0.84 / 0.87 / 0.58.
  - **r2a vs 편향 없는 r2_mott_rl (짝지은 bootstrap 95% CI):** 성공률 차 ip +0.00 · **pcb +0.08 [+0.02, +0.16] · pkg +0.12 [+0.02, +0.24]**; 개수 비 ip 1.011 [0.995, 1.027] · pcb 1.003 [0.953, 1.051] · pkg 0.987 [0.931, 1.047]. → **2단에서는 주파수 궁합 편향이 pcb·pkg 성공률을 올림 (CI가 0 밖), 개수는 같음.** 3단(r3a vs r3)에서는 유의한 차이 없음. 주의: seed 1개; r2_mott_rl은 사다리 중 pkg 성공률이 낮은 편 (0.72 vs r3 0.88)이라 r2a는 r3 수준으로 "따라잡은" 것일 수 있음 → seed로 확인 필요.
  - **r2a vs E0+:** ip 성공률 차 +0.00, 개수 비 1.029 [1.000, 1.067] → **ip에서 E0+를 넘지 못함** (2·3단 모두 같은 개수 근처). pcb 0.203 · pkg 0.583 (E0+가 종류를 섞지 못하는 판). 그대로 보고.
- **day1 eval10 r3_loop_distill 끝** (07:15, `runs/2026-10-28/r3_loop_distill/eval/`): 성공률 ip 0.82 · pcb 0.84 · pkg 0.80 → 3단은 RL이 ip 성공률을 0.82 → 0.98로 올림 (그 밖은 RESULTS 표).
- 결정: **최종 후보(QUEUE 03:10 3번 시드 대상)를 r2a로 잠정 선택, GPU 큐 끝에 r2a seed 1·2 추가** (`agent/day2_queue.json`). 이유: 지금까지 바꾼 아키텍처 중 부모 대비 CI가 0 밖인 개선은 r2a 성공률뿐 (r3a는 없음). 큐 맨 끝이라 앞 순서(선행 연구 기준선 → 사다리 seed)를 바꾸지 않음; (c) 결과로 후보가 바뀌면 이 두 줄만 교체.
- 결정: (c) "여러 수 초안 검산 루프"는 아직 만들지 않음 — 큐 앞(bp_park → bm_manoharan, ~4 h)이 QUEUE 03:30 지시이고, (c)는 PROPOSALS 33줄대로 1수 검산이 이미 입력에 있어 새것이 되려면 다수 수 rollout(CPU 비용)이 필요. 설계만 다음 라운드에 PROPOSALS에 적음.
- 다음 라운드: bp_park distill→RL OOM 확인 (RL certify 2개 시작 시 CPU 4 확인), (c) 설계 메모. bp_park 끝(~09:20) → 10칸 평가 → "선행 연구 재구현" 표.

## 라운드 43 (06:33~ UTC)
- 상태 정상: `ensure_running.py` → day2 드라이버·side16_gp 살아 있음, pod 26.2 GiB / 60, oom_kill 0. r2a RL it 100 val score 1.976 (new best), it 125.
- 결정: **e0plus16 pcb16_0 종류 0 (05:09~, 85분째)을 멈추고 r3a 10칸 평가를 먼저 돌림** (`scripts/kill_pat.py`, rc −15). 이유: 16칸 g+p가 10칸보다 문제당 ~20배 느림 (pkg16_1 98 s/inst vs 10칸 3~5 s) → pcb 10칸 종류 0이 8~16분이었으니 16칸은 ~3~5 h 예상, r3a 판정(QUEUE 03:15 3번, 2일차 1순위)이 라운드 38부터 4라운드째 그 뒤에 막혀 있었음. r3a 평가는 17분. 버린 것: CPU 85분 (기록은 끝에만 저장되므로 부분 결과 없음).
  - `agent/day2_queue.json`: 이름을 "e0plus16 eval_pcb16_0 type0 retry"로 바꿔 (드라이버 FAILED 목록 회피) 같은 명령으로 다시 돌게 하고, 짧은 "day1 eval10 r3_loop_distill"·"r2_mott_distill" (각 ~17분, 큐 맨 끝에 있던 것)을 그 앞으로 옮김 → r2a 학습 (~07:20 끝) 10칸 평가가 다시 16칸 작업 뒤에 몇 시간 막히지 않게 (드라이버는 작업 사이마다 끝난 학습 평가를 먼저 고름). CPU 수 4 그대로.
- **r3a 10칸 평가 끝** (06:51, `runs/2026-10-29/r3a_rl/eval/`, 각 판 50문제) → RESULTS 재생성 (드라이버). 성공률 / 개수÷g+p / 개수÷E0+: ip 1.00 / 0.74 / 1.02 · pcb 0.92 / 0.86 / 0.20 · pkg 0.86 / 0.83 / 0.61.
  - **r3a vs 편향 없는 r3 (짝지은 bootstrap 95% CI):** 성공률 차 ip +0.02 [+0.00, +0.06] · pcb +0.08 [−0.02, +0.20] · pkg −0.02 [−0.12, +0.08]; 개수 비 ip 1.003 [0.997, 1.013] · pcb 1.027 [0.964, 1.089] · pkg 1.040 [0.998, 1.084]. → **주파수 궁합 편향은 3단에서 유의한 차이 없음** (seed 1개; 성공률은 조금 높고 개수는 조금 많은 쪽, 모두 CI가 0·1 포함).
  - **r3a vs E0+:** ip 성공률 차 +0.00, 개수 비 1.016 [0.998, 1.042] → **ip에서 E0+를 넘지 못함** (같은 개수). pcb 0.204 [0.163, 0.252] · pkg 0.608 [0.552, 0.667] (E0+가 종류를 섞지 못하는 판). 그대로 보고.
  - 종류 비율 (성공 문제): ip 종류 0 1.00, pcb 0/1/2 = 0.08/0.19/0.73, pkg 0.14/0.18/0.68 — r3와 거의 같음 (정책 하나가 판마다 다른 종류를 고르는 것은 r3에서 이미 그랬음).
- 다음 라운드: r2a 10칸 평가 (r2a RL 끝 ~07:20 → day1 distill 평가 뒤) → r2a vs r2_mott_rl ablation 판정 → 둘 다 무의미하면 (a)는 "효과 없음"으로 정리하고 다음 아키텍처 (c: 여러 수 초안 검산 루프, PROPOSALS 33줄) 설계를 시작할지 결정. bp_park 시작·OOM 확인.

## 라운드 42 (06:18~ UTC)
- 상태 정상: `ensure_running.py` → day2 드라이버·side16_gp 살아 있음, pod 30.0 GiB / 60 (< 45), oom_kill 0. r2a RL it 50 (`runs/2026-10-29/r2a_rl/log.txt`, cert-bad 1, OOM 0).
- 새로 끝난 실험 없음 → RESULTS 재생성 없음.
- 결정: 새 실험·코드 변경 없음 — CPU 4칸 꽉 참 (라운드 40·41과 같은 이유).
- 다음 라운드: 라운드 38과 같음 (r3a 10칸 평가 기록 → ablation 표 r3a vs r3 + E0+ 대비 판정 → RESULTS 재생성; r2a RL val·OOM 확인).

## 라운드 41 (06:02~ UTC)
- 상태 정상: `ensure_running.py` → day2 드라이버·side16_gp 살아 있음, pod 26.1 GiB / 60, oom_kill 0.
- **r2a distill 끝** (`runs/2026-10-29/r2a_distill/log.txt`): 1603 iter / 0.77 h, best it 1500 score 1.843 (r3a distill과 같은 예산). 마지막 val (16문제, 보고서용 아님): pcb 1.035/fail 0.75 · pkg 1.071/0.50 · ip 0.691/0.12. RL이 best.pt에서 시작, certify 2.
- 새로 끝난 평가 없음 → RESULTS 재생성 없음.
- 결정: 새 실험·코드 변경 없음 — CPU 4칸 (r2a certify 2 + e0plus16 1 + side16_gp 1) 꽉 참. r3a 10칸 평가는 드라이버가 e0plus16 pcb16_0 종류 0 뒤에 자동 시작.
- 다음 라운드: 라운드 38과 같음 (r3a 10칸 평가 기록 → ablation 표 r3a vs r3 + E0+ 대비 판정 → RESULTS 재생성; r2a RL OOM 확인, 끝 ~07:20 예상).

## 라운드 40 (05:47~ UTC)
- 상태 정상: `ensure_running.py` → day2 드라이버·side16_gp 살아 있음, load 3.6, pod 23.7 GiB / 60, oom_kill 0. 새로 끝난 실험 없음 → RESULTS 재생성 없음.
- 결정: 새 실험·코드 변경 없음 — r3a 10칸 평가를 지금 따로 띄우면 r2a RL (certify 2) 시작 뒤 CPU 5개로 한도 4 초과. 드라이버 순서 그대로 둠.
- 다음 라운드: 라운드 38과 같음 (r3a 10칸 평가 기록 → ablation 표 r3a vs r3 + E0+ 대비 판정 → RESULTS 재생성; r2a RL 시작·OOM 확인).

## 라운드 39 (05:31~ UTC, 라운드 38 직후라 짧음)
- 상태 정상: `ensure_running.py` → day2 드라이버·side16_gp 살아 있음, pod 33.3 GiB / 60 (< 45), oom_kill 0. 새로 끝난 실험 없음 → RESULTS 재생성 없음.
- 확인: `pipeline_day2.py`는 CPU 줄 다음 항목을 고를 때 "끝난 학습의 10칸 평가"를 먼저 넣음 → r3a 10칸 평가는 지금 도는 e0plus16 pcb16_0 종류 0 (16칸 파일 하나) 바로 뒤. 10칸 pcb 종류 0이 8~16분이었으니 16칸은 더 걸림 (시작 22분째, 기록 아직 없음).
- 결정: 새 실험·코드 변경 없음 — CPU 4칸 꽉 참, 다음 판단(r3a vs r3, E0+ 대비)은 r3a 10칸 기록이 있어야 가능.
- 다음 라운드: 라운드 38과 같음 (r3a 10칸 평가 기록 → ablation 표 + E0+ 대비 판정 → RESULTS 재생성; r2a distill→RL·OOM 확인).

## 라운드 38 (05:13~ UTC)
- **r3a 학습 끝** (05:13): RL 346 iter / 1.27 h, best iter 100 (val score 1.678), replaced 307, gpu peak 17.5 GB, OOM 0 (`runs/2026-10-29/r3a_rl/log.txt`). distill + RL 모두 r3와 같은 예산. 드라이버가 r2a를 바로 시작.
- **E0+ 10칸 6/6 끝** (ip10_1 종류 2, 05:07) → `make_type_aware.py` + `collect_results.py`, RESULTS 재생성. 세 종류 모두 50문제: E0+ 성공 / 개수÷g+p — ip 1.00 / 0.736 · pcb 0.66 / 4.83 · pkg 0.82 / 1.59. 정책 vs E0+ (ip 50): 개수 비 0단 1.449 [1.314, 1.595] · 1단 1.228 [1.140, 1.335] · 2단 1.017 [0.995, 1.046] · 3단 1.013 [0.998, 1.038], 성공률 차 0.00 ~ −0.04 → ip에서 2·3단은 E0+와 같은 개수(넘지는 못함), 0·1단은 더 씀. pcb·pkg는 E0+가 종류를 섞지 못해 약함 (정책 개수 비 0.18~0.22 / 0.53~0.59).
- 결정: **1일차 side10 체인 (pid 117899, `eval_policy --methods modal_rule,ls` 10칸)을 멈추고 드라이버 큐 맨 끝으로 옮김** (`agent/day2_queue.json` "day1 base10 modal_rule"·"day1 base10 ls", 같은 명령·같은 기록 경로, 이미 있는 기록은 건너뜀). 이유: 종류 2 g+p가 끝나며 이 체인이 modal_rule·LS로 넘어갔는데, 드라이버 `EXTERNAL` 패턴이 이 명령과 맞지 않아 CPU 줄도 e0plus16을 시작 → r2a RL이 certify 2개를 띄우면 CPU 5개로 QUEUE 한도 4 초과. LS 10칸은 몇 시간짜리라 남겨 두면 r3a 10칸 평가(QUEUE 다음 판정)가 밀림. 버린 진행분: modal_rule ip10 1파일 진행 중 수 분 (pkg10_1 modal_rule 기록은 05:14에 저장됨, 재사용).
- 다음 라운드: r3a 10칸 평가 기록 (e0plus16 pcb16_0 종류 0 뒤) → ablation 표 (r3a vs r3) + E0+ 대비 판정 → RESULTS 재생성. r2a distill→RL 진행·OOM 확인.

## 라운드 37 (04:57~ UTC)
- 상태 정상: `ensure_running.py` → day2 드라이버·side16_gp 살아 있음, pod 16.3 GiB / 60, oom_kill 0. r3a RL it 275 (`runs/2026-10-29/r3a_rl/log.txt`).
- **종류 2 g+p ip10_0·pkg10_1 끝 → E0+ 기록 5/6** (`collect_results.py`, RESULTS 재생성; `runs/2026-10-29/baselines_e0plus/eval/`):
  - **ip 10 (25문제, 파일 0)**: E0+ 성공 1.00, 개수 7.92 (모두 종류 0) vs g+p 0.40 → E0+/g+p 개수 비 0.765. 정책 vs E0+: 성공률 차 모두 +0.00, **개수 비 0단 1.437 [1.260, 1.628] · 1단 1.206 [1.112, 1.304] · 2단 1.019 [0.979, 1.064] · 3단 1.026 [0.997, 1.072]** → ip에서는 E0+가 강한 기준선이고, 2·3단만 E0+와 같은 개수 (CI가 1 포함), 0·1단은 더 많이 씀. 사다리 효과가 E0+ 기준으로는 ip에서 보임 (g+p 기준으로는 안 보였던 것). 정책이 E0+를 넘지는 못함 — 그대로 보고.
  - **pkg 10 (50문제, 파일 0·1)**: E0+ 성공 0.82, 개수 12.44, E0+/g+p 개수 비 1.586 [1.372, 1.836]. 정책 vs E0+: 개수 비 0.53~0.59, 성공률 차 −0.14~+0.06 (모두 CI 0 포함).
  - "판 종류마다 맞는 decap 종류" 표: 단일 종류 최선 ip → 0 (1.00), pcb → 1 (0.64), pkg → 0 (0.74), pkg는 E0+가 고른 종류가 0·1 섞임 (0.41 / 0.54).
- 결정: 새 실험·코드 변경 없음 (CPU 4칸 꽉 참; 남은 판단은 r3a RL `done:` ~05:15 → 10칸 평가, ip10_1 종류 2 → E0+ 6/6).
- 다음 라운드: r3a 10칸 평가 기록 → ablation 표 (r3a vs r3) + E0+ 대비 판정 → RESULTS 재생성. ip10_1 종류 2 끝나면 ip E0+ 50문제로.

## 라운드 36 (04:45~ UTC, 라운드 35 직후라 짧음)
- 상태 정상: `ensure_running.py` → day2 드라이버·side16_gp 살아 있음, pod 15.9 GiB / 60, oom_kill 0. r3a RL it 225 (train_pcb10_2 cert-bad 5, OOM 0, grid capped 51). E0+ 기록 3/6 그대로 → 새로 끝난 실험 없음, RESULTS 재생성 없음.
- 결정: 새 실험·코드 변경 없음 (라운드 23~35와 같은 이유: CPU 4칸 꽉 참, 다음 판단은 r3a RL `done:` ~05:15 뒤).
- 다음 라운드: 라운드 22와 같음 (남은 종류 2: ip10_0·ip10_1·pkg10_1 → E0+ 6/6).

## 라운드 35 (04:42~ UTC, 라운드 34 직후라 짧음)
- 상태 정상: `ensure_running.py` → day2 드라이버·side16_gp 살아 있음, pod 16.6 GiB / 60, oom_kill 0. E0+ 기록 3/6 그대로 → 새로 끝난 실험 없음, RESULTS 재생성 없음.
- r3a RL it 200 val (16문제, 보고서용 아님): pcb 1.124/fail 0.62 · pkg 1.000/0.88 · ip 0.691/0.12 → it 100 (score 1.678, best 유지)보다 나쁨. 같은 시점 r3 RL: pcb 0.990/0.75 · pkg 0.964/0.56 · ip 0.691/0.19 — r3도 best가 it 100이었음 → 두 run 모두 RL 후반에 val이 나빠지는 같은 모양, 평가는 best ckpt로 하므로 조건 동일.
- 결정: 새 실험·코드 변경 없음 (라운드 23~34와 같은 이유: CPU 4칸 꽉 참, 다음 판단은 r3a RL `done:` ~05:15 뒤).
- 다음 라운드: 라운드 22와 같음 (남은 종류 2: ip10_0·ip10_1·pkg10_1 → E0+ 6/6).

## 라운드 34 (04:39~ UTC)
- 상태 정상: `ensure_running.py` → day2 드라이버·side16_gp 살아 있음, pod 22.3 GiB / 60, oom_kill 0. r3a RL 마지막 줄 it 175 그대로 (resume ckpt it 192 저장).
- **종류 2 g+p pcb10_1·pkg10_0 끝 → E0+ 기록 3/6** (`collect_results.py` → `make_type_aware.py`, RESULTS 재생성):
  - pcb 10 (50문제, 파일 0·1): E0+ 성공 0.66, 개수 45.3 vs g+p 0.90 / 12.0 → 개수 비 E0+/g+p 4.83 [3.85, 5.84]. 정책 vs E0+ 개수 비 0.18~0.22, 성공률 차 +0.18~+0.32.
  - pkg 10 (25문제, 파일 0만): E0+ 성공 0.84, 개수 12.9 vs g+p 0.90 → 개수 비 1.83 [1.46, 2.28], 성공률 차 −0.08 [−0.24, +0.08]. 정책 vs E0+ 개수 비 0.51~0.56, 성공률 차 −0.04~+0.04 (CI 0 포함).
  - 종류 2만: pcb10_1 0.08, pkg10_0 0.08. 단일 종류로 가장 잘 되는 종류: **pcb → 1, pkg → 0 (ip → 0)** → RESULTS에 "판 종류마다 다르다" 문장이 자동으로 나옴 (QUEUE 03:15 4번 대비 문장).
  - 해석: pcb·pkg 모두 "종류 하나 고정" E0+는 g+p보다 약함 (종류를 섞어야 하는 판). ip에서만 E0+가 강할 것으로 예상 (종류 0만 50/50) — ip 종류 2 끝나면 확인. PROPOSALS (d') 그대로 (사람 결정).
- 결정: 새 실험·코드 변경 없음 (CPU 4칸 꽉 참, 다음 판단은 r3a RL `done:` ~05:15 뒤).
- 다음 라운드: 라운드 22와 같음 (남은 종류 2: ip10_0·ip10_1·pkg10_1 → E0+ 6/6).

## 라운드 33 (04:37~ UTC, 라운드 32 직후라 짧음)
- 상태 정상: `ensure_running.py` → day2 드라이버·side16_gp 살아 있음, pod 16.8 GiB / 60, oom_kill 0. r3a RL it 175 (train_pcb10_0 cert-bad 2, OOM 0, gpu peak 16.4 GB). E0+ 기록 1/6 그대로 → 새로 끝난 실험 없음, RESULTS 재생성 없음.
- 결정: 새 실험·코드 변경 없음 (라운드 23~32와 같은 이유: CPU 4칸 꽉 참, 다음 판단은 r3a RL `done:` ~05:15 뒤).
- 다음 라운드: 라운드 22와 같음.

## 라운드 32 (04:34~ UTC, 라운드 31 직후라 짧음)
- 상태 정상: `ensure_running.py` → day2 드라이버·side16_gp 살아 있음, pod 17.2 GiB / 60, oom_kill 0. r3a RL 마지막 줄 it 150 그대로, E0+ 기록 1/6 그대로 → 새로 끝난 실험 없음, RESULTS 재생성 없음.
- 결정: 새 실험·코드 변경 없음 (라운드 23~31과 같은 이유: CPU 4칸 꽉 참, 다음 판단은 r3a RL `done:` ~05:15 뒤).
- 다음 라운드: 라운드 22와 같음.

## 라운드 31 (04:32~ UTC, 라운드 30 직후라 짧음)
- 상태 정상: `ensure_running.py` → day2 드라이버·side16_gp 살아 있음, CPU 바쁜 것 4개 (load 4.1), pod 17.3 GiB / 60, oom_kill 0. r3a RL it 150 (train_pcb12_1 cert-bad 3, OOM 0, grid capped 15; val은 it 100이 마지막). 종류 2 기록 1/6 그대로 → 새로 끝난 실험 없음, RESULTS 재생성 없음.
- 결정: 새 실험·코드 변경 없음 (라운드 23~30과 같은 이유: CPU 4칸 꽉 참, 다음 판단은 r3a RL `done:` ~05:15 뒤).
- 다음 라운드: 라운드 22와 같음.

## 라운드 30 (04:29~ UTC, 라운드 29 직후라 짧음)
- 상태 정상: `ensure_running.py` → day2 드라이버·side16_gp 살아 있음, CPU 바쁜 것 4개, pod 17.1 GiB / 60, oom_kill 0. r3a RL 마지막 줄 it 125 그대로. 종류 2 기록 1/6 그대로 → 새로 끝난 실험 없음, RESULTS 재생성 없음.
- 결정: 새 실험·코드 변경 없음 (라운드 23~29와 같은 이유: CPU 4칸 꽉 참, 다음 판단은 r3a RL `done:` ~05:15 뒤).
- 다음 라운드: 라운드 22와 같음.

## 라운드 29 (04:26~ UTC, 라운드 28 직후라 짧음)
- 상태 정상: `ensure_running.py` → day2 드라이버·side16_gp 살아 있음, CPU 바쁜 것 4개, pod 15.6 GiB / 60, oom_kill 0. r3a RL it 125 (train_pcb12_1 cert-bad 9, OOM 0; val은 it 100이 마지막: score 1.678).
- 새로 끝난 실험 없음 (E0+ 기록 1/6 그대로) → RESULTS 재생성 없음. 선행 연구 재구현 절·각 표 아래 차이 한 줄은 `collect_results.py`에 이미 있음 확인 (bp_park·bm_manoharan 기록이 생기면 자동 표시).
- 결정: 새 실험·코드 변경 없음 (라운드 23~27과 같은 이유: CPU 4칸 꽉 참, 다음 판단은 r3a RL `done:` ~05:15 뒤).
- 다음 라운드: 라운드 22와 같음.

## 라운드 28 (04:21~ UTC)
- 상태 정상: `ensure_running.py` → day2 드라이버·side16_gp 살아 있음, pod 15.8 GiB / 60, oom_kill 0.
- **종류 2 g+p pcb10_0 끝 → 첫 E0+ 기록** (`runs/2026-10-29/baselines_e0plus/eval/eval_pcb10_0__gp_type_aware.json`, 25문제): 종류 0·1·2만 성공률 0.04 / 0.72 / 0.00 → E0+ 성공 0.72, 개수 48.4 (모두 종류 1). **g+p (0.90) 보다 성공률 낮고, 둘 다 성공한 문제 개수 비 E0+/g+p 5.00 [3.79, 6.30]** — pcb에서는 종류를 섞어야 하므로 "종류 하나 고정"은 강한 기준선이 아님. 정책 vs E0+ 개수 비 (pcb10_0, 25문제) 0.17~0.22는 이 약점 때문. RESULTS 재생성.
- 결정: E0+ 정의 (QUEUE 03:15 4번)는 그대로 두고, g+p 대비·E0+ 대비를 둘 다 계속 보고 → "{g+p, E0+} 중 문제마다 최선"을 정식 기준으로 할지는 PROPOSALS (d')에 (사람 결정; 기준선을 바꾸는 일이라 에이전트가 정하지 않음).
- `collect_results.py` 수정 2개 (CPU 거의 안 씀): (1) **같은 단 편향 없는 버전 대비 짝지은 bootstrap CI 표** 추가 (`ABLATIONS`: r3a_rl vs r3_loop_rl, r2a_rl vs r2_mott_rl; QUEUE 03:15 3번 판정 기준) — 기록이 생기면 자동 표시, r2_mott_rl vs r3_loop_rl로 표 모양 확인. (2) "판 종류마다 맞는 decap 종류" 표가 종류 0·1(50문제)과 종류 2(25문제)를 섞어 계산하던 것을 **세 종류가 모두 끝난 풀 파일만** 쓰게 고침, 판 종류가 하나뿐이면 "같다/다르다" 문장을 내지 않음.
- 다음 라운드: 라운드 22와 같음 (r3a RL `done:` ~05:15 → r3a 10칸 평가 → 새 ablation 표·E0+ 대비로 판정).

## 라운드 27 (04:18~ UTC, 라운드 26 직후라 짧음)
- 상태 정상: `ensure_running.py` → day2 드라이버·side16_gp 살아 있음, CPU 바쁜 것 4개 (certify 2 + 종류 2 g+p + 16칸 g+p 1 worker RSS 9.1 GB), pod 16.5 GiB / 60, oom_kill 0.
- r3a RL it 100 (1254 s, OOM 0, gpu peak 14.6 GB, grid capped 5): val (보고서용 아님, 16문제) pcb 1.121/fail 0.44 · pkg 0.972/0.56 · ip 0.691/0.12, score 1.678 (new best). 같은 시점 r3 RL: pcb 0.991/0.44 · pkg 0.979/0.62 · ip 0.691/0.12, score 1.679 → 거의 같음; 판정은 10칸 eval 뒤 (`runs/2026-10-29/r3a_rl/log.txt`).
- 종류 2 g+p 기록 아직 0/6 → 새로 끝난 실험 없음, RESULTS 재생성 없음.
- 결정: 새 실험·코드 변경 없음 (라운드 23~26과 같은 이유: CPU 4칸 꽉 참, 다음 판단은 r3a RL `done:` ~05:15 뒤).
- 다음 라운드: 라운드 22와 같음.

## 라운드 26 (04:15~ UTC, 라운드 25 직후라 짧음)
- 상태 정상: `ensure_running.py` → day2 드라이버·side16_gp 살아 있음, CPU 바쁜 것 4개 (certify 2 + 종류 2 g+p pcb10_0 + 16칸 g+p 1 worker RSS 9.2 GB), pod 15.1 GiB / 60, oom_kill 0. r3a RL it 75 (gpu peak 14.4 GB, r3 RL의 같은 시점 14.6 GB와 비슷). 종류 2 기록은 아직 0/6 → 새로 끝난 실험 없음, RESULTS 재생성 없음.
- 결정: 새 실험·코드 변경 없음 (라운드 23~25와 같은 이유).
- 다음 라운드: 라운드 22와 같음.

## 라운드 25 (04:12~ UTC, 라운드 24 직후라 짧음)
- 상태 정상: `ensure_running.py` → day2 드라이버·side16_gp 살아 있음, CPU 바쁜 것 4개, oom_kill 0.
- **10칸 종류 1만 쓰는 g+p 6파일 끝** (`runs/analysis/gp_one_type/*@type1.json`, 각 25문제; 성공률 / 성공 문제 평균 개수): ip10_0 0.12 / 24.7 · ip10_1 0.24 / 33.7 · pcb10_0 0.72 / 48.4 · pcb10_1 0.56 / 42.6 · pkg10_0 0.40 / 8.7 · pkg10_1 0.56 / 10.5. (종류 0: ip 1.00, pcb 0.04·0.00, pkg 0.76·…) → 판 종류마다 맞는 decap 종류가 다르다는 QUEUE 4번 대비가 종류 1에서도 보임. E0+ 표는 종류 2 끝(~05:30) 뒤 `make_type_aware.py`로.
- 결정: 새 실험·코드 변경 없음 (CPU 4칸 꽉 참, 다음 판단은 r3a RL `done:` ~05:15 뒤). RESULTS 재생성은 E0+가 완성되는 종류 2 뒤에 한 번에 (지금 종류 1은 분석 기록이라 요약표 행이 바뀌지 않음).
- 다음 라운드: 라운드 22와 같음.

## 라운드 24 (04:10~ UTC, 라운드 23 직후라 짧음)
- 상태 정상: `ensure_running.py` → day2 드라이버·side16_gp 살아 있음, CPU 바쁜 것 4개 (certify 2 + 종류 g+p ip10_1 종류 1 + 16칸 g+p 1 worker RSS 9.2 GB), pod 14.5 GiB / 60, oom_kill 0. r3a RL it 50 (603 s, OOM 0). 새로 끝난 실험 없음 → RESULTS 재생성 없음.
- 결정: 새 실험·코드 변경 없음 (라운드 23과 같은 이유: CPU 4칸 꽉 참, 다음 판단은 r3a RL `done:` ~05:15 뒤). DAILY는 01:50 작성 → 다음은 10/30 01:50 무렵.
- 다음 라운드: 라운드 22와 같음.

## 라운드 23 (04:07~ UTC, 라운드 22 직후라 짧음)
- 상태 정상: `ensure_running.py` → day2 드라이버·side16_gp 살아 있음, 프로세스 목록에 CPU 바쁜 것 4개 (certify 2 + 종류 g+p 1 + 16칸 g+p 1), oom_kill 0. 새로 끝난 실험 없음 → RESULTS 재생성 없음.
- `analyze_gp_one_type.py`가 `--skip_existing`를 받는 것 확인 (드라이버 CPU 줄의 e0plus 항목이 같은 인자를 씀; side10_base.log의 argparse 오류는 수정 전 흔적).
- 결정: 새 실험·코드 변경 없음 — CPU 4칸이 차 있고, 다음 판단(r3a 10칸 평가, E0+ 열)은 ~05:15 이후 기록이 있어야 가능.
- 다음 라운드: 라운드 22와 같음 (r3a RL `done:` → r3a 10칸 평가 → 종류 2 끝 → `make_type_aware.py` → r3 대비·E0+ 대비 판정).

## 라운드 22 (04:03~ UTC)
- 상태 정상 (드라이버·side16_gp·종류 루프 살아 있음, oom_kill 0). 새로 끝난 실험 없음 → RESULTS 재생성 없음.
- 후보 (c) 검산 루프 코드 확인: 1수 깊이 Woodbury 검산(모든 add/remove의 정확한 V−V′·최악 여유)은 이미 1~3단 입력 + logit prior에 있음 → PROPOSALS 2일차 메모 (c). 새것이 되려면 여러 수 초안 + 모드 재추출이어야 함.
- 결정: (c)는 (a) r3a·r2a 판정 뒤로 미룸 — 한 번에 하나만 바꾸기 규칙, GPU 큐 ~20 h, CPU 4칸이 다 차서 smoke test도 못 돌림.
- 다음 라운드: r3a RL `done:` (~05:15) → 드라이버의 r3a 10칸 평가 → 종류 2 끝 (~05:30) → `make_type_aware.py`로 E0+ 열 채우기 → r3 대비·E0+ 대비 판정.

## 라운드 21 (03:24~ UTC)
- 상태 정상: day2 드라이버·side16_gp 살아 있음, oom_kill 0. side10_base.log의 `--skip_existing` argparse 오류는 이전 체인의 흔적이고, 지금 루프(pid 117899, 종류 0→1→2)는 정상 진행.
- **r3a distill 끝** (03:56): 1477 iter / 0.77 h, best iter 800 (val score 1.858), gpu peak 12.1 GB, OOM 0 (`runs/2026-10-29/r3a_distill/log.txt`). 같은 예산의 r3 distill은 1500 iter, best iter 1400 score 1.743 — val 16문제라 잡음 큼, 판정은 10칸 eval 뒤 (보고서용 아님). r3a RL 03:57 시작 → 끝 ~05:15 예상, 그다음 드라이버가 r3a 10칸 평가 (CPU 줄 최우선).
- **QUEUE B (외부 데이터) 포기, ~15분** → PROPOSALS (e). DevFormer 공개 데이터는 고정 Z 행렬 하나(201점 표, 100 포트, 문제 100개 = 탐침 + keep-out), decap 한 종류 R+C (ESL 0 → SRF 없음), 목표 임피던스 없음, 체크포인트는 저장소에 없음(Google Drive). 우리 판정 `verify.py` 700점 검사(격자 Laplacian으로 임의 f 재계산)를 표 Z에 적용할 수 없고, 정책 입력이 T=3 종류에 묶여 있음.
- 결정: 외부 데이터 어댑터를 만들지 않음 — 새 판정 기준이 필요해 규칙 2 밖이고, QUEUE가 "형식이 안 맞으면 포기"를 허용. 필요한 것 목록은 PROPOSALS (e)에 (사람 결정).
- 다음 라운드: r3a RL 진행·OOM·`done:` → r3a 10칸 평가 기록 → 종류별 g+p 1·2 끝 (~05:30) → E0+ 표 → r3 대비·E0+ 대비 판정 → RESULTS 재생성.

## 라운드 20 (02:59~ UTC) — QUEUE 2일차 지시 + 03:15 수정 반영
- **지금 도는 것:** GPU `r3a` = 3단 loop + `--fc_bias` (QUEUE 03:15 3번: `dec_from_mode`에만 편향 −|log f_k − log SRF_t|/τ_h, τ head마다 학습, init 1 decade; 나머지·COMMON·seed·distill 0.75 h + RL 1.25 h는 r3와 같음), 03:10 시작 (`runs/2026-10-29/r3a_distill/log.txt`). 큐: r3a → r2a (2단 + 같은 편향) → bp_park. CPU: ip16_1 풀 (11.9 h째) + 10칸 종류 하나 g+p (종류 0→1→2, `agent/logs/side10_base.log`) + 학습 certify 2 = 4.
- **새 드라이버 `scripts/pipeline_day2.py`** (`agent/logs/pipeline_day2.log`): `agent/day2_queue.json`을 매 단계 다시 읽음 → GPU 줄 학습 하나씩 (CERTIFY_WORKERS=2), CPU 줄 1 worker: 끝난 학습의 10칸 평가 → "cpu" 항목 순서 (E0+ 10칸 → 16칸을 풀 파일 하나씩: E0+ 종류 0·1·2 + rl 사다리 4단 → 1일차 남은 10칸 r3@R·distill). `ensure_running.py`는 day2 드라이버 + side16_gp (+ 16칸 풀 파일이 없으면 build_pools 1 worker).
- 결정: **1일차 드라이버·side_evals 멈춤** (03:02) — 학습에 certify 2칸이 필요하고 한도는 4. 버린 진행분: r0 rl 16칸 job ~20분, r3 @R4 10칸 job 수 분 (같은 명령이 day2 큐에 있음).
- 결정: **CERTIFY_WORKERS 2 유지** (QUEUE는 2 또는 1 허용) — 1로 하면 같은 시간 예산의 RL iter 수가 r3·r2와 달라져 "편향 하나만 바꾼" 비교가 흐려짐. 대신 16칸 평가는 풀 생성이 끝나 그 칸이 비면 (side16_gp: 16칸 g+p → modal_rule) 그리고 E0+ 10칸이 끝나면 (CPU 줄) 1 worker로 이어짐.
- 결정: 03:04에 시작한 `a2_fc` (2단 + 편향 + 모드←decap 역방향 attention 추가 = 두 가지 변경)는 QUEUE 03:15 수정과 달라 6분 만에 멈춤 → `runs/2026-10-29/a2_fc*_killed_031014_spec_changed`. 역방향 코드는 지움.
- **E0+ 종류 인식 greedy (정식):** `scripts/make_type_aware.py` (종류별 g+p 기록에서 문제마다 성공 해 중 최소 개수, 셋 다 실패면 실패, 시간은 종류 합) → 보통 eval 기록 `runs/2026-10-29/baselines_e0plus/eval/`. `collect_results.py`가 먼저 실행; 요약표·종류별 표에 "개수÷종류 인식 greedy" 열, 새 표 "판 종류마다 맞는 decap 종류" + 숫자에서 자동 생성되는 한 문장 (QUEUE 4번). 10칸 종류 1·2 기록이 끝나면 자동으로 채워짐 (/tmp 가짜 기록으로 선택 규칙·표 확인).
- **B-Park 구현** (`baselines_ml/park_am.py`, `train.py --model park_am --baseline rollout --imit 0`): 좌표 입력 transformer 인코더 + 문맥 glimpse + C·tanh 포인터 (채널마다 query), RL = frozen copy greedy rollout 기준선. CPU smoke 통과. RESULTS "선행 연구 재구현" 절 + 원 논문과 다른 점 4줄 (기록이 생기면 표시). QUEUE대로 r3a 결과 뒤에 학습 (큐 3번째).
- **B-Manoharan 구현** (`baselines_ml/manoharan.py`, `--model manoharan`): 포트 순서 = 최악 탐침에서 본 loop L (포트 단락, 맨판 최악 위반 주파수 f0 한 점) 오름차순, 행동 = 현재 포트에 종류 또는 빈칸, 상태 = 고정 특징 주파수 |Z|/Zt MLP + 포트 L 임베딩, decap 토큰 + 빈칸 토큰 attention 점수, 갱신은 env rank-1. 선생님 해를 고정 순서로 다시 걷는 궤적으로 distill. 확인: val ip10_0·pcb10_0 32문제에서 고정 순서 재생이 선생님 개수와 같게 성공 (1.00, 개수 비 1.000), CPU 학습 smoke (distill·rl 3 iter) rc 0. 원 논문과 다른 점 4줄은 RESULTS "선행 연구 재구현" 절에 (기록이 생기면 표시).
- **짝지은 bootstrap 95% CI** (`collect_results.py`, 2000회, seed 0): 종류 인식 greedy 대비 + g+p 대비 성공률 차·개수 비. 예: g+p 대비 ip10 3단 rl 성공률 +0.58 [+0.44, +0.70], 개수 비 0.736 [0.578, 0.887]; pcb10 3단 rl −0.06 [−0.20, +0.08], 0.845 [0.738, 0.949].
- 결정: 사다리 seed 1·2 학습 8개를 GPU 큐 끝에 미리 넣음 (QUEUE 3번; GPU가 놀지 않게). 최종 후보 seed는 r3a/r2a 판정 뒤 추가.
- 다음 라운드: r3a distill `done:` → RL 진행·OOM → r3a 10칸 평가 (CPU 줄 최우선, ~05:30 이후 시작 예상) → 10칸 종류별 g+p 끝 (~05:30) → 종류 인식 greedy 표 확인 → r3 대비·종류 인식 greedy 대비 판정 → RESULTS.

## 라운드 19 (02:22~ UTC)
- 상태 정상: 드라이버 살아 있음 (16칸 r0 rl 평가 1 worker, RSS ~10 GB, 첫 기록 아직), side_evals 0단 rl 10칸 4/6, ip16_1 풀 3.3 h째, pod mem 21 GiB / 60, oom_kill 0.
- **ip g+p 실패 원인 확인 (기록된 occ, 학습 없음)** — `scripts/analyze_ip_swap.py` → `runs/analysis/ip_swap.json`: g+p 실패 30문제에서 **위치는 그대로 두고 모든 decap을 종류 0 (100 pF)으로 바꾸면 30/30 통과** (최악 중앙값 0.42 Zt), 종류 1·2로 바꾸면 0/30 (1.66, 4.57). 3단 rl 해(29개)도 모두 종류 1로 바꾸면 0/29, 종류 2로 0/29. → 실패는 위치가 아니라 **종류 선택** 때문. g+p 기록은 cutting-plane 0 round → g+p가 자기 작업 격자에서도 불가능으로 끝남 (격자 문제 아님).
- **g+p를 종류 0만 쓰게 하면** (`scripts/analyze_gp_one_type.py`, 같은 탐욕 규칙에 `only` 제한, 같은 cutting-plane·정확 검사; 분석용) ip10 50/50 성공, 개수 7.92 / 8.68 — **3단 rl과 같은 문제에서 개수 비 rl/종류0 = 1.03, 1.00**, 시간 0.44 / 0.14 s/문제. 즉 ip에서 "정책이 g+p보다 성공률이 높다"(0.98 vs 0.40)는 **g+p 기준선의 종류 선택 규칙(지금 최악점을 가장 많이 낮추는 소자)이 약해서**이고, 종류만 고정한 단순 규칙이 정책과 같은 수준. 보고서 주장 범위에 직접 영향 → 그림 2·3 해석에 반드시 적을 것.
- 결정: 이 단일 종류 g+p는 **분석용으로만** 기록 (`runs/analysis/`, RESULTS 별도 표) — 보고 기준선을 바꾸는 것(예: "종류별 g+p 중 최선"을 E0에 추가)은 사람 결정 → PROPOSALS (d). 지금은 pcb·pkg 10칸에도 종류 0·1·2 각각 돌려 ip만의 현상인지 확인 중 (`agent/logs/side10_base.log`, 1 process; pcb10_0 종류 0만으로 0.04 성공). 이어서 같은 칸에서 10칸 modal_rule·LS 기준선 평가 (드라이버 E0 순서에 있던 것, 같은 명령·같은 기록 파일).
- **10칸 rl 사다리 4단 모두 끝** (각 50문제, 성공률 / 개수÷g+p): ip 0단 1.00/1.05 · 1단 0.96/0.88 · 2단 1.00/0.74 · 3단 0.98/0.74; pcb 0.96/0.80 · 0.98/0.75 · 0.88/0.88 · 0.84/0.84; pkg 0.76/0.79 · 0.68/0.75 · 0.72/0.88 · 0.88/0.84. 성공률에 단조 추세 없음; ip 개수만 0→2단에서 줄어듦 (1.05 → 0.74). 0단도 ip 1.00 → ip의 높은 성공률은 사다리 구조와 무관 (위 종류 선택 분석과 맞음). side_evals는 3단 @R2/4/16 10칸 진행 중.
- **16칸 첫 rl 기록** (r0 rl `eval_pkg16_1`, 2202 s / 25문제, 성공 0.72, 개수 7.94, 18.97 s/문제): **`peak_rss_mb` 23087 — chunk 5로도 ~22 GB 그대로**. 정점은 정책 rollout이 아니라 job 안 다른 곳 (16칸 F 390 격자의 25문제 bank/격자 추가 쪽으로 추정, 확인 안 함; certify는 문제마다 256 MB 조각이라 아님). ps로는 대부분 ~10 GB, 정점은 잠깐.
- 결정: **W16 = 1 유지** — 23 GB × 2 + 풀/side 칸이면 45 GiB 한도를 넘을 수 있음 (QUEUE). 16칸 rl job ~37~95 min → rl 4단 × 6풀 ~1일 (계획용).
- 결정: 4번째 CPU 칸은 위 분석·10칸 기준선에 씀 (풀 1 + 드라이버 1 + side_evals 1 + 이것 1 = 4). 처음에 10칸 LS를 띄웠다가 분석 스크립트가 5번째가 되지 않도록 2분 만에 멈추고 한 줄(같은 칸)로 합침.

## 라운드 18 (02:03~ UTC)
- 상태 정상: 드라이버 살아 있음 (16칸 r0 rl 평가, 1 worker, chunk 5, 02:00 시작), pod mem 17 GiB / 60, oom_kill 0. side_evals: **1단 rl 10칸 끝** (02:04) → 0단 rl 10칸 중. side16_gp는 ip16_1 풀 대기.
- 10칸 rl 사다리 (RESULTS.md 재생성, 각 50문제, 성공률 / 개수÷g+p): 1단 ip 0.96/0.88 · pcb 0.98/0.75 · pkg 0.68/0.75, 2단 1.00/0.74 · 0.88/0.88 · 0.72/0.88, 3단 0.98/0.74 · 0.84/0.84 · 0.88/0.84. **단이 올라갈수록 좋아지는 단조 추세는 아님** (pkg 성공률만 1→3단 증가, pcb는 감소) — 0단·distill 기록 뒤 그림 3에서 판단.
- **그림 1 초안** `reports/fig1_board_draft_r3rl_side10.png` (3단 rl, 10칸 `_0` 파일 세 종류, `--pick median` = 성공 문제 중 개수 중앙값: pcb #4 7개 0.985, pkg #11 5개 0.793, ip #13 8개 0.990). `fig_board.py` 수정: 종류마다 decap 라이브러리가 달라 범례를 행마다 (전에는 PCB 값만 보여 pkg·ip 행이 틀린 값), 행 제목에 문제 번호·방법, 실패면 "(fails)", `--title`.
- **고찰 그림** `reports/fig_ip_gp_fail_ip10_0_13.png`: 같은 ip10_0 #13에서 g+p 74개 (대부분 5 pF) → 정확 검사 최악 1.703 Zt (실패), 3단 rl 100 pF 8개 → 0.990. 두 해 모두 맨판의 ~3.3 GHz 공진은 누르지만 ~1.2 GHz에 반공진 피크가 생김; g+p는 이 피크가 1.70, 정책은 0.99. 즉 g+p의 실패는 "판을 다 채워도 못 누르는 새 피크"이고, 원인(종류 선택 순서 vs 반공진 감쇠)은 아직 미확인 — 단정하지 않음.
- **ip g+p 실패 30문제 전체** (`scripts/analyze_ip_gp_fail.py` → `runs/analysis/ip_gp_fail.json`, RESULTS.md 새 표): g+p 최악점은 **30/30 모두 맨판 최악 공진보다 낮은 주파수** (비 중앙값 0.36, 0.28~0.45; 최악 중앙값 1.83 Zt) = decap이 만든 새 반공진. 같은 문제를 통과한 3단 rl(29개)과 **총 정전용량은 비슷** (g+p/정책 중앙값 0.95) → "용량 부족"이 아니라 구성(5 pF 다수 vs 100 pF 소수)의 차이. 정책 해도 최악점이 같은 대역(0.4~1.8 GHz)에 0.90~1.00으로 붙어 있는 경우가 많음 (겨우 통과). 감쇠(ESR)·배치 위치 중 무엇이 피크 높이를 정하는지는 아직 미확인.
- 결정: 그림은 기록된 occ로만 다시 계산 (학습·평가 조건 변경 없음, 10칸 100셀 1스레드 수 초). 그림 1 최종본은 16칸 기록이 생기면 같은 스크립트로.
- 02:19: 16칸 r0 rl 평가 worker (chunk 5) 18분째 RSS 9.4 GB, pod 19.5 GiB, 첫 기록 아직 없음 (16칸 정책 job ~1.4 h → ~03:30 예상). W16은 기록의 `peak_rss_mb`를 보고 정함 (그대로 1).
- 다음 라운드: r0 rl 16칸 첫 기록 `peak_rss_mb` < ~14 GB이면 `pipeline_day1.py`의 `W16 = 2` + 드라이버 재시작 (ATTACH) → side_evals 0단 rl 10칸 → RESULTS 재생성 (10칸 rl 사다리 4단 완성) → ip16_1 풀 완성·side16_gp 시작 확인 → 그림 3 10칸 초안 (`fig_results.py`).

## 라운드 17 (01:22~ UTC, 10/29 첫 라운드 → DAILY.md 작성)
- 상태 정상: 드라이버 살아 있음, pod mem 31 → 38.7 GiB / 60 (16칸 평가 worker 둘 10.3~11.0 GB), oom_kill 0. GPU 비어 있음.
- **10칸 3단 rl 평가 끝** (side_evals, 6 파일 150문제): 성공률 / 개수÷g+p — ip 0.98 / 0.74, pcb 0.84 / 0.84, pkg 0.88 / 0.84 (g+p: 0.40 / 0.90 / 0.90). ip에서 g+p 실패 30문제는 판 100칸 중 60~99개를 채우고도 실패 (최악 |Z| 1.2~3.6 Zt), 3단 rl은 그중 29개를 3~26개로 통과. 시간/문제 rl 0.44~3.09 s vs g+p 2.2~4.9 s (`runs/2026-10-28/r3_loop_rl/eval/`, `.../baselines/eval/`). 이어서 2단 rl 10칸 평가 중.
- **`collect_results.py` 버그 수정**: (종류, 크기)마다 방법 기록 하나만 남아 `_0` 파일이 `_1`로 덮였고 (표가 25문제), 한 방법이 `_0`만, g+p가 `_1`만 있으면 **다른 문제끼리** 개수 비를 짝지을 수 있었음. 이제 파일들을 합치고 짝은 같은 파일 안에서만, 표에 문제 수 표시. (g+p pcb10 성공률 0.80 → 0.90 = 두 파일 평균.)
- **드라이버 순서 변경** (01:36 재시작, r0 distill 평가·풀 생성 ATTACH): 16칸 정책 평가 1개가 75분 넘게 기록 0 → 단 평가는 16칸 풀만, **rl 4단 먼저 → distill → 3단 @R** (10칸은 side_evals가 함). 이전 드라이버의 2-worker 평가가 살아 있으면 끝날 때까지 기다림 (한도 4 유지). mock 실행으로 순서 확인.
- **`scripts/side16_gp.py`** (01:36 시작, `agent/logs/side16_gp.log`): 풀 생성(ip16_1, 마지막 1 worker)이 끝나면 그 칸에서 16칸 g+p → modal_rule, 1 worker. 이유: g+p는 개수 비의 기준인데 드라이버 순서로는 16칸 g+p가 하루 넘게 뒤에 있었음.
- 결정: 순서만 바꿈, 실험 조건 그대로 (같은 eval_policy·풀·rounds·1스레드). 돌던 r0 distill 16칸 평가는 75분 진행분을 버리지 않으려고 그대로 둠 (끝까지 ~수 시간).
- DAILY.md 첫 작성 (10/29).
- **ip에서 g+p가 실패하는 이유 (기록된 occ, 학습 없음)**: `collect_results.py`에 decap 종류 비율 표 추가. ip 10칸 g+p 실패 30문제는 평균 78.6개, 그중 0.90이 종류 2 (5 pF / 1 pH / 0.2 Ω, ESL 가장 작음); 3단 rl 성공 49문제는 평균 8.4개, 0.99가 종류 0 (100 pF / 5 pH). 탐욕 규칙이 지금 최악 주파수를 가장 많이 낮추는 ESL 최소 소자를 계속 고르다 판을 채움 → 보고서 고찰 후보 (RESULTS.md "decap 종류 비율").
- 2단 rl 10칸 첫 기록: pkg10_1 성공 0.60 (3단 0.88, g+p 0.88).
- **첫 16칸 정책 기록** (01:47): r0 distill `eval_pcb16_0` 5183 s / 25문제, 성공 0.80, 개수 38.1 (풀 생성 때 g+p 10.8), 마지막 라운드 방법 시간 55.9 s/문제 (1스레드). 계획용: 16칸 정책 job 1개 ~1.4 h → 드라이버의 16칸 단 평가 (8단계 × 6풀 + @R 3 × 6풀) 2 workers로 ~2일. 마감(11/8) 안이지만 20칸은 그 뒤 → 20칸 범위는 사람 결정 (PROPOSALS (c)).
- r0 distill 16칸 두 번째 기록: `eval_pkg16_1` 5703 s, 성공 0.80, 개수 31.15.
- **메모리 사고 (OOM 직전)**: 01:48 pod mem **51.6 GiB / 60** (anon 55 GB). 16칸 정책 평가 worker가 job 하나 안에서 ~10 → **22 GB**까지 커짐 (25문제를 한 rollout에), 둘이 동시에 정점이면 44 GB + 풀 생성 6 GB + side 3 GB. 조치: 드라이버 정지 → r0 distill 평가 정지 (기록 2개 저장 직후, 새 job 둘은 ~8분·1분 진행분 버림) → `eval_policy.py` 수정: (1) `--inst_chunk N` (정책 greedy rollout을 N문제씩), (2) `Pool(maxtasksperchild=1)` (job마다 새 프로세스 → 정점 메모리가 다음 job에 남지 않음), (3) 기록에 `peak_rss_mb`, `inst_chunk`. **동치 확인**: 3단 rl `eval_pkg10_1`을 chunk 5로 다시 → ok·count·worst·occ·rounds·F 모두 기존 기록과 같음 (`runs/chunk_equiv/equiv_check/`, peak 4.5 GB). 드라이버 02:00 재시작: 16칸 단 평가 `--inst_chunk 5`, **1 worker** (`W16`). 02:01 pod mem 10.4 GiB.
- 결정: 16칸 정책 평가는 1 worker로 시작 — chunk 5의 16칸 정점은 아직 모름. 첫 기록의 `peak_rss_mb`가 ~14 GB 아래면 다음 라운드에 `W16 = 2`. 그동안 CPU 칸은 3개만 씀 (드라이버 1 + side_evals 1 + 풀 생성/side16_gp 1).
- **관찰: 시간 측정이 부하에 따라 흔들림** — 같은 정책·같은 풀·같은 해인데 문제당 3.09 s (01:17, CPU 4개 바쁨) vs 1.23 s (01:57). 그림 2의 시간은 지금 기록(평가 중 다른 프로세스와 같이 돈 것)을 그대로 쓰면 안 됨 → 평가가 끝난 뒤 조용한 상태에서 같은 조건으로 따로 시간만 재는 단계가 필요 (계획에 추가, 실험 조건 변경 아님).
- 다음 라운드: r0 rl 16칸 첫 기록의 `peak_rss_mb` → W16 2로 올릴지 결정 → side_evals 2·1·0단 rl 10칸 기록 → RESULTS 재생성 → 첫 16칸 정책 기록 시간·ip16_1 풀 완성·side16_gp 시작 확인 → ip g+p 실패 원인 분석 (기록된 occ, 학습 없음) → 그림 1 초안.

## 라운드 16 (23:30~01:20 UTC)
- 상태 정상: 드라이버 살아 있음, pod mem 13.4 → 26.3 GiB / 60, oom_kill 0.
- **r3 (3단 loop) RL 끝** (00:19): 352 iter / 1.27 h, best iter 100 (val score 1.679), cuda_oom_skipped 0, chunk 2, gpu peak 17.1 GB, 선생님 교체 296, λ 1.42, grid capped 151 step (`runs/2026-10-28/r3_loop_rl/metrics.json`). 최종 val (보고서용 아님): pcb 0.99/fail 0.81, pkg 1.00/0.50, ip 0.69/0.19 → best는 iter 100. **사다리 0~3단 학습 모두 끝** (모두 distill 0.75 h + RL 1.25 h, 같은 인자).
- RL best val score (낮을수록 좋음, val 16문제라 잡음 큼, 보고서용 아님): 0단 1.999 → 1단 1.737 → 2단 1.691 → 3단 1.679.
- eval 풀: `eval_ip16_0` 9187 s, `eval_ip10_1` 305 s 완성. 남은 것 `eval_ip16_1` 하나 (worker 1개).
- 드라이버 early evals 시작 (00:20): r0 distill부터 이미 만든 풀 11개, 2 workers. 16칸 정책 평가 1개가 30분 넘게 걸림 (00:49에 아직 기록 0) → 드라이버 순서(정책마다 16칸 먼저)로는 10칸 사다리 표가 오래 비어 있음.
- **4번째 CPU 칸 사용** (`scripts/side_evals.py`, 00:40~, `agent/logs/side_evals.log`): 10칸 eval 6개에 greedy+prune → 0~3단 rl → 3단 @R2/4/16 → 0~3단 distill 순서, 1 worker. 같은 `eval_policy.py`·같은 기록 파일이라 드라이버는 이미 있는 기록을 건너뜀. 첫 기록: `eval_pkg10_1` greedy+prune 513 s (ok 0.88).
- 결정: 4번째 칸은 평가에만 씀 (풀 1 + 드라이버 평가 2 + side 1 = 4, QUEUE 한도 그대로; 20칸 풀 생성이 시작되면 side는 멈춤). 실험 조건은 안 바꿈. GPU는 비어 있지만 학습에 필요한 CPU 검사 칸(2개)이 없어 새 학습은 안 띄움. 정책 평가는 CPU 1스레드 그대로 (그림 2 시간 비교 조건 "같은 계산기").
- 01:17 side: 10칸 greedy+prune 6개 끝 (2168 s, `runs/2026-10-28/baselines/eval/`; 정확 검사 ok pcb 1.00·0.80 / pkg 0.92·0.88 / ip 0.40·0.40 — ip는 cutting-plane 0 round, 풀 생성 때 g+p ok와 같음). 이어서 3단 rl 10칸 평가 중. pod mem 최고 35.2 GiB (16칸 평가 worker 둘이 커짐), oom_kill 0.
- 드라이버 16칸 정책 평가 (r0 distill): 01:20에 60분째 기록 0 → 16칸은 정책 1개 × 풀 6개가 몇 시간 단위. 11개 정책이면 16칸 단 평가만 하루 이상일 수 있음 (계획용 관찰, 숫자는 첫 기록 뒤 다시 계산).
- 다음 라운드: 첫 16칸 정책 평가 시간 기록 → 그걸로 16칸 남은 시간 계산, 너무 길면 16칸은 rl 4개 먼저 (distill·@R은 뒤로) 순서만 바꾸는 것 검토 → 10칸 사다리 RESULTS.md → 05:30 무렵 첫 DAILY.md.

## 라운드 15 (23:25 UTC~, 라운드 14 직후라 짧음)
- 상태 정상: 드라이버 살아 있음, pod mem 11.7 GiB / 60, oom_kill 0. GPU 18.6 GB (r3 RL).
- r3 (3단 loop) RL iter 100: OOM 0, gpu peak 14.7 GB, grid capped 7, resume.pt iter 93. iter 100 val (보고서용 아님): pcb 0.99/fail 0.44, pkg 0.98/0.62, ip 0.69/0.12, val score 1.679 (new best; 0~2단 RL best 1.999 / 1.737 / 1.691보다 낮음, val 16문제라 잡음 큼) (`runs/2026-10-28/r3_loop_rl/log.txt`).
- **드라이버 CPU 줄 수정** (`scripts/pipeline_day1.py`): r3 RL이 끝나면 (~00:25) GPU 줄의 certify 2개가 비는데, 옛 순서는 ip16 ×2·ip10_1 풀이 다 끝날 때(~05시 예상)까지 아무 평가도 안 해서 CPU 2칸이 ~4.5 h 놀았음. 이제 풀 생성은 별도 스레드로 두고, **모든 단 학습이 끝나면 이미 만든 eval 풀 9개**(pcb·pkg 10/16 각 2, ip10_0)에서 단 평가 (r0~r3 distill·rl, r3 @R2/4/16) → E0 gp/modal_rule을 먼저 돌림. 평가 기록은 파일이 있으면 건너뛰므로 뒤의 원래 순서는 남은 풀만 채움. 동시 CPU 프로세스는 풀 2 + 평가 2 = 4 (QUEUE 한도 그대로). `eval_policy.py --sets`는 쉼표로 여러 glob 허용. mock 실행으로 순서 확인, 드라이버 23:27 재시작 (풀 생성·r3 학습 ATTACH, 끊김 없음).
- 결정: 평가 순서만 앞당김 — 실험 조건(풀·검사·rounds·workers)은 그대로라 보수적 변경. 풀 파일은 .json이 써진 것만 (build_pools가 .pkl 다음에 씀) 골라 반쯤 쓴 풀을 읽지 않음.
- 관찰 (계획용, 보고서 숫자 아님): eval 풀 생성 시간(cutting-plane 안 g+p, 16칸 2.8~5.1 h/풀)으로 보면 16칸 E0 gp·LS는 풀마다 몇 시간씩 걸림 → 16칸 E0 전체가 10/29 하루 이상 걸릴 수 있음. 정책 평가는 롤아웃이 싸서 훨씬 빠를 것 (val 48문제가 학습 중 수십 초) → 단 평가를 E0보다 먼저 둔 이유.
- 다음 라운드: r3 RL `done:`(~00:25)·`cuda_oom_skipped` → 드라이버 로그에 `early evals while pools build` 시작 확인 → 첫 단 평가 기록 → RESULTS.md 평가 표 → 10/29 05:30 무렵 첫 DAILY.md.

## 라운드 14 (22:32~23:24 UTC)
- 상태 정상: 드라이버 살아 있음, pod mem 18.0 → 12.6 GiB / 60, oom_kill 0. CPU 프로세스 ≤4 → 새 실험 안 띄움.
- **r3 (3단 loop) distill 끝** (23:05): 1500 iter / 0.77 h, best iter 1400 (val score 1.743), gpu peak 12.1 GB, OOM 0. 같은 시간 예산이라 iter 수가 0단(2192)·1단(2109)보다 적음 (loop가 블록을 R번 돌려 ~1.6 s/iter) — 사다리 조건(시간 예산)은 같고 iter 수 차이는 metrics.json·RESULTS.md에 기록됨. best val (보고서용 아님): pcb 0.91/fail 0.50, pkg 1.00/0.56, ip 0.69/0.25 (`runs/2026-10-28/r3_loop_distill/log.txt`).
- **r3 RL 시작** (23:06): iter 75까지 OOM 0, gpu peak 13.3 (iter 25) → 14.6 GB (iter 75), ~13 s/iter, grid capped 3, 선생님 교체 134, λ 1.16 (`runs/2026-10-28/r3_loop_rl/log.txt`). 0~2단 RL과 같은 메모리 추세 → 24 GB 안에서 끝날 것으로 봄 (확인은 `done:`의 `cuda_oom_skipped`). 끝 예상 ~00:25.
- **eval 풀 `eval_pkg16_0` 완성** (18523 s = 5.1 h, F 284, g+p ok 0.64, count 25.9; `agent/logs/build_pools_v1_r3.log`) — 16칸 중 가장 김 (pcb16 2.8 h, pkg16_1 3.1 h). 남은 것: ip16 ×2, ip10_1.
- RESULTS.md 재생성 (r3 distill 행 추가, eval 기록 0개 — 풀 대기).
- 다음 라운드: r3 RL `done:` (~00:25)·`cuda_oom_skipped` 기록 → ip16 풀 시간 → 풀 다 끝나면 E0·단 평가 시작 확인 → 10/29 05:30 무렵 첫 DAILY.md.

## 라운드 13 (22:26 UTC~, 라운드 12 직후라 짧음)
- 상태 정상: 드라이버 살아 있음, pod mem 19.5~24.3 GiB / 60, oom_kill 0. CPU 프로세스 4 (풀 worker 2 + certify 2) → 새 실험 안 띄움.
- r3 (3단 loop) distill iter 425, gpu peak 12.1 GB 그대로, OOM 0, ~1.6 s/iter. iter 400 val (보고서용 아님): pcb 0.94/fail 0.75, pkg 1.17/0.88, ip 0.69/0.19, val score 2.139 (new best; iter 100 2.393) (`runs/2026-10-28/r3_loop_distill/log.txt`). 0~2단 distill과 같이 val fail이 들쭉날쭉.
- eval 풀: 새로 끝난 것 없음 (pkg16_0 ~4.2 h째, ip16 진행 중).
- 다음 라운드: r3 distill `done:`(~23:05)·RL 시작·RL gpu peak 확인 (loop는 블록을 R번 반복해 RL activation이 2단보다 클 수 있음; OOM이면 train.py가 step을 버리고 chunk를 반으로 줄임 → `chunk_final` 기록) → pkg16_0 완성 시간.

## 라운드 12 (22:22 UTC~, 라운드 11 직후라 짧음)
- 상태 정상: 드라이버 살아 있음, pod mem 17.6~19.0 GiB / 60, oom_kill 0. CPU 프로세스 4 (풀 worker 2 + certify 2) → 새 실험 안 띄움.
- r3 (3단 loop) distill iter 225, gpu peak 12.1 GB, OOM 0, ~1.6 s/iter. iter 200 val (보고서용 아님): pcb 3.13/fail 0.94, pkg 1.00/0.69, ip 0.69/0.25 (`runs/2026-10-28/r3_loop_distill/log.txt`).
- eval 풀: 새로 끝난 것 없음 (pkg16_0, ip16 진행 중).
- `scripts/fig_results.py` 수정: 기록을 풀 파일(set)별로 묶어 g+p와 **같은 파일끼리만** 짝지어 개수 비 계산. 이전 코드는 한 방법이 _0/_1 중 하나만 평가됐으면 배열 길이가 달라 그림이 죽었고, 같은 파일을 다시 평가하면 두 번 셌음. /tmp 가짜 기록으로 그림 2·3 실행 확인 (가짜 숫자는 runs/에 안 씀).
- 다음 라운드: r3 distill `done:`(~23:05)·RL 시작·OOM 확인 → pkg16_0 완성 시간 기록.

## 라운드 11 (21:51 UTC~)
- 상태 정상: 드라이버 살아 있음, pod mem 14.6~20.2 GiB / 60, oom_kill 0. 동시 CPU 프로세스 4 (풀 worker 2 + certify 2) → 새 실험 안 띄움.
- **r2 (2단 mott) 학습 끝** (22:14, rc 0): RL 354 iter / 1.27 h, best iter 300 (val score 1.691), cuda_oom_skipped 0, chunk 2, gpu peak 16.8 GB, 선생님 교체 297, λ 1.52, grid capped 149 step (`runs/2026-10-28/r2_mott_rl/metrics.json`). RL 최종 val (보고서용 아님): pcb 0.88/fail 0.44, pkg 1.00/fail 0.62, ip 0.69/fail 0.19. 2단 distill best (iter 1300, score 1.759)보다 RL이 val score로는 나음 — 0·1단(RL이 distill보다 못하거나 비슷)과 다른 첫 단. 단 val 16문제라 잡음 큼, 결론은 eval 풀 평가 뒤.
- RL best val score (낮을수록 좋음, 보고서용 아님): 0단 1.999 → 1단 1.737 → 2단 1.691. 세 단 모두 같은 인자 (teacher gp, chunk 2, rl 1.25 h, git hash만 다름).
- **r3 loop smoke 통과** (22:14~22:16, `agent/logs/smoke_loop.log`, 5단계 모두 OK) → **r3 (3단 loop) 학습 시작** 22:16. distill iter 100: gpu peak 12.1 GB, ~1.5 s/iter, OOM 0 (`runs/2026-10-28/r3_loop_distill/log.txt`). 끝 예상 ~00:20 (distill 0.75 h + RL 1.25 h + 로딩).
- eval 풀: `eval_pkg16_0`은 22:25에도 진행 중 (~4.1 h째, 지금까지 16칸 중 가장 김), 다른 worker는 21:16부터 다음 풀(ip16) 진행. 남은 것 pkg16_0, ip16 ×2, ip10_1.
- RESULTS.md 재생성 (r2 RL 행 추가, eval 기록 0개 — 풀 대기).
- 다음 라운드: r3 distill `done:`·RL 시작·OOM 확인 → pkg16_0 완성 시간 기록 → 10/29 05:30 무렵 첫 DAILY.md.

## 라운드 10 (21:15 UTC~)
- 상태 정상: 드라이버 살아 있음, pod mem 20.8 GiB / 60, oom_kill 0, GPU 17.9 GB 사용 (r2 RL). 동시 CPU 프로세스 4 (풀 worker 2 + certify 2) → 새 실험 안 띄움.
- **eval 풀 `eval_pkg16_1` 완성** (11319 s = 3.1 h, F 298, g+p ok 0.68, count 16.2; `agent/logs/build_pools_v1_r3.log`). `eval_pkg16_0`은 21:38에도 진행 중 (~3.6 h째, pcb16 2.8 h보다 김). 남은 풀: pkg16_0, ip16 ×2, ip10_1 → 16칸 풀 3.1 h 실측 기준 10·16칸 전체 끝은 10/29 01시 전후, 그다음 E0·단 평가.
- r2 (2단 mott) RL: iter 100까지 OOM 0, gpu peak 15.5 GB, ~14.4 s/iter, grid capped 8. iter 100 val (보고서용 아님): pcb 0.83/fail 0.62, pkg 0.97/fail 0.69, ip 0.69/fail 0.12, val score 1.788 (new best) (`runs/2026-10-28/r2_mott_rl/log.txt`). 0·1단과 같은 양상 (학습 배치 fail 0.01~0.23 vs val fail 0.12~0.69).
- 다음 라운드: r2 RL `done:` (~22:15) 기록 → r3 loop smoke 통과·학습 시작 확인 (`agent/logs/smoke_loop.log`) → pkg16_0·ip16 풀 시간 기록.

## 라운드 9 (19:56 UTC~)
- 상태 정상: 드라이버 살아 있음, pod mem 18.9 → 33.8 GiB / 60 (21:13, 16칸 풀 worker가 커지는 중), oom_kill 0. 동시 CPU 프로세스 = 풀 worker 2 + certify 2 = 4 (QUEUE 한도 꽉 참 → 새 실험 안 띄움).
- **r1 (1단) 학습 끝** (20:57, rc 0): RL 413 iter / 1.27 h, best iter 200 (val score 1.737), cuda_oom_skipped 0, chunk 2, gpu peak 17.2 GB, 선생님 교체 319, λ 1.64, grid capped 217 step, 27개 풀 모두 F 상한 1.5x 도달 (`runs/2026-10-28/r1_modal_rl/metrics.json`). RL best val (iter 200, 보고서용 아님): pcb 0.89/fail 0.44, pkg 0.92/fail 0.56, ip 0.66/fail 0.38 vs 1단 distill best (iter 600: 1.06/0.38, 1.10/0.44, 0.73/0.31) → 개수는 줄고 fail은 늘어남, val score로는 distill(1.714)이 RL(1.737)보다 약간 나음. r0와 같은 패턴 (작업 격자 과적합 의심). 평가는 드라이버가 distill·rl 둘 다 eval 풀 완성 후 실행.
- **r2 (2단 mott) RL 시작** (20:58): iter 50 OOM 0, gpu peak 13.8 GB, ~16.5 s/iter (0·1단 ~12 s보다 느림 → 1.25 h에 ~270 iter 예상; iter 수는 metrics.json에 기록). 끝 ~22:15 → 그다음 r3 smoke+학습.
- RESULTS.md 재생성 (`collect_results.py`, eval 기록은 아직 0개 — 10/16칸 풀 대기).
- 다음 라운드: r2 RL `done:` 기록 → r3 loop smoke 통과·학습 시작 확인 (`agent/logs/smoke_loop.log`) → pkg16 풀 완성 시간 기록.
- r1 (1단 modal) RL: iter 150까지 OOM 0, gpu peak 13.1 → 16.2 GB (r0와 같은 추세), ~11.8 s/iter, `grid capped` 1 (iter 75) → 29 (iter 150). iter 100 val (보고서용 아님): count/g+p pcb 0.92 fail 0.69 / pkg 0.90 fail 0.69 / ip 0.66 fail 0.38 (`runs/2026-10-28/r1_modal_rl/log.txt`). r0 RL iter 100 (fail 0.62 / 0.88 / 0.25)와 같은 양상: 학습 배치 fail 0.01~0.12 vs val 정확 검사 fail 0.38~0.69.
- eval 풀: pkg16 ×2 생성 중 (~18:08 시작, pcb16 실측 2.8 h 기준 ~21:00 전후 예상). 
- 다음: r1 RL `done:` (~21:00) 기록 → r2 RL 시작·OOM 확인.

## 라운드 8 (18:52 UTC~)
- 상태 정상: 드라이버 살아 있음, pod mem 15.5 GiB / 60, oom_kill 0. GPU 줄 r0 RL 진행 중 (18:25~).
- r0 RL (격자 수정 후): iter 225까지 OOM 0, gpu peak 16.6 GB (iter 100 15.2 → 천천히 증가, 상한 1.5x F0이 걸려서 더 안 커질 것), ~13 s/iter. `grid capped` 6 (iter 100) → 73 (iter 225): 상한 뒤 실패가 빠르게 쌓임.
- r0 RL iter 200 val (보고서용 아님): pcb count/g+p 1.00 fail 0.50 / pkg fail 1.00 (성공 0개라 count nan) / ip 0.85 fail 0.25. iter 100 대비 pcb fail 0.62→0.50, pkg 0.88→1.00. 학습 배치의 fail (작업 격자 기준) 0.05~0.21 vs val 정확 검사 fail 0.25~1.00 → **정책 해가 작업 격자에는 맞지만 700점 검사에서 떨어지는 간극**이 RL에서도 그대로. 보고서 고찰 후보 (격자 과적합). λ 1.0→1.62 계속 상승 (목표 0.02).
- eval 풀: `eval_pcb16_0` 9941 s, `eval_pcb16_1` 10392 s에 완성 (g+p ok 0.80·0.76). 지금 pkg16 ×2 진행 → 그다음 ip16 ×2, ip10_1. 실측으로 16칸 시간 추정을 고침 (PROPOSALS.md 갱신: 이전 6~9 h/개 추정 → 실제 ~2.8 h). 10·16칸 풀 전체 끝은 10/29 새벽 예상 → 그때 E0·단 평가 시작.
- **r0 (0단) 학습 끝** (19:41, rc 0): RL 350 iter / 1.27 h, best iter 300, cuda_oom_skipped 0, chunk 2 유지, gpu peak 17.1 GB, 선생님 교체 352, λ 1.79, grid capped 176 step. 풀별 F 처음→끝은 모두 상한 1.5x에 도달 (`metrics.json` `pool_F`, 예: pcb12_0 280→420). RL 최종 val (보고서용 아님): pcb 1.06/fail 0.56, pkg 0.67/fail 0.94, ip 0.77/fail 0.25 — distill 최종 (0.75 / 0.69 / 0.19 fail)과 비교해 pcb만 나아지고 pkg·ip는 나빠짐. 원인 후보 (학습법은 안 바꿈): (1) 작업 격자 과적합 — 상한 뒤엔 격자로 못 고침, (2) λ가 목표 0.02에 못 미쳐 계속 오름, (3) ip 선생님이 대부분 g+p로 떨어진 풀(ip8_x LS 2.1~2.5개 vs g+p 22~30개)에서 교체로 기준이 급변. 학습 기록 표는 RESULTS.md (자동 생성 갱신).
- 기록: r0 RL의 `config.json`/`metrics.json` git hash는 `31d36bc`이지만 실제로는 커밋 전 작업 트리(= `d4241f7`의 격자 수정, `grid_growth` 인자가 args에 있음)로 실행됨. r1·r2 RL은 `d4241f7` 이후 커밋에서 시작.
- **지금:** r1 (1단 modal) RL 19:41 시작, iter 50 gpu peak 13.8 GB, OOM 0 (`runs/2026-10-28/r1_modal_rl/log.txt`). GPU 줄 다음: r2 RL (~21:00) → r3 smoke+학습 (~22:20). CPU 줄: pkg16 ×2 풀 생성 중.
- 다음 라운드: r1 RL `done:` (~21:00) 확인·기록 → r2 RL 시작·OOM 없는지 확인 → pkg16 풀 완성 시간 기록.
- 결정: 아무 설정도 바꾸지 않음 — RL 중간 val이 나빠도 학습법 변경은 금지 (AGENT_PROMPT), r0 RL 끝 체크포인트로 평가한다.

- **RL OOM 원인 찾음 (이전 STUCK 해제, 확인 중):** RL은 정확 검사가 실패할 때마다 풀에 격자점을 추가하는데 (`batch.add_freqs`), `--chunk` 도입 후 chunk마다 따로 호출되어 매번 그 chunk의 가장 긴 목록으로 모든 인스턴스를 패딩 → step당 F가 chunk 수만큼 더 늘어남. 2단 RL(18:10, chunk 2)에서 pkg10_1 F 267→511 (iter 25), gpu peak 18.3 GB, 18:17 메모리 매핑 실패 경고. dense 풀과 rollout 블록 [E,F,n,k]가 모두 F에 비례하므로 chunk를 줄여도 결국 OOM. 즉 "같은 갱신"이라는 라운드 6 설명이 틀렸었음 (격자 증가가 달랐다).
- 수정 3 (`train.py`): (1) 격자점은 step 끝에 한 번만 추가 (chunk 전 1-pass와 같은 동작), (2) `--grid_growth 0.5`: 풀마다 처음 F의 1.5배까지만 (넘는 실패도 정확 검사로 실패 처리 → 보상은 정직), (3) step 안 CUDA OOM이면 그 step을 버리고 (갱신 없음) chunk 반으로, 20번 넘으면 중단. iter 로그에 `grid capped`·`oom`, metrics.json에 `pool_F`(풀별 처음/끝 F)·`cuda_oom_skipped`·`chunk_final`. CPU smoke `scripts/smoke_rl_grid.py` 통과 (OOM 주입 → step 건너뜀·chunk 2→1·F 불변 / 상한 0.5 지킴). 상한 없이도 작은 smoke에서 pcb8_0 F 187→259 (6 iter) → 상한은 필요.
- 2단 RL은 18:20에 멈춤 (12분 진행, 옛 코드라 다른 단과 조건이 달라짐 → `r2_mott_rl_killed_182000_addfreqs`). distill 3개는 그대로. 드라이버 18:25 재시작 → GPU 줄: **r0 RL (18:25~) → r1 RL → r2 RL → r3** (각 RL 1.25 h).
- 확인: r0 RL iter 25 pkg10_1 F=310, gpu peak 13.0 GB → iter 100 peak 15.2 GB, OOM 0, ~14 s/iter (1.25 h ≈ 320 iter), resume.pt iter 84 (`runs/2026-10-28/r0_coord_rl/log.txt`). 상한이 벌써 걸림 (iter 100까지 `grid capped 6`, pcb12_2 F 378/385) → 상한 뒤의 실패는 격자로 고쳐지지 않으므로 best가 아닌 샘플의 작업 격자 보상은 낙관적일 수 있음 (보고서 방법 장에 적을 것).
- r0 RL iter 100 val (보고서용 아님): pcb count/g+p 1.07 fail 0.62 / pkg 1.00 fail 0.88 / ip 0.76 fail 0.25 — distill best (iter 1600: 1.75/0.56, 1.50/0.56, 0.77/0.19)보다 fail은 아직 나음 없음. λ 1.0→1.38 (목표 fail 0.02보다 훨씬 높음), 선생님 교체 180회.
- 다음 라운드: `ensure_running.py` → r0 RL `done:` (~19:45) 확인, metrics.json `pool_F`·`cuda_oom_skipped` 기록 → r1 RL 시작 확인. RL이 val fail을 못 낮추면 (r0 끝에서 distill best보다 나쁘면) 그대로 기록하고 원인 후보(λ·선생님 교체·격자 상한)를 적되 학습법은 바꾸지 않음.
- 2단 distill 끝 (17:22~18:09, rc 0). 중간 val (보고서용 아님: val 풀 종류당 16문제, 정확 검사, checkpoint 고르는 데 쓴 세트): best iter 1300 count/g+p pcb 0.89 fail 0.56 / pkg 1.07 fail 0.56 / ip 0.69 fail 0.19 (`runs/2026-10-28/r2_mott/driver.log`).
- 결정: 상한 0.5 — 풀 9.5 GB × 1.5 ≈ 14.3 GB + chunk 2 rollout이 24 GB 안에 들어가는 쪽. 모든 단의 RL이 같은 값 (r0·r1·r2 RL 모두 새 코드로 처음부터).
- 결정: 2단 RL을 12분 만에 버리고 재시작 — 다른 단과 격자 증가 규칙이 달라지면 사다리 비교가 깨짐.
- **위험: eval 풀 시간.** 16칸 풀 하나 ~6~9 h (1스레드, 판 변 ~6제곱), 6개 2 workers로 ~18~27 h → 10칸 E0·단 평가도 10/29 낮까지 밀림; 20칸은 ~90 h로 사실상 불가. 결정: 풀 생성은 그대로 (멈추면 진행분 손실, 늘리면 QUEUE 한도 4 위반). 사람 선택지 (a)(b)(c)를 PROPOSALS.md "사람 결정이 필요한 것"에 적음.

## 라운드 6 (17:40 UTC)

- **사고 4 (=STUCK): chunk 4도 RL에서 OOM.** 1단 RL 17:11 시작 → 17:15 OOM (할당 17.2 GB + 단편화로 놀던 5.7 GB), 드라이버 순서대로 0단 RL 재실행 17:16 → 17:22 같은 OOM (15.9 + 6.5 GB). 원인: `sim.loaded_state`가 [E,F,n,k] complex128 블록을 step마다 다른 k로 여러 개 잡음 (16 env·F 251·n 144·k 64면 블록 하나 0.6 GB) + 크기가 계속 바뀌어 단편화. 1·0단 distill 체크포인트는 그대로 있음 (`runs/2026-10-28/r{0,1}_*_distill/best.pt`; 1단 distill 2109 iter, best iter 600).
- 수정 2: `--chunk` 기본 2 (16 env = distill의 선생님 replay와 같은 크기, 그건 풀 전체를 올린 채 0·1단 각 0.75 h 문제없이 돌았음) + `train.py`에서 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (단편화 제거). iter 로그·`done:`에 `gpu peak` 표시 추가. CPU smoke (`scripts/smoke_chunk.py`) 통과. 2단 distill이 GPU를 15.4 GB 쓰는 중이라 GPU에서 실제 크기 smoke는 못 함 (남은 8 GB로는 선생님 replay도 안 들어가 대표성 없음 → 지움).
- 드라이버 17:38 재시작 → GPU 줄: r2 distill (ATTACH, 17:22 시작) → r2 RL (~18:10, 새 train.py) → **r0 RL → r1 RL** (distill 재사용) → r3.
- 결정: chunk 2 (1이 아니라) — 16 env는 distill에서 실제로 검증된 크기이고, chunk 1은 RL iter 수를 더 줄임. 모든 단이 같은 값.
- eval 풀: 17:38에도 16칸 첫 파일이 안 끝남 (2시간 23분째, 5/12 그대로). pod 메모리 29 GiB (풀 worker 둘이 8.8·8.1 GB로 커지는 중), oom_kill 0.

- 라운드 6 (15:51~): 상태 정상 (mem 15.8 GiB, oom_kill 0). **0단 distill 끝** (16:22, 2192 iter / 0.77 h, best iter 1600; resume.pt 1003·1961에 저장됨 = 20분 체크포인트 동작 확인). best iter 중간 val (보고서용 아님): count/g+p pcb 1.75 fail 0.56 / pkg 1.50 fail 0.56 / ip 0.77 fail 0.19 (`runs/2026-10-28/r0_coord_distill/log.txt`). 0단 RL 시작 (`runs/2026-10-28/r0_coord_rl/`).
- **사고 3: 0단 RL이 GPU OOM** (16:23, 첫 iter; B=16×G=8=128 env를 한 번에 rollout → `sim.loaded_state`가 2 GB를 더 할당하다 실패, 풀 9.5 GB + 18.4 GB 사용). 드라이버는 그대로 1단 distill로 넘어감 (16:24). 수정: `train.py --chunk` (기본 4) — RL 한 step의 16개 문제를 4개씩 rollout+backward 하고 gradient를 누적해 optimizer step 1번. 기준선은 문제별 G=8 샘플 안에서 정규화, loss scale은 전체 배치 기준(1/(B·G), 1/B) 그대로 → 같은 갱신, 메모리만 줄어듦. `scripts/smoke_chunk.py` 통과 (CPU, chunk 2 / 0). 1단 RL(~17:10 시작)부터 적용 (이미 떠 있는 run_rung이 새 train.py를 부름).
- 드라이버 GPU 줄 순서: 살아 있는 단 먼저 ATTACH → 나머지 사다리 순서. 그래서 r1 → **r0 RL 재실행** (distill은 유지, OOM 난 rl 폴더는 `_killed_`로 옮김) → r2 → r3. 드라이버 16:39 재시작 (r1·풀 생성 ATTACH).
- 결정: chunk=4 (가장 메모리 안전한 쪽). 대가는 step당 시간이 늘어 1.25 h 안의 RL iter 수가 줄 수 있음 — 모든 단이 같은 chunk라 사다리 비교는 유지, iter 수는 metrics.json에 기록.
- 결정: 0단 RL을 1단 뒤로 미룸 — 지금 1단 distill이 GPU를 쓰고 있고, 둘을 같이 올리면 GPU OOM (15:15 사고 1).
- 관찰: distill loss는 iter 2000 무렵 0.0~0.4까지 내려가지만 val fail은 iter 200~2100 내내 0.19~0.94 사이에서 들쭉날쭉 (val 16문제라 잡음 큼) → 27개 풀의 선생님 행동을 외우는 쪽. RL 단계에서 val fail이 내려가는지가 첫 확인 포인트.
- 위험: 10·16칸 eval 풀 생성 (2 workers, 큰 판부터)이 15:15 시작 후 72분째 16칸 2개가 아직 하나도 안 끝남. 남은 7개(16칸 6 + ip10_1)가 끝나야 E0와 단 평가가 돌기 때문에, 0단 학습이 ~17:40에 끝나도 평가는 그 뒤로 밀린다. GPU 줄(1·2·3단 학습)은 영향 없음.
- 결정: 풀 생성은 그대로 둔다 — 지금 멈추면 72분 진행분을 버리고, workers를 늘리면 QUEUE 한도 4를 넘음. 다음 라운드에 첫 16칸 풀 시간이 찍히면 남은 시간을 그 숫자로 다시 계산하고, 너무 길면 (단 평가가 2시간 넘게 기다리면) 10칸 평가를 16칸 풀보다 먼저 하도록 드라이버 순서만 바꾸는 것을 검토.

- 라운드 5 (15:47~): 상태 정상 (mem 12.5 GiB, oom_kill 0). 0단 distill iter 500 중간 val (보고서용 아님): count/g+p pcb 2.66 fail 0.62 / pkg 1.50 fail 0.69 / ip 0.77 fail 0.25 (`runs/2026-10-28/r0_coord_distill/log.txt`). eval 풀 10·16칸 12개 중 5개 완료 (16칸 진행 중).
- 결정: CPU 줄 순서 변경 — E0를 (gp·modal_rule 전체) → (LS 10칸) → 단 평가가 기다리는 게 없을 때만 16칸 LS를 종류별로 하나씩. 이유: LS 4x32는 greedy를 판마다 수백 번 다시 돌리고 eval_policy가 큰 판부터 돌려서, 기존 순서면 16칸 LS가 끝날 때까지 0단 평가(~17:40 학습 끝)가 몇 시간 밀릴 수 있음. 실험 조건은 그대로, 순서만 바뀜. 드라이버 15:49 재시작 (r0·풀 생성 ATTACH, 끊김 없음).

- 라운드 4 (15:43~): 상태 정상 (mem 12 GiB, oom_kill 0). 0단 distill iter ~300, 중간 val (iter 200) fail pcb 0.44 / pkg 1.00 / ip 0.38 (`runs/2026-10-28/r0_coord_distill/log.txt`, 보고서용 아님). 10·16칸 eval 풀 12개 중 5개 완료 (ip10_1, 16칸 6개 진행 중).
- 3단 추가: 드라이버 GPU 줄에 `r3_loop` (`--model loop`, 같은 COMMON·시간 예산)을 2단 뒤에 넣음. 시작 전 같은 줄에서 `scripts/smoke_pipeline.py --arms loop --workers 2` (작은 판 전체 경로: 선생님→distill→rl→eval @R2/@R16) 통과해야 학습 시작, 실패면 3단 건너뜀 (`agent/logs/smoke_loop.log`). 3단 rl 평가는 테스트 R = 기본 8 + 2·4·16. 드라이버 재시작 15:45 (r0·풀 생성은 ATTACH, 끊김 없음). 예상: 3단 ~21:50 시작.
- 결정: 3단 smoke를 지금 따로 돌리지 않고 드라이버 GPU 줄의 2단 뒤에 둠 — 지금 돌리면 동시 프로세스가 QUEUE 한도 4를 넘음.
- 결정: 3단 학습에 `--grad_ckpt` 등 메모리 옵션을 넣지 않음 (다른 단과 조건 동일 유지). GPU OOM이 나면 다음 라운드에 기록하고 그때 판단.
- DAILY.md: 첫 라운드 05:24 UTC → 첫 DAILY.md는 10/29 05:30 UTC 무렵 라운드에서 작성.

- 사고: 13:45경 파드 두 번째 재시작 (QUEUE: 60 GiB 한도 OOM 의심). 당시 동시 실행 = eval 풀 생성 14개(20칸 포함) + 선생님 8개 = 22 프로세스. 재시작 후 oom_kill 카운터는 0 (리셋됨). 선생님 해는 27개 중 15개 완성 상태였음.
- QUEUE 반영 (14:20 지시): 드라이버 `scripts/pipeline_day1.py` 재작성 — CPU 작업 프로세스 합계 ≤4 (선생님 4 → 학습 CERTIFY_WORKERS=2 + CPU 줄 2; 20칸 eval 풀은 맨 끝에 2개), 단계마다 시작/끝 `memory.current`·`oom_kill` 로그 (`scripts/mem.py`), 45 GiB 넘으면 새 프로세스 대기. `train.py`에 `--ckpt_min 20` resume.pt (모델·옵티마이저·λ·RNG·선생님 교체 상태·추가된 격자점·사용 시간) → 재시작 시 이어서 학습 (`scripts/smoke_resume.py` 통과: iter 6→12 이어짐). `run_rung.py`는 resume.pt가 있으면 옮기지 않고 이어감.
- 선생님 완료 15:14 (27/27, 4 workers, 최대 mem 13.9 GiB, oom_kill 0). 마지막 `train_pcb12_1` 혼자 3235 s.
- 사고 2: 15:15 0·1단 동시 학습이 **GPU OOM** (27개 풀을 GPU에 dense로 올리면 프로세스당 >11 GB, 24 GB에 둘이 안 들어감) → 둘 다 rc=1, 드라이버가 2단으로 넘어감. 2단을 멈추고 (distill 1129 s 진행분, resume.pt 전이라 버림) 드라이버를 GPU 줄(0→1→2단 순차) + CPU 줄(2 workers: 10·16칸 eval 풀 → E0 → 단마다 끝나면 인증 평가 → 20칸 풀 → 20칸 평가)로 다시 짬.
- 돌고 있는 것: 0단 coord distill (15:36 시작, `runs/2026-10-28/r0_coord_distill/log.txt`), 10·16칸 eval 풀 (`agent/logs/build_pools_v1_r3.log`). 예상 순서: 0단 ~17:40, 1단 ~19:45, 2단 ~21:50 (각 distill 0.75 h + RL 1.25 h + 풀 로딩).
- 참고 (보고서용 아님): 중단된 2단 mott distill이 700 iter에서 val fail pcb 0.94 / pkg 0.56 / ip 0.25 — 2단 모델 자체는 학습 루프에서 돈다.
- 재시작 대응: 라운드 시작 때 `.venv/bin/python scripts/ensure_running.py` (죽었으면 드라이버만 다시 띄움; 모든 단계가 끝난 출력은 건너뜀).
- 끝난 것 (숫자, 보고서용 아님 — 학습 판의 선생님 통계): 27개 학습 풀에서 LS 해의 정확 검사 통과율 0.00~0.81 (g+p는 0.38~1.00) → LS가 실패한 문제는 선생님 = g+p 해. 표: RESULTS.md "선생님 해".
- 다음 라운드: `ensure_running.py` → 0단 distill/RL 중간 val (`runs/2026-10-28/r0_coord_*/log.txt`) 기록 → 10·16칸 풀 끝나면 E0 결과 RESULTS.md (`collect_results.py`는 단마다 드라이버가 자동 실행) → 1단 시작 확인.
- 결정: 단은 GPU에서 하나씩, 각 CERTIFY_WORKERS=2 + CPU 줄 2 workers = 합계 4 (QUEUE). 모든 단이 같은 학습 조건(시간 예산·workers)이라 사다리 비교 유지. 단, CPU 줄 작업(풀 생성/평가)이 단마다 달라 반복 수가 조금 다를 수 있음 → metrics.json의 iters로 기록.
- 결정: 큐 우선순위대로 2단보다 0·1단을 먼저 — 그래서 이미 돌던 2단을 멈춤.
- 결정: 20칸 eval 풀을 만든 뒤의 20칸 평가도 2 workers — 20칸 판의 메모리 사용이 OOM 원인으로 의심되므로 같은 한도.
- 결정 (이전 라운드 유지): 1단 = 좌표 + modal 특징, 정확 검사는 프로세스당 1스레드, 2단은 기존 `MoTT`, IP 난이도 EASY / PCB·PKG HARD, 풀 시드 고정.
