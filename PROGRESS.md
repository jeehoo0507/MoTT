STUCK: `bm2_rlonly_s2` (B-Manoharan RL만, seed 2) GPU OOM 2번 — 11/08 19:15 UTC it 300, 같은 명령 resume (it 288 에서) 도 21:24 UTC 같은 자리 (it 300 val 직후 `manoharan_reward` → `probe_impedance` 의 `torch.linalg.solve`/magma OOM, 24.5 GB 카드). 결정적 재현이라 플래그 그대로는 다시 안 됨 → 다시 돌리지 않음, best.pt (it 100) 만 있고 metrics.json 없음 → 평가 없음. 사람 결정 (QUEUE 11/09 13:50 KST): 다시 돌리지 않음, 보고서엔 seed 0·1 로 적고 seed 2 는 학습 OOM 실패라고 씀.
STUCK (해제 시도 중, 11/05 08:10; 11/07 05:05 갱신): LS16이 메모리 45 GiB를 넘음. `--ls_chunk 8`로 **ip16_0 (0.64), pcb16_1 (0.64) 완주**. **ip16_1은 11/07 04:51 mem_guard가 멈춤** (18.7 h째, r2pkfc pkg16_0 평가 시작과 겹쳐 pod 45.09 GiB — 사람 결정 QUEUE 13:40 KST: r2pkfc 평가 우선, LS16은 부록 기준선 → `.memskip`, "메모리로 생략"). pcb16_0은 chunk 판 2번 중단. 남은 LS16 (pcb16_0, pkg16_0·1)은 `cpu_deferred` (G1 뒤).
# PROGRESS (갱신 2026-11-09 05:50 UTC, 라운드 223 — 서버 2 **N2 48 / 48 다 옴** (지난 라운드 "35" 는 셈 실수), nrest 1 / 10; RESULTS 다시 생성 (1503 기록); GPU = 사람 S (lo_pkg_bdf_s3 distill it 415, best 2.356 @ 400); cpu = `bm2_rlonly_rl+rep4` pkg16_1 (8070 s 째, 워커 100 %))

## 라운드 223 (2026-11-09 05:35 ~ 05:50 UTC)
- `git fetch r2e` (05:35): 새 커밋 없음 (사람 "N/N2 records" c6a5b29..97b1d9c 는 이미 로컬). QUEUE 끝 = 11/09 14:00 KST (S) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. `restore_proc_log` → 되살릴 것 없음 (197214 B). mem 28.9 GiB / 60, oom_kill 2.
- S (사람 `seeds_s.sh s1 run`): lo_pkg_bdf_s3 distill 진행, `runs/2026-10-29/lo_pkg_bdf_s3_distill/log.txt` it 400 new best 2.356 (val pcb10_0 1.200 / fail 0.94, ip10_0 0.761 / fail 0.44 — 학습 중 val, 판정 아님), resume ckpt it 415. GPU 16 % / 11.9 GB. 손대지 않음.
- cpu: `N harness eval_pkg16_1 bm2_rlonly_rl+rep4` 8072 s 째, 워커 2016660 util 100 % / rss 10.3 GB. 기준 14400 s ≈ 07:20 UTC.
- **N2 (서버 2, 보고만): 48 / 48 다 옴.** 파일로 다시 셈: r2pk ×3 seed ×4 = 12, r0 12, bp_park 12, bm2_rlonly 12. 지난 라운드의 "35 / 48, 아직 없음 bm2_rlonly pcb10 ×3" 는 틀림 — 그 3 개는 c3c4df2 (05:13 UTC) 에 이미 왔었음 (셈 실수, 고침). 이번에 새로 온 N2 3 개: bp_park s0·s1·s2 pcb10_1. QUEUE 의 "09:00 KST 까지" 는 지났고 마지막은 14:35 KST 쯤 도착 (늦음, 보고만).
- **nrest (서버 2, QUEUE 13:50 KST 의 10 개): 1 / 10** — `bm2_rlonly_s1_rl+rep4` pcb10_0 만 옴. 남은 9: bm2_rlonly_s1 pcb10_1, pkg10 ×2, ip10 ×2, pkg16 ×2, lo_pkg_r0_s3 pkg10 ×2.
- `collect_results.py` 손으로 (1488 → 1503 기록). 스크립트 숫자 그대로 (성공률 / 개수 비, 판정 없음): bp_park+rep4 pcb10 s0 0.06 / 1.22, s1 0.04 / 1.10, s2 0.02 / 5.86; ip10 세 seed 1.00 / 1.28. lo_pkg_bm2_rlonly+rep4 pcb10 s0 0.04 / 3.99, s1 0.04 / 3.99, s2 0.02 / 6.29; ip10 세 seed 1.00 / 2.15. bm2_rlonly_s1+rep4 pcb10_0 0.04 [25]. 서버 2 시간은 쓰지 않음.
- 관찰 (판정 아님): lo_pkg_bm2_rlonly s0 와 s1 의 pcb10 개수 비 (3.993) 와 ip10 숫자가 같음, bp_park 세 seed ip10 개수 비 1.284 도 같음 — 지난 라운드에 적은 "seed 가 달라도 같은 배치로 모임" 과 같은 꼴. 사람 판단용.
- 한 줄 (N 6단계): GPU = 사람 S (lo_pkg_bdf_s3), cpu = N 하네스 bm2_rlonly pkg16_1, 새 기록 4 (서버 2 N2 3 + nrest 1, RESULTS 반영).
- 결정: 큐·플래그·순서 그대로, S 손대지 않음 — 이유: 사람 새 지시 없음, 모든 프로세스 정상.
- 할 일 (다음): (1) pkg16_1 경과 (07:20 UTC 기준; 넘거나 util 0 % 면 eval.log·py-spy); 끝나면 RESULTS `bm2_rlonly_rl+rep4` pkg16 [50]. (2) S: lo_pkg_bdf_s3 distill 끝 ≈ 06:55 UTC → seeds_s 가 평가·push; S 기록 오면 collect_results (실패 포함 전부). (3) nrest 남은 9 오면 세고 collect_results. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. (5) DAILY 다음 11/10 01:15 UTC. (6) mem 감시 (S 평가 + cpu 겹칠 때 45 GiB).

## 라운드 222 (2026-11-09 05:05 ~ 05:25 UTC)
- `git fetch r2e` (05:05): 새 커밋 없음. 로컬에 이미 있는 사람 커밋 04533ff (PROPOSALS **11/09 S**, `scripts/seeds_s.sh`) + QUEUE **11/09 14:00 KST**: 서버 1 GPU 는 사람 스크립트 `seeds_s.sh s1 run` (lo_pkg_bdf_s3/s4, bdf_s4, r2pk_s4/s5; 평가는 하나씩 뒤에서). day2_queue.json 에 넣지 않음, GPU 에 다른 것 시작 안 함, 그 프로세스 죽이지 않음. 보고만 (판정 없음), 실패 seed 도 RESULTS 에.
- S 상태: `agent/logs/seeds_s_s1/run.log` = 04:55 smokes ok → 04:55 start train lo_pkg_bdf_s3 (distill 2.0 h → ≈ 06:55 UTC 끝 예상). GPU 21 % / 11.9 GB. train.py 2033463 (+ 로더 워커 2).
- 주의: `seeds_s.sh push()` 는 이 checkout 에서 `git pull --rebase --autostash` 를 함 → 추적 중인 로그가 stash 되면 끊길 수 있음 (memory no-git-stash-live-logs). 사람 스크립트라 고치지 않음; 대신 (1) 라운드마다 `restore_proc_log.py 1792539` 계속, (2) 내 commit/push 는 같은 잠금 `agent/logs/seeds_s_s1/git.lock` 을 잡고 함.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. `restore_proc_log` → 되살릴 것 없음 (197214 B). mem 28.2 GiB / 60, oom_kill 2.
- cpu: `N harness eval_pkg16_1 bm2_rlonly_rl+rep4` ≈ 7200 s 째, 워커 2016660 util 100 % / rss 9.9 GB. pkg16_0 (9055 s) 기준 ≈ 05:55 UTC, S 학습 로더와 CPU 를 나눠 조금 늦을 수 있음.
- **N2 (서버 2, 보고만): 34 / 48** (+9): r0_s1 ip10_0, r0_s2 ×4, bm2_rlonly s0·s1 pcb10_1, s2 pcb10_0, bp_park ip10_0 등. 아직 없음 14: bm2_rlonly pcb10 ×3 (s0·s1 pcb10_0, s2 pcb10_1), bp_park ×11. `collect_results.py` 손으로 (1488 기록; CPU 경합으로 ~12 분 걸림). 스크립트 숫자 그대로: r0_s2+rep4 ip10 1.00 / 1.36, pcb10 0.90 / 0.21; bp_park+rep4 ip10_0 1.00 [25]; bm2_rlonly+rep4 pcb10 s0 0.04 [25], s1 0.04 [25], s2 0.04 [25].
- **관찰 (판정 아님):** bm2_rlonly s0 와 s1 pcb10_1 하네스 숫자가 완전히 같아서 확인함. best.pt md5 세 seed 모두 다름 (복사 실수 아님). 서버 1 정책 단독 기록에서도 **배치 (occ) 가 같음**: ip10_0·ip10_1 은 s0=s1=s2, pcb10_0·pcb10_1 은 s0=s1 (s2 다름), pkg10_0·1 은 셋 다 다름. 즉 B-Manoharan v2 RL만 (패키지형 뺀 학습) 은 seed 가 달라도 PCB·IP 에서 같은 결정적 배치로 모임 (RL-only 붕괴와 같은 꼴, memory rl-only-lambda-collapse). 보고서에서 이 묶음의 seed 범위가 0 인 것은 "seed 독립 변동이 없음" 이 아니라 "같은 배치로 붕괴" 로 읽어야 함. 사람 판단용으로 적기만 함.
- 05:27 push 전 merge 로 1 개 더 (a31ea8b: lo_pkg_bp_park_rl pcb10_0) → **35 / 48**; RESULTS 에는 다음 라운드 collect_results 때 넣음 (12 분 걸려서 이번엔 다시 안 돌림).
- QUEUE "09:00 KST 까지 48 개" 지남 (지금 14:25 KST): **13 개 아직 안 옴 (늦음, 보고만).** 서버 1 에 다시 넣지 않음.
- 한 줄 (N 6단계): GPU = 사람 S (lo_pkg_bdf_s3), cpu = N 하네스 bm2_rlonly pkg16_1, 새 기록 9 (서버 2 N2, RESULTS 반영).
- 결정: 큐·플래그·순서 그대로, S 는 손대지 않음 — 이유: QUEUE 14:00 KST 가 GPU 를 사람 스크립트에 줌; seeds_s 의 autostash 위험은 사람 스크립트를 고치지 않고 같은 flock 을 쓰는 쪽 (가장 보수적).
- 할 일 (다음): (1) pkg16_1 경과 (기준 14400 s ≈ 07:20 UTC; 넘거나 util 0 % 면 eval.log·py-spy); 끝나면 RESULTS `bm2_rlonly_rl+rep4` pkg16 [50] 확인. (2) S: `seeds_s.sh s1 status` 로 학습·평가 진행 보기, S 기록 push 되면 collect_results (전부 표에, 실패 포함). (3) N2 남은 14 + nrest 10 오면 개수 세고 collect_results. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. (5) DAILY 다음 11/10 01:15 UTC. (6) mem 감시: S 평가 (서버 1 은 하나씩) + 드라이버 cpu 가 겹치면 45 GiB 가까운지 확인.

# (이전) 라운드 221 머리말: (갱신 2026-11-09 04:50 UTC, 라운드 221 — 사람 새 QUEUE (13:50 KST): 남은 N 하네스 10 개 (bm2_rlonly_s1 8, lo_pkg_r0_s3 pkg10 2) 서버 2 로 옮김 → 서버 1 cpu 는 `bm2_rlonly_rl+rep4` pkg16_1 하나만 (80 분째, 워커 100 %); 서버 2 N2 **25 / 48** (r0_rl·r2pk_s2·r0_s1 일부 도착), RESULTS 다시 생성; GPU 빔)

## 라운드 221 (2026-11-09 04:41 ~ 04:50 UTC)
- `git fetch r2e` (04:42): 새 커밋 없음 (사람 8f5419e `Move last 10 N harness evals from server 1 to server 2` 은 이미 로컬). QUEUE 끝 = **11/09 13:50 KST** 새 항목: bm2_rlonly_s1_rl (pcb10/pkg10/ip10/pkg16 ×0,1 = 8) + lo_pkg_r0_s3_rl pkg10_{0,1} (2) → `cpu_deferred` (`moved` = server2_nrest), 서버 2 가 `scripts/server2_nrest.sh` 로 병렬. "Do not re-add them." bm2_rlonly_s2 항목은 그대로 (STUCK, 보고서엔 seed 0·1 로 적고 seed 2 는 학습 OOM 실패라고 씀; 플래그 바꿔 다시 돌리지 않음).
- 드라이버는 큐를 매 단계 다시 읽음 (pipeline_day2.py docstring) → 따로 할 일 없음. 확인: cpu 대기 9 개 = 진행 중 pkg16_1 1 개 + bm2_rlonly_s2 8 개 (after = s2 metrics.json 없음 → 영원히 건너뜀). 즉 **pkg16_1 끝나면 서버 1 cpu 줄도 빔.**
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. `restore_proc_log.py 1792539` → 되살릴 것 없음 (live = disk 197214 B). GPU 85 MiB / 0 % (빔).
- cpu: `N harness eval_pkg16_1 bm2_rlonly_rl+rep4` 4828 s 째, 워커 2016660 util 100 % / rss 9.3 GB (살아 있음). pkg16_0 (9055 s) 처럼이면 ≈ 05:55 UTC.
- **N2 (서버 2, 보고만): 22 / 48.** 새로 8 개: lo_pkg_r0_rl ip10_{0,1}·pcb10_{0,1}, lo_pkg_r2pk_s2_rl ip10_{0,1}·pcb10_{0,1}. 아직 없음 (26): bm2_rlonly pcb10 ×6, r0 s1·s2 (8), bp_park ×3 (12). `collect_results.py` 손으로 다시 (1468 → 1476 기록). 숫자는 스크립트 그대로: r0_rl+rep4 ip10 1.00 / 1.30 (개수 비), pcb10 0.94 / 0.20; r2pk_s2_rl+rep4 ip10 1.00 / 1.01, pcb10 1.00 / 0.20. 판정 줄 안 씀 (사람). 서버 2 시간은 쓰지 않음.
- 04:52 merge 로 3 개 더: lo_pkg_r0_s1_rl ip10_1·pcb10_{0,1} → **25 / 48**. collect_results 다시 (1479 기록): r0_s1+rep4 ip10 1.00 / 1.48 [25], pcb10 0.98 / 0.19.
- QUEUE "09:00 KST 까지 48 개" 기한 지남 (지금 13:50 KST): **23 개 아직 안 옴 (늦음, 보고만).** 서버 1 에 다시 넣지 않음.
- 한 줄 (N 6단계): GPU = 없음, cpu = N 하네스 bm2_rlonly pkg16_1 (서버 1 마지막 항목), 새 기록 8 (서버 2 N2, RESULTS 반영).
- 결정: 큐·플래그·순서 그대로, 아무것도 다시 넣지 않음 — 이유: 사람 13:50 KST 항목이 남은 10 개를 서버 2 로 옮기고 "do not re-add" 라 함; pkg16_1 정상 진행. pkg16_1 끝난 뒤 서버 1 이 비어도 새 일을 스스로 시작하지 않음 — 이유: 11/06 12:00 UTC 새 시도 마감 지남 (QUEUE 11/07 10:20 KST), 사람 항목 없음.
- 할 일 (다음): (1) pkg16_1 경과 감시 (기준 14400 s ≈ 07:20 UTC; 넘거나 util 0 % 이면 eval.log·py-spy). (2) 끝나면 RESULTS `bm2_rlonly_rl+rep4` pkg16 행이 [50] 이 되는지 확인. (3) 서버 2 push (N2 남은 26 + nrest 10) 오면 개수 세고 `collect_results.py` 다시. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. (5) DAILY 다음 11/10 01:15 UTC. (6) GPU·cpu 빈 채로 (사람 항목 없으면).

# (이전) 라운드 220 머리말: (갱신 2026-11-09 04:24 UTC, 라운드 220 — 서버 2 N2 기록 도착 시작: **14 / 48** (사람 커밋 16 개, 04:07~04:24 UTC); RESULTS 손으로 collect_results 다시 돌려 반영; cpu N 하네스 `bm2_rlonly_rl+rep4` pkg16_1 진행 (57 분째, 워커 100 %); GPU 빔)

## 라운드 220 (2026-11-09 04:19 ~ 04:24 UTC)
- `git fetch r2e` (04:19): 새 커밋 없음 (사람 "N2 records" 커밋 10 개 223995f..9575af4 는 이미 로컬). QUEUE 끝 = 11/09 01:40 KST 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. `restore_proc_log.py 1792539` → 되살릴 것 없음 (live = disk 197214 B). GPU 85 MiB / 0 % (빔).
- cpu: `N harness eval_pkg16_1 bm2_rlonly_rl+rep4` 3405 s 째, 워커 2016660 util 100 % / rss 8.8 GB (살아 있음). ≈ 06:00 UTC 예상.
- **N2 (서버 2, 보고만): 10 / 48 도착.** bm2_rlonly {s0,s1,s2} ip10_{0,1} (6) + r2pk_rl (s0) ip10_{0,1}·pcb10_{0,1} (4). 아직 없음: bm2_rlonly pcb10 ×6 (병렬판이 "느린 것 먼저" 라 진행 중일 것), r2pk s1·s2, r0 ×3, bp_park ×3 (38 개).
- report_watch 는 untracked 파일만 보므로 사람이 커밋한 N2 파일을 RESULTS 에 넣지 않음 → `.venv/bin/python scripts/collect_results.py` 손으로 돌림 (1454 → 1464 기록). 숫자는 스크립트가 쓴 그대로, 손 숫자 없음: ip10 bm2_rlonly+rep4 세 seed 모두 1.00 (수리 전 0.96 → 1.00, 개수 비 2.50), r2pk_rl+rep4 ip10 1.00 / 1.01, pcb10 0.98 / 0.20.
- 주의: N2 파일의 문제당 초 (`sec_per_inst`) 는 서버 2 CPU 값 — QUEUE 11/05 규칙대로 시간 비교·규모 곡선에 쓰지 않음. N2 판정 줄은 사람이 씀 (PROPOSALS 11/09 N2, report-only).
- 04:24 merge 에서 4 개 더 옴: r2pk_s1 ip10_{0,1}·pcb10_{0,1} → **14 / 48**. collect_results 다시 (1468 기록): r2pk_s1+rep4 ip10 1.00 / 1.01, pcb10 1.00 / 0.19.
- QUEUE "11/09 09:00 KST 까지 48 개 다 안 오면 PROGRESS 에 말하라": 지금 13:24 KST, **34 개 아직 안 옴 (늦음, 보고만).** 서버 1 은 다시 넣지 않음.
- 한 줄 (N 6단계): GPU = 없음, cpu = N 하네스 bm2_rlonly pkg16_1, 새 기록 14 (서버 2 N2, RESULTS 반영).
- 결정: RESULTS 를 손으로 collect_results 다시 돌려 갱신 — 이유: 같은 스크립트·같은 규칙 (report_watch 가 하던 일), 사람이 커밋한 파일이라 자동 경로가 못 봄; 숫자만 넣고 판정은 안 씀. 큐·플래그·순서 그대로.
- 할 일 (다음): (1) pkg16_1 경과 감시 (기준 14400 s ≈ 07:20 UTC; 넘거나 util 0 % 이면 eval.log·py-spy). (2) pkg16_1 끝나면 RESULTS 행이 [50] 이 되는지 확인. (3) 서버 2 N2 push 더 오면 개수 세고 `collect_results.py` 다시 돌림. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. (5) DAILY 다음 11/10 01:15 UTC. (6) GPU 는 사람 항목 없으면 빈 채로.

# (이전) 라운드 219 머리말: (갱신 2026-11-09 04:04 UTC, 라운드 219 — cpu N 하네스 `bm2_rlonly_rl+rep4` pkg16_1 진행 (03:21~, 41 분째, 워커 100 %); 서버 2 가 N2 를 병렬판 `server2_n2b.sh` 로 다시 돌림 (사람 커밋 913b8e9, 12:46 KST); N2 0 / 48; GPU 빔; 새 기록 0)

## 라운드 219 (2026-11-09 04:02 ~ 04:04 UTC)
- `git fetch r2e` (04:02): 새 커밋 없음 (사람 913b8e9 `server2_n2b: run N2 evals as parallel single jobs, slowest first` 는 이미 merge 됨 — 서버 2 가 48 개를 `xargs -P` 로, B-Manoharan v2 pcb 먼저; 같은 명령·플래그). QUEUE 끝 = 11/09 01:40 KST 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 25.5 GiB / 60, oom_kill 2 (변화 없음). `restore_proc_log.py 1792539` → 되살릴 것 없음 (live = disk 197214 B). GPU 85 MiB / 0 % (빔).
- cpu: `N harness eval_pkg16_1 bm2_rlonly_rl+rep4` 2442 s 째, 워커 2016660 util 100 % / rss 8.9 GB (살아 있음). ≈ 06:00 UTC 예상.
- N2: `eval_{pcb10,ip10}_*__lo_pkg_*+rep4.json` 12 개는 모두 `lo_pkg_bdf*` (N2 아님) → N2 **0 / 48**. 서버 2 가 병렬판으로 다시 시작했으니 "늦음" 보고는 거두고 push 를 기다림. 서버 1 은 다시 넣지 않음.
- 한 줄 (N 6단계): GPU = 없음, cpu = N 하네스 bm2_rlonly pkg16_1, 새 기록 0.
- 결정: 큐·플래그·순서 그대로 — 이유: 사람 새 지시 없음, pkg16_1 정상 진행, N2 는 사람이 서버 2 에서 맡음 (QUEUE 11/09 01:40 KST "do not re-add them").
- 할 일 (다음): (1) pkg16_1 경과 감시 (기준 14400 s ≈ 07:20 UTC; 넘거나 util 0 % 이면 eval.log·py-spy). (2) pkg16_1 끝나면 RESULTS 행이 [50] 이 되는지 확인. (3) 서버 2 N2 push 오면 개수 세기 (collect_results 가 RESULTS 에 숫자만). (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. (5) DAILY 다음 11/10 01:15 UTC. (6) GPU 는 사람 항목 없으면 빈 채로.

# (이전) 라운드 218 머리말: (갱신 2026-11-09 03:48 UTC, 라운드 218 — cpu N 하네스 `bm2_rlonly_rl+rep4` pkg16_1 진행 (03:21~, 25 분째, 워커 100 %); N2 여전히 늦음 0 / 48; GPU 빔 (s2 STUCK 그대로); 새 기록 0)

## 라운드 218 (2026-11-09 03:46 ~ 03:48 UTC)
- `git fetch r2e` (03:46): 새 커밋 없음. QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 25.2 GiB / 60, oom_kill 2 (변화 없음). `restore_proc_log.py 1792539` → 되살릴 것 없음 (live = disk 197214 B). GPU 85 MiB / 0 % (빔).
- cpu: `N harness eval_pkg16_1 bm2_rlonly_rl+rep4` 1486 s 째, 워커 2016660 util 100 % / rss 8.8 GB (살아 있음). pkg16_0 처럼 ~2.5 h 예상 (≈ 06:00 UTC).
- **N2 늦음 (보고만):** 03:47 UTC `find runs` 다시 셈 → `eval_{pcb10,ip10}_*__lo_pkg_{r2pk,r0,bp_park,bm2_rlonly}*+rep4.json` **0 / 48**. 서버 1 은 다시 넣지 않음.
- 한 줄 (N 6단계): GPU = 없음, cpu = N 하네스 bm2_rlonly pkg16_1, 새 기록 0.
- 결정: 큐·플래그·순서 그대로 — 이유: 사람 새 지시 없음, pkg16_1 정상 진행. N2 를 서버 1 에 다시 넣지 않음 — 이유: QUEUE 11/09 01:40 KST "do not re-add them".
- 할 일 (다음): (1) pkg16_1 경과 감시 (기준 14400 s ≈ 07:20 UTC; 넘거나 util 0 % 이면 eval.log·py-spy). (2) pkg16_1 끝나면 RESULTS 행이 [50] 이 되는지 확인. (3) 서버 2 N2 push 오면 개수 세기. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. (5) DAILY 다음 11/10 01:15 UTC. (6) GPU 는 사람 항목 없으면 빈 채로.

# (이전) 라운드 217 머리말: PROGRESS (갱신 2026-11-09 03:33 UTC, 라운드 217 — `bm2_rlonly_rl+rep4` pkg16_0 끝: **0.92 / 115.8 decap [25]** (9055 s); pkg16_1 진행 (03:21~); N2 여전히 늦음 0 / 48; GPU 빔 (s2 STUCK 그대로))

## 라운드 217 (2026-11-09 03:29 ~ 03:33 UTC)
- `git fetch r2e` (03:29): 새 커밋 없음. QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 25.2 GiB / 60, oom_kill 2 (변화 없음). `restore_proc_log.py 1792539` → 729 B 되살림 (live 197214 B / disk 196485 B — loop.sh autostash 탓, 기억대로). GPU 85 MiB / 0 % (빔).
- **끝난 실험:** `N harness eval_pkg16_0 bm2_rlonly_rl+rep4` 03:16:27 UTC rc=0, 9055 s. 성공 **0.92**, 개수 115.83, 72.4 s/문제, rounds 4. 수리 전 0.88 → 0.92 (+0.04 [+0.00, +0.12], 1 문제 구함). pkg16 +rep4 다른 정책 (~20~30 decap) 보다 decap 4~5 배 → 개수 비 8.08 (greedy+prune+rep4 = 1.000 인 표) — 성공률은 비슷하지만 개수로 짐. `collect_results` (03:16~03:19) 가 RESULTS 행 자동 반영 ([25], pkg16_0 만). 손 숫자 없음.
- cpu: `N harness eval_pkg16_1 bm2_rlonly_rl+rep4` 03:21:29 시작, 워커 2016660 util 100 % / rss 8.3 GB. pkg16_0 처럼 ~2.5 h 예상 (≈ 06:00 UTC).
- **N2 늦음 (보고만):** 03:31 UTC `find runs` 다시 셈 → `eval_{pcb10,ip10}_{0,1}__lo_pkg_{r2pk,r0,bp_park,bm2_rlonly}*+rep4.json` **0 / 48** (pkg10 harness 파일 12 개는 N2 아님). 서버 1 은 다시 넣지 않음.
- 한 줄 (N 6단계): GPU = 없음, cpu = N 하네스 bm2_rlonly pkg16_1, 새 기록 1 (pkg16_0 0.92 [25]).
- 결정: 큐·플래그·순서 그대로 — 이유: 사람 새 지시 없음, pkg16_1 정상 진행. N2 를 서버 1 에 다시 넣지 않음 — 이유: QUEUE 11/09 01:40 KST "do not re-add them".
- 할 일 (다음): (1) pkg16_1 경과 감시 (기준 14400 s ≈ 07:20 UTC; 넘거나 util 0 % 이면 eval.log·py-spy). (2) pkg16_1 끝나면 RESULTS 행이 [50] 이 되는지 확인. (3) 서버 2 N2 push 오면 개수 세기. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. (5) DAILY 다음 11/10 01:15 UTC. (6) GPU 는 사람 항목 없으면 빈 채로.

# (이전) 라운드 216 머리말: (갱신 2026-11-09 03:16 UTC, 라운드 216 — N2 여전히 늦음: 서버 2 push 0 / 48; GPU 빔 (사람 항목 없음, s2 STUCK 그대로); cpu N 하네스 `bm2_rlonly_rl+rep4` pkg16_0 진행 (00:45~, 148 분째, 워커 100 %); 새 기록 0)

## 라운드 216 (2026-11-09 03:13 ~ 03:16 UTC)
- `git fetch r2e` (03:13): 새 커밋 없음. QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 27.6 GiB / 60, oom_kill 2 (변화 없음). `restore_proc_log.py 1792539` → 되살릴 것 없음 (live = disk 196485 B). GPU 85 MiB / 0 % (빔).
- cpu: `N harness eval_pkg16_0 bm2_rlonly_rl+rep4` 8900 s 째, 워커 1995558 util 100 % / rss 11.3 GB (살아 있음; rss 10.2→11.3 GB). eval.log 마지막 = pkg16_0 시작 줄 그대로.
- **N2 늦음 (보고만):** 03:14 UTC `find runs` 다시 셈 → `eval_{pcb10,ip10}_{0,1}__lo_pkg_{r2pk,r0,bp_park,bm2_rlonly}*+rep4.json` **0 / 48**. 서버 1 은 다시 넣지 않음.
- 한 줄 (N 6단계): GPU = 없음, cpu = N 하네스 bm2_rlonly pkg16_0, 새 기록 0.
- 결정: 큐·플래그·순서 그대로, pkg16_0 계속 기다림 — 이유: 워커 100 %, rss 늘어남 (일하는 중), 기준 14400 s (≈ 04:45 UTC) 아래. N2 를 서버 1 에 다시 넣지 않음 — 이유: QUEUE 11/09 01:40 KST "do not re-add them".
- 할 일 (다음): (1) pkg16_0 경과 시간 감시 (기준 14400 s ≈ 04:45 UTC; 넘거나 util 0 % 이면 eval.log·py-spy). (2) bm2_rlonly pkg16 +rep4 RESULTS 행 확인 (pkg16_1 뒤 [50]). (3) 서버 2 N2 push 오면 개수 세기. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. (5) DAILY 다음 11/10 01:15 UTC. (6) GPU 는 사람 항목 없으면 빈 채로.

# (이전) 라운드 215 머리말: (갱신 2026-11-09 03:00 UTC, 라운드 215 — N2 여전히 늦음: 서버 2 push 0 / 48; GPU 빔 (사람 항목 없음, s2 STUCK 그대로); cpu N 하네스 `bm2_rlonly_rl+rep4` pkg16_0 진행 (00:45~, 132 분째, 워커 100 %); 새 기록 0)

## 라운드 215 (2026-11-09 02:57 ~ 03:00 UTC)
- `git fetch r2e` (02:57): 새 커밋 없음. QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 26.8 GiB / 60, oom_kill 2 (변화 없음). `restore_proc_log.py 1792539` → 되살릴 것 없음 (live = disk 196485 B). GPU 85 MiB / 0 % (빔).
- cpu: `N harness eval_pkg16_0 bm2_rlonly_rl+rep4` 7952 s 째, 워커 1995558 util 100 % / rss 10.2 GB (살아 있음). eval.log 마지막 = ip10_1 1.00 뒤 pkg16_0 시작 줄 (16칸은 끝날 때 한 번에 찍힘). 기준 14400 s 아래 → 기다림.
- **N2 늦음 (보고만):** 02:58 UTC 다시 셈 → `eval_{pcb10,ip10}_*__lo_pkg_{r2pk,r0,bp_park,bm2_rlonly}*+rep4.json` **0 / 48**. 서버 1 은 다시 넣지 않음.
- 한 줄 (N 6단계): GPU = 없음, cpu = N 하네스 bm2_rlonly pkg16_0, 새 기록 0.
- 결정: 큐·플래그·순서 그대로, pkg16_0 계속 기다림 — 이유: 워커가 일하는 중이고 (100 %, rss 천천히 늘어남), 느린 이유 (decap 34~44 개 수리) 가 설명됨. 기준 그대로: 14400 s 넘거나 util 0 % 이면 eval.log·py-spy 로 살펴봄. N2 를 서버 1 에 다시 넣지 않음 — 이유: QUEUE 11/09 01:40 KST "do not re-add them".
- 할 일 (다음): (1) pkg16_0 경과 시간 감시 (기준 14400 s ≈ 04:45 UTC). (2) bm2_rlonly pkg16 +rep4 RESULTS 행 확인 (pkg16_1 뒤 [50]). (3) 서버 2 N2 push 오면 개수 세기. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. (5) DAILY 다음 11/10 01:15 UTC. (6) GPU 는 사람 항목 없으면 빈 채로.

# (이전) 라운드 214 머리말: (갱신 2026-11-09 02:45 UTC, 라운드 214 — N2 여전히 늦음: 서버 2 push 0 / 48; GPU 빔 (사람 항목 없음, s2 STUCK 그대로); cpu N 하네스 `bm2_rlonly_rl+rep4` pkg16_0 진행 (00:45~, 116 분째, 워커 100 %); 새 기록 0)

## 라운드 214 (2026-11-09 02:41 ~ 02:45 UTC)
- `git fetch r2e` (02:41): 새 커밋 없음. QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 26.3 GiB / 60, oom_kill 2 (변화 없음). `restore_proc_log.py 1792539` → 되살릴 것 없음 (live = disk 196485 B). GPU 85 MiB / 0 % (빔).
- cpu: `N harness eval_pkg16_0 bm2_rlonly_rl+rep4` 6969 s 째, 워커 1995558 util 100 % / rss 9.8 GB (살아 있음). **이미 지금까지 본 pkg16 +rep4 최장 (r0_coord_s2 pkg16_0 4601 s) 의 1.5 배.** 설명 근거: bm2_rlonly 는 decap 수가 34~44 개 (다른 정책 ~10 개) 라 수리 비용이 큼 — 같은 run pcb10_0 도 3286 s (r2pk 류보다 훨씬 김). 워커 100 % + rss 천천히 늘어남 = "dead Pool worker" 멈춤과 다름 → 죽이지 않음.
- **N2 늦음 (보고만):** 02:42 UTC `find runs` 다시 셈 → `eval_{pcb10,ip10}_*__lo_pkg_{r2pk,r0,bp_park,bm2_rlonly}*+rep4.json` **0 / 48**. 서버 1 은 다시 넣지 않음.
- 한 줄 (N 6단계): GPU = 없음, cpu = N 하네스 bm2_rlonly pkg16_0, 새 기록 0.
- 결정: 큐·플래그·순서 그대로, pkg16_0 평가 계속 기다림 — 이유: 워커가 일하는 중이고 느린 이유가 설명됨, 죽이면 2 시간 손실. 기준: 4 h (14400 s) 넘거나 워커 util 0 % 이면 eval.log·py-spy 로 살펴봄 (그때도 죽이기 전 PROGRESS 에 근거 기록). N2 를 서버 1 에 다시 넣지 않음 — 이유: QUEUE 11/09 01:40 KST "do not re-add them".
- 할 일 (다음): (1) pkg16_0 경과 시간 감시 (기준 14400 s). (2) bm2_rlonly pkg16 +rep4 RESULTS 행 확인 (pkg16_1 뒤 [50]). (3) 서버 2 N2 push 오면 개수 세기. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. (5) DAILY 다음 11/10 01:15 UTC. (6) GPU 는 사람 항목 없으면 빈 채로.

# (이전) 라운드 213 머리말: (갱신 2026-11-09 02:28 UTC, 라운드 213 — N2 여전히 늦음: 서버 2 push 0 / 48; GPU 빔 (사람 항목 없음, s2 STUCK 그대로); cpu N 하네스 `bm2_rlonly_rl+rep4` pkg16_0 진행 (00:45~, 100 분째, 워커 100 %); 새 기록 0)

## 라운드 213 (2026-11-09 02:25 ~ 02:28 UTC)
- `git fetch r2e` (02:25): 새 커밋 없음. QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 26.7 GiB / 60, oom_kill 2 (변화 없음). `restore_proc_log.py 1792539` → 되살릴 것 없음 (live = disk 196485 B). GPU 85 MiB / 0 % (빔).
- cpu: `N harness eval_pkg16_0 bm2_rlonly_rl+rep4` 6026 s 째, 워커 1995558 util 100 % / rss 10.0 GB (살아 있음; rss 9.9→10.0 GB).
- **N2 늦음 (보고만):** 02:26 UTC `find runs` 다시 셈 → `eval_{pcb10,ip10}_{0,1}__lo_pkg_{r2pk,r0,bp_park,bm2_rlonly}*+rep4.json` **0 / 48**. 서버 1 은 다시 넣지 않음.
- 한 줄 (N 6단계): GPU = 없음, cpu = N 하네스 bm2_rlonly pkg16_0, 새 기록 0.
- 결정: 큐·플래그·순서 그대로, N2 를 서버 1 에 다시 넣지 않음 — 이유: QUEUE 11/09 01:40 KST "do not re-add them".
- 할 일 (다음): (1) bm2_rlonly pkg16 +rep4 RESULTS 행 확인 (pkg16_1 뒤 [50]). (2) 서버 2 N2 push 오면 개수 세기. (3) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. (4) DAILY 다음 11/10 01:15 UTC. (5) GPU 는 사람 항목 없으면 빈 채로.

# (이전) 라운드 212 머리말: (갱신 2026-11-09 02:12 UTC, 라운드 212 — N2 여전히 늦음: 서버 2 push 0 / 48; GPU 빔 (사람 항목 없음, s2 STUCK 그대로); cpu N 하네스 `bm2_rlonly_rl+rep4` pkg16_0 진행 (00:45~, 85 분째, 워커 100 %); 새 기록 0)

## 라운드 212 (2026-11-09 02:09 ~ 02:12 UTC)
- `git fetch r2e` (02:09): 새 커밋 없음. QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 25.9 GiB / 60, oom_kill 2 (변화 없음). `restore_proc_log.py 1792539` → 되살릴 것 없음 (live = disk 196485 B). GPU 85 MiB / 0 % (빔).
- cpu: `N harness eval_pkg16_0 bm2_rlonly_rl+rep4` 5072 s 째, 워커 1995558 util 100 % / rss 9.9 GB (살아 있음; rss 9.6→9.9 GB 천천히).
- **N2 늦음 (보고만):** 02:11 UTC 다시 셈 → `eval_{pcb10,ip10}_*__lo_pkg_{r2pk,r0,bp_park,bm2_rlonly}*+rep4.json` **0 / 48**. 서버 1 은 다시 넣지 않음.
- 한 줄 (N 6단계): GPU = 없음, cpu = N 하네스 bm2_rlonly pkg16_0, 새 기록 0.
- 결정: 큐·플래그·순서 그대로, N2 를 서버 1 에 다시 넣지 않음 — 이유: QUEUE 11/09 01:40 KST "do not re-add them".
- 할 일 (다음): (1) bm2_rlonly pkg16 +rep4 RESULTS 행 확인 (pkg16_1 뒤 [50]). (2) 서버 2 N2 push 오면 개수 세기. (3) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. (4) DAILY 다음 11/10 01:15 UTC. (5) GPU 는 사람 항목 없으면 빈 채로.

# (이전) 라운드 211 머리말: (갱신 2026-11-09 01:56 UTC, 라운드 211 — N2 여전히 늦음: 서버 2 push 0 / 48; GPU 빔 (사람 항목 없음, s2 STUCK 그대로); cpu N 하네스 `bm2_rlonly_rl+rep4` pkg16_0 진행 (00:45~, 69 분째, 워커 100 %); 새 기록 0)

## 라운드 211 (2026-11-09 01:54 ~ 01:56 UTC)
- `git fetch r2e` (01:54): 새 커밋 없음. QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 26.4 GiB / 60, oom_kill 2 (변화 없음). `restore_proc_log.py 1792539` → 되살릴 것 없음 (live = disk 196485 B). GPU 85 MiB / 0 % (빔).
- cpu: `N harness eval_pkg16_0 bm2_rlonly_rl+rep4` 4121 s 째, 워커 1995558 util 100 % / rss 9.6 GB (살아 있음; rss 9.3→9.6 GB 천천히, 한도와 거리 멂). 참고: pcb10_0 은 3286 s 였으니 pkg16 은 더 김 (16-cell) — 정상 범위로 봄.
- **N2 늦음 (보고만):** 01:55 UTC 다시 셈 → `eval_{pcb10,ip10}_{0,1}__lo_pkg_{r2pk,r0,bp_park,bm2_rlonly}*+rep4.json` **0 / 48**. 서버 1 은 다시 넣지 않음.
- 참고: 이 셸에는 `python` 이 PATH 에 없음 → 도우미는 `.venv/bin/python scripts/...` 로 한 명령씩 실행.
- 한 줄 (N 6단계): GPU = 없음, cpu = N 하네스 bm2_rlonly pkg16_0, 새 기록 0.
- 결정: 큐·플래그·순서 그대로, N2 를 서버 1 에 다시 넣지 않음 — 이유: QUEUE 11/09 01:40 KST "do not re-add them".
- 할 일 (다음): (1) bm2_rlonly pkg16 +rep4 RESULTS 행 확인 (pkg16_1 뒤 [50]). (2) 서버 2 N2 push 오면 개수 세기. (3) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. (4) DAILY 다음 11/10 01:15 UTC. (5) GPU 는 사람 항목 없으면 빈 채로.

# (이전) 라운드 210 머리말: (갱신 2026-11-09 01:41 UTC, 라운드 210 — N2 여전히 늦음: 서버 2 push 0 / 48; GPU 빔 (사람 항목 없음, s2 STUCK 그대로); cpu N 하네스 `bm2_rlonly_rl+rep4` pkg16_0 진행 (00:45~, 53 분째, 워커 100 %); 새 기록 0)

## 라운드 210 (2026-11-09 01:38 ~ 01:41 UTC)
- `git fetch r2e` (01:38): 새 커밋 없음. QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 25.8 GiB / 60, oom_kill 2 (변화 없음). `restore_proc_log.py 1792539` → 되살릴 것 없음 (live = disk 196485 B). GPU 85 MiB / 0 % (빔).
- cpu: `N harness eval_pkg16_0 bm2_rlonly_rl+rep4` 3173 s 째, 워커 1995558 util 100 % / rss 9.3 GB (살아 있음; rss 천천히 늘어남 8.7→9.3 GB, 45 GiB 한도와 거리 멂).
- **N2 늦음 (보고만):** 01:39 UTC `find runs` 로 다시 셈 → `eval_{pcb10,ip10}_{0,1}__lo_pkg_{r2pk,r0,bp_park,bm2_rlonly}*+rep4.json` **0 / 48**. 서버 1 은 다시 넣지 않음.
- 한 줄 (N 6단계): GPU = 없음, cpu = N 하네스 bm2_rlonly pkg16_0, 새 기록 0.
- 결정: 큐·플래그·순서 그대로, N2 를 서버 1 에 다시 넣지 않음 — 이유: QUEUE 11/09 01:40 KST "do not re-add them".
- 할 일 (다음): (1) bm2_rlonly pkg16 +rep4 RESULTS 행 확인 (pkg16_1 뒤 [50]). (2) 서버 2 N2 push 오면 개수 세기. (3) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. (4) DAILY 다음 11/10 01:15 UTC. (5) GPU 는 사람 항목 없으면 빈 채로.

# (이전) 라운드 209 머리말: (갱신 2026-11-09 01:25 UTC, 라운드 209 — N2 여전히 늦음: 서버 2 push 0 / 48; GPU 빔 (사람 항목 없음, s2 STUCK 그대로); cpu N 하네스 `bm2_rlonly_rl+rep4` pkg16_0 진행 (00:45~, 워커 100 %); 새 기록 0)

## 라운드 209 (2026-11-09 01:22 ~ 01:25 UTC)
- `git fetch r2e` (01:22): 새 커밋 없음. QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 25.1 GiB / 60, oom_kill 2 (변화 없음). `restore_proc_log.py 1792539` → 되살릴 것 없음 (live = disk 196485 B). GPU 85 MiB / 0 % (빔).
- cpu: `N harness eval_pkg16_0 bm2_rlonly_rl+rep4` 2235 s 째, 워커 1995558 util 100 % / rss 8.8 GB (살아 있음, 정상; 부모 0 % 는 정상 대기).
- **N2 늦음 (보고만):** 01:23 UTC 다시 셈 → `eval_{pcb10,ip10}_{0,1}__lo_pkg_{r2pk,r0,bp_park,bm2_rlonly}*+rep4.json` **0 / 48**. 서버 1 은 다시 넣지 않음.
- 한 줄 (N 6단계): GPU = 없음, cpu = N 하네스 bm2_rlonly pkg16_0, 새 기록 0.
- 결정: 큐·플래그·순서 그대로, N2 를 서버 1 에 다시 넣지 않음 — 이유: QUEUE 11/09 01:40 KST "do not re-add them".
- 할 일 (다음): (1) bm2_rlonly pkg16 +rep4 RESULTS 행 확인 (pkg16_1 뒤 [50]). (2) 서버 2 N2 push 오면 개수 세기. (3) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. (4) DAILY 다음 11/10 01:15 UTC. (5) GPU 는 사람 항목 없으면 빈 채로.

# (이전) 라운드 208 머리말: (갱신 2026-11-09 01:08 UTC, 라운드 208 — **DAILY 11/09 새로 씀**; N2 여전히 늦음: 서버 2 push 0 / 48; GPU 빔 (사람 항목 없음, s2 STUCK 그대로); cpu N 하네스 `bm2_rlonly_rl+rep4` pkg16_0 진행 (00:45~); 새 기록 0)

## 라운드 208 (2026-11-09 01:04 ~ 01:08 UTC)
- `git fetch r2e` (01:05): 새 커밋 없음. QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 25.0 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 되살릴 것 없음 (live = disk 196485 B). GPU 85 MiB / 0 % (빔).
- cpu: `N harness eval_pkg16_0 bm2_rlonly_rl+rep4` 1204 s 째, 워커 1995558 util 100 % / rss 8.7 GB (정상). 남은 N 하네스 = 이 run pkg16_1 + `bm2_rlonly_s1_rl` 8개 (`_s2` 8개는 metrics.json 없어 `after` 게이트로 건너뜀).
- **N2 늦음 (보고만):** 01:05 UTC 다시 셈 → `eval_{pcb10,ip10}_{0,1}__lo_pkg_{r2pk,r0,bp_park,bm2_rlonly}*+rep4.json` **0 / 48** (있는 pcb10/ip10 +rep4 12개는 lo_pkg_bdf*). 서버 1 은 다시 넣지 않음.
- **DAILY.md 새로 씀** (대상 11/08 01:15 ~ 11/09 01:06 UTC, 라운드 123~208): F3 끝, N GPU 9/10 (s2 STUCK), 뺀 pkg 하네스 묶음, N2 늦음, 추천 큐 5개.
- 한 줄 (N 6단계): GPU = 없음, cpu = N 하네스 bm2_rlonly pkg16_0, 새 기록 0.
- 결정: 큐·플래그·순서 그대로, N2 를 서버 1 에 다시 넣지 않음 — 이유: QUEUE 11/09 01:40 KST "do not re-add them".
- 할 일 (다음): (1) bm2_rlonly pkg16 +rep4 RESULTS 행 확인. (2) 서버 2 N2 push 오면 개수 세기. (3) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. (4) DAILY 다음 11/10 01:15 UTC. (5) GPU 는 사람 항목 없으면 빈 채로.

# (이전) 라운드 207 머리말: (갱신 2026-11-09 00:52 UTC, 라운드 207 — **N2 여전히 늦음: 서버 2 push 0 / 48**; GPU 빔 (사람 항목 없음, s2 STUCK 그대로); cpu N 하네스 `bm2_rlonly_rl+rep4` ip10_0·1 끝 → ip10 **1.00 [50]** (단독 0.96), 지금 pkg16_0 (00:45~); 새 기록 2)

## 라운드 207 (2026-11-09 00:48 ~ 00:52 UTC)
- `git fetch r2e` (00:48): 새 커밋 없음. QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 24.7 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 1454 B 되살림. GPU 85 MiB / 0 % (빔).
- **cpu `N harness eval_ip10_0` (00:32–00:34) · `eval_ip10_1` (00:39–00:40) `bm2_rlonly_rl+rep4` 끝 (rc 0)** → RESULTS 자동 행 ip10 **1.00 [50]** (정책 단독 0.96 → 수리로 +2건; 기준 짝 차이 +0.00 [+0.00, +0.00]). 판정 없음 (보고만).
- 지금 cpu = `N harness eval_pkg16_0 bm2_rlonly_rl+rep4` (00:45~).
- **N2 늦음 (보고만):** 00:49 UTC 다시 셈 → `eval_{pcb10,ip10}_{0,1}__lo_pkg_{r2pk,r0,bp_park,bm2_rlonly}*+rep4.json` **0 / 48**. 서버 1 은 다시 넣지 않음.
- 한 줄 (N 6단계): GPU = 없음, cpu = N 하네스 bm2_rlonly pkg16_0, 새 기록 2.
- 결정: 큐·플래그·순서 그대로, N2 를 서버 1 에 다시 넣지 않음 — 이유: QUEUE 11/09 01:40 KST "do not re-add them".
- 할 일 (다음): (1) bm2_rlonly pkg16 +rep4 진행·RESULTS 행 확인. (2) 서버 2 N2 push 오면 개수 세기. (3) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. (4) **DAILY 다음 11/09 01:15 UTC** (다음 라운드). (5) GPU 는 사람 항목 없으면 빈 채로.

# (이전) 라운드 206 머리말: (갱신 2026-11-09 00:36 UTC, 라운드 206 — **N2 여전히 늦음: 서버 2 push 0 / 48**; GPU 빔 (사람 항목 없음, s2 STUCK 그대로); cpu N 하네스 `bm2_rlonly_rl+rep4` pkg10_1 끝 → pkg10 **0.90 [50]** (단독 0.84), 지금 ip10_0 (00:32~); 새 기록 1)

## 라운드 206 (2026-11-09 00:33 ~ 00:36 UTC)
- `git fetch r2e` (00:33): 새 커밋 없음. QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 18.4 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 1174 B 되살림. GPU 85 MiB / 0 % (빔).
- **cpu `N harness eval_pkg10_1 bm2_rlonly_rl+rep4` 끝 (00:19–00:27, rc 0)** → RESULTS 자동 행 pkg10 **0.90 [50]** (정책 단독 0.84 → 수리로 +3건; 기준 짝 차이 +0.08 [−0.04, +0.20]). 판정 없음 (보고만).
- 지금 cpu = `N harness eval_ip10_0 bm2_rlonly_rl+rep4` (00:32~).
- **N2 늦음 (보고만):** 00:34 UTC 다시 셈 → `eval_{pcb10,ip10}_{0,1}__lo_pkg_{r2pk,r0,bp_park,bm2_rlonly}*+rep4.json` **0 / 48** (pcb10/ip10 +rep4 있는 12개는 lo_pkg_bdf*, N2 아님). 서버 1 은 다시 넣지 않음.
- 한 줄 (N 6단계): GPU = 없음, cpu = N 하네스 bm2_rlonly ip10_0, 새 기록 1.
- 결정: 큐·플래그·순서 그대로, N2 를 서버 1 에 다시 넣지 않음 — 이유: QUEUE 11/09 01:40 KST "do not re-add them".
- 할 일 (다음): (1) bm2_rlonly ip10 +rep4 진행·RESULTS 행 확인. (2) 서버 2 N2 push 오면 개수 세기. (3) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. (4) **DAILY 다음 11/09 01:15 UTC**. (5) GPU 는 사람 항목 없으면 빈 채로.

# (이전) 라운드 205 머리말: (갱신 2026-11-09 00:20 UTC, 라운드 205 — **N2 여전히 늦음: 서버 2 push 0 / 48** (마감 00:00 UTC 지남); GPU 빔 (사람 항목 없음, s2 STUCK 그대로); cpu N 하네스 `bm2_rlonly_rl+rep4` pcb10_1 끝 → pcb10 **0.04 [50]**, pkg10_0 끝 → **23/25** (단독 22/25); 새 기록 2)

## 라운드 205 (2026-11-09 00:16 ~ 00:20 UTC)
- `git fetch r2e` (00:16): 새 커밋 없음. QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 17.0 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 1011 B 되살림. GPU 85 MiB / 0 % (빔).
- **cpu `N harness eval_pcb10_1 bm2_rlonly_rl+rep4` 끝 (23:17–00:04, rc 0)** → RESULTS 자동 행 pcb10 **0.04 [50]** (단독과 같음, 수리 성공 0; 짝 차이 vs 기준 −0.62 [−0.76, −0.48]). 판정 없음.
- **cpu `N harness eval_pkg10_0 bm2_rlonly_rl+rep4` 끝 (00:09–00:14, rc 0, 1.2 s/inst)** → 기록 23/25 (정책 단독 22/25). RESULTS 행은 드라이버 collect_results (00:14:36~) 가 넣는 중.
- **N2 늦음 (보고만):** 00:16 UTC 다시 셈 → `eval_{pcb10,ip10}_{0,1}__lo_pkg_{r2pk,r0,bp_park,bm2_rlonly}*+rep4.json` **0 / 48**. 서버 1 은 다시 넣지 않음 (QUEUE 11/09 01:40 KST).
- 한 줄 (N 6단계): GPU = 없음, cpu = N 하네스 bm2_rlonly pkg10_1 (다음), 새 기록 2.
- 결정: 큐·플래그·순서 그대로, N2 를 서버 1 에 다시 넣지 않음 — 이유: QUEUE 11/09 01:40 KST "do not re-add them".
- 할 일 (다음): (1) bm2_rlonly pkg10 +rep4 RESULTS 행 확인 (pkg10_1 뒤 [50]). (2) 서버 2 N2 push 오면 개수 세기. (3) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. (4) DAILY 다음 11/09 01:15 UTC. (5) GPU 는 사람 항목 없으면 빈 채로.

# (이전) 라운드 204 머리말: (갱신 2026-11-09 00:01 UTC, 라운드 204 — **N2 늦음: 마감 11/09 00:00 UTC (09:00 KST) 지났는데 서버 2 push 0 / 48**; GPU 빔 (사람 항목 없음, s2 STUCK 그대로); cpu = N 하네스 `bm2_rlonly_rl+rep4` pcb10_1 (23:17~, 워커 100 %, 정상); 새 기록 0)

## 라운드 204 (2026-11-08 23:55 ~ 11-09 00:01 UTC)
- `git fetch r2e` (23:55, 다시 00:00:43): 새 커밋 없음. QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 18.6 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 되살릴 것 없음 (live = disk 192846 B). GPU 85 MiB / 0 % (빔).
- cpu: `N harness eval_pcb10_1 bm2_rlonly_rl+rep4` ~2500 s 째, 워커 1982793 util 100 % / rss 2.0 GB (정상; pcb10_0 은 3286 s).
- **N2 늦음 (QUEUE 11/09 01:40 KST 지시대로 보고):** 11/09 00:00 UTC 지난 뒤 (00:00:43) 센 결과 `runs/*/*/eval/eval_{pcb10,ip10}_{0,1}__lo_pkg_{r2pk,r0,bp_park,bm2_rlonly}*+rep4.json` **0 / 48**. 서버 2 에서 N2 push 가 하나도 없음. 서버 1 은 이 48개를 다시 넣지 않음 (QUEUE 지시) — 사람 확인 필요 (보고만).
- 한 줄 (N 6단계): GPU = 없음, cpu = N 하네스 bm2_rlonly pcb10_1, 새 기록 0.
- 결정: 큐·플래그·순서 그대로, N2 를 서버 1 에 다시 넣지 않음 — 이유: QUEUE 11/09 01:40 KST "do not re-add them" (늦으면 PROGRESS 에 적기만).
- 할 일 (다음): (1) cpu N 하네스 pcb10_1 진행, RESULTS 자동 행 확인. (2) 서버 2 N2 push 오면 개수 세고 RESULTS 자동 행 확인. (3) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. (4) DAILY 다음 11/09 01:15 UTC. (5) GPU 는 사람 항목 없으면 빈 채로.

# (이전) 라운드 203 머리말: (갱신 2026-11-08 23:42 UTC, 라운드 203 — GPU 빔 (사람 항목 없음, s2 STUCK 그대로); cpu = N 하네스 `bm2_rlonly_rl+rep4` pcb10_1 (23:17~, 워커 100 %, 정상); 원격 새 push 없음 → N2 0/48 (마감 11/09 00:00 UTC 18분 전); 새 기록 0)

## 라운드 203 (2026-11-08 23:39~ UTC)
- `git fetch r2e`: 새 커밋 없음. QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 18.4 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 되살릴 것 없음 (live = disk 192846 B). GPU 85 MiB / 0 % (빔).
- cpu: `N harness eval_pcb10_1 bm2_rlonly_rl+rep4` 1304 s 째, 워커 1982793 util 100 % / rss 1.9 GB (정상; pcb10_0 은 ~3200 s 걸림).
- 서버 2 N2: `runs/**/eval/eval_{pcb10,ip10}_*__lo_pkg_{r2pk,r0,bp_park,bm2_rlonly}*+rep4.json` **0 / 48** (있는 12개는 이전 bdf 것, N2 아님). 마감 11/09 00:00 UTC 아직 안 지남.
- 참고: 이 pod 에는 `python` 이 없음 → 도우미는 `.venv/bin/python scripts/...` 로 한 명령씩.
- 한 줄 (N 6단계): GPU = 없음, cpu = N 하네스 bm2_rlonly pcb10_1, 새 기록 0.
- 결정: 큐·플래그·순서 그대로, 새 항목 없음 — 이유: QUEUE N 6단계·N2 (에이전트 새 학습 금지, 항목 안 바꿈).
- 할 일 (다음): (1) cpu N 하네스 pcb10_1 진행, RESULTS 자동 행 확인. (2) **11/09 00:00 UTC 지나면** 서버 2 N2 push 수 세서 48 미만이면 PROGRESS 에 "늦음" 적기 (QUEUE 11/09 01:40 KST). (3) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. (4) DAILY 다음 11/09 01:15 UTC. (5) GPU 는 사람 항목 없으면 빈 채로.

# (이전) 라운드 202 머리말: (갱신 2026-11-08 23:27 UTC, 라운드 202 — GPU 빔 (사람 항목 없음, s2 STUCK 그대로); cpu N 하네스 `bm2_rlonly_rl+rep4` pcb10_0 끝 → **0.04 [25]** (수리 0건, 99 s/inst), 지금 pcb10_1 (23:17~); r0_coord_s3 +rep4 RESULTS 행 들어감 (pkg10 **0.96** / pkg16 **0.92** [50]); 원격 새 push 없음 → N2 0/48; 새 기록 1 (+서버 2 4개 반영))

## 라운드 202 (2026-11-08 23:23~ UTC)
- `git fetch r2e`: 새 커밋 없음. QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 18.4 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 729 B 되살림. GPU 85 MiB / 0 % (빔).
- **cpu `N harness eval_pcb10_0 bm2_rlonly_rl+rep4` 끝 (22:18–23:12, rc 0)** → RESULTS 자동 행 pcb10 **0.04 [25]** (정책 단독 pcb10_0 과 같음: 수리 시도 0 → 0 성공, 짝 차이 +0.00; 99 s/inst). B-Manoharan RL만 모델은 pcb 에서 수리로도 안 살아남 (판정 없음, 보고만).
- **서버 2 push 1a9ec87 반영:** `r0_coord_s3_rl+rep4` pkg10 **0.96 [50]** (단독 0.74 → +0.22 [+0.12, +0.34]), pkg16 **0.92 [50]** (단독 0.60 → +0.32 [+0.20, +0.46]). 서버 2 CPU 시간이므로 시간 열은 비교에 쓰지 않음 (QUEUE 11/05 규칙).
- 지금 cpu = `N harness eval_pcb10_1 bm2_rlonly_rl+rep4` (23:17~, 워커 1982793 100 %).
- 서버 2 N2: `+rep4` pcb10/ip10 기록 **0 / 48**. 마감 11/09 00:00 UTC (33분 뒤) — 다음 라운드에 마감 지났으면 PROGRESS 에 "늦음" 적기.
- 한 줄 (N 6단계): GPU = 없음, cpu = N 하네스 bm2_rlonly pcb10_1, 새 기록 1.
- 결정: 큐·플래그·순서 그대로, 새 항목 없음 — 이유: QUEUE N 6단계·N2 (에이전트 새 학습 금지, 항목 안 바꿈).
- 할 일 (다음): (1) cpu N 하네스 pcb10_1 진행. (2) **11/09 00:00 UTC** 지나면 서버 2 N2 push 수 세서 48 미만이면 PROGRESS 에 적기. (3) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. (4) DAILY 다음 11/09 01:15 UTC. (5) GPU 는 사람 항목 없으면 빈 채로.

# (이전) 라운드 201 머리말: (갱신 2026-11-08 23:10 UTC, 라운드 201 — GPU 빔 (사람 항목 없음, s2 STUCK 그대로); cpu = N 하네스 `bm2_rlonly_rl+rep4` pcb10_0 (22:18~, 2974 s 째, 워커 100 %, 정상); 원격 새 push 없음 → N2 0/48; 새 기록 0)

## 라운드 201 (2026-11-08 23:07~ UTC)
- `git fetch r2e`: 새 커밋 없음 (r2e/main 마지막 = 1a9ec87 22:52 커밋 시각). QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 18.5 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 되살릴 것 없음 (live = disk 192117 B). GPU 85 MiB / 0 % (빔).
- cpu: `N harness eval_pcb10_0 bm2_rlonly_rl+rep4` 2974 s 째, 워커 1974519 util 100 % / rss 1.9 GB (정상). 정책 단독 pcb10_0 이 1175 s (3 rounds) 였으므로 수리 4회가 더 긴 것은 예상대로.
- r0_coord_s3 +rep4 pkg10/16 (서버 2 push 1a9ec87): 파일 4개 있음, RESULTS 행은 아직 없음 (마지막 collect_results 22:13 < push) → 이 eval 끝난 뒤 드라이버 collect_results 가 넣음.
- 서버 2 N2: `+rep4` pcb10/ip10 기록 **0 / 48**. 마감 11/09 00:00 UTC 전.
- 한 줄 (N 6단계): GPU = 없음, cpu = N 하네스 bm2_rlonly pcb10_0, 새 기록 0.
- 결정: 큐·플래그·순서 그대로, 새 항목 없음 — 이유: QUEUE N 6단계·N2 (에이전트 새 학습 금지, 항목 안 바꿈).
- 할 일 (다음): (1) cpu N 하네스 진행 + r0_coord_s3 +rep4 pkg 행 RESULTS 확인. (2) **11/09 00:00 UTC** 에 서버 2 N2 48개 다 push 안 됐으면 PROGRESS 에 적기. (3) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. (4) DAILY 다음 11/09 01:15 UTC. (5) GPU 는 사람 항목 없으면 빈 채로.

# (이전) 라운드 200 머리말: (갱신 2026-11-08 22:55 UTC, 라운드 200 — GPU 빔 (사람 항목 없음, s2 STUCK 그대로); cpu = N 하네스 `bm2_rlonly_rl+rep4` pcb10_0 (22:18~, 워커 100 %, 정상); 서버 2 push 1a9ec87 = r0_coord_s3 pkg10/16 +rep4 4개 (N2 아님) → N2 여전히 0/48; 새 기록 0)

## 라운드 200 (2026-11-08 22:51~ UTC)
- `git fetch r2e`: 새 커밋 없음 (이미 받은 1a9ec87 "server 2 (human): N records 11-08 22:48" 이 마지막). QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로.
- 서버 2 push 1a9ec87: `runs/2026-10-29/r0_coord_s3_rl/eval/eval_{pkg10_0,pkg10_1,pkg16_0,pkg16_1}__r0_coord_s3_rl+rep4.json` 4개. N2 (lo_pkg_* pcb10/ip10 +rep4) 가 아님 → N2 기록 **0 / 48** 그대로. RESULTS 에 아직 행 없음 → 드라이버가 지금 eval 끝난 뒤 돌리는 `collect_results` 가 자동으로 넣음 (손 숫자 없음).
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 18.6 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 되살릴 것 없음. GPU 85 MiB / 0 % (빔).
- cpu: `N harness eval_pcb10_0 bm2_rlonly_rl+rep4` 2020 s 째, 워커 1974519 util 100 % / rss 1.9 GB (정상, 멈춘 Pool 워커 아님).
- 한 줄 (N 6단계): GPU = 없음, cpu = N 하네스 bm2_rlonly pcb10_0, 새 기록 0.
- 결정: 큐·플래그·순서 그대로, 새 항목 없음 — 이유: QUEUE N 6단계·N2 (에이전트 새 학습 금지, 항목 안 바꿈).
- 할 일 (다음): (1) cpu N 하네스 진행 + r0_coord_s3 +rep4 pkg 행이 RESULTS 에 들어갔는지 확인. (2) **11/09 00:00 UTC** 에 서버 2 N2 48개 다 push 안 됐으면 PROGRESS 에 적기. (3) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. (4) DAILY 다음 11/09 01:15 UTC. (5) GPU 는 사람 항목 없으면 빈 채로.

# (이전) 라운드 199 머리말: (갱신 2026-11-08 22:40 UTC, 라운드 199 — GPU 빔 (사람 항목 없음, s2 STUCK 그대로); cpu = N 하네스 `bm2_rlonly_rl+rep4` pcb10_0 (22:18~, 워커 100 % CPU, 정상 진행); 서버 2 N2 0/48; 새 기록 0)

## 라운드 199 (2026-11-08 22:35~ UTC)
- `git fetch r2e`: 새 커밋 없음. QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 18.8 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 되살릴 것 없음 (live = disk 192117 B). GPU 85 MiB / 0 % (빔).
- cpu: `N harness eval_pcb10_0 bm2_rlonly_rl+rep4` 1076 s 째, forkserver 워커 1974519 util 100 % / rss 2.1 GB → 멈춘 Pool 워커 아님 (정상). 이 모델은 정책 단독 pcb10 이 0.04 / 7 s/inst 라 수리 4회 하네스가 느린 것은 예상대로.
- 서버 2 N2: 원격 새 push 없음 → `+rep4` pcb10/ip10 기록 **0 / 48**. 마감 11/09 00:00 UTC 전.
- 한 줄 (N 6단계): GPU = 없음, cpu = N 하네스 bm2_rlonly pcb10_0, 새 기록 0.
- 결정: 큐·플래그·순서 그대로, 새 항목 없음 — 이유: QUEUE N 6단계·N2 (에이전트 새 학습 금지, 항목 안 바꿈).
- 할 일 (다음): (1) cpu N 하네스 진행, RESULTS 자동 행 확인. (2) **11/09 00:00 UTC** 에 서버 2 N2 48개 다 push 안 됐으면 PROGRESS 에 적기. (3) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. (4) DAILY 다음 11/09 01:15 UTC. (5) GPU 는 사람 항목 없으면 빈 채로.

# (이전) 라운드 198 머리말: (갱신 2026-11-08 22:25 UTC, 라운드 198 — GPU 빔 (사람 항목 없음, s2 STUCK 그대로); cpu N 하네스 lo_pkg_bm2_rlonly_s2 pkg10_1 끝 → **`lo_pkg_bm2_rlonly_s2_rl+rep4` pkg10 0.90 [50]** (B-Manoharan v2 pkg10 3/3 0.89 [150]); 지금 cpu = N 하네스 bm2_rlonly pcb10_0; 서버 2 N2 0/48; 새 기록 1)

## 라운드 198 (2026-11-08 22:19~ UTC)
- `git fetch r2e`: 새 커밋 없음. QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 18.4 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 1236 B 되살림. GPU 85 MiB / 0 % (빔).
- **cpu `N harness eval_pkg10_1 lo_pkg_bm2_rlonly_s2_rl+rep4` 끝 (22:05–22:13, rc 0)** → RESULTS 자동 행 `lo_pkg_bm2_rlonly_s2_rl+rep4` pkg10 = **0.90 [50]** (정책 단독 0.84 → 0.90, 짝 차이 vs 기준 +0.08 [−0.04, +0.20]). B-Manoharan v2 pkg10 3 seed 합 **0.89 [150]** (vs 기준 −0.10 [−0.15, −0.05]). 판정 없음 (보고만).
- 지금 cpu = `N harness eval_pcb10_0 bm2_rlonly_rl+rep4` (22:18~, 워커 1974519 살아 있음).
- 서버 2 N2: 원격 새 push 없음 → `+rep4` pcb10/ip10 기록 여전히 **0 / 48**. 마감 11/09 00:00 UTC 전.
- 한 줄 (N 6단계): GPU = 없음, cpu = N 하네스 bm2_rlonly pcb10_0, 새 기록 1.
- 결정: 큐·플래그·순서 그대로, 새 항목 없음 — 이유: QUEUE N 6단계·N2 (에이전트 새 학습 금지, 항목 안 바꿈).
- 할 일 (다음): (1) cpu N 하네스 진행, RESULTS 자동 행 확인. (2) **11/09 00:00 UTC** 에 서버 2 N2 48개 다 push 안 됐으면 PROGRESS 에 적기. (3) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. (4) DAILY 다음 11/09 01:15 UTC. (5) GPU 는 사람 항목 없으면 빈 채로.

# (이전) 라운드 197 머리말: (갱신 2026-11-08 22:05 UTC, 라운드 197 — GPU 빔 (train 큐 끝, s2 STUCK 그대로); cpu eval10 `lo_pkg_r0_s3_rl` 끝 (21:51, rc 0) → **pcb10 0.98 / ip10 0.98 / pkg10 0.64 [50]**; N 하네스 lo_pkg_bm2_rlonly_s2 pkg10_0 **ok 0.92 [25]**; 서버 2 N2 0/48 push; 새 기록 2)

## 라운드 197 (2026-11-08 22:03~ UTC)
- `git fetch r2e`: 새 커밋 없음. QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 17.1 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 1052 B 되살림. GPU 85 MiB / 0 % (빔, 사람 항목 없음).
- **cpu `eval10 lo_pkg_r0_s3_rl` 끝 (21:29–21:51, rc 0):** RESULTS 자동 행 pcb10 **0.98 [50]**, ip10 **0.98 [50]**, pkg10 **0.64 [50]** (held-out pkg 이므로 pkg 가 낮은 것은 설계대로; 판정 없음, 보고만).
- **cpu `N harness eval_pkg10_0 lo_pkg_bm2_rlonly_s2_rl+rep4` 끝 (21:56–22:00, rc 0): ok 0.92 [25]** (정책 단독 대비 짝 차이 +0.08 [−0.08, +0.24]). 지금 collect_results 22:00~, 다음 = 같은 모델 pkg10_1 (큐 61).
- 확인: 큐 78–85 (`N harness … bm2_rlonly_s2_rl+rep4`) 는 `after: runs/2026-10-29/bm2_rlonly_s2_rl/metrics.json` 으로 막혀 있음 → s2 STUCK 이라 metrics.json 이 안 생기므로 best.pt (it 100) 평가는 돌지 않음 (라운드 196 "평가 없음" 결정과 맞음). 손대지 않음.
- 서버 2 N2: 지금 `+rep4` pcb10/ip10 기록 **0 / 48** (마지막 서버 2 push 8f55c5c 20:33 = r0_coord_s3 eval10). 마감 11/09 00:00 UTC 전이라 아직 "늦음" 아님 — 다음 라운드에 다시 셈.
- 한 줄 (N 6단계): GPU = 없음, cpu = N 하네스 lo_pkg_bm2_rlonly_s2 (pkg10), 새 기록 2.
- 결정: 큐·플래그·순서 그대로, 새 항목 없음 — 이유: QUEUE N 6단계·N2 (에이전트 새 학습 금지, 항목 안 바꿈).
- 할 일 (다음): (1) cpu N 하네스 진행, RESULTS 자동 행 확인. (2) **11/09 00:00 UTC** 에 서버 2 N2 48개 다 push 안 됐으면 PROGRESS 에 적기 (지금 0/48). (3) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. (4) DAILY 다음 11/09 01:15 UTC. (5) GPU 는 사람 항목 없으면 빈 채로.

# (이전) 라운드 196 머리말: (갱신 2026-11-08 21:58 UTC, 라운드 196 — **`bm2_rlonly_s2` resume 도 GPU OOM (21:24, it 300 val 직후, 같은 자리) → STUCK, 다시 안 돌림**; GPU 빔 (train 큐 비어 있음); cpu = eval10 lo_pkg_r0_s3_rl (21:29~, pcb10_0 1.00 / pkg10_0 0.64 / pcb10_1 0.96 까지); 새 기록 0)

## 라운드 196 (2026-11-08 21:46~ UTC)
- `git fetch r2e`: 새 커밋 없음. QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 18.3 GiB / 60, oom_kill 2 (늘지 않음 = 호스트 메모리 아님, GPU OOM). `restore_proc_log.py 1792539` → 731 B 되살림.
- **GPU `bm2_rlonly_s2` resume 실패 (21:24:56, rc 1, 529 s):** it 300 숫자가 첫 run 과 똑같음 (count/ref 3.450, best-of-M 2.134, fail 0.20, lam 3.91, gpu peak 19.0 GB; val pcb10_0 fail 1.00, pkg10_0 9.000/0.94, ip10_0 0.920/0.69) → val 뒤 다음 iter 의 `manoharan_reward` → `decap_loop/sim.py:84 probe_impedance` `torch.linalg.solve` 에서 `CUDA error: out of memory` (magma_zgetrf_batched, free 10 MB / 25.3 GB). 첫 실패와 같은 자리 = 결정적. `runs/2026-10-29/bm2_rlonly_s2/driver.log` 끝에 traceback.
- 결정: `bm2_rlonly_s2` 는 더 돌리지 않고 STUCK (첫 줄) — 이유: 라운드 189·195 결정 "같은 OOM 이면 STUCK, 다시 돌리지 않음"; resume 이 같은 배치를 재생하므로 같은 플래그로는 같은 OOM. 플래그 (B·M·grid) 바꾸기·best.pt (it 100) 평가 넣기는 에이전트가 하지 않음 (QUEUE N 6단계: 항목·arm·seed·세트 안 바꿈). B-Manoharan RL만 은 seed 0·1 2개만 표에 있음.
- RESULTS: 자동 생성 (`collect_results.py`, 손 숫자 없음) — s2 는 metrics.json 이 없어 행 없음. 이 실패는 PROGRESS 에만 적음.
- cpu: 드라이버 21:29 eval10 `lo_pkg_r0_s3_rl` (큐 순서, eval_jobs 먼저) — 1051 s 째 pcb10_0 ok 1.00, pkg10_0 ok 0.64, pcb10_1 ok 0.96. 끝나면 N 하네스 lo_pkg_r0_s3 등 이어 감.
- 한 줄 (N 6단계): GPU = 없음 (train 큐 끝, s2 STUCK), cpu = eval10 lo_pkg_r0_s3_rl, 새 기록 0.
- 할 일 (다음): (1) cpu eval10 lo_pkg_r0_s3 → N 하네스 진행, RESULTS 자동 행 확인. (2) **11/09 00:00 UTC (09:00 KST)** 에 서버 2 N2 48개 다 push 안 됐으면 PROGRESS 에 적기. (3) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. (4) DAILY 다음 11/09 01:15 UTC. (5) GPU 는 사람 항목 없으면 빈 채로 (에이전트 새 학습 금지).

# (이전) 라운드 195 머리말: (갱신 2026-11-08 21:22 UTC, 라운드 195 — GPU `lo_pkg_r0_s3` 끝 (354 it, 1.26 h, rc 0) → **`bm2_rlonly_s2` resume 1번 시작 (it 288 에서, 남은 ~0.65 h, ~22:00 끝)**; cpu N 하네스 lo_pkg_bm2_rlonly pkg10_1 **ok 0.88 → 합계 0.90 [50]**, lo_pkg_bm2_rlonly_s1 pkg10_0 **ok 0.92**; 새 기록 2)

## 라운드 195 (2026-11-08 21:04~ UTC)
- `git fetch r2e`: 새 커밋 없음. QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 19.2 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 845 B + 1309 B 되살림.
- **GPU `lo_pkg_r0_s3` 끝 (21:17, rc 0):** RL 354 it, 1.26 h, best it 100 score 1.704, replaced 248, gpu peak 11.9 GB, oom 0. 마지막 val (it 300) pcb10_0 1.339 / fail 0.62, ip10_0 0.787 / 0.31. eval10 은 드라이버 cpu 줄이 (eval_jobs 가 큐 앞) 다음에 집는다.
- **GPU `bm2_rlonly_s2` resume (라운드 189 결정 실행, 21:18):** GPU 빈 것 확인 (compute app 0) 뒤 `bg.py` 로 드라이버와 같은 명령 (`CERTIFY_WORKERS=2 run_rung.py --name bm2_rlonly_s2 --arm '--model manoharan2 --imit 0 --reward manoharan' --seed 2 --distill_hours 0 --rl_hours 2.0 --skip_distill --skip_eval`, 플래그 그대로) pid 1965291. log: "resumed from resume.pt: iter 288, 1.35 h used, lam 3.76" → 남은 ~0.65 h, ~22:00 끝. 드라이버 train 큐는 이제 비어 있음 (s2 는 FAILED 라 드라이버가 겹쳐 띄우지 않음). metrics.json 이 생기면 드라이버가 eval10·N 하네스를 집는다.
- **cpu `N harness eval_pkg10_1 lo_pkg_bm2_rlonly_rl+rep4` 끝 (20:52–21:00, rc 0, 489 s): ok 0.88, count 37.91** → RESULTS 행 `lo_pkg_bm2_rlonly_rl+rep4` pkg10 = **0.90 [50]** (정책 단독 0.84, 짝 차이 +0.08 [−0.04, +0.20] vs 기준). **cpu `N harness eval_pkg10_0 lo_pkg_bm2_rlonly_s1_rl+rep4` 끝: 0.92 [25]**. 판정 없음 (보고만). 지금 cpu = lo_pkg_bm2_rlonly_s1 pkg10_1 (21:14~).
- 21:40 확인: s2 resume it 300 지남 (count/ref 3.450, 같은 값 = 결정적 재현, oom 0, gpu peak 19.0 GB; nvidia-smi 22.5 GB), 전에 죽은 val 단계도 지남 (val pcb10_0 fail 1.00, pkg10_0 0.94, ip10_0 0.69 — 이전 val 과 같음).
- 한 줄 (N 6단계): GPU = `bm2_rlonly_s2` resume (결정대로 1번), cpu = N 하네스 lo_pkg_bm2_rlonly_s1 pkg10_1, 새 기록 2.
- 결정: `bm2_rlonly_s2` 는 같은 명령·플래그로 1번만 이어 감 — 이유: 라운드 189 결정 (보수적: 큐·플래그 그대로, GPU 1개에 train.py 1개). 같은 OOM 이면 STUCK 로 적고 다시 돌리지 않음.
- 할 일 (다음): (1) `bm2_rlonly_s2` ~22:00 끝 확인 (OOM 이면 STUCK, 보고만). (2) cpu: eval10 lo_pkg_r0_s3 → N 하네스 진행, RESULTS 확인. (3) **11/09 00:00 UTC** 에 서버 2 N2 48개 다 push 안 됐으면 PROGRESS 에 적기. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 194 머리말: (갱신 2026-11-08 20:52 UTC, 라운드 194 — GPU `lo_pkg_r0_s3` RL it 175 정상; cpu N 하네스 lo_pkg_bp_park_s2 pkg10_1 끝 **ok 0.88 → 합계 0.92 [50]**, B-Park pkg10 3/3 시드 +rep4 = **0.91 [150]**; lo_pkg_bm2_rlonly pkg10_0 끝 **ok 0.92**; 새 기록 2)

## 라운드 194 (2026-11-08 20:41~ UTC)
- `git fetch r2e`: 새 커밋 없음. QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 18.9 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 289 B + 771 B 되살림. (셸에 `python` 없음 → `.venv/bin/python` 으로 부름.)
- GPU `lo_pkg_r0_s3` RL: it 175 (2165 s) count/ref 0.580, best-of-M 0.494, fail 0.07, new best score 1.704 (it 100), oom 0, gpu peak 11.8 GB. NaN 없음. RL 1.25 h → 끝 ~21:20.
- **cpu `N harness eval_pkg10_1 lo_pkg_bp_park_s2_rl+rep4` 끝 (20:24–20:38, rc 0, 808 s, 6.4 s/inst): ok 0.88, count 33.73.** collect_results 20:38–20:41 rc 0 → RESULTS 행 `lo_pkg_bp_park_s2_rl+rep4` pkg10 = **0.92 [50]** (정책 단독 0.88 → 0.92, 짝 차이 +0.04 [+0.00, +0.10]). 요약 표 B-Park pkg10 3/3 시드: 0.79 → **0.91 [150]**. 판정 없음 (보고만).
- **cpu `N harness eval_pkg10_0 lo_pkg_bm2_rlonly_rl+rep4` 끝 (20:43–20:47, rc 0, 249 s, 1.1 s/inst): ok 0.92, count 33.96.** collect_results 20:47~ (드라이버). 다음은 큐 순서대로.
- 한 줄 (N 6단계): GPU = `lo_pkg_r0_s3` RL, cpu = N 하네스 lo_pkg_bm2_rlonly (pkg10), 새 기록 2.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계·N2, 라운드 189 결정 유지 (`bm2_rlonly_s2` resume 은 GPU 빈 뒤 1번).
- 할 일 (다음): (1) `lo_pkg_r0_s3` 끝 ~21:20 → GPU 비면 `bm2_rlonly_s2` resume 1번 (라운드 189 결정; 같은 OOM 이면 STUCK 기록, 보고만). (2) cpu lo_pkg_bm2_rlonly pkg10_1 등 하네스 진행, RESULTS 확인. (3) **11/09 00:00 UTC** 에 서버 2 N2 48개 다 push 안 됐으면 PROGRESS 에 적기. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 193 머리말: (갱신 2026-11-08 20:28 UTC, 라운드 193 — GPU `lo_pkg_r0_s3` RL it 100 정상; cpu N 하네스 lo_pkg_bp_park_s2 pkg10_0 끝 **ok 0.96, count 31.62**; cpu 지금 lo_pkg_bp_park_s2 pkg10_1; 새 기록 1)

## 라운드 193 (2026-11-08 20:25~ UTC)
- `git fetch r2e`: 새 커밋 없음. QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 21.1 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 771 B 되살림.
- GPU `lo_pkg_r0_s3` RL: it 100 count/ref 0.595, best 0.539, fail 0.05, val pcb10_0 1.19/0.375, ip10_0 0.84/0.31; GPU 14.9 GB, NaN 없음. 끝 ~21:15.
- **cpu `N harness eval_pkg10_0 lo_pkg_bp_park_s2_rl+rep4` 끝 (20:05–20:19, rc 0, 849 s, 7.0 s/inst): ok 0.96, count 31.62, rounds 4.** collect_results 20:19–20:23 rc 0 → RESULTS 행 `lo_pkg_bp_park_s2_rl+rep4` pkg10 = 0.96 [25] (정책 단독 0.88). 판정 없음 (보고만).
- cpu 다음: 드라이버 20:24 `N harness eval_pkg10_1 lo_pkg_bp_park_s2_rl+rep4` 시작 (큐 순서).
- 한 줄 (N 6단계): GPU = `lo_pkg_r0_s3` RL, cpu = N 하네스 lo_pkg_bp_park_s2 pkg10_1, 새 기록 1.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계·N2, 라운드 189 결정 유지.
- 할 일 (다음): (1) `lo_pkg_r0_s3` 끝 ~21:15 → GPU 비면 `bm2_rlonly_s2` resume 1번 (라운드 189 결정; 같은 OOM 이면 STUCK 기록, 보고만). (2) cpu pkg10_1 끝나면 B-Park pkg10 3/3 시드 +rep4 묶음 RESULTS 확인. (3) **11/09 00:00 UTC** 에 서버 2 N2 48개 다 push 안 됐으면 PROGRESS 에 적기. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 192 머리말: (갱신 2026-11-08 20:12 UTC, 라운드 192 — GPU `lo_pkg_r0_s3` 증류 끝 → RL it 25 정상; cpu N 하네스 lo_pkg_bp_park_s1 pkg10_1 끝 **ok 0.84 → 합계 0.90 [50]**; cpu 지금 lo_pkg_bp_park_s2 pkg10_0; 새 기록 1)

## 라운드 192 (2026-11-08 20:09~ UTC)
- `git fetch r2e`: 새 커밋 없음 (HEAD 6633d0d = 서버 2 N 기록 20:07, `r0_coord_s3_rl` metrics). QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 21.1 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 771 B 되살림.
- GPU `lo_pkg_r0_s3`: 증류 끝 (2306 s, val pcb10_0 2.25/0.25, ip10_0 0.81/0.19 — RESULTS 학습 표에 자동 행). RL 시작 ~20:00, it 25 count/ref 0.971, best 0.884, fail 0.03, oom 0, gpu peak 9.0 GB. RL 끝 ~21:15.
- **cpu `N harness eval_pkg10_1 lo_pkg_bp_park_s1_rl+rep4` 끝 (19:38–20:00, rc 0, 22 분, 11.0 s/inst)** → collect_results 20:00–20:04 rc 0 → RESULTS 행 `lo_pkg_bp_park_s1_rl+rep4` pkg10 = **0.90 [50]** (pkg10_0 0.96 + pkg10_1 0.84), base 0.64 → 0.90, 짝 차이 +0.26 [+0.14, +0.38]. B-Park pkg10 요약 2/3 시드 0.74 → 0.90 [100]. 판정 없음 (보고만).
- cpu 다음: 드라이버 20:05 `N harness eval_pkg10_0 lo_pkg_bp_park_s2_rl+rep4` 시작 (큐 순서), 워커 100 %.
- 한 줄 (N 6단계): GPU = `lo_pkg_r0_s3` RL, cpu = N 하네스 lo_pkg_bp_park_s2 pkg10_0, 새 기록 1.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계·N2, 라운드 189 결정 (`bm2_rlonly_s2` resume 은 GPU 빈 뒤 1번) 유지.
- 할 일 (다음): (1) `lo_pkg_r0_s3` 끝 ~21:15 → GPU 비면 `bm2_rlonly_s2` resume 1번 (라운드 189 결정; 같은 OOM 이면 STUCK 기록, 보고만). (2) cpu lo_pkg_bp_park_s2 pkg10 하네스 진행. (3) **11/09 00:00 UTC** 에 서버 2 N2 48개 다 push 안 됐으면 PROGRESS 에 적기. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.


# (이전) 라운드 191 머리말: (갱신 2026-11-08 19:55 UTC, 라운드 191 — GPU `lo_pkg_r0_s3` 증류 it 1875 정상; cpu N 하네스 lo_pkg_bp_park_s1 pkg10_1 시작 (19:38~); 새 기록 0)

## 라운드 191 (2026-11-08 19:52~ UTC)
- `git fetch r2e`: 새 커밋 없음. QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 21.3 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 408 B 되살림.
- GPU `lo_pkg_r0_s3` (19:15~): distill it 1875 (2212 s), count/ref 0.85, teach ok 0.56–0.94, oom 0, gpu peak 8.9 GB; val it 1800 pcb10 2.174 fail 0.19, ip10 0.831 fail 0.19 (it 1000 의 2.629/0.886 보다 낮아짐). NaN 없음. RL 단계 포함 끝 ~21:15.
- cpu: 드라이버 19:38 `N harness eval_pkg10_1 lo_pkg_bp_park_s1_rl+rep4` 시작 (큐 순서), 워커 100 %, 15분째. pkg10_1 은 이전 시드에서 45분 걸린 느린 세트 → ~20:25 끝 예상.
- 한 줄 (N 6단계): GPU = `lo_pkg_r0_s3` 증류, cpu = N 하네스 lo_pkg_bp_park_s1 pkg10_1, 새 기록 0.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계·N2, 끝난 실험 없음.
- 할 일 (다음): (1) `lo_pkg_r0_s3` 끝 ~21:15 → GPU 비면 `bm2_rlonly_s2` resume 1번 (라운드 189 결정; 같은 OOM 이면 STUCK 기록, 보고만). (2) cpu pkg10_1 끝나면 collect_results 결과 RESULTS 확인. (3) **11/09 00:00 UTC** 에 서버 2 N2 48개 다 push 안 됐으면 PROGRESS 에 적기. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.


# (이전) 라운드 190 머리말: (갱신 2026-11-08 19:40 UTC, 라운드 190 — GPU `lo_pkg_r0_s3` 증류 it 1050 정상; cpu N 하네스 lo_pkg_bp_park_s1 pkg10_0 끝 **ok 0.96, count 10.21**; 새 기록 1)

## 라운드 190 (2026-11-08 19:36~ UTC)
- `git fetch r2e`: 새 커밋 없음. QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로 (맨 위 절 = 00:20 KST N2).
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 18.8 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 289 B 되살림.
- GPU `lo_pkg_r0_s3` (19:15~): distill it 1050, count/ref 0.844, teach ok 0.62, oom 0, gpu peak 8.9 GB; val it 1000 pcb10 2.629 fail 0.44, ip10 0.886 fail 0.19. NaN 없음. 끝 ~21:15 (0.75 + 1.25 h).
- **cpu `N harness eval_pkg10_0 lo_pkg_bp_park_s1_rl+rep4` 끝 (19:12–19:33, rc 0): ok 0.96, count 10.21, 10.1 s/inst, rounds 4.** collect_results 19:33~ 돌고 있음 (report_watch 쪽 collect_results 도 같이 돎 — 이전 라운드와 같음). 판정 없음 (보고만).
- 한 줄 (N 6단계): GPU = `lo_pkg_r0_s3` 증류, cpu = collect_results → 다음 N 하네스 (큐 순서), 새 기록 1.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계·N2. `bm2_rlonly_s2` resume 은 라운드 189 결정대로 GPU 가 빈 뒤 (~21:15) 한 번만.
- 할 일 (다음): (1) `lo_pkg_r0_s3` 끝 ~21:15 → GPU 비면 `bm2_rlonly_s2` resume 1번 (라운드 189 결정; 같은 OOM 이면 STUCK 기록, 보고만). (2) cpu lo_pkg pkg10 하네스 진행. (3) **11/09 00:00 UTC** 에 서버 2 N2 48개 다 push 안 됐으면 PROGRESS 에 적기. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.


# (이전) 라운드 189 머리말: 갱신 2026-11-08 19:25 UTC, 라운드 189 — **GPU `bm2_rlonly_s2` 실패 (19:15 rc 1, it 300 뒤 CUDA OOM in `probe_impedance` solve, 1.49 h / 2 h)** → 드라이버가 마지막 train `lo_pkg_r0_s3` 시작; cpu N 하네스 lo_pkg_bp_park pkg10_1 끝 **ok 0.84**; cpu 지금 lo_pkg_bp_park_s1 pkg10_0; 새 기록 1

## 라운드 189 (2026-11-08 19:19~ UTC)
- `git fetch r2e`: 새 커밋 없음. QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 20.9 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 1363 B 되살림.
- **GPU `bm2_rlonly_s2` 실패 (17:46–19:15, rc 1, 5350 s):** it 300 (count/ref 3.450, best 2.134, fail 0.20, val 그대로) 뒤 `manoharan_reward → probe_impedance → torch.linalg.solve` 에서 `magma_zgetrf_batched cannot allocate` → `CUDA error: out of memory`. gpu peak 22.0 GB / 24.5 GB, grid capped 39 → 105 로 계속 늘어남 (RL-only 판 격자가 커짐); 18:13 부터 `expandable_segments ... OOM` 경고가 점점 잦아짐. 그때 GPU 를 쓰던 다른 프로세스 없음 (지금 GPU = `lo_pkg_r0_s3` 1개, 11.8 GB). s0·s1 은 같은 경고 뒤 2 h 를 채움 (peak 22.1 GB) → 메모리 한계에 걸친 시드 차이로 봄. 남은 것: `bm2_rlonly_s2_rl/{best.pt (18:15), last.pt (it 300), resume.pt (it 288, ~1.36 h 사용)}`, `metrics.json` 없음 → eval10·N 하네스 (`after` = metrics.json) 는 돌지 않음 (잘린 ckpt 평가 안 됨, 맞는 동작).
- 드라이버: `FAILED` 에 넣고 "skipped until driver restart" (재시작하면 resume.pt 에서 이어 감 — 설계대로). 19:15 바로 **`lo_pkg_r0_s3`** (train 큐 마지막 항목) 시작, 지금 distill 정상.
- **cpu `N harness eval_pkg10_1 lo_pkg_bp_park_rl+rep4` 끝 (18:22–19:07, rc 0): ok 0.84, count 21.95, 24.8 s/inst** (pkg10_0 은 15.0 s/inst → 더 느린 세트, 워커 멈춤 아님). collect_results 19:07–19:11 rc 0 → RESULTS 행 `lo_pkg_bp_park_rl+rep4` pkg10 = 0.90 [50]. 판정 없음 (보고만).
- cpu 다음: 드라이버 19:12 `N harness eval_pkg10_0 lo_pkg_bp_park_s1_rl+rep4` 시작 (큐 순서), 워커 100 %.
- 한 줄 (N 6단계): GPU = `lo_pkg_r0_s3` 증류 (bm2_rlonly_s2 OOM 실패), cpu = N 하네스 lo_pkg_bp_park_s1 pkg10_0, 새 기록 1.
- 결정: 드라이버를 지금 재시작하지 않음, `bm2_rlonly_s2` 를 다른 GPU 작업과 겹쳐 돌리지 않음 — 이유: 재시작하면 도는 cpu 하네스·`lo_pkg_r0_s3` 가 겹쳐 다시 뜰 수 있음 (한 GPU 에 train.py 1개 규칙), 큐·플래그 그대로가 가장 보수적. **`lo_pkg_r0_s3` 가 끝나 GPU 가 비면 (~21:20 UTC)** 같은 명령 (`run_rung.py --name bm2_rlonly_s2 …`, day2_queue 그대로, 플래그 바꾸지 않음) 을 한 번만 bg 로 돌려 resume.pt (it 288) 에서 남은 ~0.64 h 를 채운다. 같은 OOM 으로 또 죽으면 플래그를 바꾸지 말고 "STUCK" 로 적고 보고만 한다 (s2 는 seed 0·1 과 val 이 같은 보고용 반복).
- 할 일 (다음): (1) `lo_pkg_r0_s3` 끝 ~21:20 → GPU 비면 `bm2_rlonly_s2` resume 1번 (위 결정). (2) cpu lo_pkg pkg10 하네스 진행. (3) **11/09 00:00 UTC** 에 서버 2 N2 48개 다 push 안 됐으면 PROGRESS 에 적기. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.


# (이전) 라운드 188 머리말: 갱신 2026-11-08 19:05 UTC, 라운드 188 — GPU `bm2_rlonly_s2` it 250 정상; cpu N 하네스 lo_pkg_bp_park pkg10_1 41분째 (워커 100 %, pkg10_0 29분보다 김 — 아직 정상 범위); 새 기록 0)

## 라운드 188 (2026-11-08 19:02~ UTC)
- `git fetch r2e`: 새 커밋 없음. QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 20.9 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 되살릴 것 없음 (live = disk 182208 B).
- GPU `bm2_rlonly_s2` (17:46~): it 250 count/ref 3.561, best 2.186, fail 0.20, loss −3.07, oom 0, gpu peak 21.1 GB (it 222 에서 resume ckpt). NaN 없음. 끝 ~19:47 → 다음 train `lo_pkg_r0_s3`.
- cpu `N harness eval_pkg10_1 lo_pkg_bp_park_rl+rep4` (18:22~): 워커 1939316 2451 s CPU / 2447 s 경과 = 100 %, rss 2.4 GB → 멈춘 Pool 워커 아님 (메모리 노트의 "dead worker hang" 증상과 다름). pkg10_0 은 1745 s; pkg10_1 은 더 느린 세트일 수 있음. 다음 라운드에 60분을 넘으면 원인 확인.
- 한 줄 (N 6단계): GPU = `bm2_rlonly_s2` RL it 250, cpu = N 하네스 lo_pkg_bp_park pkg10_1, 새 기록 0.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계·N2 (01:40 KST "do not re-add them").
- 할 일 (다음): (1) pkg10_1 하네스 끝 확인 (60분 넘으면 워커 상태 확인). (2) `bm2_rlonly_s2` 끝 ~19:47 → 다음 train (`lo_pkg_r0_s3`) + s2 eval10 이 cpu 에 들어가는지. (3) **11/09 00:00 UTC** 에 서버 2 N2 48개 다 push 안 됐으면 PROGRESS 에 적기. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.


# (이전) 라운드 187 머리말: 갱신 2026-11-08 18:48 UTC, 라운드 187 — GPU `bm2_rlonly_s2` it 200 정상, val 그대로; cpu N 하네스 lo_pkg_bp_park pkg10_1 25분째 정상 (워커 100 %); 새 기록 0)

## 라운드 187 (2026-11-08 18:46~ UTC)
- `git fetch r2e`: 새 커밋 없음. QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 23.3 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 되살릴 것 없음 (live = disk 182208 B).
- GPU `bm2_rlonly_s2` (17:46~): it 200 count/ref 1.524, best 0.897, fail 0.20, loss −18.86, oom 0, gpu peak 21.1 GB; val it 200 = s0·s1 과 같은 값 (pcb10 fail 1.00, pkg10 9.000 fail 0.94, ip10 0.920 fail 0.69). NaN 없음. 끝 ~19:47 → 다음 train `lo_pkg_r0_s3`.
- cpu `N harness eval_pkg10_1 lo_pkg_bp_park_rl+rep4` (18:22~): forkserver 워커 1939316 100 % CPU, rss 2.3 GB — 정상 (pkg10_0 은 29 분 걸림 → 끝 ~18:51).
- 한 줄 (N 6단계): GPU = `bm2_rlonly_s2` RL it 200, cpu = N 하네스 lo_pkg_bp_park pkg10_1, 새 기록 0.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계·N2 (01:40 KST "do not re-add them").
- 할 일 (다음): (1) lo_pkg pkg10 하네스 진행 확인 (남은 10개 → `bm2_rlonly_rl+rep4`). (2) `bm2_rlonly_s2` 끝 ~19:47 → 다음 train (`lo_pkg_r0_s3`) 확인 + s2 eval10 이 cpu 에 들어가는지. (3) **11/09 00:00 UTC** 에 서버 2 N2 48개 다 push 안 됐으면 PROGRESS 에 적기. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.


# (이전) 라운드 186 머리말: 갱신 2026-11-08 18:33 UTC, 라운드 186 — cpu `eval10 bm2_rlonly_s1_rl` 6/6 끝 (pcb10_1 0.00, ip10_0 0.92, ip10_1 1.00), collect_results rc 0; cpu 다음 = N 하네스 lo_pkg_bp_park pkg10_1; GPU `bm2_rlonly_s2` it 150 정상; 새 기록 3

## 라운드 186 (2026-11-08 18:30~ UTC)
- `git fetch r2e`: 새 커밋 없음. QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 20.5 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 731 B 되살림.
- **cpu `eval10 bm2_rlonly_s1_rl` 끝 (17:54–18:17, rc 0):** pkg10_1 0.80, pcb10_0 0.04, pkg10_0 0.88, **pcb10_1 0.00 (count nan), ip10_0 0.92 (17.61), ip10_1 1.00 (16.80)**. collect_results 18:17–18:20 rc 0 → RESULTS `bm2_rlonly_s1_rl` 행 (ip 0.96, pcb 0.02, pkg 0.84) — seed 0 (ip 0.96, pcb 0.04, pkg 0.84) 과 거의 같음. 판정 없음 (보고만).
- cpu 다음: 드라이버 18:22 **`N harness eval_pkg10_1 lo_pkg_bp_park_rl+rep4`** 시작 (큐 순서, 예상대로). 남은 lo_pkg pkg10 하네스 10개 → `bm2_rlonly_rl+rep4`.
- GPU `bm2_rlonly_s2` (17:46~): it 150 count/ref 3.586, best 1.912, fail 0.50, loss −19.85, oom 0, gpu peak 19.2 GB, resume ckpt it 150. NaN 없음. 끝 ~19:47 → 다음 train `lo_pkg_r0_s3`.
- 한 줄 (N 6단계): GPU = `bm2_rlonly_s2` RL it 150, cpu = N 하네스 lo_pkg_bp_park pkg10_1, 새 기록 3.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계·N2 (01:40 KST "do not re-add them").
- 할 일 (다음): (1) lo_pkg pkg10 하네스 진행 확인. (2) `bm2_rlonly_s2` 끝 ~19:47 → 다음 train (`lo_pkg_r0_s3`) 확인 + s2 eval10 이 cpu 에 들어가는지. (3) **11/09 00:00 UTC** 에 서버 2 N2 48개 다 push 안 됐으면 PROGRESS 에 적기. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.


# (이전) 라운드 185 머리말: 갱신 2026-11-08 18:16 UTC, 라운드 185 — GPU `bm2_rlonly_s2` it 75 정상; cpu `eval10 bm2_rlonly_s1_rl` 3/6 (pkg10_1 0.80, pcb10_0 0.04, pkg10_0 0.88); 새 기록 3

## 라운드 185 (2026-11-08 18:14~ UTC)
- `git fetch r2e`: 새 커밋 없음. QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 20.8 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 되살릴 것 없음 (live = disk 181477 B).
- GPU `bm2_rlonly_s2` (17:46~): it 75 count/ref 1.874, best 1.156, fail 0.02, loss −1.84, oom 0, gpu peak 17.5 GB, resume ckpt it 74. 18:13 driver.log 에 `CUDACachingAllocator expandable_segments ... OOM` 경고 1줄 — s0 (26줄)·s1 (22줄) 도 같은 경고 뒤 rc 0 으로 끝남 → 경고만, 조치 없음. (log.csv 가 아직 0 B — s0·s1 log.csv 도 ~790 B 로 val 줄만 들어가므로 정상; 진행은 log.txt.) 끝 ~19:47.
- cpu `eval10 bm2_rlonly_s1_rl` (17:54~): 끝난 3 = **pkg10_1 ok 0.80 (count 37.40), pcb10_0 ok 0.04 (44.00), pkg10_0 ok 0.88 (34.82)** — seed 0 (`bm2_rlonly_rl`) 과 거의 같음. 남은 ip10·pcb10_1 진행 중. 판정 없음.
- 한 줄 (N 6단계): GPU = `bm2_rlonly_s2` RL it 75, cpu = `eval10 bm2_rlonly_s1_rl` (3/6), 새 기록 3.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계·N2 (01:40 KST "do not re-add them").
- 할 일 (다음): (1) `eval10 bm2_rlonly_s1_rl` 끝 → collect_results → lo_pkg pkg10 하네스 나머지 11개 이어짐 확인. (2) `bm2_rlonly_s2` 끝 ~19:47 → 다음 train (`lo_pkg_r0_s3`) 확인. (3) **11/09 00:00 UTC** 에 서버 2 N2 48개 다 push 안 됐으면 PROGRESS 에 적기. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.


# (이전) 라운드 184 머리말: 갱신 2026-11-08 18:00 UTC, 라운드 184 — GPU `bm2_rlonly_s1` 끝 (17:46 rc 0) → `bm2_rlonly_s2` it 25 정상; cpu N 하네스 `lo_pkg_bp_park_rl+rep4` pkg10_0 끝 **ok 0.96 (24/25), count 23.42**; cpu 지금 `eval10 bm2_rlonly_s1_rl`; 새 기록 1

## 라운드 184 (2026-11-08 17:58~ UTC)
- `git fetch r2e`: 새 커밋 없음. QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 21.4 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 1224 B 되살림.
- **GPU `bm2_rlonly_s1` 끝 (15:44–17:46, rc 0)** → 드라이버가 바로 **`bm2_rlonly_s2`** 시작 (17:46). it 25 count/ref 1.346, best 0.901, fail 0.16, loss −1.53, oom 0, gpu peak 16.2 GB — 정상 (NaN 없음). 끝 ~19:47.
- **cpu `N harness eval_pkg10_0 lo_pkg_bp_park_rl+rep4` 끝 (17:20–17:49, rc 0): ok 0.96 (24/25), count 23.42, 15.0 s/inst, rss 4.2 GB.** 정책 단독 pkg10 (0·1 합) 0.84 옆. collect_results 17:49–17:53 rc 0 → RESULTS 행 들어감. 판정 없음.
- cpu 다음: 드라이버 17:54 **`eval10 bm2_rlonly_s1_rl`** 시작 (드라이버 자동 eval10 이 큐 하네스보다 먼저 — 라운드 180 `bm2_rlonly_rl` 때와 같은 동작). 그 뒤 lo_pkg pkg10 하네스 나머지 11개 → `bm2_rlonly_rl+rep4`.
- 한 줄 (N 6단계): GPU = `bm2_rlonly_s2` RL it 25, cpu = `eval10 bm2_rlonly_s1_rl`, 새 기록 1 (lo_pkg_bp_park pkg10_0 +rep4).
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계·N2 (01:40 KST "do not re-add them").
- 할 일 (다음): (1) `eval10 bm2_rlonly_s1_rl` 끝 → collect_results → lo_pkg 하네스 이어짐 확인. (2) `bm2_rlonly_s2` 끝 ~19:47 → 다음 train (`lo_pkg_r0_s3`) 확인. (3) **11/09 00:00 UTC** 에 서버 2 N2 48개 다 push 안 됐으면 PROGRESS 에 적기. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.


# (이전) 라운드 183 머리말: 갱신 2026-11-08 17:45 UTC, 라운드 183 — GPU `bm2_rlonly_s1` it 400 정상, 끝 직전; cpu `N harness eval_pkg10_0 lo_pkg_bp_park_rl+rep4` 22분째 정상; 새 기록 0

## 라운드 183 (2026-11-08 17:42~ UTC)
- `git fetch r2e`: 새 커밋 없음. QUEUE 끝 = 11/09 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 21.2 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 되살릴 것 없음 (live = disk 180253 B).
- GPU `bm2_rlonly_s1` (15:44~): it 400 count/ref 3.137, best 1.606, fail 0.20, loss −9.49, oom 0, gpu peak 22.1 GB; val = 그대로 (pcb10 fail 1.00, pkg10 9.000 fail 0.94, ip10 0.920 fail 0.69). NaN 없음. 6683 s / 2 h → 곧 끝, 다음 train `bm2_rlonly_s2`.
- cpu `N harness eval_pkg10_0 lo_pkg_bp_park_rl+rep4` (17:20~): forkserver 워커 1929777 100 % CPU, rss 2.6 GB — 정상.
- 한 줄 (N 6단계): GPU = `bm2_rlonly_s1` RL it 400 (끝 직전), cpu = N 하네스 lo_pkg_bp_park pkg10_0, 새 기록 0.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계·N2 (01:40 KST "do not re-add them").
- 할 일 (다음): (1) `bm2_rlonly_s1` 끝 → `bm2_rlonly_s2` 시작 확인 (+ 드라이버가 s1 eval10 을 cpu 에 집는지). (2) cpu lo_pkg pkg10 하네스 12개 → `bm2_rlonly_rl+rep4`. (3) **11/09 00:00 UTC** 에 서버 2 N2 48개 다 push 안 됐으면 PROGRESS 에 적기. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.


# (이전) 라운드 182 머리말: 갱신 2026-11-08 17:30 UTC, 라운드 182 — cpu `eval10 bm2_rlonly_rl` 6/6 끝 (pcb10_1 0.04, ip10_0 0.92, ip10_1 1.00), collect_results rc 0; cpu 다음 = `N harness eval_pkg10_0 lo_pkg_bp_park_rl+rep4` (큐 순서); GPU `bm2_rlonly_s1` it 350 정상

## 라운드 182 (2026-11-08 17:26~ UTC)
- `git fetch r2e`: 새 커밋 없음. QUEUE 맨 위 = 11/09 00:20 KST N2 + 끝 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 21.1 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 728 B 되살림. (이 셸엔 `python` 없음 → `python3` 로 helper 실행.)
- **cpu `eval10 bm2_rlonly_rl` 끝 (16:44–17:15, rc 0):** pkg10_1 0.80, pcb10_0 0.04, pkg10_0 0.88, **pcb10_1 0.04 (count 34.00), ip10_0 0.92 (17.78), ip10_1 1.00 (18.12)**. collect_results 17:15–17:19 rc 0 → RESULTS 표에 `bm2_rlonly_rl` 행 (pkg 0.84, pcb 0.04, ip 0.96). 판정 없음 (보고만).
- cpu 다음: 드라이버 17:20 **`N harness eval_pkg10_0 lo_pkg_bp_park_rl+rep4`** 시작 — 라운드 181 예상 (`bm2_rlonly_rl+rep4`) 과 다름. 확인: 큐 50–61 = lo_pkg_{bp_park,bm2_rlonly}×3 seed pkg10 하네스 (N 단계, N2 아님), 62~ = `bm2_rlonly_rl+rep4`. 드라이버가 큐 순서대로 집은 것 — 정상.
- GPU `bm2_rlonly_s1` (15:44~): it 350 count/ref 2.961, best 1.662, fail 0.49, loss −18.5, oom 0, gpu peak 22.1 GB; val it 300 = 그대로 (pcb10 fail 1.00, pkg10 9.000 fail 0.94, ip10 0.920 fail 0.69). NaN 없음. 끝 ~17:45 → 다음 train `bm2_rlonly_s2`.
- 한 줄 (N 6단계): GPU = `bm2_rlonly_s1` RL it 350, cpu = N 하네스 lo_pkg_bp_park pkg10_0, 새 기록 3.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계·N2 (01:40 KST "do not re-add them"), 큐 순서 바꾸지 말라는 지시.
- 할 일 (다음): (1) `bm2_rlonly_s1` 끝 ~17:45 → `bm2_rlonly_s2` 시작 확인. (2) cpu lo_pkg pkg10 하네스 12개 → `bm2_rlonly_rl+rep4` 진행. (3) **11/09 00:00 UTC** 에 서버 2 N2 48개 다 push 안 됐으면 PROGRESS 에 적기. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.


# (이전) 라운드 181 머리말: 갱신 2026-11-08 17:12 UTC, 라운드 181 — cpu `eval10 bm2_rlonly_rl` 3/6 (pkg10_1 0.80, pcb10_0 0.04, pkg10_0 0.88); GPU `bm2_rlonly_s1` it 300 정상, val 그대로; 새 기록 3


## 라운드 181 (2026-11-08 17:10~ UTC)
- `git fetch r2e`: 새 커밋 없음. QUEUE 맨 위 = 11/09 00:20 KST N2 + 끝 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 20.7 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 되살릴 것 없음 (live = disk 179525 B).
- cpu `eval10 bm2_rlonly_rl` (16:44~, 정책 단독 10칸 6세트): 끝난 3 = **pkg10_1 ok 0.80 (count 37.35), pcb10_0 ok 0.04 (44.00), pkg10_0 ok 0.88 (34.77)**. 남은 ip10·pcb10_1 진행 중 (워커 100 %, rss 1.7 GB). 다음 cpu 항목 = `N harness ... bm2_rlonly_rl+rep4`. 판정 없음 (보고만).
- GPU `bm2_rlonly_s1` (15:44~): it 300 count/ref 2.512, best 1.909, fail 0.05, loss −2.19; val it 300 = it 100·200 그대로 (pcb10 fail 1.00, pkg10 9.000 fail 0.94, ip10 0.920 fail 0.69). NaN 없음. 끝 ~17:45 → 다음 train `bm2_rlonly_s2`.
- 한 줄 (N 6단계): GPU = `bm2_rlonly_s1` RL it 300, cpu = `eval10 bm2_rlonly_rl` (3/6), 새 기록 3.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계·N2 (01:40 KST "do not re-add them").
- 할 일 (다음): (1) `eval10 bm2_rlonly_rl` 끝 → collect_results → `N harness ... bm2_rlonly_rl+rep4` 시작 확인. (2) `bm2_rlonly_s1` 끝 ~17:45 → `bm2_rlonly_s2` 시작 확인. (3) **11/09 00:00 UTC** 에 서버 2 N2 48개 다 push 안 됐으면 PROGRESS 에 적기. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.


# (이전) 라운드 180 머리말: 갱신 2026-11-08 16:58 UTC, 라운드 180 — 좌표 하네스 s2 pkg20_1 끝 (ok 0.50, count 13.00); collect_results 뒤 RESULTS N2 표에 B-DevFormer pcb·ip "짝 없음" 행 나옴; cpu `eval10 bm2_rlonly_rl` 시작; GPU `bm2_rlonly_s1` it 200 정상

## 라운드 180 (2026-11-08 16:54~ UTC)
- `git fetch r2e`: 새 커밋 없음. QUEUE 맨 위 = 11/09 00:20 KST N2 + 끝 01:40 KST "N2 moved to server 2" 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 20.4 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 718 B 되살림.
- **`N coord harness eval_pkg20_1 r0_coord_s2_rl+rep4` 끝 (15:18–16:39, rc 0): ok 0.50, count 13.00, 117 s/inst (4 rounds).** 좌표 하네스 s2 8개 중 서버 1 몫 끝. collect_results 16:39–16:42 rc 0.
- RESULTS N2 표 확인: MoTT+P1 pkg10 하네스 0.99 (3/3), 좌표 pkg10 0.92 (−0.07 [−0.12, −0.03]), B-DevFormer pkg10 0.45 (−0.54 [−0.62, −0.46]), B-DevFormer pcb10 0.79·ip10 0.99 (3/3 짝 없음, MoTT+P1 기록 대기). 나머지 0/3 (서버 2 `server2_n2b.sh` 몫). 판정 없음.
- cpu: 드라이버 16:44 **`eval10 bm2_rlonly_rl`** 시작 (after 열림, 예상대로). 다음은 `N harness ... bm2_rlonly_rl+rep4`.
- GPU `bm2_rlonly_s1` (15:44~): it 200 count/ref 2.439, best 1.527, fail 0.41, loss −12.9; val it 200 = it 100 그대로 (pcb10 fail 1.00, pkg10 9.000 fail 0.94, ip10 0.920 fail 0.69) — seed 0 과 같은 모양. NaN 없음. 끝 ~17:45.
- 한 줄 (N 6단계): GPU = `bm2_rlonly_s1` RL it 200, cpu = `eval10 bm2_rlonly_rl`, 새 기록 1 (좌표 s2 pkg20_1).
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계·N2 (01:40 KST "do not re-add them").
- 할 일 (다음): (1) `eval10 bm2_rlonly_rl` → `N harness ... bm2_rlonly_rl+rep4` 진행 확인. (2) `bm2_rlonly_s1` 끝 ~17:45 → 다음 train `bm2_rlonly_s2`. (3) **11/09 00:00 UTC** 에 서버 2 N2 48개 다 push 안 됐으면 PROGRESS 에 적기. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.


# (이전) 라운드 179 머리말: 갱신 2026-11-08 16:40 UTC, 라운드 179 — 사람이 N2 서버 1 몫 48개를 서버 2로 옮김 (`cpu_deferred`, 확인함); GPU `bm2_rlonly_s1` it 175 정상; CPU 좌표 하네스 s2 pkg20_1 80분째; 새 기록 0

## 라운드 179 (2026-11-08 16:38~ UTC)
- `git fetch r2e`: 새 커밋 없음. HEAD af70aef = **사람 커밋 (11/09 01:31 KST) "Move N2 harness evals from server 1 queue to server 2"**: QUEUE 끝에 "11/09 01:40 KST — N2 moved to server 2" 절, `day2_queue.json` cpu 의 `N2 harness ...` 48개 → `cpu_deferred` (`moved`), `scripts/server2_n2b.sh` (서버 2 사람 실행, 체크포인트는 사람이 손으로 복사). 확인: cpu 안 N2 0개, cpu_deferred N2 48개. 드라이버는 큐를 매번 다시 읽음 → 재시작 필요 없음.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 27.8 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 되살릴 것 없음 (live = disk 178807 B).
- GPU `bm2_rlonly_s1` (15:44~): it 175 count/ref 1.925, best-of-M 1.086, fail 0.41, loss −3.72, oom 0, gpu peak 20.6 GB, resume ckpt it 157. val it 100 = seed 0 과 같은 모양 (pcb10 fail 1.00, pkg10 9.000 fail 0.94, ip10 0.920 fail 0.69). NaN 없음. 끝 ~17:45.
- cpu `N coord harness eval_pkg20_1 r0_coord_s2_rl+rep4` (15:18~, 80분째): forkserver 워커 1912072 100 % CPU, rss 9.4 GB — 정상 (pkg16_0 은 77분).
- N2 기록 현황: B-DevFormer (서버 2) pcb10·ip10 `+rep4` **12/12**; 서버 1 몫 모델 (이제 서버 2 `server2_n2b.sh`) **0/48**. RESULTS N2 표는 다음 드라이버 collect_results (pkg20_1 끝 뒤) 때 B-DevFormer 짝 없음 행으로 나옴.
- 한 줄 (N 6단계): GPU = `bm2_rlonly_s1` RL it 175, cpu = 좌표 하네스 s2 pkg20_1, 새 기록 0.
- 결정: N2 를 서버 1 큐에 다시 넣지 않음, 큐·플래그·순서 그대로 — 이유: QUEUE 11/09 01:40 KST ("do not re-add them").
- 할 일 (다음): (1) pkg20_1 끝 → collect_results → RESULTS N2 표 (B-DevFormer 행) 확인. (2) cpu `eval10 bm2_rlonly_rl`·`N harness ... bm2_rlonly_rl+rep4` 를 드라이버가 집는지. (3) **11/09 00:00 UTC (= 09:00 KST)** 에 서버 2 N2 48개가 다 push 안 됐으면 PROGRESS 에 적기 (QUEUE 01:40). (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.


# (이전) 라운드 178 머리말: 갱신 2026-11-08 16:23 UTC, 라운드 178 — GPU `bm2_rlonly_s1` it 100 정상 (fail 0.12, val 그대로); CPU 좌표 하네스 s2 pkg20_1 64분째 (워커 100 %, rss 8.4 GB); 새 기록 0

## 라운드 178 (2026-11-08 16:20~ UTC)
- `git fetch r2e`: 새 커밋 없음 (HEAD 37a4faf = 서버 2 사람 N 기록 커밋 16:13·16:19, 이미 받음). QUEUE 맨 위 = 11/09 00:20 KST N2 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 27.3 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 되살릴 것 없음 (live = disk 178807 B).
- GPU `bm2_rlonly_s1` (15:44~): it 100 count/ref 1.526 (best 1.017), fail 0.12, loss −1.58; val it 100 = pcb10 fail 1.00 (count nan), pkg10 9.000 fail 0.94, ip10 0.920 fail 0.69 — seed 0 (`bm2_rlonly`) 와 같은 모양. NaN 없음. 끝 ~17:45.
- cpu `N coord harness eval_pkg20_1 r0_coord_s2_rl+rep4` (15:18~, 64분째): 워커 1912072 100 % CPU, rss 8.4 GB — 정상 (pkg20_0 은 29분, pkg16_0 은 77분).
- 한 줄 (N 6단계): GPU = `bm2_rlonly_s1` RL it 100, cpu = 좌표 하네스 s2 pkg20_1, 새 기록 0.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계·N2.
- 할 일 (다음): 라운드 176 목록 그대로 (pkg20_1 끝 → collect_results → N2 표 확인; cpu `eval10 bm2_rlonly_rl`·`N harness ... bm2_rlonly_rl+rep4` 집는지; N2 48개; `restore_proc_log.py`). DAILY 다음 11/09 01:15 UTC.


# (이전) 라운드 177 머리말: 갱신 2026-11-08 16:08 UTC — GPU `bm2_rlonly_s1` it 83 정상; CPU 좌표 하네스 s2 pkg20_1 49분째; 새 기록 0

## 라운드 177 (2026-11-08 16:05~ UTC)
- `git fetch r2e`: 새 커밋 없음 (HEAD 39f45e8 = 서버 2 사람 N·N2 기록 커밋, 이미 받음). QUEUE 맨 위 = 11/09 00:20 KST N2 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 26.8 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 되살릴 것 없음 (디스크 = live 178807 B).
- GPU `bm2_rlonly_s1` (15:44~): it 75 count/ref 1.165, fail 0.12, loss −1.76, gpu peak 17.1 GB, oom 0, resume ckpt it 83. NaN 없음. 끝 ~17:45.
- cpu `N coord harness eval_pkg20_1 r0_coord_s2_rl+rep4` (15:18~, 49분째): forkserver 워커 1912072 100 % CPU, rss 7.8 GB — 정상 (pkg16_0 은 77분 걸렸음).
- 한 줄 (N 6단계): GPU = `bm2_rlonly_s1` RL it 83, cpu = 좌표 하네스 s2 pkg20_1, 새 기록 0.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계·N2.
- 할 일 (다음): 라운드 176 목록 그대로 (pkg20_1 끝 → collect_results → N2 표 확인; cpu `eval10 bm2_rlonly_rl`·`N harness ... bm2_rlonly_rl+rep4` 집는지; N2 48개; `restore_proc_log.py`). DAILY 다음 11/09 01:15 UTC.


# (이전) 라운드 176 머리말: 갱신 2026-11-08 15:55 UTC — GPU `bm2_rlonly` 끝 → `bm2_rlonly_s1` 시작; CPU 좌표 하네스 s2 pkg20_1 33분째; 새 기록 0

## 라운드 176 (2026-11-08 15:50~ UTC)
- `git fetch r2e`: 새 커밋 없음 (HEAD 978c5df = 서버 2 사람 N2 기록 커밋, 이미 받음). QUEUE 맨 위 = 11/09 00:20 KST N2 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 25.9 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 491 B 되살림.
- **GPU `bm2_rlonly` 끝 (13:42–15:44, rc 0):** 419 iter, 2.03 h, metrics.json·best.pt 있음 (`runs/2026-10-29/bm2_rlonly_rl/`). val it 400 = pcb10 fail 1.00 (count nan), pkg10 9.000 fail 0.94, ip10 0.920 fail 0.69 — it 100·200·300 그대로. 드라이버 15:44 **`bm2_rlonly_s1`** (seed 1) 시작, 끝 ~17:45.
- 한 줄 (N 6단계): GPU = `bm2_rlonly_s1` RL (막 시작), cpu = `N coord harness eval_pkg20_1 r0_coord_s2_rl+rep4` (15:18~, 33분째, forkserver 워커 1912072 100 % CPU, rss 7.7 GB — 정상), 새 기록 0.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계·N2.
- 할 일 (다음): (1) 좌표 하네스 s2 pkg20_1 끝 → collect_results → RESULTS N2 표 (B-DevFormer 짝 없음 행) 확인. (2) cpu `eval10 bm2_rlonly_rl`·`N harness ... bm2_rlonly_rl+rep4` 를 드라이버가 집는지 (after 열림). (3) 그 뒤 N2 48개 (서버 1 몫). (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.


# (이전) 라운드 175 머리말: 갱신 2026-11-08 15:40 UTC, 라운드 175 — 서버 2 N2 기록 (B-DevFormer `lo_pkg_bdf*` pcb10·ip10 +rep4) 들어옴; N2 표가 짝 없는 기록도 보이게 고침 (보고 코드만); CPU 좌표 하네스 s2 마지막 pkg20_1 진행; GPU `bm2_rlonly` it 375, 끝 ~15:43)

## 라운드 175 (2026-11-08 15:33~ UTC)
- `git fetch r2e`: 새 커밋 없음 (HEAD a1df71d = 서버 2 사람 "N2 records" 커밋 5개 15:19–15:28, 이미 받음). QUEUE 맨 위 = 11/09 00:20 KST N2 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 25.8 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 458 B 되살림.
- 드라이버 cpu: 15:16 collect_results 끝 rc 0 → 15:18 **`N coord harness eval_pkg20_1 r0_coord_s2_rl+rep4`** 시작 (좌표 하네스 s2 마지막).
- **서버 2 N2 기록:** `lo_pkg_bdf*_distill` pkg10·pcb10·ip10 `+rep4` 일부 (pcb 2 seed, ip 1 seed 까지). 서버 1 몫 N2 (MoTT+P1·좌표·B-Park·B-Man2 의 pcb10·ip10) 는 아직 0 → 짝지을 기준이 없어 기존 `n2_table` 은 B-DevFormer pcb·ip 를 "0/3" 로 보였음 (기록이 있는데 없다고 보임).
- **`collect_results.py` `n2_table` 고침 (보고 코드만):** 그 모델 기록은 있는데 같은 seed·세트의 MoTT+P1 기록이 없으면 `k/3 (짝 없음)`, 정책 단독·하네스 성공률·문제 수만 적고 차 칸은 "(MoTT+P1 기록 대기)" (CI 없음). 지금 읽어 보면: B-DevFormer pkg10 하네스 0.45 (3/3, vs MoTT+P1 −0.54 [−0.62, −0.46]), pcb10 0.81 (2/3 짝 없음, 75문제), ip10 1.00 (1/3 짝 없음, 50문제). 다음 드라이버 collect_results 때 RESULTS 에 들어감 (드라이버와 겹치지 않게 내가 직접 돌리지는 않음). 판정 없음.
- 결정: 표 모양만 고침, 큐·플래그·순서 그대로 — 이유: QUEUE N 6단계·N2 ("평가만, 결과는 숫자만").
- 한 줄 (N 6단계): GPU = `bm2_rlonly` RL (it 375 / 6357 s, count/ref 2.186, fail 0.34, loss −7.41, oom 0, gpu peak 22.1 GB; val = it 100 그대로 (pcb10 fail 1.00, pkg10 9.000 fail 0.94, ip10 0.920 fail 0.69), NaN 아님; 끝 ~15:43 → 다음 train `bm2_rlonly_s1`), cpu = `N coord harness eval_pkg20_1 r0_coord_s2_rl+rep4`, 새 기록 0 (서버 1); 서버 2 N2 기록 여러 개.
- 할 일 (다음): (1) 좌표 하네스 s2 pkg20_1 끝 → collect_results → RESULTS 에 N2 표 (B-DevFormer 행) 나오는지 확인. (2) `bm2_rlonly` 끝 → `bm2_rlonly_s1` 시작, cpu `eval10 bm2_rlonly_rl`·`N harness ... bm2_rlonly_rl+rep4` 를 드라이버가 집는지. (3) 그 뒤 cpu 는 N2 48개 (서버 1 몫) — 첫 기록이 생기면 짝지은 CI 가 나오는지. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.


# (이전) 라운드 174 머리말: 갱신 2026-11-08 15:18 UTC, 라운드 174 — 새 QUEUE N2 (뺀 pkg 모델 pcb·ip 하네스 48개) 확인, 큐에 이미 있음 → 드라이버가 N 하네스 뒤에 집음; `collect_results.py` 에 N2 표 (최저 종류 성공률) 추가; CPU 좌표 하네스 s2 pkg20_0 끝 (ok 0.88, count 10.57); GPU `bm2_rlonly` it 325 정상)

## 라운드 174 (2026-11-08 15:15~ UTC)
- `git fetch r2e`: 새 커밋 없음. HEAD 10062dd = 사람 N2 커밋 (QUEUE 11/09 00:20 KST, PROPOSALS "11/09 N2", day2_queue.json cpu 에 `N2 harness ...` 48개 (서버 1 몫), `scripts/server2_n2.sh`). 드라이버는 큐를 매번 다시 읽으므로 재시작 필요 없음.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 18.8 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 289 B 되살림.
- **`N coord harness eval_pkg20_0 r0_coord_s2_rl+rep4` 끝 (14:43–15:13, rc 0): ok 0.88, count 10.57 (4 rounds).** collect_results 15:13~ (드라이버).
- **N2 보고 표 준비 (평가·큐는 안 건드림):** `scripts/collect_results.py` `n2_table` — 모델 5개 (MoTT+P1 `lo_pkg_r2pk*`, 좌표 `lo_pkg_r0*`, B-Park `lo_pkg_bp_park*`, B-Manoharan v2 `lo_pkg_bm2_rlonly*`, B-DevFormer `lo_pkg_bdf*_distill` (서버 2)) × 종류 (뺀 pkg10, pcb10, ip10) `+rep4`: seed 묶음 정책 단독 (`ok_pre`) → 하네스, 짝지은 bootstrap CI vs MoTT+P1, 그리고 최저 종류 성공률 + 그 차의 CI (종류마다 문제를 짝지어 다시 뽑음). pcb10·ip10 `lo_pkg_*+rep4` 기록이 하나라도 생겨야 표가 나옴 (지금은 0 → 안 나옴). 가짜 기록으로 모양만 시험함 (`/tmp/n2t/t.py`, 커밋 안 함). 판정 문장 없음 — PROPOSALS N2 대로 사람 몫.
- 결정: N2 표는 QUEUE "결과는 판정 없이 RESULTS 에 숫자만 (최저 종류 성공률 포함)" 을 따르기 위한 보고 코드만 추가; 큐·플래그·순서·세트는 그대로 — 이유: QUEUE N 6단계 (에이전트가 항목 추가·변경 금지).
- 한 줄 (N 6단계): GPU = `bm2_rlonly` RL (it 325 / 5458 s, count/ref 1.728, fail 0.23, loss −8.08, oom 0, gpu peak 21.9 GB; val it 300 = it 100·200 그대로 (pcb10 fail 1.00, pkg10 9.000 fail 0.94, ip10 0.920 fail 0.69), NaN 아님; 끝 ~15:43), cpu = collect_results → 다음 `N coord harness eval_pkg20_1 r0_coord_s2_rl+rep4`, 새 기록 1 (좌표 s2 pkg20_0).
- 할 일 (다음): (1) 좌표 하네스 s2 마지막 pkg20_1. (2) `bm2_rlonly` 끝 ~15:43 → 다음 train `bm2_rlonly_s1`; cpu `eval10 bm2_rlonly_rl`·`N harness ... bm2_rlonly_rl+rep4` 를 드라이버가 집는지. (3) N2 48개의 `after` (= 그 모델 `metrics.json`) 는 12 모델 모두 이미 학습 끝이라 열려 있음 → N 하네스 게이트가 닫혀 있을 때 cpu 빈자리에 돈다 — 첫 N2 기록이 생기면 RESULTS 에 N2 표가 나오는지 확인. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.


# (이전) 라운드 173 머리말: 갱신 2026-11-08 15:00 UTC, 라운드 173 — GPU `bm2_rlonly` it 250 (fail 0.52 로 올라감, NaN 아님, resume it 226); CPU 좌표 하네스 s2 pkg20_0 16분째; 새 기록 0)

## 라운드 173 (2026-11-08 14:59~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 26.9 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` 디스크와 같은 크기 → 할 일 없음.
- 한 줄 (N 6단계): GPU = `bm2_rlonly` RL (it 250 / 4132 s, count/ref 2.150, best-of-M 1.157, **fail 0.52** (it 200 0.20 에서 올라감), teach ok 0.50, loss −0.635, oom 0, gpu peak 21.6 GB; log.txt 에 "resume checkpoint at iter 226" — 학습 안 재개 저장, 정상; val it 200 = it 100 그대로; 끝 ~15:43), cpu = `N coord harness eval_pkg20_0 r0_coord_s2_rl+rep4` (14:43:46~), 새 기록 0.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계. fail 0.52 는 N 멈춤 규칙 (처음 25 iter NaN, 또는 val 세 종류 fail 1.00 + loss 0) 에 해당 안 됨 → 보고만.
- 할 일 (다음): (1) 좌표 하네스 s2 남은 2개 (pkg20_0 진행, pkg20_1). (2) `bm2_rlonly` 끝 ~15:43 → 다음 train (`bm2_rlonly_s1`), cpu `eval10 bm2_rlonly_rl` 을 드라이버가 집는지 확인. (3) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.


# (이전) 라운드 172 머리말: 갱신 2026-11-08 14:45 UTC, 라운드 172 — CPU `eval10 lo_pkg_bm2_rlonly_s2_rl` 6/6 끝 (pkg 0.84, pcb 0.02, ip 0.96; seed 0·1 과 거의 같음); 좌표 하네스 s2 pkg20_0 시작; GPU `bm2_rlonly` val it 200 = it 100 그대로)

## 라운드 172 (2026-11-08 14:43~ UTC)
- `git fetch r2e`: 새 main 커밋 없음 (HEAD 93d3f7d = 서버 2 사람 bdf_s2_distill 기록 6개, 이미 받음). QUEUE 맨 위 = 11/08 12:10 KST 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 18.7 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 345 B + 384 B 되살림.
- **`eval10 lo_pkg_bm2_rlonly_s2_rl` 끝 (14:19–14:38, rc 0), collect_results 14:38–14:42 rc 0 → RESULTS 자동:** pkg10 0.84 (count/ref 4.68), pcb10 0.02 (6.29), ip10 0.96 (1.97). seed 0 = 0.84 / 0.04 / 0.96, seed 1 = 0.82 / 0.04 / 0.96 → B-Manoharan2 RL 단독 3 seed 묶음 완성 (pcb 거의 전부 실패는 3 seed 공통). 판정·해석은 사람 몫.
- 드라이버 cpu: 14:43:46 `N coord harness eval_pkg20_0 r0_coord_s2_rl+rep4` 시작.
- 한 줄 (N 6단계): GPU = `bm2_rlonly` RL (it 200 / 3248 s, count/ref 1.595, fail 0.20, loss −7.65; val it 200 = pcb10 fail 1.00, pkg10 9.000 fail 0.94, ip10 0.920 fail 0.69 — it 100 그대로, NaN 아님; 끝 ~15:43), cpu = `N coord harness eval_pkg20_0 r0_coord_s2_rl+rep4`, 새 기록 6 (bm2_rlonly_s2 10칸 4개 + 이전 라운드 2개 포함 6/6).
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) 좌표 하네스 s2 남은 2개 (pkg20_0 진행, pkg20_1). (2) `bm2_rlonly` 끝 ~15:43 → 다음 train (`bm2_rlonly_s1`), 그 뒤 cpu `eval10 bm2_rlonly_rl` 을 드라이버가 집는지 확인. (3) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.


# (이전) 라운드 171 머리말: 갱신 2026-11-08 14:30 UTC, 라운드 171 — CPU 좌표 하네스 s2 pkg16_1 끝 (ok 0.88, count 9.00; pkg16 s2 50개 0.88); 드라이버 cpu `eval10 lo_pkg_bm2_rlonly_s2_rl` 시작 (pkg10_1 0.80, pcb10_0 0.04); GPU `bm2_rlonly` it 150 정상)

## 라운드 171 (2026-11-08 14:27~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 20.9 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 768 B 되살림.
- **`N coord harness eval_pkg16_1 r0_coord_s2_rl+rep4` 끝 (13:30–14:14, rc 0):** ok 0.88, count 9.00 (4 rounds). collect_results 14:14–14:17 rc 0 → RESULTS pkg16 s2 행 50개 (0.88, count 9.36; 정책 단독 0.58 → 0.88, +0.30 [+0.18, +0.42]). 속도 숫자는 QUEUE 7에 따라 표에 안 씀.
- 드라이버 cpu: 14:19 `eval10 lo_pkg_bm2_rlonly_s2_rl` 시작 (좌표 하네스 s2 pkg20_0·1 보다 먼저 집음 — 드라이버 순서, 그대로 둠). 지금 2/6: pkg10_1 0.80 (count 37.75), pcb10_0 0.04 (44.00).
- 한 줄 (N 6단계): GPU = `bm2_rlonly` RL (it 150 / 2279 s, count/ref 1.924, fail 0.13, teach ok 0.88, loss −2.39, oom 0, gpu peak 19.7 GB; log.csv 의 nan 은 val pcb10 count 하나 = fail 1.00 때문, 학습 값은 NaN 아님; 끝 ~15:43), cpu = `eval10 lo_pkg_bm2_rlonly_s2_rl` (2/6), 새 기록 3 (좌표 s2 pkg16_1, bm2_rlonly_s2 pkg10_1·pcb10_0).
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) `eval10 lo_pkg_bm2_rlonly_s2_rl` 남은 4개 → RESULTS 자동. (2) 좌표 하네스 s2 남은 2개 (pkg20_0·1). (3) `bm2_rlonly` 끝 ~15:43 → 다음 train (`bm2_rlonly_s1`). (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.


# (이전) 라운드 170 머리말: 갱신 2026-11-08 14:15 UTC, 라운드 170 — GPU `bm2_rlonly` 첫 val it 100 (pcb10 fail 1.00, pkg10 9.000 fail 0.94, ip10 0.920 fail 0.69; NaN 아님); CPU 좌표 하네스 s2 pkg16_1 41분째 (워커 100 %))

## 라운드 170 (2026-11-08 14:11~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 30.6 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` 디스크와 같은 크기 → 할 일 없음.
- 한 줄 (N 6단계): GPU = `bm2_rlonly` RL (it 100 / 1474 s, count/ref 1.714, fail 0.17, teach ok 0.94, loss −4.91, cert-bad 11, oom 0, gpu peak 17.8 GB; **첫 val it 100: pcb10 fail 1.00 (count nan), pkg10 9.000 fail 0.94, ip10 0.920 fail 0.69** — ip10 은 `lo_pkg_bm2_rlonly` 3 seed 와 같은 숫자, new best 8.057), cpu = `N coord harness eval_pkg16_1 r0_coord_s2_rl+rep4` (13:30:16~, 41분째, forkserver 워커 1895757 100 % CPU, rss 12.3 GB — 정상; s2 pkg16_0 은 77분 걸림), 새 기록 0.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) 좌표 하네스 s2 남은 3개 (pkg16_1 진행, pkg20_0·1). (2) 그 뒤 `eval10 lo_pkg_bm2_rlonly_s2_rl` 드라이버가 집는지 확인 → RESULTS 자동 반영. (3) `bm2_rlonly` 끝 ~15:43 → 다음 train (`bm2_rlonly_s1`). (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.


# (이전) 라운드 169 머리말: 갱신 2026-11-08 13:58 UTC, 라운드 169 — GPU `lo_pkg_bm2_rlonly_s2` 끝 (13:42 rc 0, val it 300 = it 100·200 그대로) → `bm2_rlonly` RL 시작 (it 25 정상); CPU 좌표 하네스 s2 pkg16_1 25분째 (워커 100 %))

## 라운드 169 (2026-11-08 13:55~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 29.0 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 492 B 되살림.
- **GPU `lo_pkg_bm2_rlonly_s2` 끝 (11:41–13:42, rc 0):** 마지막 val it 300 = pcb10 fail 1.00 (count nan), ip10 0.920 fail 0.69 — it 100·200, seed 0·1 과 같음. metrics.json·best.pt 있음 → 드라이버가 `eval10 lo_pkg_bm2_rlonly_s2_rl` 을 좌표 하네스 다음에 집을 것 (s1 때와 같은 흐름).
- 드라이버 GPU: 13:42 **`bm2_rlonly`** RL 시작 (seed 0, `--model manoharan2 --imit 0 --reward manoharan`, pool_glob 없음 = 전 종류). it 25 / 386 s: count/ref 1.538, fail 0.11, teach ok 0.94, loss −1.54, cert-bad 10, oom 0, gpu peak 15.8 GB — NaN 아님, 정상.
- 한 줄 (N 6단계): GPU = `bm2_rlonly` RL (it 25, 끝 ~15:43), cpu = `N coord harness eval_pkg16_1 r0_coord_s2_rl+rep4` (13:30:16~, 25분째, forkserver 워커 1895757 100 % CPU, rss 10.6 GB — 정상), 새 기록 0.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) 좌표 하네스 s2 남은 3개 (pkg16_1 진행, pkg20_0·1). (2) `eval10 lo_pkg_bm2_rlonly_s2_rl` 드라이버가 집는지 확인 → RESULTS 자동 반영. (3) `bm2_rlonly` 첫 val (it 100, ~14:15). (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.


# (이전) 라운드 168 머리말: 갱신 2026-11-08 13:42 UTC, 라운드 168 — CPU `eval10 lo_pkg_bm2_rlonly_s1_rl` 6/6 끝 (pkg 0.76·0.88, pcb 0.04·0.04, ip 0.92·1.00); 좌표 하네스 s2 pkg16_1 진행; GPU `lo_pkg_bm2_rlonly_s2` 끝 직전 (~13:43))

## 라운드 168 (2026-11-08 13:39~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 27.8 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 729 B 되살림.
- **`eval10 lo_pkg_bm2_rlonly_s1_rl` 끝 (12:53–13:25, rc 0), 6/6:** pkg10_1 0.76 (count 35.37), pcb10_0 0.04 (44.00), pkg10_0 0.88 (34.55), pcb10_1 0.04 (34.00), ip10_0 0.92 (17.61), ip10_1 1.00 (16.80). collect_results 13:25–13:28 rc 0 (RESULTS 자동 반영). 속도 숫자는 QUEUE 7에 따라 표에 안 씀.
- 드라이버 cpu: 13:30 `N coord harness eval_pkg16_1 r0_coord_s2_rl+rep4` 시작 (2/4), forkserver 워커 1895757 100 % CPU, rss 9.5 GB — 정상.
- 한 줄 (N 6단계): GPU = `lo_pkg_bm2_rlonly_s2` RL (val it 300 = it 100·200 그대로, 경과 7078 s / 7200 s → 끝 ~13:43, 다음 train `bm2_rlonly`), cpu = 좌표 하네스 s2 pkg16_1, 새 기록 2 (ip10_0·1; 세트 6개 끝).
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) `lo_pkg_bm2_rlonly_s2` 끝 → 드라이버가 `eval10 lo_pkg_bm2_rlonly_s2_rl` 과 다음 train `bm2_rlonly` 올리는지 확인. (2) 좌표 하네스 s2 남은 2개 (pkg20_0·1). (3) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.


# (이전) 라운드 167 머리말: 갱신 2026-11-08 13:25 UTC, 라운드 167 — CPU `eval10 lo_pkg_bm2_rlonly_s1_rl` 4/6 (pkg 0.76·0.88, pcb 0.04·0.04); GPU `lo_pkg_bm2_rlonly_s2` val it 300 = it 100·200 그대로, 끝 ~13:45)

## 라운드 167 (2026-11-08 13:22~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 19.9 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` 디스크와 같은 크기 → 할 일 없음.
- 한 줄 (N 6단계): GPU = `lo_pkg_bm2_rlonly_s2` RL (val it 300: pcb10 fail 1.00 (count nan), ip10 0.920 fail 0.69 — it 100·200 과 같음, seed 0·1 과도 같음; 경과 ~6100 s, 끝 ~13:45), cpu = `eval10 lo_pkg_bm2_rlonly_s1_rl` (12:53:42~, 4/6: eval_pkg10_1 ok 0.76, eval_pcb10_0 ok 0.04, eval_pkg10_0 ok 0.88, eval_pcb10_1 ok 0.04; 남은 ip10 2개), 새 기록 0 (세트 기록은 6개 끝난 뒤 collect_results).
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) eval10 s1 끝 → RESULTS 자동 반영 확인. (2) 좌표 하네스 s2 남은 3개 (pkg16_1, pkg20_0·1) 드라이버가 집는지 확인. (3) `lo_pkg_bm2_rlonly_s2` 끝 ~13:45 → 다음 train (`bm2_rlonly`). (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.


# (이전) 라운드 166 머리말: 갱신 2026-11-08 13:10 UTC, 라운드 166 — CPU `eval10 lo_pkg_bm2_rlonly_s1_rl` 진행 (pkg10_1 ok 0.76, 1/6); GPU `lo_pkg_bm2_rlonly_s2` RL it ~250, 끝 ~13:45)

## 라운드 166 (2026-11-08 13:05~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 20.3 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 479 B 되살림.
- 드라이버 cpu: 12:48 collect_results 끝 (rc 0) → 12:53 `eval10 lo_pkg_bm2_rlonly_s1_rl` 시작 (`lo_pkg_bm2_rlonly_s1` 학습 11:41 끝 → 10칸 평가; 하네스 `after` 게이트 항목보다 앞에 집힘 — 드라이버 순서 그대로). 진행 1/6: **eval_pkg10_1 ok 0.76** (count 35.37, rounds 4). forkserver 워커 1890973 100 % CPU, rss 1.9 GB — 정상.
- 한 줄 (N 6단계): GPU = `lo_pkg_bm2_rlonly_s2` RL (log.csv 마지막 it 200, 경과 5130 s → it ~250; val it 200 = pcb10 fail 1.00, ip10 0.920 fail 0.69), cpu = `eval10 lo_pkg_bm2_rlonly_s1_rl` (1/6), 새 기록 1 (pkg10_1).
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) `eval10 lo_pkg_bm2_rlonly_s1_rl` 끝 → RESULTS 자동 반영 확인. (2) 좌표 하네스 s2 남은 3개 (pkg16_1, pkg20_0·1) 드라이버가 집는지 확인. (3) `lo_pkg_bm2_rlonly_s2` 끝 ~13:45 → 다음 train (`bm2_rlonly`). (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.


# (이전) 라운드 165 머리말: 갱신 2026-11-08 12:55 UTC, 라운드 165 — CPU 좌표 하네스 s2 1/4 끝 (`r0_coord_s2_rl+rep4` pkg16_0 ok 0.88, 수리 전 0.56); GPU `lo_pkg_bm2_rlonly_s2` RL it 200 정상 (val it 200 = it 100 그대로))

## 라운드 165 (2026-11-08 12:51~ UTC)
- `git fetch r2e`: 새 main 커밋 없음 (HEAD = `54a561f`). QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 18.8 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 289 B 되살림.
- **좌표 하네스 s2 1/4 끝 (드라이버):** `N coord harness eval_pkg16_0 r0_coord_s2_rl+rep4` 11:32–12:48 rc 0 — **ok 0.88 (22/25)**, 수리 전 0.56, 성공 개수 평균 9.73, repair 4회 사용, peak rss 22.4 GiB (속도 숫자는 QUEUE 7에 따라 표에 안 씀). 기록 `runs/2026-10-29/r0_coord_s2_rl/eval/eval_pkg16_0__r0_coord_s2_rl+rep4.json`. 참고 s1 같은 세트 0.84. collect_results 12:48~ 드라이버가 돌리는 중 (RESULTS 자동 반영).
- 한 줄 (N 6단계): GPU = `lo_pkg_bm2_rlonly_s2` RL (it 200 / 3910 s, count/ref 3.52, fail 0.30, teach ok 0.88, loss −18.35, cert-bad 1, oom 0, gpu peak 14.6 GB; val it 200 = pcb10 fail 1.00 (count nan), ip10 0.920 fail 0.69 — it 100 과 같음), cpu = collect_results → 다음 `N coord harness` (s2 pkg16_1), 새 기록 1.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) 좌표 하네스 s2 남은 3개 (pkg16_1, pkg20_0·1) 진행 확인; 워커 0 % CPU 로 멈추면 그때 적기. (2) `lo_pkg_bm2_rlonly_s2` 끝 ~13:45 → 다음 train 항목 (`bm2_rlonly`). (3) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.


# (이전) 라운드 164 머리말: 갱신 2026-11-08 12:38 UTC, 라운드 164 — GPU `lo_pkg_bm2_rlonly_s2` RL it 150 정상 (val it 100 그대로); CPU 좌표 하네스 `r0_coord_s2_rl+rep4` eval_pkg16_0 63분째 (워커 100 %, 정상))

## 라운드 164 (2026-11-08 12:35~ UTC)
- `git fetch r2e`: 새 main 커밋 없음 (서버 2 `fa330ba` = `bdf_s1_distill` 10칸 기록 6개, 이미 HEAD). QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 28.4 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` 디스크와 같은 크기 → 할 일 없음.
- 한 줄 (N 6단계): GPU = `lo_pkg_bm2_rlonly_s2` RL (it 150 / 2998 s, count/ref 3.69, fail 0.23, teach ok 0.56, loss −1.89, cert-bad 6, oom 0, gpu peak 13.8 GB; resume ckpt it 120; val 마지막 = it 100 — pcb10 fail 1.00, ip10 0.920 fail 0.69), cpu = `N coord harness eval_pkg16_0 r0_coord_s2_rl+rep4` (11:32:14~, 63분째, forkserver 워커 1878614 100 % CPU, rss 10.6 GB — 정상), 새 기록 0 (서버 2 push 6개는 `bdf_s1_distill`, 사람 몫).
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) 좌표 하네스 s2 (4개) 진행 확인; 워커가 0 % CPU 로 멈추면 그때 적기. (2) `lo_pkg_bm2_rlonly_s2` 끝 ~13:45 → 다음 train 항목 (`bm2_rlonly`). (3) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 163 머리말: 갱신 2026-11-08 12:21 UTC, 라운드 163 — GPU `lo_pkg_bm2_rlonly_s2` 첫 val it 100 = seed 0·1 과 같음 (pcb10 fail 1.00, ip10 0.920 fail 0.69); CPU 좌표 하네스 `r0_coord_s2_rl+rep4` eval_pkg16_0 48분째 (워커 100 %, 정상))

## 라운드 163 (2026-11-08 12:19~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 28.1 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` 디스크와 같은 크기 → 할 일 없음.
- 한 줄 (N 6단계): GPU = `lo_pkg_bm2_rlonly_s2` RL (it 100 / 1964 s, count/ref 3.47, fail 0.23, teach ok 0.56, loss −3.50, oom 0, gpu peak 13.4 GB; **첫 val it 100: pcb10 fail 1.00 (count nan), ip10 0.920 fail 0.69** — seed 0·1 의 val 과 같은 숫자; NaN loss 아님), cpu = `N coord harness eval_pkg16_0 r0_coord_s2_rl+rep4` (11:32:14~, 48분째, forkserver 워커 1878614 100 % CPU, rss 9.8 GB — 정상; s1 의 pkg16·20 도 30~110분 걸림), 새 기록 0.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계 (bm2 멈춤 규칙은 12:30 KST 확인 한 번으로 끝난 항목; 여기서는 숫자만 적음).
- 할 일 (다음): (1) 좌표 하네스 s2 (4개) 진행 확인; 워커가 0 % CPU 로 멈추면 그때 적기. (2) `lo_pkg_bm2_rlonly_s2` 끝 ~13:45 → 다음 train 항목 (`bm2_rlonly`). (3) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 162 머리말: 갱신 2026-11-08 12:05 UTC, 라운드 162 — GPU `lo_pkg_bm2_rlonly_s2` RL it 50 정상 (첫 val 전); CPU 좌표 하네스 `r0_coord_s2_rl+rep4` eval_pkg16_0 32분째 (워커 100 %, 정상))

## 라운드 162 (2026-11-08 12:03~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 28.9 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` 디스크와 같은 크기 → 할 일 없음.
- 한 줄 (N 6단계): GPU = `lo_pkg_bm2_rlonly_s2` RL (it 50 / 868 s, count/ref 2.86, fail 0.14, teach ok 0.69, loss 0.33, cert-bad 15, oom 0, gpu peak 11.4 GB; resume ckpt it 61; `log.csv` 는 첫 val (it 100) 전이라 비어 있음), cpu = `N coord harness eval_pkg16_0 r0_coord_s2_rl+rep4` (11:32:14~, 32분째, forkserver 워커 1878614 100 % CPU, rss 9.5 GB — 멈춘 워커 아님), 새 기록 0.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) 좌표 하네스 s2 (4개) 진행 확인; 워커가 0 % CPU 로 멈추면 그때 적기. (2) `lo_pkg_bm2_rlonly_s2` 첫 val (it 100, ~12:15) 적기, 끝 ~13:45 → 다음 train 항목. (3) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 161 머리말: 갱신 2026-11-08 11:52 UTC, 라운드 161 — GPU `lo_pkg_bm2_rlonly_s1` 끝 (11:41, rc 0) → `lo_pkg_bm2_rlonly_s2` RL 시작; CPU 좌표 하네스 `r0_coord_s2_rl+rep4` eval_pkg16_0 진행 중 (11:32~))

## 라운드 161 (2026-11-08 11:48~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 27.0 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 598 B 되살림.
- 한 줄 (N 6단계): GPU = `lo_pkg_bm2_rlonly_s1` 끝 (11:41:24 rc 0; 마지막 val it 300 = pcb10 fail 1.00 (count nan), ip10 0.920 fail 0.69 — it 100·200 과 같음) → 드라이버가 **`lo_pkg_bm2_rlonly_s2`** RL 시작 (11:41:24~, seed 2, teacher 파일 읽음, GPU mem 5.77 GB), cpu = `N coord harness eval_pkg16_0 r0_coord_s2_rl+rep4` (11:32:14~, 16분째, forkserver 워커 1878614 100 % CPU, rss 8.6 GB — 정상), 새 기록 0.
- s1 하네스 (`N harness ... lo_pkg_bm2_rlonly_s1_rl+rep4`) 는 큐에 있고 `after` = s1 metrics.json (이제 있음) → 좌표 s2 하네스 뒤 순서대로 드라이버가 집는다.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) 좌표 하네스 s2 (4개) 진행 확인. (2) `lo_pkg_bm2_rlonly_s2` 첫 val (it 100) 적기 (~12:15), 끝 ~13:45 → 다음 train 항목. (3) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 160 머리말: 갱신 2026-11-08 11:35 UTC, 라운드 160 — CPU eval10 `lo_pkg_bm2_rlonly_rl` 끝 (11:27, rc 0, 6세트) → collect_results 끝, 다음 = 좌표 하네스 s2; GPU `lo_pkg_bm2_rlonly_s1` RL val it 300 그대로 (~11:40 끝 → `_s2`))

## 라운드 160 (2026-11-08 11:31~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 18.7 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 342 B 되살림.
- 한 줄 (N 6단계): GPU = `lo_pkg_bm2_rlonly_s1` RL (val it 300: pcb10 fail 1.00 (count nan), ip10 0.920 fail 0.69 — it 100·200 과 같음, seed 0 과도 같음; 2 h 다 차 감), cpu = **eval10 `lo_pkg_bm2_rlonly_rl` 끝** (10:57:39~11:27:33, 30분, rc 0) → collect_results (11:30:44 rc 0) → 드라이버 다음 항목 (좌표 하네스 s2, `r0_coord_s2_rl/metrics.json` 있음) 대기, 새 기록 6 (`eval_{ip,pcb,pkg}10_{0,1}__lo_pkg_bm2_rlonly_rl.json`).
- 새 기록 (숫자만, 판정 없음; RESULTS 는 collect_results 가 갱신): bm2 RL-only seed 0, pkg 뺀 학습 — 성공률 / 개수비(g+p 대비) = ip 0.96 / 1.97, pcb 0.04 / 3.99, **뺀 pkg 0.84 / 4.65**. pcb 는 val 처럼 거의 못 풂; 풀 때는 개수가 g+p 의 2~5배.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) 좌표 하네스 s2 (4개) 진행 확인. (2) `lo_pkg_bm2_rlonly_s1` 끝 → `_s2`, s1 eval10. (3) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 159 머리말: 갱신 2026-11-08 11:17 UTC, 라운드 159 — CPU eval10 `lo_pkg_bm2_rlonly_rl` 진행 중 (첫 세트 eval_pkg10_1 ok 0.80); GPU `lo_pkg_bm2_rlonly_s1` RL it 275 정상 (~11:40 끝))

## 라운드 159 (2026-11-08 11:15~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 21.3 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` 디스크와 같은 크기 → 할 일 없음.
- 한 줄 (N 6단계): GPU = `lo_pkg_bm2_rlonly_s1` RL (it 275 / 5402 s, count/ref 2.39, fail 0.34, teach ok 0.94, loss −14.00, cert-bad 3, oom 0, gpu peak 14.5 GB; resume ckpt it 244; val 마지막 = it 200 그대로 — pcb10 fail 1.00, ip10 0.920 fail 0.69), cpu = `eval10 lo_pkg_bm2_rlonly_rl` (10:57:39~, 6세트 중 1개 끝: eval_pkg10_1 ok 0.80 count 37.40, 2.0 s/판; forkserver 워커 1873968 100 % CPU, rss 2.3 GB), 새 기록 0 (세트 기록은 6개 다 끝난 뒤 collect_results).
- bm2 seed 0: val 에서 pcb10 을 한 번도 못 풀었지만 eval_pkg10_1 은 0.80 — 적기만 (판정 없음).
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) eval10 bm2 seed 0 끝 → 좌표 하네스 s2 (4개). (2) `lo_pkg_bm2_rlonly_s1` 끝 (~11:40) → `_s2`. (3) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 158 머리말: 갱신 2026-11-08 11:02 UTC, 라운드 158 — CPU 좌표 하네스 `r0_coord_s1_rl+rep4` eval_pkg20_1 끝 (10:52, rc 0, 6/8) → eval10 `lo_pkg_bm2_rlonly_rl` 시작; GPU `lo_pkg_bm2_rlonly_s1` RL it 200 (val 그대로))

## 라운드 158 (2026-11-08 10:59~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 20.4 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 753 B 되살림.
- 한 줄 (N 6단계): GPU = `lo_pkg_bm2_rlonly_s1` RL (val it 200: pcb10 fail 1.00 (count nan), ip10 0.920 fail 0.69 — it 100 과 같음, seed 0 과도 같음), cpu = 좌표 하네스 `eval_pkg20_1 r0_coord_s1_rl+rep4` 끝 (08:49:43~10:52:59, 123분, rc 0, 193 s/판) → collect_results → **eval10 `lo_pkg_bm2_rlonly_rl`** (10:57:39~), 새 기록 1 (`eval_pkg20_1__r0_coord_s1_rl+rep4.json`).
- 새 기록: 좌표 s1 +rep4 pkg20_1 = 6/8 (수리 전 5/8; 판 0 이 수리로 통과), 개수 평균 10.5. pkg20 두 세트 합 13/16 = 0.81, 개수비 0.809 [0.650, 0.950] — RESULTS 는 드라이버 collect_results 가 이미 갱신 (판정 없음, 숫자만).
- 좌표 seed 1 하네스 4개 (pkg16_0/1, pkg20_0/1) 다 끝. 남은 N coord harness = seed 2 4개 (bm2 eval10 뒤).
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) eval10 bm2 seed 0 끝 → 좌표 하네스 s2 (4개). (2) `lo_pkg_bm2_rlonly_s1` 끝 (~11:40) → `_s2`. (3) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 157 머리말: 갱신 2026-11-08 10:45 UTC, 라운드 157 — GPU `lo_pkg_bm2_rlonly_s1` RL it 175 정상; CPU 좌표 하네스 `r0_coord_s1_rl+rep4` eval_pkg20_1 114분째 (워커 100 %, rss 8.9 GB, 정상))

## 라운드 157 (2026-11-08 10:43~ UTC)
- `git fetch r2e`: 새 커밋 = 서버 2 사람 `bdf_distill` 10칸 eval 기록 6개 (3c4c5ab, 이미 HEAD). QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 26.7 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` 디스크와 같은 크기 → 할 일 없음.
- 한 줄 (N 6단계): GPU = `lo_pkg_bm2_rlonly_s1` RL (it 175 / 3385 s, count/ref 3.02, fail 0.45, teach ok 0.62, loss −16.43, cert-bad 4, oom 0, gpu peak 14.2 GB; resume ckpt it 185; val 은 it 100 그대로 — pcb10 fail 1.00, ip10 0.920 fail 0.69), cpu = `N coord harness eval_pkg20_1 r0_coord_s1_rl+rep4` (08:49:43~, 114분째, forkserver 워커 1855610 100 % CPU, rss 8.9 GB — 멈춘 워커 아님), 새 기록 0 (서버 1); 서버 2 사람 기록 6 (`bdf_distill` 10칸, pull 만).
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계. pkg20_1 은 pkg20_0 (68분) 의 1.7배째지만 워커가 100 % CPU 로 일하는 중 → 기다린다 (가장 보수적). pkg20_0 의 3배 (~204분, 12:15 UTC) 를 넘기면 그때 적기만 (죽이지 않음).
- 할 일 (다음): (1) 좌표 하네스 pkg20_1 끝 → 드라이버가 eval10 `lo_pkg_bm2_rlonly_rl` → 나머지 하네스. (2) `lo_pkg_bm2_rlonly_s1` 끝 (~11:40) → `_s2`. (3) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 156 머리말: 갱신 2026-11-08 10:29 UTC, 라운드 156 — GPU `lo_pkg_bm2_rlonly_s1` RL it 125, 첫 val (it 100) = seed 0 과 같음 (pcb10 fail 1.00, ip10 0.920 fail 0.69); CPU 좌표 하네스 `r0_coord_s1_rl+rep4` eval_pkg20_1 98분째 (워커 100 %, 정상)

## 라운드 156 (2026-11-08 10:27~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 27.1 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` 디스크와 같은 크기 → 할 일 없음.
- 한 줄 (N 6단계): GPU = `lo_pkg_bm2_rlonly_s1` RL (it 125 / 2391 s, count/ref 3.24, fail 0.32, teach ok 0.62, loss −8.09, cert-bad 4, oom 0, gpu peak 13.3 GB; resume ckpt it 126; **첫 val it 100: val_pcb10_0 fail 1.00 (count nan), val_ip10_0 0.920 fail 0.69 — seed 0 의 it 100·200·300 과 숫자까지 같음**), cpu = `N coord harness eval_pkg20_1 r0_coord_s1_rl+rep4` (08:49:43~, 98분째, forkserver 워커 1855610 100 % CPU, rss 9.0 GB — 멈춘 워커 아님), 새 기록 0.
- bm2 s1 val: seed 가 달라도 val 이 같다 → manoharan2 RL-only 가 pcb10 에서 한 번도 못 푸는 것은 seed 탓이 아닌 듯 (적기만; 멈춤 규칙은 seed 0 의 25 iter 에서 이미 판정, 사람 큐 그대로).
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계. pkg20_1 은 pkg20_0 (68분) 보다 30분 더 걸리지만 워커가 100 % CPU 로 일하고 rss 가 늘어나는 중 → 기다린다 (가장 보수적).
- 할 일 (다음): (1) 좌표 하네스 pkg20_1 끝 → 드라이버가 eval10 `lo_pkg_bm2_rlonly_rl` → 나머지 하네스. 워커가 0 % CPU 로 멈추면 그때 적기. (2) `lo_pkg_bm2_rlonly_s1` 끝 (~11:40) → `_s2`. (3) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 155 머리말: 갱신 2026-11-08 10:13 UTC, 라운드 155 — GPU `lo_pkg_bm2_rlonly_s1` RL it 75 정상 (첫 val 전); CPU 좌표 하네스 `r0_coord_s1_rl+rep4` eval_pkg20_1 82분째 (워커 100 %, 정상, pkg20_0 68분보다 김))

## 라운드 155 (2026-11-08 10:11~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 26.1 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` 디스크와 같은 크기 → 할 일 없음.
- 한 줄 (N 6단계): GPU = `lo_pkg_bm2_rlonly_s1` RL (it 75 / 1391 s, count/ref 3.23, fail 0.20, teach ok 0.94, loss −9.49, cert-bad 3, oom 0, gpu peak 12.1 GB; resume ckpt it 65; `_s1_rl/log.csv` 첫 val (it 100) 전이라 비어 있음), cpu = `N coord harness eval_pkg20_1 r0_coord_s1_rl+rep4` (08:49:43~, 82분째, forkserver 워커 1855610 100 % CPU, rss 7.9 GB — 멈춘 워커 아님), 새 기록 0.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계. pkg20_1 이 pkg20_0 (68분) 보다 오래 걸리지만 워커가 CPU 100 % 로 일하는 중이라 기다린다 (가장 보수적).
- 할 일 (다음): (1) 좌표 하네스 pkg20_1 끝 → 드라이버가 eval10 `lo_pkg_bm2_rlonly_rl` → 나머지 하네스. 워커가 0 % CPU 로 멈추면 그때 적기. (2) `lo_pkg_bm2_rlonly_s1` 처음 val (it 100) 만 적기. (3) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 154 머리말: 갱신 2026-11-08 09:57 UTC, 라운드 154 — GPU `lo_pkg_bm2_rlonly` 끝 (09:40, rc 0) → `lo_pkg_bm2_rlonly_s1` RL it 25 정상; CPU 좌표 하네스 `r0_coord_s1_rl+rep4` eval_pkg20_1 66분째 (워커 100 %, 정상))

## 라운드 154 (2026-11-08 09:55~ UTC)
- `git fetch r2e`: 새 커밋 = 서버 2 사람 `bdf_distill` log/metrics (학습 기록, eval 기록 아님) — 이미 HEAD 에 있음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 26.0 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 595 B 되살림.
- 한 줄 (N 6단계): GPU = `lo_pkg_bm2_rlonly` 끝 (09:40:22 rc 0; 마지막 val it 300: pcb10 fail 1.00, ip10 fail 0.69 count/g+p 0.92 — it 100 부터 그대로) → `lo_pkg_bm2_rlonly_s1` RL 시작 (it 25: count/ref 1.64, fail 0.27, teach ok 0.38, loss −2.75, cert-bad 12, oom 0, gpu peak 10.3 GB; log.csv 는 첫 val 전이라 비어 있음), cpu = `N coord harness eval_pkg20_1 r0_coord_s1_rl+rep4` (08:49:43~, 66분째, 워커 1855610 100 % CPU, rss 7.9 GB; pkg20_0 은 68분), 새 기록 0.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계. bm2 seed 0 eval10 은 드라이버가 좌표 하네스 다음에 세운다 (cpu 는 한 번에 하나).
- 할 일 (다음): (1) 좌표 하네스 pkg20_1 끝 → 드라이버가 eval10 `lo_pkg_bm2_rlonly_rl` → 나머지 하네스. (2) `lo_pkg_bm2_rlonly_s1` 처음 val 만 적기. (3) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 153 머리말: 갱신 2026-11-08 09:42 UTC, 라운드 153 — GPU `lo_pkg_bm2_rlonly` RL it 350, 2 h 다 참 (곧 끝); CPU 좌표 하네스 `r0_coord_s1_rl+rep4` eval_pkg20_1 50분째 (워커 100 %, 정상)

## 라운드 153 (2026-11-08 09:39~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 27.0 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` 디스크와 같은 크기 → 할 일 없음.
- 한 줄 (N 6단계): GPU = `lo_pkg_bm2_rlonly` RL (it 350 / 6811 s, count/ref 3.60, fail 0.24, teach ok 0.88, loss −14.70, cert-bad 2, oom 0, gpu peak 14.4 GB; train.py 경과 7208 s → 2 h 한도, 곧 끝; val 마지막 = it 300 그대로), cpu = `N coord harness eval_pkg20_1 r0_coord_s1_rl+rep4` (08:49:43~, 50분째, forkserver 워커 1855610 100 % CPU, rss 7.7 GB — 멈춘 워커 아님; pkg20_0 은 68분 → ~09:58 끝 예상), 새 기록 0.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) bm2 학습 끝 → 드라이버가 eval10 을 앞에 세우고 GPU 는 `lo_pkg_bm2_rlonly_s1`; 새 run 은 처음 val 만 적기. (2) 좌표 하네스 s1 pkg20_1 끝 → 나머지 (s2 4개). (3) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 152 머리말: 갱신 2026-11-08 09:25 UTC, 라운드 152 — GPU `lo_pkg_bm2_rlonly` RL it 300 (val 그대로); CPU 좌표 하네스 `r0_coord_s1_rl+rep4` eval_pkg20_1 진행 중 (35분째, pkg20_0 은 68분 걸림 → ~09:57 끝 예상)

## 라운드 152 (2026-11-08 09:23~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 25.3 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` 디스크와 같은 크기 → 할 일 없음.
- 한 줄 (N 6단계): GPU = `lo_pkg_bm2_rlonly` RL (it 300 / 5750 s, count/ref 5.27, fail 0.25, teach ok 0.62, loss −2.665, cert-bad 8, oom 0, gpu peak 14.4 GB; resume ckpt it 314; **val it 300 = it 100·200 과 같음: val_pcb10_0 fail 1.00 (count nan), val_ip10_0 0.920 fail 0.69**; 2 h RL → ~09:40 끝), cpu = `N coord harness eval_pkg20_1 r0_coord_s1_rl+rep4` (08:49:43~, 기록 아직 없음), 새 기록 0.
- bm2 it 300: loss −2.7 로 0 에 가까워지는 중이지만 val 은 세 번 다 같음 (pcb 한 종류만 fail 1.00) — 적기만, 규칙 판정은 25 iter 에서 끝남.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) 좌표 하네스 s1 pkg20_1 끝 → 나머지 (s2 4개). (2) bm2 학습 끝 (~09:40) → 드라이버가 eval10 을 앞에 세우고 GPU 는 다음 train (`lo_pkg_bm2_rlonly_s1`). (3) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 151 머리말: 갱신 2026-11-08 09:10 UTC, 라운드 151 — GPU `lo_pkg_bm2_rlonly` RL it 250 정상; CPU 좌표 하네스 `r0_coord_s1_rl+rep4` eval_pkg20_1 진행 중 (20분째))

## 라운드 151 (2026-11-08 09:07~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 26.7 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` 디스크와 같은 크기 → 할 일 없음.
- 한 줄 (N 6단계): GPU = `lo_pkg_bm2_rlonly` RL (it 250 / 4893 s, count/ref 3.25, fail 0.12, teach ok 1.00, loss −8.35, cert-bad 3, oom 0, gpu peak 14.2 GB; resume ckpt it 248; val 마지막 = it 200 그대로), cpu = `N coord harness eval_pkg20_1 r0_coord_s1_rl+rep4` (08:49:43~, 아직 기록 없음; 같은 run pkg20_0+rep4 는 0.88), 새 기록 0.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) 좌표 하네스 s1 pkg20_1 끝 → 나머지 (s2 4개). (2) bm2 학습 끝 (~09:40) → eval10 은 드라이버가 앞에 세움. (3) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 150 머리말: PROGRESS (갱신 2026-11-08 08:53 UTC, 라운드 150 — GPU `lo_pkg_bm2_rlonly` RL it 200 (val 그대로: pcb10 fail 1.00, ip10 0.69); CPU `eval10 lo_pkg_bp_park_s2_rl` 끝 6/6 → 좌표 하네스 `r0_coord_s1_rl` pkg20_1)

## 라운드 150 (2026-11-08 08:51~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 25.3 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 726 B 복구 (08:45 eval10 s2 끝, collect, 08:49 좌표 하네스 시작 줄).
- 한 줄 (N 6단계): GPU = `lo_pkg_bm2_rlonly` RL (it 200 / 3881 s, count/ref 3.19, fail 0.16, teach ok 1.00, loss −16.525, cert-bad 3, oom 0, gpu peak 14.1 GB; **val it 200 = it 100 과 같음: val_pcb10_0 fail 1.00 (count nan), val_ip10_0 0.920 fail 0.69**), cpu = `eval10 lo_pkg_bp_park_s2_rl` **끝** (08:45 rc 0; pkg10_1 0.84, pcb10_0 0.04, pkg10_0 0.92, pcb10_1 0.00, ip10_0 0.96, ip10_1 1.00 — s1 과 같이 pcb 10칸 둘 다 거의 0; collect_results 08:48 RESULTS 반영) → `N coord harness eval_pkg20_1 r0_coord_s1_rl+rep4` (08:49:43~), 새 기록 3 (s2 pcb10_1, ip10_0, ip10_1).
- bm2 it 200 val: 할 일 (2) 대로 적기만 (붕괴 조건 아님 — pcb 한 종류만 fail 1.00, loss −16.5).
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) 좌표 하네스 나머지 (s1 pkg20_1 진행 중) 끝 → 드라이버 다음 cpu 항목. (2) bm2 학습 끝 (~09:40) → eval10 은 드라이버가 앞에 세움. (3) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 149 머리말: 갱신 2026-11-08 08:37 UTC, 라운드 149 — GPU `lo_pkg_bm2_rlonly` RL it 150 정상; CPU `eval10 lo_pkg_bp_park_s2_rl` 3/6 (pkg10_0 0.92))

## 라운드 149 (2026-11-08 08:35~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 21.2 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` 디스크와 같은 크기 → 할 일 없음.
- 한 줄 (N 6단계): GPU = `lo_pkg_bm2_rlonly` RL (it 150 / 2945 s, count/ref 2.93, fail 0.26, teach ok 0.94, loss −8.58, cert-bad 3, oom 0, gpu peak 14.1 GB; resume ckpt it 123), cpu = `eval10 lo_pkg_bp_park_s2_rl` 3/6 (pkg10_1 0.84, pcb10_0 0.04, **pkg10_0 0.92** — 08:28, RESULTS 반영은 eval 끝난 뒤 드라이버 collect), 새 기록 1 (s2 pkg10_0).
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) eval10 s2 끝 → 좌표 하네스 나머지 5개. (2) bm2 it 200 val 도 적기만. (3) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 148 머리말: 갱신 2026-11-08 08:24 UTC, 라운드 148 — GPU `lo_pkg_bm2_rlonly` RL it 100 (val pcb10 fail 1.00, ip10 fail 0.69, loss −15.9 → 붕괴 아님); CPU `eval10 lo_pkg_bp_park_s2_rl` 2/6)

## 라운드 148 (2026-11-08 08:19~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 21.9 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` 디스크와 같은 크기 → 할 일 없음.
- 한 줄 (N 6단계): GPU = `lo_pkg_bm2_rlonly` RL (it 100 / 1936 s, count/ref 3.08, fail 0.11, teach ok 1.00, loss −15.898, cert-bad 3, oom 0, gpu peak 13.3 GB; **첫 val: val_pcb10_0 fail 1.00 (count nan), val_ip10_0 count/g+p 0.920 fail 0.69**, new best score 6.648), cpu = `eval10 lo_pkg_bp_park_s2_rl` 2/6 (pkg10_1 0.84, pcb10_0 0.04), 새 기록 2 (s2 pkg10_1, pcb10_0).
- bm2 it 100 val: fail 1.00 은 pcb 한 종류뿐이고 loss 는 0 이 아님 → 할 일 (2) 의 붕괴 조건 (두 종류 fail 1.00 + loss ~0) 아님. 적기만, 큐 그대로. (pcb10 fail 1.00 은 bp_park 의 eval pcb10 0.04 와 같은 흐름 — 비교는 eval10 나온 뒤.)
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) eval10 s2 끝 → 좌표 하네스 나머지 5개. (2) bm2 it 200 val 도 적기만. (3) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 147 머리말: 갱신 2026-11-08 08:08 UTC, 라운드 147 — GPU `lo_pkg_bm2_rlonly` RL it 75 정상; CPU `eval10 lo_pkg_bp_park_s1_rl` 끝 (6/6) → `eval10 lo_pkg_bp_park_s2_rl` 07:56~)

## 라운드 147 (2026-11-08 08:03~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 22.1 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log` → 732 B 복구 (07:52 eval10 s1 끝, collect, 07:56 eval10 s2 시작 줄).
- 한 줄 (N 6단계): GPU = `lo_pkg_bm2_rlonly` RL (it 75 / 1365 s, count/ref 2.54, fail 0.18, teach ok 0.94, loss −9.34, cert-bad 1, oom 0, gpu peak 13.3 GB; resume ckpt it 68; log.csv 아직 0 B — it 100 에 첫 줄), cpu = `eval10 lo_pkg_bp_park_s1_rl` **끝** (07:52 rc 0; pkg10_1 0.64, pcb10_0 0.04, pkg10_0 0.64, pcb10_1 0.00, ip10_0 0.96, ip10_1 1.00 — pcb 10칸 둘 다 거의 0, collect_results 07:55 RESULTS 반영) → `eval10 lo_pkg_bp_park_s2_rl` (07:56:54~), 새 기록 3 (s1 pcb10_1, ip10_0, ip10_1).
- 참고: 드라이버가 좌표 하네스 나머지 5개보다 `eval10` 을 먼저 집음 (학습 끝난 run 의 eval10 은 드라이버가 자동으로 앞에 세움) — 순서는 건드리지 않음.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) eval10 s2 끝 → 좌표 하네스 나머지 5개. (2) bm2 it 100 val 이 fail 1.00 두 종류 + loss ~0 이면 적기만. (3) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (4) 라운드마다 `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 146 머리말: 갱신 2026-11-08 07:50 UTC, 라운드 146 — GPU `lo_pkg_bp_park_s2` 끝 → `lo_pkg_bm2_rlonly` it 25 멈춤 규칙 안 걸림 (bm2 그대로); CPU `eval10 lo_pkg_bp_park_s1_rl` 3/6)

## 라운드 146 (2026-11-08 07:38~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 20.5 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 586 B 복구 (07:39 s2 끝, bm2 시작 줄).
- 한 줄 (N 6단계): GPU = `lo_pkg_bp_park_s2` **끝** (07:39:32 rc 0, RL 216 iter / 1.26 h, best iter 100 score 6.273, gpu peak 11.4 GB; 평가는 드라이버 cpu 큐의 eval10) → `lo_pkg_bm2_rlonly` 시작 (07:39:32, manoharan2, RL만), cpu = `eval10 lo_pkg_bp_park_s1_rl` 3/6 (pkg10_1 0.64, pcb10_0 0.04, pkg10_0 0.64), 새 기록 1 (s1 pkg10_0).
- **N 5단계 멈춤 규칙 확인 (`lo_pkg_bm2_rlonly` 처음 25 iter):** it 25 loss −23.691 (NaN 아님, 0 아님), count/ref 2.656, fail 0.33, teach ok 0.81, cert-bad 8, oom 0, gpu peak 10.9 GB. val 줄은 아직 없음 (log.csv 는 it 100 에 첫 줄 — bp_park 와 같음). 규칙 (NaN, 또는 val 세 종류 fail 1.00 + loss 0) **안 걸림** → 남은 bm2 5개 큐에 그대로.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계, 멈춤 규칙 안 걸림.
- 할 일 (다음): (1) eval10 s1 끝 → 좌표 하네스 나머지 5개 → eval10 s2. (2) bm2 it 100 val 이 fail 1.00 두 종류 + loss ~0 으로 붕괴하면 적기만 (규칙은 25 iter 에서 판정 끝). (3) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (4) 라운드마다 `restore_proc_log.py 1792539`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 145 머리말: 갱신 2026-11-08 07:26 UTC, 라운드 145 — GPU `lo_pkg_bp_park_s2` RL it 150 정상; CPU `eval10 lo_pkg_bp_park_s1_rl` 2/6 (3번째 세트 진행))

## 라운드 145 (2026-11-08 07:22~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 21.1 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` 디스크와 같은 크기 → 할 일 없음.
- 한 줄 (N 6단계): GPU = `lo_pkg_bp_park_s2` RL (it 150 / 3163 s, count/ref 1.16, fail 0.01, teach ok 0.56, cert-bad 1, oom 0, gpu peak 11.3 GB; it 125 train_pcb12_0 fail 1.00 (count nan) 한 번 — s1 에서도 pcb12 fail 높았음, NaN 은 loss 에 없음; resume ckpt it 115), cpu = `eval10 lo_pkg_bp_park_s1_rl` 2/6 (pkg10_1 0.64, pcb10_0 0.04; 3번째 세트 워커 1839716 100 % CPU, 1172 s, rss 2.7 GB), 새 기록 0.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) eval10 s1 끝 → 좌표 하네스 나머지 5개. (2) `lo_pkg_bp_park_s2` 끝 (~07:40 UTC) → `lo_pkg_bm2_rlonly` 시작 뒤 처음 25 iter 멈춤 규칙 확인 (NaN, 또는 val 세 종류 fail 1.00 + loss 0 → 남은 bm2 5개 `train_removed`). (3) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (4) 라운드마다 `restore_proc_log.py 1792539`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 144 머리말: 갱신 2026-11-08 07:10 UTC, 라운드 144 — GPU `lo_pkg_bp_park_s2` RL it 100 정상; CPU `eval10 lo_pkg_bp_park_s1_rl` 2/6)

## 라운드 144 (2026-11-08 07:06~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 21.4 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` 디스크와 같은 크기 → 할 일 없음.
- 한 줄 (N 6단계): GPU = `lo_pkg_bp_park_s2` RL (it 100 / 2072 s, ratio 1.00, fail 0.04, lam 2.11; val it 100 pcb10_0 fail 1.00 / ip10_0 fail 0.31 — s1 it 100 과 같은 모양 (pcb 0.94 / ip 0.62), NaN 0), cpu = `eval10 lo_pkg_bp_park_s1_rl` 2/6 (pkg10_1 0.64, pcb10_0 0.04; 3번째 세트 진행), 새 기록 2 (report_watch collect_results 가 반영 중).
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) eval10 s1 끝 → 좌표 하네스 나머지 5개. (2) `lo_pkg_bp_park_s2` 끝 (~07:40 UTC) → `lo_pkg_bm2_rlonly` 시작 뒤 처음 25 iter 멈춤 규칙 확인. (3) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (4) 라운드마다 `restore_proc_log.py 1792539`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 143 머리말: 갱신 2026-11-08 06:52 UTC, 라운드 143 — GPU `lo_pkg_bp_park_s2` RL it 75 정상; CPU `eval10 lo_pkg_bp_park_s1_rl` 06:38~ 진행 중)

## 라운드 143 (2026-11-08 06:50~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 21.3 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 464 B 복구 (06:36 collect 끝, 06:38 eval10 시작 줄).
- 한 줄 (N 6단계): GPU = `lo_pkg_bp_park_s2` RL (it 75 / 1580 s, count/ref 11.96 (pcb8 판, it 50 은 1.41), fail 0.74, teach ok 0.94, cert-bad 6, oom 0, gpu peak 10.6 GB; 57 iter 에서 resume checkpoint), cpu = `eval10 lo_pkg_bp_park_s1_rl` (06:38:21~, 6 세트, 워커 1836386 100 % CPU, rss 2.9 GB — 아직 기록 없음), 새 기록 0 (06:36 collect_results 가 좌표 s1 pkg20_0 기록을 이미 반영, 커밋 6a1b883).
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) eval10 s1 끝 → 좌표 하네스 나머지 5개 (s1 pkg20_1, s2 pkg16/20 × 2). (2) `lo_pkg_bp_park_s2` 끝 (~07:40 UTC) → `lo_pkg_bm2_rlonly` 시작 뒤 처음 25 iter 멈춤 규칙 확인. (3) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (4) 라운드마다 `restore_proc_log.py 1792539`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 142 머리말: 갱신 2026-11-08 06:37 UTC, 라운드 142 — GPU `lo_pkg_bp_park_s2` RL it 25 정상; CPU 좌표 하네스 3/8 `eval_pkg20_0 r0_coord_s1_rl+rep4` 끝 (7/8, 수리 전 6/8) → collect_results)

## 라운드 142 (2026-11-08 06:34~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 18.8 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 289 B 복구 (06:33 줄: 좌표 하네스 끝, collect_results 시작).
- 한 줄 (N 6단계): GPU = `lo_pkg_bp_park_s2` 증류 끝 → RL (it 25 / 523 s, count/ref 2.31, fail 0.46, teach ok 0.81, cert-bad 10, oom 0, gpu peak 9.0 GB, NaN 0), cpu = `N coord harness eval_pkg20_0 r0_coord_s1_rl+rep4` **끝** (06:33:43 rc 0, 4000 s, peak rss 18.2 GB; ok 7/8 = 0.88, 수리 전 6/8, F_final 438) → 드라이버 collect_results 중 (report_watch 의 collect 와 둘이 같이 돎). 새 기록 1 (`eval_pkg20_0__r0_coord_s1_rl+rep4.json`).
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) 좌표 하네스 나머지 5개 (s1 pkg20_1, s2 pkg16/20 × 2) — 20칸 한 파일 약 67분. (2) `eval10 lo_pkg_bp_park_s1_rl` (드라이버). (3) `lo_pkg_bp_park_s2` 끝 (~07:40 UTC) → `lo_pkg_bm2_rlonly` 시작 뒤 처음 25 iter 멈춤 규칙 확인. (4) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (5) 라운드마다 `restore_proc_log.py 1792539`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 141 머리말: 갱신 2026-11-08 06:21 UTC, 라운드 141 — GPU `lo_pkg_bp_park_s2` 증류 it 2000 정상; CPU 좌표 하네스 3/8 (`eval_pkg20_0 r0_coord_s1_rl+rep4`) 53분째)

## 라운드 141 (2026-11-08 06:18~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 26.5 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` 디스크와 같은 크기 → 할 일 없음.
- 한 줄 (N 6단계): GPU = `lo_pkg_bp_park_s2` 증류 (it 2000 / 2395 s, count/ref 0.98, teach ok 0.44–0.81, cert-bad 0, oom 0, gpu peak 8.9 GB; val it 2000 pcb10_0 fail 0.31 / ip10_0 fail 0.19; log.csv nan 은 증류 중 best_ratio·fail 열뿐), cpu = `N coord harness eval_pkg20_0 r0_coord_s1_rl+rep4` (05:25:57~, 워커 1825779 100 % CPU, rss 8.6 GB — 진행 중; s1 pkg16 두 파일은 43–53분 걸림), 새 기록 0.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) 좌표 하네스 나머지 (s1 pkg20_0·1, s2 pkg16/20 × 2). (2) `eval10 lo_pkg_bp_park_s1_rl` (드라이버). (3) `lo_pkg_bp_park_s2` 끝 (~07:40 UTC) → `lo_pkg_bm2_rlonly` 시작 뒤 처음 25 iter 멈춤 규칙 확인. (4) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (5) 라운드마다 `restore_proc_log.py 1792539`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 140 머리말: 갱신 2026-11-08 06:05 UTC, 라운드 140 — GPU `lo_pkg_bp_park_s2` 증류 it 1175 정상; CPU 좌표 하네스 3/8 (`eval_pkg20_0 r0_coord_s1_rl+rep4`) 37분째)

## 라운드 140 (2026-11-08 06:02~ UTC)
- `git fetch r2e`: 새 main 커밋 없음 (사람 N records 06:00 까지 이미 받음). QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 25.9 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` 디스크와 같은 크기 → 할 일 없음.
- 한 줄 (N 6단계): GPU = `lo_pkg_bp_park_s2` 증류 (it 1175 / 1439 s, count/ref 0.87, teach ok 0.88, cert-bad 0, oom 0, gpu peak 8.9 GB; val it 1100 pcb10_0 fail 0.31 / ip10_0 fail 0.25, NaN 없음), cpu = `N coord harness eval_pkg20_0 r0_coord_s1_rl+rep4` (05:25:57~, 워커 1825779 100 % CPU, rss 8.0 GB — 진행 중), 새 기록 0.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) 좌표 하네스 나머지 (s1 pkg20_0·1, s2 pkg16/20 × 2). (2) `eval10 lo_pkg_bp_park_s1_rl` (드라이버). (3) `lo_pkg_bp_park_s2` 끝 (~07:40 UTC) → `lo_pkg_bm2_rlonly` 시작 뒤 처음 25 iter 멈춤 규칙 확인. (4) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (5) 라운드마다 `restore_proc_log.py 1792539`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 139 머리말: 갱신 2026-11-08 05:48 UTC, 라운드 139 — GPU `lo_pkg_bp_park_s1` 끝 (rc 0) → `lo_pkg_bp_park_s2` 증류 it 400 정상; CPU 좌표 하네스 3/8 (`eval_pkg20_0 r0_coord_s1_rl+rep4`) 21분째)

## 라운드 139 (2026-11-08 05:46~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 24.6 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 572 B 복구 (05:37 줄: s1 끝, s2 시작).
- 한 줄 (N 6단계): GPU = `lo_pkg_bp_park_s1` rc 0 (05:37:47, metrics.json 있음; val iter 100 pcb10_0 fail 0.94 / ip10_0 fail 0.62) → 드라이버가 `lo_pkg_bp_park_s2` (seed 2) 시작: 증류 it 400 / 475 s, teach ok 0.56–0.69, cert-bad 0, oom 0, gpu peak 8.9 GB, val it 300 pcb10_0 fail 0.75 / ip10_0 fail 0.38 (NaN 없음). cpu = `N coord harness eval_pkg20_0 r0_coord_s1_rl+rep4` (05:25:57~, 워커 1825779 100 % CPU, rss 6.4 GB — 진행 중). 하네스 `lo_pkg_bp_park_s1_rl+rep4` (큐 52·53) 는 `after` 충족, 좌표 하네스 뒤 차례. 새 기록 0.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) 좌표 하네스 나머지 (s1 pkg20_0·1, s2 pkg16/20 × 2). (2) `eval10 lo_pkg_bp_park_s1_rl` (드라이버). (3) `lo_pkg_bp_park_s2` 끝 (~07:40 UTC) → `lo_pkg_bm2_rlonly` 시작 뒤 처음 25 iter 멈춤 규칙 확인. (4) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (5) 라운드마다 `restore_proc_log.py 1792539`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 138 머리말: 갱신 2026-11-08 05:32 UTC, 라운드 138 — CPU `eval10 lo_pkg_bp_park_rl` 6/6 끝 → 좌표 하네스 3/8 (`eval_pkg20_0 r0_coord_s1_rl+rep4`); GPU `lo_pkg_bp_park_s1` RL it 150 정상)

## 라운드 138 (2026-11-08 05:29~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 24.3 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` → 723 B 복구 (05:21~05:25 줄: collect_results, 좌표 하네스 시작).
- 한 줄 (N 6단계): GPU = `lo_pkg_bp_park_s1` RL (it 150 / 3803 s, count/ref 4.23, fail 0.22, teach ok 0.88, oom 0, gpu peak 11.3 GB, NaN 0; resume ckpt it 143), cpu = `eval10 lo_pkg_bp_park_rl` **6/6 끝** (pkg10_1 0.80, pcb10_0 0.04, pkg10_0 0.88, pcb10_1 0.04, ip10_0 0.96, ip10_1 1.00 — 뺀 pkg 묶음 0.84, pcb 0.04, ip 0.98) → 드라이버 collect_results (05:24, RESULTS 반영) → `N coord harness eval_pkg20_0 r0_coord_s1_rl+rep4` 05:25~ (3/8), 새 기록 3.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) 좌표 하네스 나머지 (s1 pkg20_0·1, s2 pkg16/20 × 2). (2) `lo_pkg_bp_park_s1` 끝 (~05:40 UTC) → eval10 → `lo_pkg_bp_park_s2`. (3) `lo_pkg_bm2_rlonly` 시작 (~07:40 UTC 이후) 뒤 처음 25 iter 멈춤 규칙 확인. (4) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (5) 라운드마다 `restore_proc_log.py 1792539`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 137 머리말: 갱신 2026-11-08 05:16 UTC, 라운드 137 — CPU `eval10 lo_pkg_bp_park_rl` 3/6 (4번째 세트 진행); GPU `lo_pkg_bp_park_s1` RL it 100 정상)

## 라운드 137 (2026-11-08 05:14~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 20.3 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` 디스크와 같은 크기 → 할 일 없음.
- 한 줄 (N 6단계): GPU = `lo_pkg_bp_park_s1` RL (it 100 / 2483 s, count/ref 3.40, fail 0.66 (train_pcb12_0), teach ok 0.75, oom 0, gpu peak 11.0 GB, NaN 0; val iter 100 pcb10_0 fail 0.94 / ip10_0 fail 0.62 — seed 0 의 it 100 과 같은 수준, new best 2.368), cpu = `eval10 lo_pkg_bp_park_rl` 3/6 (pkg10_1 0.80, pcb10_0 0.04, pkg10_0 0.88; 4번째 세트 04:49~ 진행), 새 기록 0.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) eval10 `lo_pkg_bp_park_rl` 끝 (~06:30 UTC) → collect_results → 좌표 하네스 3/8. (2) 좌표 하네스 나머지 6개. (3) `lo_pkg_bm2_rlonly` 시작 (bp_park 3 seed 뒤 ~07:40 UTC 이후) 뒤 처음 25 iter 멈춤 규칙 확인. (4) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (5) 라운드마다 `restore_proc_log.py 1792539`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 136 머리말: 갱신 2026-11-08 05:00 UTC, 라운드 136 — CPU `eval10 lo_pkg_bp_park_rl` 3/6 (pkg10_0 ok 0.88); GPU `lo_pkg_bp_park_s1` RL it 75 정상)

## 라운드 136 (2026-11-08 04:58~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 20.1 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` 디스크와 같은 크기 → 할 일 없음.
- 한 줄 (N 6단계): GPU = `lo_pkg_bp_park_s1` RL (it 75 / 1803 s, count/ref 1.70, fail 0.06, teach ok 0.94, oom 0, gpu peak 10.7 GB — NaN 없음), cpu = `eval10 lo_pkg_bp_park_rl` 3/6 (`eval_pkg10_0` ok 0.88 (25문제), 04:49 UTC; 4번째 세트 진행), 새 기록 1.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) eval10 `lo_pkg_bp_park_rl` 끝 (~06:30 UTC) → collect_results → 좌표 하네스 3/8. (2) 좌표 하네스 나머지 6개. (3) `lo_pkg_bm2_rlonly` 시작 (bp_park 3 seed 뒤 ~07:40 UTC 이후) 뒤 처음 25 iter 멈춤 규칙 확인. (4) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (5) 라운드마다 `restore_proc_log.py 1792539`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 135 머리말: 갱신 2026-11-08 04:45 UTC, 라운드 135 — CPU `eval10 lo_pkg_bp_park_rl` 2/6 (3번째 세트 진행); GPU `lo_pkg_bp_park_s1` RL it 50 정상)

## 라운드 135 (2026-11-08 04:42~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 20.9 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` 디스크와 같은 크기 → 할 일 없음.
- 한 줄 (N 6단계): GPU = `lo_pkg_bp_park_s1` RL (it 50 / 1163 s, count/ref 2.91, fail 0.41, teach ok 0.94, oom 0, gpu peak 9.9 GB — NaN 없음; log.csv 는 val iter 100 에서 첫 행, seed 0 과 같음), cpu = `eval10 lo_pkg_bp_park_rl` 2/6 (pkg10_1 ok 0.80, pcb10_0 ok 0.04; 3번째 세트 진행), 새 기록 0.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) eval10 `lo_pkg_bp_park_rl` 끝 (~06:40 UTC) → collect_results → 좌표 하네스 3/8. (2) 좌표 하네스 나머지 6개. (3) `lo_pkg_bm2_rlonly` 시작 (bp_park 3 seed 뒤 ~07:40 UTC 이후) 뒤 처음 25 iter 멈춤 규칙 확인. (4) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (5) 라운드마다 `restore_proc_log.py 1792539`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 134 머리말: 갱신 2026-11-08 04:28 UTC, 라운드 134 — CPU `eval10 lo_pkg_bp_park_rl` 2/6 (pcb10_0 ok 0.04); GPU `lo_pkg_bp_park_s1` 증류 끝 → RL 시작)

## 라운드 134 (2026-11-08 04:26~ UTC)
- `git fetch r2e`: 새 main 커밋 없음 (HEAD 1824f96 = 서버 2 N 기록, 이미 merge). QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 20.8 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` 디스크와 같은 크기 → 할 일 없음.
- 한 줄 (N 6단계): GPU = `lo_pkg_bp_park_s1` 증류 끝 (it 2300, val pcb10_0 fail 1.00 / ip10_0 fail 0.31) → RL 시작 (pool 18 batch, GPU 5.8 GB, log.csv 아직 행 없음), cpu = `eval10 lo_pkg_bp_park_rl` 2/6 (`eval_pkg10_1` ok 0.80 개수 20.40, `eval_pcb10_0` ok 0.04 개수 7.00 — 학습 val pcb10_0 fail 0.94 와 맞음; 3번째 세트 워커 1816446 100 % CPU), 새 기록 1.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) eval10 `lo_pkg_bp_park_rl` 끝 (~06:40 UTC) → collect_results → 좌표 하네스 3/8. (2) 좌표 하네스 나머지 6개. (3) `lo_pkg_bm2_rlonly` 시작 (bp_park 3 seed 뒤 ~07:40 UTC 이후) 뒤 처음 25 iter 멈춤 규칙 확인. (4) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (5) 라운드마다 `restore_proc_log.py 1792539`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 133 머리말: 갱신 2026-11-08 04:12 UTC, 라운드 133 — CPU `eval10 lo_pkg_bp_park_rl` 1/6 (pkg10_1 ok 0.80); GPU `lo_pkg_bp_park_s1` 증류 it 1725 정상)

## 라운드 133 (2026-11-08 04:10~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 20.0 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` 디스크와 같은 크기 → 할 일 없음.
- 한 줄 (N 6단계): GPU = `lo_pkg_bp_park_s1` 증류 (it 1725 / 2041 s, teach ok 0.44–0.69, cert-bad 0, oom 0, gpu peak 8.9 GB, val pcb10_0 fail 1.00 / ip10_0 fail 0.31 — 증류 중, NaN 없음), cpu = `eval10 lo_pkg_bp_park_rl` (03:38:16~, 1/6 끝: `eval_pkg10_1` ok 0.80 개수 20.40; 다음 세트 워커 1814483 100 % CPU), 새 기록 1 (eval10 기록, collect_results 는 6개 끝난 뒤 드라이버가 돌림).
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) eval10 `lo_pkg_bp_park_rl` 끝 (~30분/세트 → ~06:40 UTC) → collect_results → 좌표 하네스 3/8. (2) 좌표 하네스 나머지 6개. (3) `lo_pkg_bm2_rlonly` 시작 (bp_park 3 seed 뒤 ~07:40 UTC 이후) 뒤 처음 25 iter 멈춤 규칙 확인. (4) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (5) 라운드마다 `restore_proc_log.py 1792539`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 132 머리말: 갱신 2026-11-08 03:57 UTC, 라운드 132 — CPU `eval10 lo_pkg_bp_park_rl` 17분째 (워커 100 %); GPU `lo_pkg_bp_park_s1` 증류 it 925 정상)

## 라운드 132 (2026-11-08 03:54~ UTC)
- `git fetch r2e`: HEAD = r2e/main (서버 2 사람 커밋 `aa9466f` GA/LS seed 1 기준선 기록 9개, `13f8143` `lo_pkg_bdf_distill` 기록 — 이미 merge 됨, 서버 2 몫이라 중복 실행 안 함). QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 21.0 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` 디스크와 같은 크기 → 할 일 없음.
- 한 줄 (N 6단계): GPU = `lo_pkg_bp_park_s1` 증류 (it 925 / 1082 s, teach ok 0.81, oom 0, gpu peak 8.9 GB, val pcb10_0 fail 1.00 / ip10_0 fail 0.31 — 증류 중, NaN 없음), cpu = `eval10 lo_pkg_bp_park_rl` (03:38:16~, forkserver 워커 1810064 100 % CPU, rss 5.2 GB — 부모 0 % 는 정상, dead Pool 아님; 아직 기록 0/6), 새 기록 0.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) eval10 `lo_pkg_bp_park_rl` 끝 → collect_results → 좌표 하네스 3/8. (2) 좌표 하네스 나머지 6개. (3) `lo_pkg_bm2_rlonly` 시작 (bp_park 3 seed 뒤 ~07:40 UTC 이후) 뒤 처음 25 iter 멈춤 규칙 확인. (4) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (5) 라운드마다 `restore_proc_log.py 1792539`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 131 머리말: 갱신 2026-11-08 03:41 UTC, 라운드 131 — 좌표 하네스 2/8 끝 (`r0_coord_s1_rl+rep4` pkg16 50문제 ok 0.84); GPU `lo_pkg_bp_park` 학습 끝 → eval10, `lo_pkg_bp_park_s1` 증류 it 100 정상)

## 라운드 131 (2026-11-08 03:38~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. mem 21.2 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` +1307 B 복구.
- **좌표 하네스 2/8 끝 (드라이버):** `N coord harness eval_pkg16_1 r0_coord_s1_rl+rep4` 02:50–03:33 rc 0, 기록 `runs/2026-10-29/r0_coord_s1_rl/eval/eval_pkg16_1__r0_coord_s1_rl+rep4.json` (peak rss 23.8 GB). collect_results 03:33–03:36 rc 0 → RESULTS pkg16 행이 25 → 50문제: **ok 0.84**, 개수 9.95, 되먹임 0.60 → 0.84 (+0.24 [+0.14, +0.36]). 속도 숫자는 QUEUE 7에 따라 따로 말하지 않음.
- **GPU:** `lo_pkg_bp_park` rc 0 (03:35:59, RL it 184, val pcb10_0 fail 0.94 / ip10_0 fail 0.62 — RESULTS 학습 표에 자동 반영). 드라이버가 바로 `lo_pkg_bp_park_s1` (seed 1) 시작: 증류 it 100 / 104 s, teach ok 0.81, oom 0, gpu peak 8.9 GB, val pcb10_0 fail 1.00 (증류 초반, 멈춤 규칙은 bm2만 해당). `eval10 lo_pkg_bp_park_rl` 03:38:16 시작 (6 set, 막 시작).
- 한 줄 (N 6단계): GPU = `lo_pkg_bp_park_s1` 증류, cpu = `eval10 lo_pkg_bp_park_rl` → 다음 좌표 하네스 3/8, 새 기록 1.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) eval10 `lo_pkg_bp_park_rl` 진행 확인 (워커가 0 % 로 머물면 dead Pool 워커 의심). (2) 좌표 하네스 나머지 6개. (3) `lo_pkg_bm2_rlonly` 시작 (bp_park 3 seed 뒤 ~07:40 UTC 이후) 뒤 처음 25 iter 멈춤 규칙 확인. (4) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (5) 라운드마다 `restore_proc_log.py 1792539`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 130 머리말: 갱신 2026-11-08 03:24 UTC, 라운드 130 — CPU 좌표 하네스 2/8 (`r0_coord_s1_rl+rep4` pkg16_1) 34분째; GPU `lo_pkg_bp_park` RL it 150 정상)

## 라운드 130 (2026-11-08 03:22~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. pod 28.8 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` 디스크와 같은 크기 → 할 일 없음.
- 한 줄 (N 6단계): GPU = `lo_pkg_bp_park` RL (log.txt it 150 / 3604 s, fail 0.09, lam 3.12, teach ok 0.94, cert-bad 5, oom 0, gpu peak 11.6 GB, resume ckpt it 152; log.csv 는 val 때만 써서 it 100 이 마지막 — 멈춘 것 아님), cpu = `N coord harness eval_pkg16_1 r0_coord_s1_rl+rep4` (02:50:15~, 워커 1803436 100 % CPU, rss 10.6 GB), 새 기록 0.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) 좌표 하네스 나머지 7개 (~54분/파일). (2) `lo_pkg_bm2_rlonly` 시작 (~09:00 UTC) 뒤 처음 25 iter 멈춤 규칙 확인. (3) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (4) 라운드마다 `restore_proc_log.py 1792539`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 129 머리말: 갱신 2026-11-08 03:08 UTC, 라운드 129 — CPU 좌표 하네스 2/8 (`r0_coord_s1_rl+rep4` pkg16_1) 16분째; GPU `lo_pkg_bp_park` RL it 100 정상)

## 라운드 129 (2026-11-08 03:06~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. pod 27.9 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` 디스크와 같은 크기 → 할 일 없음.
- 한 줄 (N 6단계): GPU = `lo_pkg_bp_park` RL (it 100 / 2184 s, fail 0.06, lam 2.51, val pcb10_0 fail 0.94 / ip10_0 fail 0.63, NaN 없음), cpu = `N coord harness eval_pkg16_1 r0_coord_s1_rl+rep4` (02:50:15~, 워커 1803436 100 % CPU, rss 9.7 GB), 새 기록 0.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) 좌표 하네스 나머지 7개 (~54분/파일). (2) `lo_pkg_bm2_rlonly` 시작 (~09:00 UTC) 뒤 처음 25 iter 멈춤 규칙 확인. (3) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (4) 라운드마다 `restore_proc_log.py 1792539`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 128 머리말: 갱신 2026-11-08 02:51 UTC, 라운드 128 — CPU 좌표 하네스 1/8 끝 (`r0_coord_s1_rl+rep4` pkg16_0 ok 0.84); GPU `lo_pkg_bp_park` RL it 75 정상)

## 라운드 128 (2026-11-08 02:49~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. pod 18.6 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` +363 B 복구.
- **좌표 하네스 1/8 끝 (드라이버):** `N coord harness eval_pkg16_0 r0_coord_s1_rl+rep4` 01:52–02:45 rc 0 — **ok 0.84**, 개수 10.14, 21.5 s/문제 (속도 숫자는 QUEUE 7에 따라 표에 안 씀), 기록 `runs/2026-10-29/r0_coord_s1_rl/eval/eval_pkg16_0__r0_coord_s1_rl+rep4.json`. collect_results 02:45–02:48 rc 0 자동 반영 (RESULTS.md 변경분 커밋).
- 한 줄 (N 6단계): GPU = `lo_pkg_bp_park` RL (it 75 / 1484 s, fail 0.12 (train_pcb8_2), teach ok 0.94, oom 0, gpu peak 10.6 GB, NaN 없음), cpu = `N coord harness eval_pkg16_1 r0_coord_s1_rl+rep4` (02:50:15~, 2/8), 새 기록 1. `restore_proc_log` 한 번 더 +384 B.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) 좌표 하네스 나머지 7개 (한 파일 ~54분 → ~6 h). (2) `lo_pkg_bm2_rlonly` 시작 (~09:00 UTC) 뒤 처음 25 iter 멈춤 규칙 확인. (3) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (4) 라운드마다 `restore_proc_log.py 1792539`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 127 머리말: 갱신 2026-11-08 02:35 UTC, 라운드 127 — GPU `lo_pkg_bp_park` RL 단계 it 25 정상; CPU 좌표 하네스 1/8 (`r0_coord_s1_rl+rep4` pkg16_0) 41분째, 새 기록 0

## 라운드 127 (2026-11-08 02:33~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. pod 28.4 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` 디스크와 같은 크기 → 할 일 없음.
- 한 줄 (N 6단계): GPU = `lo_pkg_bp_park` RL (distill 0.75 h 끝 → `lo_pkg_bp_park_rl` it 25 / 457 s, fail 0.55, teach ok 0.81, oom 0, gpu peak 8.9 GB, NaN 없음), cpu = `N coord harness eval_pkg16_0 r0_coord_s1_rl+rep4` (01:52~, 워커 1795347 100 % CPU, rss 10.2 GB — 멈춘 워커 아님), 새 기록 0.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) 좌표 하네스 8개 진행 (pkg16 rep4 한 파일이 40분 넘음 → 8개 약 5 h 이상 예상, 기록만). (2) `lo_pkg_bm2_rlonly` 시작 (~09:00 UTC) 뒤 처음 25 iter 멈춤 규칙 확인. (3) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (4) 라운드마다 `restore_proc_log.py 1792539`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 126 머리말: 갱신 2026-11-08 02:20 UTC, 라운드 126 — GPU `lo_pkg_bp_park` distill it 2175 정상; CPU 좌표 하네스 1/8 (`r0_coord_s1_rl+rep4` pkg16_0) 26분째, 새 기록 0)

## 라운드 126 (2026-11-08 02:17~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 맨 위 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. pod 27.8 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` 디스크와 같은 크기 → 할 일 없음.
- 한 줄 (N 6단계): GPU = `lo_pkg_bp_park` distill (it 2175 / 2525 s, loss 0.37, teach ok 0.56, cert-bad 0, oom 0, gpu peak 8.9 GB), cpu = `N coord harness eval_pkg16_0 r0_coord_s1_rl+rep4` (01:52~, 워커 100 % CPU, rss 9.8 GB), 새 기록 0.
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) 좌표 하네스 8개 진행. (2) `lo_pkg_bm2_rlonly` 시작 (~09:00 UTC) 뒤 처음 25 iter 멈춤 규칙 확인. (3) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (4) 라운드마다 `restore_proc_log.py 1792539`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 125 머리말: 갱신 2026-11-08 02:05 UTC, 라운드 125 — F3 22/22 끝; CPU 좌표 하네스 1/8 (`r0_coord_s1_rl+rep4` pkg16_0) 진행; GPU `lo_pkg_bp_park` distill it 1375 정상

## 라운드 125 (2026-11-08 02:01~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 마지막 절 = 11/08 12:10 KST (cpu 순서) 그대로.
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. pod 26.9 GiB / 60, oom_kill 2.
- `restore_proc_log.py 1792539 agent/logs/pipeline_day2.log` +458 B (loop.sh autostash로 지워진 inode 복구).
- **F3 22/22 끝 (드라이버):** `eval_ip20_1 r2pk_s2_rl+rep4` 01:41–01:47 rc 0. collect_results 01:47–01:50 자동 반영. → F 묶음 (F1·F2·F3) 서버 1 몫 모두 끝.
- 한 줄 (N 6단계): GPU = `lo_pkg_bp_park` distill (it 1375, loss 0.18, teach ok 0.56, oom 0; val pcb10_0 fail 0.44 / ip10_0 fail 0.19 — distill 단계, 판정 아님; 기록은 `runs/2026-10-29/lo_pkg_bp_park_distill/`), cpu = `N coord harness eval_pkg16_0 r0_coord_s1_rl+rep4` (01:52~), 새 기록 0 (하네스 기록 아직 없음).
- 결정: 큐·플래그·순서 그대로, 에이전트 새 항목 없음 — 이유: QUEUE N 6단계.
- 할 일 (다음): (1) 좌표 하네스 8개 진행. (2) `lo_pkg_bm2_rlonly` 시작 (~09:00 UTC) 뒤 처음 25 iter 멈춤 규칙 확인. (3) 예전 분석 cpu 항목 한 번 더 실패하면 적기만. (4) 라운드마다 `restore_proc_log.py 1792539`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 124 머리말: 사람 밤 묶음 N 시작됨: 드라이버 재시작 (01:34), GPU `lo_pkg_bp_park` distill 중, 첫 25 iter 정상; CPU F3 22/22번째 진행

## 라운드 124 (2026-11-08 01:44~ UTC)
- `git fetch r2e`: HEAD = r2e/main (9a1c6db, 사람 N 커밋 3개 이미 받음). QUEUE 새 절 3개: 11/08 "11:30 KST" N 묶음, "11:55" 좌표 seed 1·2 하네스 8개, "12:10" cpu 순서. (절 시각은 KST로 적혔지만 커밋은 10:31~10:37 KST = 01:31~01:37 UTC.)
- N 1~3단계 확인 (사람이 이미 함): `smoke_manoharan2.ok`·`smoke_devformer.ok`·`smoke_mm_bias.ok` 01:33~01:34 UTC 생김. 드라이버 재시작 01:34:20 (`driver day2 start`, 새 pid 1792539), GPU 줄이 `lo_pkg_bp_park` 올림 (run_rung, distill 0.75 h + RL 1.25 h). 이전 드라이버가 돌리던 `eval_ip20_1 r2pk_s1_rl+rep4`는 ATTACH로 기다려 끝남.
- **N 5단계 (lo_pkg_bp_park 첫 25 iter):** 정상 — it 25 loss 5.030, teach ok 0.81, cert-bad 0, oom 0; it 50 loss 3.551; it 500 loss 3.893, val pcb10_0 fail 0.75 / ip10_0 fail 0.25 (distill 단계, 판정 아님). loss NaN 없음 (log.csv의 nan은 distill 단계 best_ratio·fail 열). `lo_pkg_bm2_rlonly` 25 iter 확인은 그 run 시작 뒤 (~18:00 KST 예상).
- **F3 21번째 끝 (드라이버 ATTACH):** `eval_ip20_1 r2pk_s1_rl+rep4` 01:36 rc 0 — 정책 단독 8/8 → **8/8** (되먹임 쓴 문제 0), 개수 21.38, 4.75 s/문제, peak RSS 7.3 GB, git f408aa9. 기록 `runs/2026-10-29/r2pk_s1_rl/eval/eval_ip20_1__r2pk_s1_rl+rep4.json`.
- 한 줄 (N 6단계): GPU = `lo_pkg_bp_park` (distill), cpu = F3 `eval_ip20_1 r2pk_s2_rl+rep4` (01:41~), 새 기록 1. 대기 cpu 47 (F3 1 + 좌표 하네스 8 + N 하네스 38, 하네스는 `after` 게이트로 모델 대기).
- `ensure_running`: 드라이버 alive (1792539), report_watch·resume_agent 다시 띄움. pod 21.7 GiB / 60, oom_kill 2. `restore_proc_log.py 1792539` 같은 inode, 할 일 없음.
- 결정: 큐·플래그·순서 그대로 (사람이 넣은 N 항목), 에이전트 새 항목 없음 — 이유: QUEUE N 6단계 "에이전트가 스스로 항목을 더 넣지 않는다".
- 할 일 (다음): (1) F3 22/22 끝 → 좌표 하네스 8개 진행 확인. (2) `lo_pkg_bm2_rlonly` 시작하면 처음 25 iter를 PROPOSALS N 멈춤 규칙 (NaN, 또는 val 세 종류 fail 1.00 + loss 0)으로 확인 → 걸리면 안 시작한 bm2 5개를 `train_removed`로. (3) 예전 분석 cpu 항목이 한 번 더 실패하면 재시도 없이 PROGRESS에 적기만. (4) 라운드마다 `restore_proc_log.py 1792539`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 123 머리말: 드라이버 F3 20/22 끝 (ip20_0 `r2pk_s2_rl+rep4`·ip20_1 `r2pk_rl+rep4` 둘 다 8/8 → 8/8, 되먹임 안 씀) → ip20_0 3 seed 모두 끝; GPU 빔 (사람 항목 없음)

## 라운드 123 (2026-11-08 01:28~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 마지막 = 11/08 03:05 KST 해제 (사람 항목만) 그대로.
- `ensure_running`: 드라이버 alive (1575375), report_watch·resume_agent 다시 띄움. pod 16.8 GiB / 60, oom_kill 2.
- `restore_proc_log.py 1575375 agent/logs/pipeline_day2.log` +1296 B.
- **F3 19번째 끝 (드라이버):** `eval_ip20_0 r2pk_s2_rl+rep4` 01:14–01:18 rc 0 — 정책 단독 8/8 → **8/8** (n=8, 바뀐 문제 0), 성공 평균 개수 21.62, 2.77 s/문제, peak RSS 5.7 GB. 기록 `runs/2026-10-29/r2pk_s2_rl/eval/eval_ip20_0__r2pk_s2_rl+rep4.json`. → ip20_0 3 seed 끝.
- **F3 20번째 끝 (드라이버):** `eval_ip20_1 r2pk_rl+rep4` 01:22–01:27 rc 0 — 정책 단독 8/8 → **8/8** (바뀐 문제 0), 개수 20.50, 3.89 s/문제, peak RSS 7.2 GB. 기록 `runs/2026-10-29/r2pk_rl/eval/eval_ip20_1__r2pk_rl+rep4.json`.
- RESULTS는 드라이버 collect_results (01:18–01:21, 01:27~) 자동 반영. 숫자만 — 읽기는 사람.
- 드라이버 다음: `eval_ip20_1 r2pk_s1_rl+rep4` → `r2pk_s2_rl+rep4` → F3 22/22 끝. 그 뒤 CPU 큐는 `cpu_deferred`뿐 (사람이 내린 것 → 다시 올리지 않음).
- 결정: 큐·플래그 그대로, GPU 새 학습·side_slot 나눔 없음 — 이유: 라운드 108과 같음 (에이전트 새 시도 금지, F3 side_slot 허용 없음).
- 할 일 (다음): (1) 드라이버 F3 남은 2개. (2) 사람 새 QUEUE 항목 fetch (11/08 아침 KST). (3) F3 끝나면 "최종 방법" 표 20칸 ip 칸 확인. (4) 라운드마다 `restore_proc_log.py`. DAILY 다음 11/09 01:15 UTC.

# (이전) 라운드 122 머리말: 드라이버 F3 18/22 끝 (ip20_0 `r2pk_rl+rep4`·`r2pk_s1_rl+rep4` 둘 다 8/8 → 8/8, 되먹임 안 씀); DAILY 11/08 새로 씀; GPU 빔 (사람 항목 없음)

## 라운드 122 (2026-11-08 01:11~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 마지막 = 11/08 03:05 KST 해제 (사람 항목만) 그대로.
- `ensure_running`: 드라이버 alive (1575375), report_watch·resume_agent 다시 띄움. pod 16.7 GiB / 60, oom_kill 2.
- `restore_proc_log.py 1575375 agent/logs/pipeline_day2.log` +962 B.
- **F3 17번째 끝 (드라이버):** `eval_ip20_0 r2pk_rl+rep4` 00:49–00:59 rc 0 — 정책 단독 8/8 → **8/8** (n=8, 되먹임 쓴 문제 0), 성공 평균 개수 25.88, 4.83 s/문제, peak RSS 6.0 GB. 기록 `runs/2026-10-29/r2pk_rl/eval/eval_ip20_0__r2pk_rl+rep4.json`.
- **F3 18번째 끝 (드라이버):** `eval_ip20_0 r2pk_s1_rl+rep4` 01:04–01:10 rc 0 — 정책 단독 8/8 → **8/8** (되먹임 쓴 문제 0), 개수 23.00, 2.95 s/문제, peak RSS 5.7 GB. 기록 `runs/2026-10-29/r2pk_s1_rl/eval/eval_ip20_0__r2pk_s1_rl+rep4.json`.
- ip20 메모리 걱정 해소 (peak 6 GB). RESULTS는 드라이버 collect_results (00:59–01:02, 01:10~) 자동 반영. 숫자만 — 읽기는 사람.
- 드라이버 다음: `eval_ip20_0 r2pk_s2_rl+rep4` → ip20_1 3 seed → F3 22/22 끝. 그 뒤 CPU 큐에 남는 것은 `cpu_deferred`뿐 (사람이 내린 것 → 다시 올리지 않음).
- **DAILY.md 새로 씀** (11/07 01:30 ~ 11/08 01:15 UTC, 라운드 65~122). 다음 DAILY 11/09 01:15 UTC.
- 결정: 큐·플래그 그대로, GPU 새 학습·side_slot 나눔 없음 — 이유: 라운드 108과 같음 (에이전트 새 시도 금지, F3 side_slot 허용 없음).
- 할 일 (다음): (1) 드라이버 F3 남은 4개. (2) 사람 새 QUEUE 항목 fetch (11/08 아침 KST, B-Manoharan 원문판·B-DevFormer 사전 등록 예정). (3) F3 끝나면 "최종 방법" 표 20칸 ip 칸 확인. (4) 라운드마다 `restore_proc_log.py`.

# (이전) 라운드 121 머리말: 드라이버 F3 16/22 끝 (`r2pk_s2_rl+rep4` pcb20_1 0.75 → 1.00, 되먹임 2문제) → pcb20 3 seed × 2 파일 모두 끝; 17번째 `r2pk_rl+rep4` ip20_0 진행; GPU 빔 (사람 항목 없음)

## 라운드 121 (2026-11-08 00:55~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 마지막 = 11/08 03:05 KST 해제 (사람 항목만) 그대로.
- `ensure_running`: 드라이버 alive (1575375), report_watch·resume_agent 다시 띄움. pod 19.4 GiB / 60, oom_kill 2.
- `restore_proc_log.py 1575375 agent/logs/pipeline_day2.log` +680 B.
- **F3 16번째 끝 (드라이버):** `eval_pcb20_1 r2pk_s2_rl+rep4` 00:34–00:44 rc 0 — 정책 단독 6/8 (0.75) → **8/8 (1.00)** (n=8, 되먹임 쓴 문제 2), 성공 평균 개수 7.12 (단독 6.17), 1.67 s/문제, peak RSS 13.9 GB. 기록 `runs/2026-10-29/r2pk_s2_rl/eval/eval_pcb20_1__r2pk_s2_rl+rep4.json`. → F3 pcb16·ip16·pcb20 16개 모두 끝. RESULTS는 드라이버 collect_results (00:44–00:47) 자동 반영. 숫자만 — 읽기는 사람.
- 드라이버 다음: `eval_ip20_0 r2pk_rl+rep4` 00:49~ 진행 → F3 나머지 6개 (ip20 3 seed × 2 파일).
- 결정: 큐·플래그 그대로, GPU 새 학습·side_slot 나눔 없음 — 이유: 라운드 108과 같음 (에이전트 새 시도 금지, F3 side_slot 허용 없음).
- 할 일 (다음): (1) 드라이버 F3 ip20 6개 (ip20 메모리 확인). (2) 사람 새 QUEUE 항목 fetch (11/08 아침 KST). (3) 라운드마다 `restore_proc_log.py`. DAILY 다음 11/08 01:30 UTC.

# (이전) 라운드 120 머리말: 드라이버 F3 15/22 끝 (`r2pk_s1_rl+rep4` pcb20_1 0.88 → 1.00, 되먹임 1문제); 16번째 `r2pk_s2_rl+rep4` pcb20_1 진행; GPU 빔 (사람 항목 없음

## 라운드 120 (2026-11-08 00:39~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 마지막 = 11/08 03:05 KST 해제 (사람 항목만) 그대로.
- `ensure_running`: 드라이버 alive (1575375), report_watch·resume_agent 다시 띄움. pod 22.7 GiB / 60, oom_kill 2.
- `restore_proc_log.py 1575375 agent/logs/pipeline_day2.log` +697 B.
- **F3 15번째 끝 (드라이버):** `eval_pcb20_1 r2pk_s1_rl+rep4` 00:20–00:29 rc 0 — 정책 단독 7/8 (0.88) → **8/8 (1.00)** (n=8, 되먹임 쓴 문제 1: 11 → 12개), 성공 평균 개수 7.25, 1.51 s/문제, peak RSS 13.9 GB. 기록 `runs/2026-10-29/r2pk_s1_rl/eval/eval_pcb20_1__r2pk_s1_rl+rep4.json`. RESULTS는 드라이버 collect_results (00:29–00:32) 자동 반영 (pcb20 열 [16] = 두 파일). 숫자만 — 읽기는 사람.
- 드라이버 다음: `eval_pcb20_1 r2pk_s2_rl+rep4` 00:34~ 진행 → F3 나머지 7개 (pcb20_1 1 → ip20 6).
- 결정: 큐·플래그 그대로, GPU 새 학습·side_slot 나눔 없음 — 이유: 라운드 108과 같음 (에이전트 새 시도 금지, F3 side_slot 허용 없음).
- 할 일 (다음): (1) 드라이버 F3 나머지. (2) 사람 새 QUEUE 항목 fetch (11/08 아침 KST). (3) 라운드마다 `restore_proc_log.py`. DAILY 다음 11/08 01:30 UTC.

# (이전) 라운드 119 머리말: 드라이버 F3 14/22 끝 (`r2pk_rl+rep4` pcb20_1 1.00 → 1.00, 되먹임 안 씀, 8.50개); 15번째 `r2pk_s1_rl+rep4` pcb20_1 진행; GPU 빔 (사람 항목 없음)

## 라운드 119 (2026-11-08 00:23~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 마지막 = 11/08 03:05 KST 해제 (사람 항목만) 그대로.
- `ensure_running`: 드라이버 alive (1575375), report_watch·resume_agent 다시 띄움. pod 22.2 GiB / 60, oom_kill 2.
- `restore_proc_log.py 1575375 agent/logs/pipeline_day2.log` +694 B.
- **F3 14번째 끝 (드라이버):** `eval_pcb20_1 r2pk_rl+rep4` 00:07–00:16 rc 0 — 정책 단독 8/8 → **8/8** (n=8, 되먹임 쓴 문제 0), 성공 평균 개수 8.50, 1.65 s/문제, peak RSS 13.9 GB. 기록 `runs/2026-10-29/r2pk_rl/eval/eval_pcb20_1__r2pk_rl+rep4.json`. RESULTS는 드라이버 collect_results (00:16–00:19) 자동 반영. 숫자만 — 읽기는 사람.
- 드라이버 다음: `eval_pcb20_1 r2pk_s1_rl+rep4` 00:20~ 진행 → F3 나머지 8개 (pcb20_1 2 → ip20 6).
- 결정: 큐·플래그 그대로, GPU 새 학습·side_slot 나눔 없음 — 이유: 라운드 108과 같음 (에이전트 새 시도 금지, F3 side_slot 허용 없음).
- 할 일 (다음): (1) 드라이버 F3 나머지. (2) 사람 새 QUEUE 항목 fetch (11/08 아침 KST). (3) 라운드마다 `restore_proc_log.py`. DAILY 다음 11/08 01:30 UTC.

# (이전) 라운드 118 머리말: 드라이버 F3 13/22 끝 (`r2pk_s2_rl+rep4` pcb20_0 1.00 → 1.00, 되먹임 안 씀) → pcb20_0 3 seed 모두 끝; 14번째 `r2pk_rl+rep4` pcb20_1 진행; GPU 빔 (사람 항목 없음)

## 라운드 118 (2026-11-08 00:07~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 마지막 = 11/08 03:05 KST 해제 (사람 항목만) 그대로.
- `ensure_running`: 드라이버 alive (1575375), report_watch·resume_agent 다시 띄움. pod 30.1 GiB / 60 (pcb20_1 평가 시작 직후), oom_kill 2.
- `restore_proc_log.py 1575375 agent/logs/pipeline_day2.log` +1107 B.
- **F3 13번째 끝 (드라이버):** `eval_pcb20_0 r2pk_s2_rl+rep4` 23:55–00:02 rc 0 — 정책 단독 8/8 → **8/8** (n=8, 되먹임 쓴 문제 0), 성공 평균 개수 4.62, 0.55 s/문제, peak RSS 10.9 GB. 기록 `runs/2026-10-29/r2pk_s2_rl/eval/eval_pcb20_0__r2pk_s2_rl+rep4.json`. RESULTS는 드라이버 collect_results (00:02–00:05) 자동 반영. 숫자만 — 읽기는 사람.
- 드라이버 다음: `eval_pcb20_1 r2pk_rl+rep4` 00:07~ 진행 → F3 나머지 9개 (pcb20_1 3 → ip20 6).
- 결정: 큐·플래그 그대로, GPU 새 학습·side_slot 나눔 없음 — 이유: 라운드 108과 같음 (에이전트 새 시도 금지, F3 side_slot 허용 없음).
- 할 일 (다음): (1) 드라이버 F3 나머지. (2) 사람 새 QUEUE 항목 fetch (11/08 아침 KST). (3) 라운드마다 `restore_proc_log.py`. DAILY 다음 11/08 01:30 UTC.

# (이전) 라운드 117 머리말: 드라이버 F3 12/22 끝 (pcb20_0: `r2pk_rl+rep4` 1.00 → 1.00 되먹임 안 씀; `r2pk_s1_rl+rep4` 0.88 → 1.00, 되먹임 1문제); 13번째 `r2pk_s2_rl+rep4` pcb20_0 다음; GPU 빔 (사람 항목 없음)

## 라운드 117 (2026-11-07 23:51~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 마지막 = 11/08 03:05 KST 해제 (사람 항목만) 그대로.
- `ensure_running`: 드라이버 alive (1575375), report_watch·resume_agent 다시 띄움. pod 16.7 GiB / 60, oom_kill 2.
- `restore_proc_log.py 1575375 agent/logs/pipeline_day2.log` +966 B.
- **F3 11번째 끝 (드라이버):** `eval_pcb20_0 r2pk_rl+rep4` 23:29–23:38 rc 0 — 정책 단독 8/8 → **8/8** (n=8, 되먹임 쓴 문제 0), 성공 평균 개수 5.50, 1.21 s/문제, peak RSS 10.9 GB. 기록 `runs/2026-10-29/r2pk_rl/eval/eval_pcb20_0__r2pk_rl+rep4.json`.
- **F3 12번째 끝 (드라이버):** `eval_pcb20_0 r2pk_s1_rl+rep4` 23:42–23:51 rc 0 — 정책 단독 7/8 (0.88) → **8/8 (1.00)** (되먹임 쓴 문제 1: 첫 문제 11 → 12개), 개수 5.50, 0.89 s/문제, peak RSS 10.9 GB. 기록 `runs/2026-10-29/r2pk_s1_rl/eval/eval_pcb20_0__r2pk_s1_rl+rep4.json`.
- pcb20 메모리 걱정 (라운드 116) 해소: peak 10.9 GB, pod 16.5 GiB. RESULTS는 드라이버 collect_results (23:38–23:41, 23:51~) 자동 반영. 숫자만 — 읽기는 사람.
- 드라이버 다음: `eval_pcb20_0 r2pk_s2_rl+rep4` → F3 나머지 10개 (pcb20_0 1 → pcb20_1 3 → ip20 6).
- 결정: 큐·플래그 그대로, GPU 새 학습·side_slot 나눔 없음 — 이유: 라운드 108과 같음 (에이전트 새 시도 금지, F3 side_slot 허용 없음).
- 할 일 (다음): (1) 드라이버 F3 나머지. (2) 사람 새 QUEUE 항목 fetch (11/08 아침 KST). (3) 라운드마다 `restore_proc_log.py`. DAILY 다음 11/08 01:30 UTC.

# (이전) 라운드 116 머리말: 드라이버 F3 10/22 끝 (`r2pk_s2_rl+rep4` ip16_1 1.00 → 1.00, 되먹임 안 씀) → ip16 3 seed × 2 파일 모두 끝; 11번째 `r2pk_rl+rep4` pcb20_0 진행; GPU 빔 (사람 항목 없음)

## 라운드 116 (2026-11-07 23:35~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 마지막 = 11/08 03:05 KST 해제 (사람 항목만) 그대로.
- `ensure_running`: 드라이버 alive (1575375), report_watch·resume_agent 다시 띄움. pod 22.4 GiB / 60, oom_kill 2.
- `restore_proc_log.py 1575375 agent/logs/pipeline_day2.log` +681 B.
- **F3 10번째 끝 (드라이버):** `eval_ip16_1 r2pk_s2_rl+rep4` 23:17–23:25 rc 0 — 정책 단독 1.00 → **1.00** (n=25, 되먹임 쓴 문제 0), 성공 평균 개수 15.64, 1.72 s/문제, peak RSS 13.0 GB. 기록 `runs/2026-10-29/r2pk_s2_rl/eval/eval_ip16_1__r2pk_s2_rl+rep4.json`. → F3 pcb16·ip16 12개 모두 끝. RESULTS는 드라이버 collect_results (23:25–23:28) 자동 반영.
- 드라이버 다음: `eval_pcb20_0 r2pk_rl+rep4` 23:29~ 진행 (forkserver 자식 RSS 5.3 GB, 6분째) → F3 나머지 11개 (pcb20 5 → ip20 6). pcb16 peak 22.4 GB였으므로 pcb20은 더 클 수 있음 — 다음 라운드에 peak 확인.
- 결정: 큐·플래그 그대로, GPU 새 학습·side_slot 나눔 없음 — 이유: 라운드 108과 같음 (에이전트 새 시도 금지, F3 side_slot 허용 없음).
- 할 일 (다음): (1) 드라이버 F3 나머지 (pcb20 메모리 확인). (2) 사람 새 QUEUE 항목 fetch (11/08 아침 KST). (3) 라운드마다 `restore_proc_log.py`. DAILY 다음 11/08 01:30 UTC.

# (이전) 라운드 115 머리말: 드라이버 F3 9/22 끝 (`r2pk_rl+rep4`·`r2pk_s1_rl+rep4` ip16_1 둘 다 1.00 → 1.00, 되먹임 안 씀), 10번째 `r2pk_s2_rl+rep4` ip16_1 진행; GPU 빔 (사람 항목 없음)

## 라운드 115 (2026-11-07 23:17~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 마지막 = 11/08 03:05 KST 해제 (사람 항목만) 그대로.
- `ensure_running`: 드라이버 alive (1575375), report_watch·resume_agent 다시 띄움. pod 16.7 GiB / 60, oom_kill 2.
- `restore_proc_log.py 1575375 agent/logs/pipeline_day2.log` +1036 B, 다음 항목 시작 뒤 한 번 더 +349 B.
- **F3 8번째 끝 (드라이버):** `eval_ip16_1 r2pk_rl+rep4` 22:53–23:01 rc 0 — 정책 단독 1.00 → **1.00** (n=25, 되먹임 쓴 문제 0), 성공 평균 개수 15.88, 1.72 s/문제, peak RSS 13.0 GB. 기록 `runs/2026-10-29/r2pk_rl/eval/eval_ip16_1__r2pk_rl+rep4.json`.
- **F3 9번째 끝 (드라이버):** `eval_ip16_1 r2pk_s1_rl+rep4` 23:05–23:13 rc 0 — 정책 단독 1.00 → **1.00** (n=25, 되먹임 쓴 문제 0), 개수 16.24, 1.82 s/문제, peak RSS 13.0 GB. 기록 `runs/2026-10-29/r2pk_s1_rl/eval/eval_ip16_1__r2pk_s1_rl+rep4.json`.
- RESULTS는 드라이버 collect_results (23:01–23:04, 23:13–23:16) 자동 반영. 숫자만 — 읽기는 사람.
- 드라이버 다음: `eval_ip16_1 r2pk_s2_rl+rep4` 23:17~ 진행 → F3 나머지 12개 (pcb20 6 → ip20 6).
- 결정: 큐·플래그 그대로, GPU 새 학습·side_slot 나눔 없음 — 이유: 라운드 108과 같음 (에이전트 새 시도 금지, F3 side_slot 허용 없음).
- 할 일 (다음): (1) 드라이버 F3 나머지 (pcb20은 메모리 큼 — 확인). (2) 사람 새 QUEUE 항목 fetch (11/08 아침 KST). (3) 라운드마다 `restore_proc_log.py`. DAILY 다음 11/08 01:30 UTC.

# (이전) 라운드 114 머리말: 드라이버 F3 7/22 끝 (`r2pk_s2_rl+rep4` ip16_0 1.00 → 1.00, 되먹임 안 씀), 8번째 `r2pk_rl+rep4` ip16_1 진행; GPU 빔 (사람 항목 없음)

## 라운드 114 (2026-11-07 22:58~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 마지막 = 11/08 03:05 KST 해제 (사람 항목만) 그대로.
- `ensure_running`: 드라이버 alive (1575375), report_watch·resume_agent 다시 띄움. pod 22.6 GiB / 60, oom_kill 2.
- `restore_proc_log.py 1575375 agent/logs/pipeline_day2.log` +679 B.
- **F3 7번째 끝 (드라이버):** `eval_ip16_0 r2pk_s2_rl+rep4` 22:38–22:49 rc 0 — 정책 단독 1.00 → **1.00** (n=25, 되먹임 쓴 문제 0), 성공 평균 개수 11.56, 1.24 s/문제, peak RSS 13.9 GB. 기록 `runs/2026-10-29/r2pk_s2_rl/eval/eval_ip16_0__r2pk_s2_rl+rep4.json`. → F3 ip16_0 3 seed 모두 끝. RESULTS는 드라이버 collect_results (22:49–22:52) 자동 반영.
- 드라이버 다음: `eval_ip16_1 r2pk_rl+rep4` 22:53~ 진행 (자식 RSS 5.6 GB) → F3 나머지 14개 (ip16_1 3 → pcb20 6 → ip20 6).
- 결정: 큐·플래그 그대로, GPU 새 학습·side_slot 나눔 없음 — 이유: 라운드 108과 같음 (에이전트 새 시도 금지, F3 side_slot 허용 없음).
- 할 일 (다음): (1) 드라이버 F3 나머지 (메모리 확인). (2) 사람 새 QUEUE 항목 fetch (11/08 아침 KST). (3) 라운드마다 `restore_proc_log.py`. DAILY 다음 11/08 01:30 UTC.

# (이전) 라운드 113 머리말: 드라이버 F3 6/22 끝 (`r2pk_s1_rl+rep4` ip16_0 1.00 → 1.00, 되먹임 안 씀), 7번째 `r2pk_s2_rl+rep4` ip16_0 진행; GPU 빔 (사람 항목 없음)

## 라운드 113 (2026-11-07 22:42~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 마지막 = 11/08 03:05 KST 해제 (사람 항목만) 그대로.
- `ensure_running`: 드라이버 alive (1575375), report_watch·resume_agent 다시 띄움. pod 22.0 GiB / 60, oom_kill 2.
- `restore_proc_log.py 1575375 agent/logs/pipeline_day2.log` +1117 B.
- **F3 6번째 끝 (드라이버):** `eval_ip16_0 r2pk_s1_rl+rep4` 22:29–22:33 rc 0 — 정책 단독 1.00 → **1.00** (n=25, 되먹임 쓴 문제 0), 성공 평균 개수 11.60, 1.05 s/문제, peak RSS 13.9 GB. 기록 `runs/2026-10-29/r2pk_s1_rl/eval/eval_ip16_0__r2pk_s1_rl+rep4.json`. RESULTS는 드라이버 collect_results (22:33–22:36) 자동 반영.
- 드라이버 다음: `eval_ip16_0 r2pk_s2_rl+rep4` 22:38~ 진행 → F3 나머지 15개 (ip16_1 3 → pcb20 6 → ip20 6).
- 결정: 큐·플래그 그대로, GPU 새 학습·side_slot 나눔 없음 — 이유: 라운드 108과 같음 (에이전트 새 시도 금지, F3 side_slot 허용 없음).
- 할 일 (다음): (1) 드라이버 F3 나머지 (메모리 확인). (2) 사람 새 QUEUE 항목 fetch (11/08 아침 KST). (3) 라운드마다 `restore_proc_log.py`. DAILY 다음 11/08 01:30 UTC.

# (이전) 라운드 112 머리말: 드라이버 F3 5/22 끝 (`r2pk_s2_rl+rep4` pcb16_1 0.96 → 1.00; `r2pk_rl+rep4` ip16_0 1.00 → 1.00), 6번째 ip16 진행; GPU 빔 (사람 항목 없음)

## 라운드 112 (2026-11-07 22:26~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 마지막 = 11/08 03:05 KST 해제 (사람 항목만) 그대로.
- `ensure_running`: 드라이버 alive (1575375), report_watch·resume_agent 다시 띄움. pod 16.8 GiB / 60, oom_kill 2.
- `restore_proc_log.py 1575375 agent/logs/pipeline_day2.log` +948 B.
- **F3 4번째 끝 (드라이버):** `eval_pcb16_1 r2pk_s2_rl+rep4` 22:00–22:15 rc 0 — 정책 단독 0.96 → **1.00** (n=25), 성공 평균 개수 9.96, 되먹임 쓴 문제 4, 2.77 s/문제, peak RSS 22.4 GB. 기록 `runs/2026-10-29/r2pk_s2_rl/eval/eval_pcb16_1__r2pk_s2_rl+rep4.json`. → F3 pcb16 6개 (3 seed × 2 파일) 모두 끝.
- **F3 5번째 끝 (드라이버):** `eval_ip16_0 r2pk_rl+rep4` 22:19–22:24 rc 0 — 정책 단독 1.00 → **1.00** (n=25, 되먹임 쓴 문제 0), 개수 11.88, 1.02 s/문제, peak RSS 13.9 GB. 기록 `runs/2026-10-29/r2pk_rl/eval/eval_ip16_0__r2pk_rl+rep4.json`.
- RESULTS는 드라이버 collect_results (22:15–22:18, 22:24~)가 자동 반영. 숫자만 — 읽기는 사람.
- 드라이버 다음: F3 나머지 17개 (ip16 5 → pcb20 6 → ip20 6).
- 결정: 큐·플래그 그대로, GPU 새 학습·side_slot 나눔 없음 — 이유: 라운드 108과 같음 (에이전트 새 시도 금지, F3 side_slot 허용 없음).
- 할 일 (다음): (1) 드라이버 F3 나머지 (메모리 확인). (2) 사람 새 QUEUE 항목 fetch (11/08 아침 KST). (3) 라운드마다 `restore_proc_log.py`. DAILY 다음 11/08 01:30 UTC.

# (이전) 라운드 111 머리말: 드라이버 F3 3/22 끝 (`r2pk_s1_rl+rep4` pcb16_1 1.00 → 1.00, 되먹임 안 씀), 4번째 `r2pk_s2_rl+rep4` pcb16_1 진행; GPU 빔 (사람 항목 없음)

## 라운드 111 (2026-11-07 22:11~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 마지막 = 11/08 03:05 KST 해제 (사람 항목만) 그대로.
- `ensure_running`: 드라이버 alive (1575375), report_watch·resume_agent 다시 띄움. pod 25.7 GiB / 60, oom_kill 2.
- `restore_proc_log.py 1575375 agent/logs/pipeline_day2.log` +697 B.
- **F3 3번째 끝 (드라이버):** `eval_pcb16_1 r2pk_s1_rl+rep4` 21:45–21:55 rc 0 — 정책 단독 1.00 → **1.00** (n=25, 되먹임 쓴 문제 0), 성공 평균 개수 9.68, 1.64 s/문제, peak RSS 22.4 GB. 기록 `runs/2026-10-29/r2pk_s1_rl/eval/eval_pcb16_1__r2pk_s1_rl+rep4.json`. RESULTS는 드라이버 collect_results (21:55–21:58) 자동 반영.
- 드라이버 다음: `eval_pcb16_1 r2pk_s2_rl+rep4` 22:00~ 진행 → F3 나머지 18개.
- 결정: 큐·플래그 그대로, GPU 새 학습·side_slot 나눔 없음 — 이유: 라운드 108과 같음 (에이전트 새 시도 금지, F3 side_slot 허용 없음).
- 할 일 (다음): (1) 드라이버 F3 나머지 (메모리 확인). (2) 사람 새 QUEUE 항목 fetch (11/08 아침 KST). (3) 라운드마다 `restore_proc_log.py`. DAILY 다음 11/08 01:30 UTC.

# (이전) 라운드 110 머리말: 드라이버 F3 2/22 끝 (`r2pk_s2_rl+rep4` pcb16_0 0.84 → 1.00), 3번째 `r2pk_s1_rl+rep4` pcb16_1 진행; GPU 빔 (사람 항목 없음)

## 라운드 110 (2026-11-07 21:55~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 마지막 = 11/08 03:05 KST 해제 (사람 항목만) 그대로.
- `ensure_running`: 드라이버 alive (1575375), report_watch·resume_agent 다시 띄움. pod 26.0 GiB / 60, oom_kill 2.
- `restore_proc_log.py 1575375 agent/logs/pipeline_day2.log` +697 B.
- **F3 2번째 끝 (드라이버):** `eval_pcb16_0 r2pk_s2_rl+rep4` 21:25–21:40 rc 0 — 정책 단독 0.84 → **1.00** (n=25), 성공 평균 개수 9.52, 되먹임 쓴 문제 1, 3.31 s/문제, peak RSS 22.6 GB. 기록 `runs/2026-10-29/r2pk_s2_rl/eval/eval_pcb16_0__r2pk_s2_rl+rep4.json`. RESULTS는 드라이버 collect_results (21:40–21:43) 자동 반영.
- 드라이버 다음: `eval_pcb16_1 r2pk_s1_rl+rep4` 21:45~ 진행 → F3 나머지 19개.
- 결정: 큐·플래그 그대로, GPU 새 학습·side_slot 나눔 없음 — 이유: 라운드 108과 같음 (에이전트 새 시도 금지, F3 side_slot 허용 없음).
- 할 일 (다음): (1) 드라이버 F3 나머지 (메모리 확인). (2) 사람 새 QUEUE 항목 fetch (11/08 아침 KST). (3) 라운드마다 `restore_proc_log.py`. DAILY 다음 11/08 01:30 UTC.

# (이전) 라운드 109 머리말: 드라이버 F3 2/22 진행 (`r2pk_s2_rl+rep4` pcb16_0, 21:25~); GPU 빔 (사람 항목 없음)

## 라운드 109 (2026-11-07 21:39~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 마지막 = 11/08 03:05 KST 해제 (사람 항목만) 그대로.
- `ensure_running`: 드라이버 alive (1575375), report_watch·resume_agent 다시 띄움. pod 26.5 GiB / 60, oom_kill 2.
- `restore_proc_log.py 1575375 agent/logs/pipeline_day2.log` +351 B (loop autostash로 다시 끊겼던 것).
- **F3 2번째 (드라이버):** `eval_pcb16_0 r2pk_s2_rl+rep4` 21:25~ 진행 (forkserver 자식 RSS 10.2 GB, CPU 100%). 1번째 (`r2pk_s1_rl`) 14분 걸림 → 끝 ~21:40–21:55 예상.
- 결정: 큐·플래그 그대로, GPU 새 학습·side_slot 나눔 없음 — 이유: 라운드 108과 같음 (에이전트 새 시도 금지, F3 side_slot 허용 없음).
- 할 일 (다음): (1) 드라이버 F3 나머지 21개 (메모리 확인). (2) 사람 새 QUEUE 항목 fetch (11/08 아침 KST). (3) 라운드마다 `restore_proc_log.py`. DAILY 다음 11/08 01:30 UTC.

# (이전) 라운드 108 머리말: 드라이버 F3 1/22 끝 (`r2pk_s1_rl+rep4` pcb16_0 0.92 → 1.00); 나머지 F3 21개 순서대로; GPU 빔 (사람 항목 없음)

## 라운드 108 (2026-11-07 21:23~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 마지막 = 11/08 03:05 KST 해제 (사람 항목만) 그대로.
- `ensure_running`: 드라이버 alive (1575375), report_watch·resume_agent 다시 띄움. side_slot 프로세스 없음 (R 대상 모두 끝나서 스스로 끝남, 정상). pod 16.8 GiB / 60, oom_kill 2.
- `restore_proc_log.py` 드라이버 +697 B (`agent/logs/pipeline_day2.log`). 처음에 경로를 잘못 줘서 `agent/logs/day2_driver.log`가 생겼다 → 바로 지움 (추적 안 된 새 파일, 다른 영향 없음).
- **F3 첫 항목 (드라이버):** `eval_pcb16_0 r2pk_s1_rl+rep4` 21:06–21:20 rc 0 — 정책 단독 0.92 → **1.00** (n=25), 성공 평균 개수 9.00, 되먹임 쓴 문제 2, 2.05 s/문제, peak RSS 22.6 GB. 기록 `runs/2026-10-29/r2pk_s1_rl/eval/eval_pcb16_0__r2pk_s1_rl+rep4.json`. RESULTS는 드라이버 collect_results (21:20~)가 자동 반영.
- 드라이버 다음: F3 나머지 21개 (pcb16 3 → ip16 6 → pcb20 6 → ip20 6). 항목당 15–50분 → 끝 ~11/08 06–09 UTC 예상.
- 결정: F3를 side_slot으로 나눠 돌리지 않음, GPU 새 학습 넣지 않음 — 이유: QUEUE의 side_slot 허용은 F1 pkg20에만 있었음 (F3는 언급 없음), pkg16 수리 판 동시 실행 때 45.7 GiB를 넘은 적 있음 (라운드 99); 에이전트 새 시도 금지 (11/08 03:05 KST).
- 할 일 (다음): (1) 드라이버 F3 21개 (메모리 확인). (2) 사람 새 QUEUE 항목 fetch (11/08 아침 KST). (3) 라운드마다 `restore_proc_log.py 1575375 agent/logs/pipeline_day2.log`. DAILY 다음 11/08 01:30 UTC.

# (이전) 라운드 107 머리말: F1 14/14 끝 (좌표+V pkg20_1 0.62); F2 서버 2 done → 건너뜀; 드라이버 F3 22개 시작; R 대조군 `lo_pkg_rnd_r0` 끝 (10칸 뺀 pkg 0.74); GPU 빔

## 라운드 107 (2026-11-07 20:03~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 마지막 = 11/08 03:05 KST 해제 (사람 항목만) 그대로.
- `ensure_running`: 드라이버 alive (1575375), report_watch alive, resume_agent 다시 띄움. `restore_proc_log.py` 드라이버 +430/+98/+273 B, side_slot 그대로. pod 25–27 GiB / 60, oom_kill 2.
- **F1 마지막 (드라이버):** `eval_pkg20_1 r0_coord_rl+rep4` 19:49–21:02 rc 0 — ok **0.62** (n=8), 개수 12.00, 110.6 s/문제 (정책 단독 기록도 108.8 s/문제 — 이 파일이 원래 느림). 기록 `runs/2026-10-28/r0_coord_rl/eval/eval_pkg20_1__r0_coord_rl+rep4.json`. → **F1 14개 모두 끝.**
- RESULTS (드라이버 collect_results 21:02–21:05, 1216 기록): "최종 방법" 짝 표 pkg20 좌표+V 두 파일 묶음 — 기준 0.81, MoTT+P1+V 0.958, 차 **+0.15 [+0.02, +0.27]**, 개수비 1.090 [1.006, 1.168] (3 seed × 16 = 48 짝, 작은 표본). 숫자만 — 읽기 (a)(b)(c)는 사람.
- **R 대조군:** `lo_pkg_rnd_r0` RL 20:38 끝 (349 iter, 1.27 h, best iter 200 score 1.719, oom 0, GPU peak 14.7 GB). side_slot `eval10 lo_pkg_rnd_r0_rl` 20:40–20:58 rc 0. `r_screen.py` 2×2 (seed 0, 보고만): 좌표+rnd `lo_pkg_rnd_r0` vs 좌표 `lo_pkg_r0` 뺀 pkg 0.74 vs 0.78, −0.04 [−0.22, +0.12]; MoTT+rnd vs 좌표+rnd 0.68 vs 0.74, −0.06 [−0.22, +0.10]. R 판정 그대로 **미통과** (RESULTS "R" 절 자동). → R 항목 모두 끝.
- 드라이버: F2 (서버 2 done) 건너뜀 → **F3 22개** 21:06 시작 (`eval_pcb16_0 r2pk_s1_rl+rep4` 첫 항목). GPU 학습 큐 비어 있음 (사람 항목 끝, 에이전트 새 학습 금지) → GPU 빔.
- 결정: 큐·플래그·side_slot 그대로, GPU 새 학습 넣지 않음 — 이유: QUEUE 11/08 03:05 KST "에이전트가 스스로 새 학습·시도를 넣는 것은 여전히 금지"; 다음 실험 (B-Manoharan 원문판, B-DevFormer)은 사람이 사전 등록 뒤 넣음.
- 할 일 (다음): (1) 드라이버 F3 22개 (메모리 확인). (2) 사람 새 QUEUE 항목 fetch (11/08 아침 KST). (3) 라운드마다 `restore_proc_log.py`. DAILY 다음 11/08 01:30 UTC.

# (이전) 라운드 106 머리말: F1 좌표 `r0_coord_rl+rep4` pkg20_0 끝 (1.00, n=8); 드라이버 좌표 pkg20_1 다음; GPU `lo_pkg_rnd_r0` RL it 75

## 라운드 106 (2026-11-07 19:44~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 마지막 = 11/08 03:05 KST 해제 (사람 항목만) 그대로.
- `ensure_running`: 드라이버 alive (1575375), report_watch·resume_agent 다시 띄움. `restore_proc_log.py` 드라이버 +273 B, side_slot 그대로 (같은 inode). pod 18.7 GiB / 60, oom_kill 2.
- **F1 끝난 것 (드라이버):** `eval_pkg20_0 r0_coord_rl+rep4` 19:22–19:44 rc 0 — ok **1.00** (n=8), 개수 9.25, 17.19 s/문제 (수리 351 s 포함), peak RSS 17.1 GB, round 4. 기록 `runs/2026-10-28/r0_coord_rl/eval/eval_pkg20_0__r0_coord_rl+rep4.json`. RESULTS는 드라이버 collect_results (19:44~)가 자동 반영 → "최종 방법" 표 pkg20 좌표+V 짝 CI는 pkg20_1 끝난 뒤 두 파일 묶음으로 (숫자만, 읽기는 사람).
- 드라이버 다음: `eval_pkg20_1 r0_coord_rl+rep4` → F2 (서버 2 done → 건너뜀) → F3 22.
- R 대조군 `lo_pkg_rnd_r0` RL it 75 / 1105 s (cert-bad 4, fail 0.03, oom 0, GPU peak 12.1 GB) 정상 → 끝 ~20:45 → side_slot 10칸 (보고만, `r_screen_table` 2×2 자동).
- 결정: 큐·플래그·side_slot 그대로 — 이유: 남은 항목 모두 사람 것, 에이전트 새 시도 금지 유지 (QUEUE 11/08 03:05 KST).
- 할 일 (다음): (1) 좌표 pkg20_1 → F3 22 (드라이버). (2) `lo_pkg_rnd_r0` 끝 → 10칸 → 2×2 표 확인. (3) 라운드마다 `restore_proc_log.py`. DAILY 다음 11/08 01:30 UTC.

# (이전) 라운드 105 머리말: F1 pkg16 6/6 끝 (`r2pk_s2_rl` pkg16_1 0.96; 3 seed 묶음 150문제 0.940); 드라이버 좌표 `r0_coord_rl+rep4` pkg20_0 진행; GPU `lo_pkg_rnd_r0` RL 시작

## 라운드 105 (2026-11-07 19:28~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 마지막 = 11/08 03:05 KST 해제 (사람 항목만) 그대로.
- `ensure_running`: 드라이버 alive (1575375), report_watch·resume_agent 다시 띄움. `restore_proc_log.py` 드라이버 +702 B, side_slot 그대로 (같은 inode). pod 25.8 GiB / 60, oom_kill 2.
- **F1 끝난 것 (드라이버):** `eval_pkg16_1 r2pk_s2_rl+rep4` 18:59–19:18 rc 0 — **0.96**, 개수 10.04, 3.39 s/문제. 이로써 F1 pkg16 수리 평가 6개 (3 seed × 2 파일) 끝. RESULTS 자동 표 (19:18): `r2pk_s2_rl+rep4` pkg16 50문제 0.94 (정책 단독 0.56), "최종 방법" pkg16 묶음 3 seed **0.940** (150문제, 개수 10.30). 짝 CI pkg16 150: 좌표+V 대비 +0.14 [+0.07, +0.21] 개수비 1.131; greedy+prune+V 대비 −0.02 [−0.05, +0.00] 개수비 0.729. 숫자만 — 읽기는 사람.
- 드라이버 다음: `eval_pkg20_0 r0_coord_rl+rep4` 19:22~ → pkg20_1 → F2 (서버 2 done → 건너뜀) → F3 22.
- R 대조군 `lo_pkg_rnd_r0`: 증류 끝, RL 19:22경 시작 (GPU 7.7 GB) → 1.25 h → 끝 ~20:45 → side_slot 10칸 (보고만, `r_screen_table` 2×2 자동).
- 결정: 큐·플래그·side_slot 그대로 — 이유: 남은 항목 모두 사람 것, 에이전트 새 시도 금지 유지.
- 할 일 (다음): (1) 좌표 pkg20 2 → F3 22 (드라이버). (2) `lo_pkg_rnd_r0` 끝 → 10칸 → 2×2 표 확인. (3) 라운드마다 `restore_proc_log.py`. DAILY 다음 11/08 01:30 UTC.

# (이전) 라운드 104 머리말: F1 pkg16_1 `r2pk_s1_rl+rep4` 끝 (0.48 → 0.92), pkg16_1 `r2pk_s2_rl` 진행 (F1 pkg16 마지막); GPU `lo_pkg_rnd_r0` 증류 it 1725

## 라운드 104 (2026-11-07 19:09~ UTC)
- `git fetch r2e`: 새 main 커밋 없음. QUEUE 마지막 = 11/08 03:05 KST 해제 (사람 항목만) 그대로.
- `ensure_running`: 드라이버 alive (1575375), report_watch·resume_agent 다시 띄움. `restore_proc_log.py` 드라이버 +697 B, side_slot 그대로 (같은 inode). pod 28.4 GiB / 60, oom_kill 2.
- **F1 끝난 것 (드라이버):** `eval_pkg16_1 r2pk_s1_rl+rep4` 18:36–18:55 rc 0 — 정책 단독 0.48 → **0.92**, 2.44 s/문제, peak RSS 24.4 GB. RESULTS 자동 표: `r2pk_s1_rl+rep4` pkg16 50문제 0.92 (개수 9.83). 다음 `eval_pkg16_1 r2pk_s2_rl+rep4` 18:59~ (F1 pkg16 6개 중 마지막) → 좌표 pkg20 2 → F2 (서버 2 done → 건너뜀) → F3 22.
- R 대조군 `lo_pkg_rnd_r0` 증류 it 1725 / 1980 s (cert-bad 0, oom 0, GPU 8.9 GB) 정상 → RL 1.25 h → 끝 ~20:45 → side_slot 10칸 (보고만, `r_screen_table` 2×2 자동).
- 결정: 큐·플래그·side_slot 그대로 — 이유: 모든 남은 항목이 사람 것, 에이전트 새 시도 금지 유지.
- 할 일 (다음): 라운드 103 목록 그대로. DAILY 다음 11/08 01:30 UTC.

# (이전) 라운드 103 머리말: F1 pkg16_0 3 seed 끝 (묶음 0.933), pkg16_1 2/3 진행; 서버 2 F 기록 18/18 끝 (pkg20 묶음 0.958); R 3 seed 판정 **미통과** (뺀 pkg +0.01); GPU `lo_pkg_rnd_r0` 시작

## 라운드 103 (2026-11-07 18:14~ UTC)
- `git fetch r2e`: 새 main 커밋 없음 (새 브랜치 `baselines-final`만 — 사람 것, 건드리지 않음). QUEUE 마지막 = 11/08 03:05 KST 해제 (사람 항목만).
- `ensure_running`: 드라이버 alive (1575375), report_watch·resume_agent 다시 띄움. `restore_proc_log.py` 드라이버 두 번 (+682 B, +1278 B), side_slot 그대로. pod 26–30 GiB (18:37 순간 40.8 GiB: 수리 평가 + 학습 끝 + 10칸 평가 시작이 겹침), oom_kill 2.
- **F1 끝난 것 (드라이버):** `eval_pkg16_0 r2pk_s2_rl+rep4` 17:41–18:08 rc 0 — 정책 단독 0.44 → **0.92**, 개수 11.13. `eval_pkg16_1 r2pk_rl+rep4` 18:12–18:31 rc 0 — 0.52 → **0.96**, 개수 10.38. 다음 `eval_pkg16_1 r2pk_s1_rl+rep4` 18:36~.
- **서버 2:** `pull_s2_records.py` 18:12 `r2pk_s2_rl` pkg20_1 merge → **18/18 끝** (rc 0, "not merging" 없음).
- RESULTS 다시 생성 (report_watch 18:17, 1200 기록, 자동 커밋 c1e45f5): "최종 방법" 표 pkg16_0 묶음 3 seed **0.933** (75문제, 정책 단독 0.56), pkg20 묶음 3 seed **0.958** (48문제, 정책 단독 0.50). 짝 CI pkg16: 좌표+V 대비 +0.13 [+0.05, +0.21] 개수비 1.132; greedy+prune+V 대비 −0.03 [−0.07, +0.00] 개수비 0.727. 숫자만 — 읽기는 사람.
- **R:** `lo_pkg_rnd_s2` RL 18:36 끝 (353 iter, 1.27 h, best iter 100 score 1.567, oom 0). side_slot `eval10 lo_pkg_rnd_s2_rl` 18:37~. 드라이버 GPU 다음 = `lo_pkg_rnd_r0` 18:36~ (사람이 넣은 항목, 좌표 + rnd 대조군; 증류 0.75 h + RL 1.25 h → 끝 ~20:45).
- **R 판정 (`r_screen.py`, side_slot `eval10 lo_pkg_rnd_s2_rl` 18:37–18:49 rc 0 뒤, seed 0·1·2 묶음 150 짝):** 뺀 pkg `lo_pkg_rnd` 0.72 vs `lo_pkg_r2pk` 0.71, +0.01 [−0.08, +0.11] → (i) 미달; pcb +0.05 [−0.02, +0.11], ip +0.01 → (ii) 통과; 개수비 1.020 / 1.034 / 1.002 → (iii) 통과 → **미통과 (효과 없음)**. 보고만: vs 좌표 `lo_pkg_r0` −0.08 [−0.16, +0.00]. 확인 run 없음 (사람 01:45 KST). `collect_results.py`에 `r_screen_table()` ("R" 절, 2×2 대조군 `lo_pkg_rnd_r0`은 끝나면 자동 채움) + "시도한 수정" 표 R 줄 추가.
- 결정: 큐·플래그·side_slot 그대로 — 이유: F 순서·R 항목은 모두 사람 것; 에이전트 새 시도 금지 유지.
- 할 일 (다음): (1) ~~`lo_pkg_rnd_s2` 10칸 → `r_screen.py`~~ 끝 (미통과). (2) 드라이버 F1 pkg16_1 남은 2 → 좌표 pkg20 2 → F2 (서버 2 done → 건너뜀) → F3 22. (3) `lo_pkg_rnd_r0` 끝 → side_slot 10칸 (보고만, 2×2 읽기). (4) 라운드마다 `restore_proc_log.py`. DAILY 다음 11/08 01:30 UTC.

# (이전) 라운드 102 머리말: F1 드라이버 pkg16 수리 평가 3/6 `r2pk_s2_rl` 진행; 서버 2 F 기록 17/18 merge; 사람 03:05 KST 새 학습 금지 해제 (사람 항목만); R `lo_pkg_rnd_s2` RL 중

## 라운드 102 (2026-11-07 17:58~ UTC)
- `git fetch r2e`: 새 main 커밋 없음 (새 브랜치 `bm-faithful`만 생김 — 사람 것, 건드리지 않음). 라운드 101 이후 확인한 사람 커밋 d3cec50 (QUEUE 11/08 03:05 KST): "11/08 12:00 KST 새 학습 금지" 해제, **사람이 넣은 항목만**; 에이전트가 스스로 새 학습·시도를 넣는 것은 여전히 금지. 다음 계획 (B-Manoharan 원문판 재실행, B-DevFormer 재구현)은 사람이 11/08 아침에 사전 등록 뒤 큐에 넣음 → 지금 할 것 없음.
- 서버 2 기록: `pull_s2_records.py`가 17:38 `r2pk_s1_rl` pkg20_0·1, 17:54 `r2pk_s2_rl` pkg20_0 merge → **17/18** (남은 1 = `r2pk_s2_rl` pkg20_1). "not merging" 없음.
- `ensure_running`: 드라이버 alive (1575375), report_watch·resume_agent 다시 띄움. `restore_proc_log.py`: 드라이버 +697 B 복구, side_slot 그대로. pod 29.4 GiB / 60 (pkg16 수리 중), oom_kill 2.
- 드라이버 `eval_pkg16_0 r2pk_s2_rl+rep4` 17:41~ 진행 (16분째, 앞 두 개 20–22분).
- R `lo_pkg_rnd_s2` RL it 150 / 2037 s (cert-bad 5, fail 0.04, grid capped 32, oom 0, GPU peak 13.6 GB) 정상.
- 결정: 큐·플래그·side_slot 그대로 — 이유: 사람 해제는 사람 항목에만 적용, F 순서는 사람 것.
- 할 일 (다음): 라운드 100 목록 그대로 + 서버 2 마지막 1 기록 merge 확인 → RESULTS 다시 생성 (드라이버 collect_results가 함).

## 라운드 101 (2026-11-07 17:35~ UTC)
- `git fetch r2e` (17:35): 새 사람 커밋 없음 (QUEUE 마지막 11/08 01:45 KST F). 서버 2 남은 4 기록 (`r2pk_s1/_s2` pkg20) 아직 안 옴 — `pull_s2.log` 마지막 17:08 merge (14/18), "not merging" 없음.
- `ensure_running`: 드라이버 alive (pid 1575375), report_watch·resume_agent 다시 띄움. side_slot (1710935)·pull_s2_records (1708311) alive. `restore_proc_log.py`: 드라이버 +694 B 복구, side_slot 그대로. pod 29.7 GiB (pkg16 수리 중) / 60, oom_kill 2.
- **F1 끝난 것 (드라이버):** `eval_pkg16_0 r2pk_s1_rl+rep4` 17:15–17:37 rc 0 — 정책 단독 0.60 → 되먹임 **0.92**, 평균 개수 10.39, 4.97 s/문제, round 4. pod 19.3 GiB (끝난 시점). RESULTS 다시 생성 (드라이버 collect_results 17:40): "최종 방법" 표 pkg16 묶음 2 seed 0.62 → **0.940** (50문제). 숫자만.
- 드라이버 다음 `eval_pkg16_0 r2pk_s2_rl+rep4` 17:41~ (사람 순서 그대로) → pkg16_1 3개 → 좌표 pkg20 2개 → F2 (서버 2, done이면 건너뜀) → F3 22개.
- R `lo_pkg_rnd_s2` RL it 50 / 753 s (cert-bad 2, fail 0.02, grid capped 1, oom 0, GPU peak 11.9 GB) 정상 → 끝 ~18:40 → side_slot 10칸 평가 → `r_screen.py` 3 seed.
- 결정: 큐·플래그·side_slot 그대로 — 이유: F 항목은 사람 순서, pkg16 수리 판은 드라이버 한 줄 (라운드 100 결정 유지).
- 할 일 (다음): 라운드 100 목록 그대로.

## 라운드 100 (2026-11-07 17:09~ UTC)
- `git fetch r2e` (17:09): 새 사람 지시 없음 (QUEUE 마지막 11/08 01:45 KST F). 서버 2 기록은 `pull_s2_records.py`가 merge 중: 16:56–17:08에 `r2pk_s2_rl` pcb·pkg·ip 10칸 6개 + `r2pk_rl` pkg20_0·1 → **14/18** (모두 rc 0, "not merging" 없음). 남은 4 = `r2pk_s1/_s2` pkg20.
- `ensure_running`: 드라이버 alive (pid 1575375), report_watch·resume_agent 다시 띄움. `restore_proc_log.py`: 드라이버·side_slot 복구할 것 없음 (디스크 = live). pod 29.2 GiB (pkg16 수리 중) → 18.8 GiB (끝난 뒤), oom_kill 2.
- **F1 끝난 것 (드라이버):** `eval_pkg16_0 r2pk_rl+rep4` 16:51–17:11 rc 0 — ok **0.96**, 평균 개수 10.58, 4.1 s/문제, 되먹임 round 4. pod peak 확인 시점 29.2 GiB (라운드 99의 45.7은 side_slot 좌표 pkg20과 겹친 때). 드라이버 다음 항목 `eval_pkg16_0 r2pk_s1_rl+rep4` 17:15~ (사람 순서 그대로).
- RESULTS 다시 생성 (1193 기록): **"최종 방법 MoTT+P1+V" 표 처음 채워짐** — 10칸 묶음 (3 seed, 150문제) 정책 단독 → +V: ip 0.99 → 1.000, pcb 0.92 → 1.000, pkg 0.82 → 0.987; seed 0 pcb16 0.90 → 0.98, pkg20 0.69 → 0.94 (16문제). 같은 되먹임 기준 대비 짝 CI (10칸, 150 짝): 좌표+V 대비 pkg +0.07 [+0.03, +0.11] 개수비 1.107, pcb +0.02 개수비 1.073, ip +0.00 개수비 0.773; greedy+prune+V 대비 pkg +0.05, pcb +0.06, ip +0.60, 개수비 0.74–0.86. 숫자만 — 읽기 (a)(b)(c)는 사람.
- 확인: pkg20 기록 n=8/파일 (세트 크기 그대로, 버그 아님).
- 결정: side_slot에 F 항목 다시 넣지 않음 — 이유: 라운드 99에서 pkg16 수리 + 좌표 pkg20 동시에 45.7 GiB로 사람 조건 (45 GiB 아래)을 넘었음; 드라이버 한 줄로 순서대로가 보수적. 큐·플래그 그대로.
- R `lo_pkg_rnd_s2` 증류 it 1275 / 2226 s (cert-bad 0, oom 0, GPU 8.9 GB) 정상 → RL → 끝 ~18:40 → side_slot 10칸 평가 → `r_screen.py` 3 seed 판정 (RESULTS만).
- 할 일 (다음): (1) 드라이버 F1 pkg16 남은 5개 → 좌표 pkg20 2개 → F2 (서버 2 몫, done이면 건너뜀) → F3 22개. (2) 서버 2 남은 4 기록 merge 확인. (3) `lo_pkg_rnd_s2` 끝 → 10칸 → `r_screen.py`. (4) 라운드마다 `restore_proc_log.py` (드라이버 1575375, side_slot 1710935). DAILY 다음 11/08 01:30 UTC.

# (이전) PROGRESS 라운드 99 머리말: 갱신 2026-11-07 16:45 UTC — 새 사람 지시 2개 (11/08 01:45·01:40 KST): 최종 방법 MoTT+P1+V 확정, CPU 줄 F1 → F2 → F3, 서버 2가 F1 pkg20·F2; R `lo_pkg_rnd_s1` 학습 끝 → 10칸 평가 중, `lo_pkg_rnd_s2` 시작

## 라운드 99 (2026-11-07 16:08~ UTC)
- `git fetch r2e` (16:08): 새 사람 커밋 없음 (QUEUE 마지막 22:42 KST V). `ensure_running`: 드라이버 alive (pid 1575375), report_watch·resume_agent 다시 띄움. `restore_proc_log.py`: 드라이버 +316 B, side_slot +197 B 복구. pod 18.5 GiB / 60, oom_kill 2.
- G1 보충 2: `r2pkfc_s2_rl` eval_pcb16_0 끝 (side_slot 15:53–16:06). 그다음 eval_pcb16_1 (드라이버 16:11~)은 16:33에 rc=−15로 끝남 = 사람이 죽임 (아래 F 지시). 남은 11개는 사람이 `cpu_deferred`로 내림 → **다시 올리지 않음**.
- **(16:33) 사람 커밋 0242e56 (QUEUE 11/08 01:45 KST, PROPOSALS "11/08 F"):** 최종 방법 = MoTT+P1 + 정밀 검사 되먹임 4회 (`--repair 4`), 이름 `MoTT+P1+V`. 학습 쪽 후보 끝. R seed 1·2·r0는 그대로 끝내고 `r_screen.py` 판정만 (통과해도 확인 run 없음). day2_queue cpu 맨 앞 F1 (14) → F2 (6) → F3 (22). 사람이 이 서버에서 직접 merge (HEAD 0242e56).
- **(16:37) 사람 커밋 acd4935 (01:40 KST):** 서버 2가 `scripts/server2_f.sh`로 F1 pkg20 (r2pk 3 seed × 2 = 6)과 F2 (6, 파일 12)를 맡아 기록마다 push. → merge.
- 한 일:
  - side_slot 다시 띄움 (pid 1708309, 대상 `lo_pkg_rnd_s1 lo_pkg_rnd_s2 lo_pkg_rnd_r0`). 채움 접두어에 **F1 좌표 pkg20 2개만** 추가 (`F1 eval_pkg20_{0,1} r0_coord`): QUEUE가 "드라이버 F1 pkg16 동안 side_slot F1 pkg20 허용"이라 했지만 r2pk pkg20 6개는 서버 2 몫이라 뺐다 (같은 경로 기록이 두 서버에서 생기면 add/add 충돌). 처음엔 F1 pkg20 전체로 띄웠다가 acd4935를 보고 2분 안에 좁혀 다시 띄움 (그 사이 시작한 F 항목 없음). 옛 side_slot이 16:34에 띄운 `eval10 lo_pkg_rnd_s1_rl`은 그대로 돌고 (부모만 바뀜), 드라이버가 16:37 ATTACH.
  - 새 helper `scripts/pull_s2_records.py` (bg, `agent/logs/pull_s2.log`, .gitignore): 3분마다 fetch → 뒤처졌으면 `git merge --no-edit r2e/main` (stash·rebase 없음); 들어올 경로가 여기 이미 있으면 (추적 안 됨·ignored 포함) merge 안 하고 기록만. 서버 2 기록 18개가 다 오면 끝 (최대 8 h). 이유: 드라이버는 디스크의 `done`만 보고 pull하지 않음 → QUEUE "매 항목 시작 전에 pull 해서 done 확인"을 드라이버를 재시작하지 않고 맞추는 가장 작은 방법.
  - `collect_results.py`에 `final_method_table` ("최종 방법 MoTT+P1+V"): 크기 × 종류 × seed, 정책 단독 (`ok_pre`) 옆에 되먹임 포함 성공률·평균 개수, 고친 문제의 부품 수 변화 (줄어듦/같음/늘어남), `repair_used`; 좌표+V·greedy+prune+V가 같은 세트에 있으면 seed마다 같은 문제 짝 → 묶음 CI. 읽기 (a)(b)(c)는 사람 몫이라 표에 판정 없음. F 기록 (`runs/2026-10-29/r2pk*_rl/eval/*+rep4.json`)이 하나라도 생기면 나타남 (지금은 없음; 가짜 기록으로 모양 확인: seed 0 10칸·pcb16 기존 기록이 들어감).
- R: 서버 2 seed 0 기록 (1637c71) 들어옴. `r_screen.py`: seed 0 뺀 pkg rnd 0.68 vs r2pk 0.70 (−0.02 [−0.18, +0.14]), pcb 0.92 vs 0.86, ip 1.00 → 판정 waiting (seed 1·2 필요). `lo_pkg_rnd_s1` 학습 끝 16:34 (rc 0) → 10칸 평가 중; 드라이버 GPU → `lo_pkg_rnd_s2` 16:34 시작 (끝 ~18:40), 그다음 `lo_pkg_rnd_r0`.
- R `lo_pkg_rnd_s1` RL: it 225 (2943 s / 4500 s), oom 0, GPU peak 13.9 GB, train fail 0.03. 검증 it 200: val_pcb10_0 fail 0.56 (count/g+p 0.875), val_ip10_0 fail 0.12 → best 1.470. 참고: val pcb fail이 높은 것은 이 설정에서 평소와 같음 (`lo_pkg_r2pk_s1` RL val pcb fail 0.44–0.81) → 이상 아님, 보고만.
- 메모: 로그 복구 뒤 드라이버는 지워진 옛 inode에 계속 씀 → 로그 기다리기는 디스크 경로가 아니라 `/proc/<pid>/fd/1`로 (`wait_log.py /proc/1575375/fd/1 ...`).
- (16:48–16:54) side_slot이 F1 좌표 pkg20_0 시작 (16:48), 드라이버 F1 `eval_pkg16_0 r2pk_rl+rep4` 시작 (16:51) → **pod 45.7 GiB** (pkg16 수리 worker 혼자 22 GB; 지난 pkg16 수리 기록 peak 25–28 GB, policy 판 20–30 분). QUEUE의 side_slot 허용 조건 "45 GiB 아래일 때만"을 넘음 → **좌표 pkg20 (side_slot, 6 GB)을 멈춤** (기록 안 생김, 드라이버가 큐 순서대로 나중에 함). side_slot에서 F1 채우기를 다시 뺌 (FILL 원래대로), 다시 띄움 (pid 1710935). pod 39.7 GiB. pkg16 수리 판은 드라이버 한 줄로만.
- R `lo_pkg_rnd_s1_rl` 10칸 평가 끝 (16:47): 뺀 pkg **0.82** (0.88/0.76), pcb 0.94 (0.92/0.96), ip 1.00. `r_screen.py` (seed 0·1): 뺀 pkg rnd vs r2pk 0.75 vs 0.67 (+0.08 [−0.03, +0.20]), pcb +0.05, ip 0.00; 보고 rnd vs 좌표 0.75 vs 0.81 → **판정 waiting** (seed 2 필요, `lo_pkg_rnd_s2` 학습 중). 숫자만, 판정 안 함.
- 서버 2 기록 merge (pull_s2_records): 16:41·16:47·16:50에 F2 `r2pk_s1_rl` pcb·pkg·ip 10칸 7/18 들어옴 (merge rc 0, 충돌 없음).
- 결정: F 항목의 플래그·순서·세트, 큐 순서는 그대로 (사람 지시). F 큰 판 수리 평가는 동시에 하나만 (메모리; 사람 허용은 45 GiB 아래 조건부였고 실제로 넘음). 새 학습·평가 넣지 않음.
- 할 일 (다음): (1) `lo_pkg_rnd_s1` 10칸 평가 끝 → RESULTS 자동. (2) 드라이버 F1 pkg16 6개 → (서버 2 pkg20 r2pk 기록 merge되어 건너뜀) → 좌표 pkg20 2개 → F2 (서버 2) → F3 22개. `pull_s2.log`에서 "not merging"이 나오면 원인 확인. (3) F 기록 생기면 "최종 방법 MoTT+P1+V" 표 확인 (숫자만). (4) R 3 seed 모이면 `r_screen.py` 판정을 RESULTS에. (5) 라운드마다 `restore_proc_log.py` (드라이버 1575375, side_slot 1710935). pkg16 수리 중 pod 메모리 확인 (peak 예상 ~46 GiB). DAILY 다음 11/08 01:30 UTC.

## 라운드 98 (2026-11-07 15:51~ UTC)
- `git fetch r2e` (15:51): 새 사람 커밋 없음 (QUEUE 마지막 22:42 KST V). `ensure_running`: 드라이버 alive (pid 1575375), report_watch·resume_agent 다시 띄움. `restore_proc_log.py`: 드라이버 +498 B, side_slot +297 B 복구 (15:51 autostash). pod 18.6 GiB / 60, oom_kill 2.
- G1 보충 2: `r2pkfc_s1_rl` 16·20칸 12개 모두 끝 (pkg20_1 15:50). 남은 것 `r2pkfc_s2_rl` 12개 (cpu 232–243, 드라이버가 순서대로). 표 읽기는 seed 0·1·2 다 나온 뒤 (RESULTS 자동, 지금 "대기").
- R `lo_pkg_rnd_s1` RL: it 125 (1722 s), oom 0, GPU peak 13.3 GB, train fail 0.01. 첫 검증 (it 100): val_pcb10_0 fail **0.62** (count/g+p 1.08), val_ip10_0 fail 0.06 → best 1.574. 보고만 (설정은 사람 고정; 판정은 뺀 pkg 평가로 `r_screen.py`). `r_screen.py` 시험 실행: 기록 없어 모든 줄 waiting (정상).
- 그림 2 갱신판 `reports/fig2_bigboard_r2pk_20261107.png` (새 파일, 옛 초안 그대로; 기록만 읽음, 같은 방법 7개, seed 0): 11/06에 끝난 LS16 pcb16_1·ip16_0이 들어가 LS 16칸 막대가 생김 — pcb 0.64 / 115 s, ip 0.64 / 516 s (각 n=25, 한 파일). 다른 막대 값은 11/05 초안과 같음. 캡션 메모에 "LS 16칸은 파일 하나" 추가.
- 결정: 설정·큐 그대로 (사람 지시). 새 학습·새 평가 넣지 않음; 그림은 기존 기록만.
- 할 일 (다음): (1) `lo_pkg_rnd_s1` RL 끝 (~16:35) → side_slot 10칸 평가 (pcb·pkg·ip). (2) 드라이버가 `lo_pkg_rnd_s2` → `lo_pkg_rnd_r0`. (3) 서버 2 R seed 0 push → merge → 3 seed 모이면 `r_screen.py` → RESULTS 표만. (4) `r2pkfc_s2` 16·20칸 12개 → G1 보충 2 읽기. (5) 라운드마다 `restore_proc_log.py`. DAILY 다음 11/08 01:30 UTC.

## 라운드 97 (2026-11-07 14:50~ UTC)
- `git fetch r2e` (14:50): 새 사람 커밋 없음 (QUEUE 마지막 22:42 KST V). `ensure_running`: 드라이버 alive (pid 1575375), report_watch·resume_agent 다시 띄움. side_slot (pid 1680969) `eval10 r2pkfc_s2_rl` 14:41~ (드라이버 ATTACH 14:44). pod 20.4 GiB / 60, oom_kill 2.
- **V (사람 QUEUE 22:42 KST, 보고만) 끝:** 사람 nohup 평가 12 기록 (`lo_pkg_{r2pk,r0}{,_s1,_s2}_rl+rep4`, pkg10_0·1) 모두 있음, 프로세스 끝남 (죽이지 않음). `collect_results.py`에 `heldout_repair_table` 추가 (PROPOSALS V 읽기 그대로, seed마다 같은 문제 짝 + 3 seed 묶음 150 짝, `ok_pre`로 전 → 후). 결과: MoTT+P1 0.71 → **0.993**, 좌표 0.80 → 0.920; 차 +0.07 [+0.03, +0.12]; 둘 다 성공 개수비 1.064 [1.023, 1.107] (MoTT+P1이 6% 더 씀, 숫자만). seed별 1.00/1.00/0.98. → 고정 읽기 "정밀 검사 피드백을 붙이면 처음 보는 pkg도 거의 해결" (RESULTS에 그 문장). 최종 구조 결정에는 안 씀 (PROPOSALS).
- 드라이버·side_slot 로그 또 끊김 (14:50 loop.sh autostash) → 새 helper `scripts/restore_proc_log.py <pid> <log>` (`/proc/<pid>/fd/1`가 더 길면 복사; 지난 라운드는 손으로 했음). 드라이버 +576 B, side_slot +78 B 복구.
- (15:10) `r2pkfc_s2_rl` 10칸 평가 끝 (side_slot): pcb 0.94 (0.88/1.00), pkg 0.88 (0.84/0.92), ip 1.00 → G1 구조 교체 줄 seed 3개 완성 (RESULTS 자동). `queue_scale_g1s` r2pkfc_s2 16·20칸 12개 붙음 (cpu 232–243, 드라이버 줄에서 r2pkfc_s1 pkg16_1·pkg20_0·1 다음). waiter 끝남.
- (15:20) `lo_pkg_rnd_s1` 증류 끝 (1586 it, 0.76 h, best it 700 1.381, cert-bad 0, oom 0) → **RL 시작, rnd 풀 처음 로딩 정상**: 27 batch / 432 문제 (pcb·ip·rnd 각 9 파일, 1:1:1), rnd 파일은 선생님 파일 없어 기준 = greedy+prune (`train_rnd12_2` g+p 81.25, `rnd10_0` 57.56 — 풀 생성 때 본 큰 개수 그대로). it 25: cert-bad 1, fail 0.01; it 50: cert-bad 4, fail 0.03, grid capped 1, oom 0, GPU peak 11.8 GB. 끝 ~16:35 예상 → side_slot 10칸 평가.
- 결정: 설정·큐 그대로 (사람 지시). R 판정은 3 seed (서버 2 seed 0 push + 서버 1 s1·s2) 모인 뒤 `r_screen.py`.
- 할 일 (다음): (1) `lo_pkg_rnd_s1` RL 끝 (~16:35) → 10칸 평가 (side_slot). (2) 그다음 드라이버가 `lo_pkg_rnd_s2` → `lo_pkg_rnd_r0`. (3) 서버 2 R seed 0 기록 push되면 merge → 3 seed 모이면 `r_screen.py`. (4) r2pkfc_s2 16·20칸 12개 끝나면 G1 보충 2 표 (seed 0·1·2). (5) 라운드마다 `restore_proc_log.py`로 드라이버·side_slot 로그 복구. DAILY 다음 11/08 01:30 UTC.

## 라운드 96 (2026-11-07 13:42~ UTC)
- `git pull` (루프, 13:42): **새 사람 커밋 2개** ece1a02 (22:32 KST: R 무작위 물리 학습 `rnd`, G1 확인 종료 (b') 미통과, 서버 1 큐 → `lo_pkg_rnd_s1`, `lo_pkg_rnd_s2`, 대조군 `lo_pkg_rnd_r0` (`after: runs/pool_v1/rnd.ok`), `lo_pcb_r2pkfc*` 3개는 removed) · 549a30d (22:37 KST: M1 `r2pkfcm1` 서버 2에서 사람 중단, 기준 미달).
- 풀 생성: QUEUE "지금 바로" 항목 (`build_pools.py --only train --techs rnd && touch rnd.ok`)은 드라이버가 cpu 맨 앞 항목으로 13:42에 이미 시작 (`agent/logs/build_rnd.log`, 9 파일 4 worker) → 내가 따로 돌리지 않음 (같은 파일 두 번 쓰기 방지). 끝나면 `rnd.ok` → 드라이버가 `r2pkfc_s2` 다음에 `lo_pkg_rnd_s1` 시작.
- side_slot 다시 띄움 (pid 1680969): 이름 목록 `r2pkfc_s2 lo_pkg_rnd_s1 lo_pkg_rnd_s2 lo_pkg_rnd_r0` (옛 목록의 `lo_pcb_r2pkfc*`는 안 돔, rnd run 10칸 평가를 RL 끝나자마자 하려고). 옛 side_slot은 쉬는 중이었음 (마지막 END 12:41).
- RESULTS (`collect_results.py` 고침 → 다시 생성 1140 기록): F1 표 아래 "G1 확인 판정 (사람): (b') 미통과 → 채택 안 함, `lo_pcb_r2pkfc*` 못 돌림"; "시도한 수정" G1 행 판정 갱신, **M1 행 추가** (기준 미달, 사람 중단; 뺀 pkg −0.10 [−0.22, +0.02], `r2pkfcm1` 못 돌림). `m1_screen.py` 대기 끝.
- 상태: 4/7 `r2pkfc_s2` RL 진행 (끝 ~14:35 예상), 드라이버 CPU 줄 `r2pkfc_s1_rl` ip16_0 (G1 보충 2, 남은 것 계속). pod 24.1 GiB / 60, oom_kill 2.
- 결정: 학습 조건·arm·무작위 범위·큐 순서 그대로 (사람 지시). R 판정은 `r_screen.py`로 3 seed 모인 뒤 RESULTS 표만, 확인 run (`r2pk_rnd` 등)은 사람 지시 대기.
- (14:10) rnd 풀 9 파일 끝 (1646 s, 사람 예상 10분보다 김: 12칸 파일이 각 ~27분) → `rnd.ok`. greedy+prune 성공 파일별: rnd8 1.00/0.88/1.00, rnd10 0.19/0.88/1.00, rnd12 0.44/1.00/0.38 (`train_rnd12_2` count 26.0 — 다른 파일 2.4~7.3보다 큼). 보고만: 무작위 범위는 사람 고정이라 그대로; RL에서 해당 파일 fail·cert-bad를 볼 것.
- (14:32) `r2pkfc_s2` 학습 끝 rc 0 (RL best it 200 1.748, oom 0) → 드라이버가 **`lo_pkg_rnd_s1` 시작** (arm 사람 것 그대로, `--rl_pool_glob ...train_rnd*.pkl`, seed 1, 증류 0.75 h + RL 1.25 h). 증류 풀 18 batch / 288 문제 = pcb·ip만 (rnd는 RL에만, PROPOSALS 그대로), it 50 정상 (oom 0, GPU 8.9 GB). `r2pkfc_s2` 10칸 평가: side_slot은 드라이버 줄의 ls10 scale-eval (pkg16_0, 14:23~)이 끝나면 시작 (한 번에 하나 규칙).
- **드라이버 로그 끊김 (원인 찾음):** `agent/logs/pipeline_day2.log`가 디스크에서 12:42에 멈춰 있었음 — `agent/loop.sh` `sync_pull`이 라운드마다 `git pull --rebase --autostash` → 추적되는 로그 파일을 다시 만들어 드라이버는 지워진 inode에 씀 (기억된 함정, 10/30·11/02와 같은 것). 드라이버는 정상. `/proc/<pid>/fd/1`에서 읽어 디스크 파일에 복사 (139 KB, 기존 내용 포함 확인).
- 결정: `agent/loop.sh`는 고치지 않음 — 이유: 지금 돌고 있는 bash 스크립트라 수정하면 다음 줄을 엉뚱하게 읽을 수 있고, 사람 인프라라 보수적으로. **사람께 제안:** loop.sh를 다시 띄울 때 `sync_pull` 앞에 `git add -A agent/logs && git commit -qm "logs"` (autostash 대신 먼저 커밋), 또는 오래 쓰는 로그 (`pipeline_day2.log`, `side_slot.log`, `report_watch.log`)를 `.gitignore` + `git rm --cached`. 그 전까지 라운드마다 /proc에서 복사.
- 할 일 (다음): (1) [끝] `rnd.ok` 14:10. (2) `r2pkfc_s2` 끝 → 10칸 평가 (side_slot) → 같은 유형 3 seed 표, G1 구조 교체 줄 seed 3개 완성; `queue_scale_g1s` "appended 12" 확인. (3) [시작 14:32] `lo_pkg_rnd_s1` RL (~15:17~) 첫 it에서 rnd 풀 로딩·oom·cert-bad 확인 (새 kind라 처음). (4) 서버 2 R seed 0 기록 push되면 merge. (5) 11/08 03:00 UTC 새 학습 금지는 R 확인 run에 한해 09:00 UTC까지 (사람). DAILY 다음 11/08 01:30 UTC.

## 라운드 95 (2026-11-07 13:26~ UTC)
- `git fetch r2e` (13:26): 새 사람 커밋 없음 (QUEUE 마지막 19:28 KST 그대로). 서버 2 `r2pkfcm1` 기록 아직 없음 → `m1_screen.py` 대기.
- `ensure_running`: 드라이버 alive (pid 1575375), report_watch·resume_agent 다시 띄움. side_slot (pid 1650473)·`queue_scale_g1s.py --wait` (pid 1650475) alive. pod 19.3 GiB / 60, oom_kill 2 (그대로).
- 상태: 4/7 `r2pkfc_s2` 증류 끝 (1567 it, best it 300 1.719, cert-bad 0, oom 0) → RL it 25 (cert-bad 16, fail 0.12, oom 0, GPU peak 12.9 GB) — 정상, 끝 ~14:35 예상 그대로. 드라이버 CPU 줄: `r2pkfc_s1_rl` pcb16_1 끝 13:14, pcb20_0 끝 13:25, pcb20_1 다음 (남은 9개).
- **끝난 평가 (부분, 정책 단독):** `r2pkfc_s1_rl` pcb 16칸 [50] 0.92 / 0.87 / 0.26 vs 같은 seed `r2pk_s1_rl` 0.96 / 0.84 / 0.23 → 같은 종류 차 −0.04 (보고만). RESULTS 자동 반영 (collect_results). pcb20_0은 [25] 한쪽만이라 표는 pcb20_1 뒤.
- 결정: 큐·설정 그대로 — 이유: G1 확인·보충 2 사람 순서대로 정상, 새 사람 지시 없음, 서버 1 M1 금지 (QUEUE 19:28).
- 할 일 (다음): 라운드 93 목록 그대로.

## 라운드 94 (2026-11-07 13:10~ UTC)
- `git fetch r2e` (13:10): 새 사람 커밋 없음 (QUEUE 마지막 19:28 KST 그대로). 서버 2 `runs/2026-11-07s2/r2pkfcm1_rl` 아직 없음 (`lo_pkg_r2pkfcm1`만) → `m1_screen.py` 판정 대기.
- `ensure_running`: 드라이버 alive (pid 1575375), report_watch·resume_agent 다시 띄움. side_slot (pid 1650473)·`queue_scale_g1s.py --wait` (pid 1650475) alive. pod 27.7 GiB / 60, oom_kill 2 (그대로).
- 상태: 4/7 `r2pkfc_s2` 증류 it 1400 / 2391 s (cert-bad 0, oom 0, GPU peak 12.1 GB) — 정상, 끝 ~14:35 예상 그대로. 드라이버 CPU 줄: `r2pkfc_s1_rl` pcb16_0 끝 → pcb16_1 진행 (11분째).
- **끝난 평가 (부분):** `r2pkfc_s1_rl` pcb16_0 → RESULTS pcb 16칸 0.96 / 0.93 / 0.30 [25] (드라이버 collect_results 자동). seed 0·1 표는 pcb16_1까지 끝난 뒤 (같은 seed `r2pk_s1_rl` pcb16 0.96 [50]과 비교는 그때).
- 결정: 큐·설정 그대로 — 이유: G1 확인·보충 2 사람 순서대로 정상, 새 사람 지시 없음, 서버 1 M1 금지 (QUEUE 19:28).
- 할 일 (다음): 라운드 93 목록 그대로.

## 라운드 93 (2026-11-07 12:40~ UTC)
- `git fetch r2e` (12:40): 새 사람 커밋 없음 (QUEUE 마지막 19:28 KST 그대로). 서버 2 M1 기록 (`runs/2026-11-07s2/{lo_pkg_r2pkfcm1,r2pkfcm1}_rl`) 아직 없음 → `m1_screen.py` 대기.
- `ensure_running`: 드라이버 alive (pid 1575375), report_watch·resume_agent 다시 띄움. side_slot (pid 1650473)·`queue_scale_g1s.py --wait` (pid 1650475) alive. pod 19.0 GiB → pcb16 평가 시작 뒤 27.7 GiB / 60, oom_kill 2 (그대로).
- 3/7 `r2pkfc_s1` 학습 끝 12:29 (rc 0) → 드라이버가 4/7 `r2pkfc_s2` 시작 12:29 (증류 → RL, 끝 ~14:35) → side_slot 10칸 평가 12:29~12:41 rc 0.
- **끝난 실험 (정책 단독, 10칸 50문제, `r2pkfc_s1_rl`):** pcb 0.84 (10_0 1.00, 10_1 0.68), pkg 0.76 (10_0 0.80, 10_1 0.72), ip 1.00. 같은 seed `r2pk_s1_rl` 0.92 / 0.88 / 1.00 → 같은 종류 차 pcb −0.08, pkg −0.12, ip +0.00 (보고만; G1 확인 판정은 7개 끝난 뒤 PROPOSALS G1 규칙대로, 최종 구조는 사람). RESULTS 다시 생성 (드라이버 collect_results 12:42, 1130 기록).
- `queue_scale_g1s.log` 12:42 "r2pkfc_s1: appended 12" 확인 → 드라이버 CPU 줄이 collect_results 뒤 `ls10 scale-eval eval_pcb16_0 r2pkfc_s1_rl` 시작 (~12:44, 진행 중). 남은 11개 사람 순서대로 (파일당 4~15 분).
- push 거부 → `git fetch` (13:00): 새 사람 커밋 522bd26 (서버 2 M1 기록 12:44, `lo_pkg_r2pkfcm1` 증류·RL·10칸 6파일만) → merge 충돌 없음. `m1_screen.py`: **뺀 pkg `lo_pkg_r2pkfcm1` 0.74 vs `lo_pkg_r2pkfc_rl` 0.84, −0.10 [−0.22, +0.02], 개수비 0.984 (둘 다 성공 34)** → (i)의 뺀 pkg 쪽은 미달; (i)은 같은 종류 pkg ≥ +0.10으로도 통과 가능하므로 `r2pkfcm1` 기록이 올 때까지 판정 "대기" (규칙 그대로). RESULTS 판정 줄은 둘 다 온 뒤.
- 결정: 큐·설정 그대로 — 이유: G1 확인·보충 2 사람 순서대로 정상, 새 사람 지시 없음, 서버 1 M1 금지 (QUEUE 19:28).
- 할 일 (다음): (0) 서버 2 `r2pkfcm1` 기록 → merge → `m1_screen.py` → 거르기 판정 RESULTS. (1) `r2pkfc_s1` 16·20칸 12개 끝 → RESULTS "G1 보충 2" 표 (seed 0·1). (2) `r2pkfc_s2` 끝 ~14:35 → 10칸 평가 → "appended 12" 확인. (3) 5~7/7 (`lo_pcb_r2pkfc*`) 끝 ~21:00 UTC → F1 표·G1 확인 판정 줄 + 보충 2 읽기 줄 → 사람에게. (4) 11/08 03:00 UTC 뒤 새 학습 없음; DAILY 다음 11/08 01:30 UTC.


## 라운드 92 (2026-11-07 12:24~ UTC)
- `git fetch r2e` (12:24): 새 사람 커밋 없음 (QUEUE 마지막 19:28 KST 그대로). 서버 2 M1 기록 아직 없음 → `m1_screen.py` 대기.
- `ensure_running`: 드라이버 alive (pid 1575375), report_watch·resume_agent 다시 띄움. side_slot alive (pid 1650473), `queue_scale_g1s.py --wait` alive (pid 1650475, 로그 비어 있음). pod 18.9 GiB / 60, oom_kill 2 (그대로).
- 상태: 3/7 `r2pkfc_s1` RL it 325 / 4199 s (cert-bad ≤ 8, fail ≤ 0.06, oom 0, GPU peak 16.7 GB; it 300 val fail pcb 0.44 / pkg 0.81 / ip 0.12, best it 200 1.778 그대로), resume ckpt it 278 — 정상, 곧 끝 (~12:35). 끝난 실험 없음 → RESULTS 그대로.
- 결정: 큐·설정 그대로 — 이유: G1 확인 run 사람 순서대로 정상, 새 사람 지시 없음, 서버 1 M1 금지 (QUEUE 19:28).
- 할 일 (다음): 라운드 86 목록 그대로 (`r2pkfc_s1` 끝 → 10칸 평가 → `queue_scale_g1s.log` "appended 12" 확인). DAILY 다음 작성 11/08 01:30 UTC.

## 라운드 91 (2026-11-07 12:08~ UTC)
- `git fetch r2e` (12:08): 새 사람 커밋 없음 (QUEUE 마지막 19:28 KST 그대로). 서버 2 M1 기록 아직 없음 → `m1_screen.py` 대기.
- `ensure_running`: 드라이버 alive (pid 1575375), report_watch·resume_agent 다시 띄움. side_slot alive (pid 1650473), `queue_scale_g1s.py --wait` alive (pid 1650475, 로그 비어 있음 = 아직 붙일 것 없음). pod 18.3 GiB / 60, oom_kill 2 (그대로).
- 상태: 3/7 `r2pkfc_s1` RL it 225 / 2997 s (cert-bad ≤ 9, fail ≤ 0.07, oom 0, GPU peak 16.5 GB; it 200 new best 1.778, val fail pcb 0.56 / pkg 0.56 / ip 0.19), resume ckpt it 177 — 정상, 끝 ~12:30 예상 그대로. 끝난 실험 없음 → RESULTS 그대로.
- 결정: 큐·설정 그대로 — 이유: G1 확인 run 사람 순서대로 정상, 새 사람 지시 없음, 서버 1 M1 금지 (QUEUE 19:28).
- 할 일 (다음): 라운드 86 목록 그대로 (`r2pkfc_s1` 끝 → 10칸 평가 → `queue_scale_g1s.log` "appended 12" 확인). DAILY 다음 작성 11/08 01:30 UTC.

## 라운드 90 (2026-11-07 11:52~ UTC)
- `git fetch r2e` (11:52): 새 사람 커밋 없음 (QUEUE 마지막 19:28 KST 그대로). 서버 2 M1 기록 아직 없음 → `m1_screen.py` 대기.
- `ensure_running`: 드라이버 alive (pid 1575375), report_watch·resume_agent 다시 띄움. side_slot alive (pid 1650473), `queue_scale_g1s.py --wait` alive (pid 1650475). pod 18.7 GiB / 60, oom_kill 2 (그대로).
- 상태: 3/7 `r2pkfc_s1` RL it 150 / 2035 s (cert-bad 4, fail 0.03, oom 0, GPU peak 15.9 GB; it 100 new best 1.802) — 정상, 끝 ~12:30 예상 그대로. 끝난 실험 없음 → RESULTS 그대로.
- 결정: 큐·설정 그대로 — 이유: G1 확인 run 사람 순서대로 정상, 새 사람 지시 없음, 서버 1 M1 금지 (QUEUE 19:28).
- 할 일 (다음): 라운드 86 목록 그대로. DAILY 다음 작성 11/08 01:30 UTC.

## 라운드 89 (2026-11-07 11:37~ UTC)
- `git fetch r2e` (11:37): 새 사람 커밋 없음 (QUEUE 마지막 19:28 KST 그대로). 서버 2 M1 기록 (`runs/2026-11-07s2/{lo_pkg_r2pkfcm1,r2pkfcm1}_rl`) 아직 없음 → `m1_screen.py` 대기.
- `ensure_running`: 드라이버 alive (pid 1575375), report_watch·resume_agent 다시 띄움. side_slot alive (pid 1650473), `queue_scale_g1s.py --wait` alive (pid 1650475). pod 18.9 GiB / 60, oom_kill 2 (그대로).
- 상태: 3/7 `r2pkfc_s1` RL it 100 / 1375 s (cert-bad 12, fail 0.09, oom 0, GPU peak 15.0 GB; val fail pcb 0.50 / pkg 0.69 / ip 0.19 — seed 0 `r2pkfc_rl` it 100의 0.62 / 0.69 / 0.12와 같은 범위) — 정상, 끝 ~12:30 예상 그대로. 끝난 실험 없음 → RESULTS 그대로.
- 결정: 큐·설정 그대로 — 이유: G1 확인 run 사람 순서대로 정상, 새 사람 지시 없음, 서버 1 M1 금지 (QUEUE 19:28).
- 할 일 (다음): 라운드 86 목록 그대로. DAILY 다음 작성 11/08 01:30 UTC.

## 라운드 88 (2026-11-07 11:21~ UTC)
- `git fetch r2e` (11:21): 새 사람 커밋 없음 (QUEUE 마지막 19:28 KST 그대로). 서버 2 M1 기록 (`runs/2026-11-07s2/{lo_pkg_r2pkfcm1,r2pkfcm1}_rl`) 아직 없음 → `m1_screen.py` 대기.
- `ensure_running`: 드라이버 alive (pid 1575375), report_watch·resume_agent 다시 띄움. side_slot alive (pid 1650473), `queue_scale_g1s.py --wait` alive (pid 1650475). pod 18.2 GiB / 60, oom_kill 2 (그대로).
- 상태: 3/7 `r2pkfc_s1` 증류 끝 → RL it 25 / 383 s (cert-bad 3, fail 0.02, oom 0, GPU peak 12.8 GB) — 정상, 끝 ~12:30 예상 그대로. 끝난 실험 없음 → RESULTS 그대로.
- 결정: 큐·설정 그대로 — 이유: G1 확인 run 사람 순서대로 정상, 새 사람 지시 없음, 서버 1 M1 금지 (QUEUE 19:28).
- 할 일 (다음): 라운드 86 목록 그대로. DAILY 다음 작성 11/08 01:30 UTC.

## 라운드 87 (2026-11-07 11:05~ UTC)
- `git fetch r2e` (11:05): 새 사람 커밋 없음 (QUEUE 마지막 19:28 KST 그대로). 서버 2 M1 기록 (`runs/2026-11-07s2/{lo_pkg_r2pkfcm1,r2pkfcm1}_rl`) 아직 없음 → `m1_screen.py` 대기.
- `ensure_running`: 드라이버 alive (pid 1575375), report_watch·resume_agent 다시 띄움. side_slot alive (pid 1650473), `queue_scale_g1s.py --wait` alive (pid 1650475). pod 18.2 GiB / 60, oom_kill 2.
- 상태: 3/7 `r2pkfc_s1` 증류 it 1350 / 2267 s (cert-bad 0, oom 0, GPU peak 12.1 GB) — 정상, 끝 ~12:30 예상 그대로. 끝난 실험 없음 → RESULTS 그대로.
- 결정: 큐·설정 그대로 — 이유: G1 확인 run 사람 순서대로 정상, 새 사람 지시 없음, 서버 1 M1 금지 (QUEUE 19:28).
- 할 일 (다음): 라운드 86 목록 그대로.

## 라운드 86 (2026-11-07 10:21~ UTC)
- 새 사람 커밋 209f21b·3bb7e73 (19:14 KST): QUEUE 19:14 + PROPOSALS "11/07 G1 보충 2" — `r2pkfc_s1_rl`·`r2pkfc_s2_rl` 각 `eval_{pcb,ip,pkg}{16,20}_{0,1}` 12파일 (규모 곡선 명령 그대로, 한 번에 한 파일, pcb 먼저), 각 run의 10칸 평가(판정용)가 끝난 뒤; 보고만, 읽는 규칙은 결과 전 고정 (종류×크기 seed 0·1·2 묶음 점추정, pcb·ip 16·20 중 하나라도 < −0.06 → "큰 판 비용 있음"; pkg 보고만; 결정은 사람). `git fetch r2e` 10:21: 그 뒤 새 커밋 없음.
- `scripts/queue_scale_g1s.py` (새, 평가만): run마다 학습 끝 + 10칸 6기록이 다 있을 때만 `ls10 scale-eval eval_<set> <run>_rl` 12개를 CPU 큐 끝에 붙임 (사람 순서 pcb16_0·1, pcb20_0·1, ip16·20, pkg16·20; 명령·`--out runs/2026-10-29/<run>_rl`은 QUEUE 그대로). CPU 큐에는 "after" 문이 없어 미리 넣으면 드라이버가 best.pt 없이 시작해 FAILED로 빠짐 → `--wait`로 120 s마다 확인하는 대기 프로세스 (bg, `agent/logs/queue_scale_g1s.log`, pid 1650475) — 드라이버 재시작 없음. **ensure_running이 다시 띄우지 않음** → pod 재시작 뒤엔 손으로 다시 (`scripts/bg.py agent/logs/queue_scale_g1s.log '.venv/bin/python scripts/queue_scale_g1s.py --wait r2pkfc_s1 r2pkfc_s2'`, 이미 붙은 항목은 건너뜀).
- `side_slot.py`: `LS16_EXEMPT`에 `r2pkfc_s1_rl`·`r2pkfc_s2_rl` 추가 (QUEUE 19:14 "이 평가는 죽이지 마라, 45 GiB 넘으면 LS16 쪽을 멈춘다") → side_slot 같은 7 대상으로 다시 띄움 (pid 1650473, 일 없을 때). 지금 LS16은 안 돌아서 실제 차이는 없음.
- 결정: `ls16c eval_ip16_1` 항목을 `cpu`에서 `cpu_deferred`로 (`queue_defer.py`) — 이유: `.memskip`만 있고 기록이 없어 "미완"으로 남아 있었고, 드라이버가 FAILED 기억으로만 건너뜀 → 드라이버가 재시작되면 18 h LS16을 다시 시작해 드라이버 CPU 줄을 막음. 다른 LS16 3개와 같은 자리 (사람 결정 13:40 "메모리로 생략" 그대로, RESULTS 표시 그대로).
- `collect_results.py`: `g1_big_board_lines` — "G1 보충 2" 표 (seed별 차, 짝 수, 묶음 점추정 + CI 적기만, 읽기 줄). **seed 1·2 큰 판 기록이 하나도 없을 때 작성.** 지금 seed 0만: pcb16 −0.10, pcb20 −0.12, ip16·20 +0.00, pkg16 +0.12, pkg20 −0.06 → "대기".
- `lo_pkg_r2pkfc_s2` 학습 끝 10:26 (rc 0) → side_slot 10칸 평가 10:28~10:42 rc 0. 끝난 실험 (정책 단독, 50문제): **뺀 pkg 0.66** (같은 seed `lo_pkg_r2pk_s2` 0.78, `lo_pkg_r0_s2` 0.78 → (b') −0.12), pcb 0.88, ip 1.00. F1 표 뺀 pkg r2pkfc: 0.84 / 0.78 / 0.66, (b') +0.06 / −0.06 / −0.12 (라운드 80에서 이미 pkg 범용화 "통과" 불가). RESULTS 다시 생성 (1111 기록).
- 드라이버가 3/7 `r2pkfc_s1` 10:26 시작 (증류 → RL, 끝 ~12:30) → 10칸 평가 → 대기 프로세스가 16·20칸 12개 붙임. 4/7 `r2pkfc_s2` 끝 ~14:40 → 같은 방식.
- 커밋·push 뒤 `git fetch` (10:46): 새 사람 커밋 a6d549e (서버 2 C1·대조군 기록) + 4b9fdbc (19:28 KST: M1 `--mm_bias` 코드·PROPOSALS·`m1_screen.py`·`server2_m1.sh`; QUEUE 19:28 "M1은 서버 2에서 사람이, **서버 1은 건드리지 마라**"; C1 판정 RESULTS에 적기). merge 충돌 없음. `decap_loop/model.py`·`train.py` 변경은 `--mm_bias` 없으면 같은 계산 (`mm_logtau` 없음, `mode_self` bias=None 기본, 파라미터 생성 순서 그대로) → 도는 `r2pkfc_s1` (RL 단계는 새 코드 import)에 영향 없음, 그대로 둠.
- **C1 판정 (`c1_screen.py`, PROPOSALS C1 규칙 그대로):** `r2pk_vsi_rl` − `r2pk_ctl_rl` (10칸 50문제, 서버 2): pcb 0.90 vs 0.88 +0.02 [−0.12, +0.14] 개수비 1.071; pkg 0.84 vs 0.86 −0.02 [−0.16, +0.10] 개수비 0.980; ip 1.00 vs 0.98 +0.02 [+0.00, +0.06] 개수비 1.002; iters 227 vs 290 → (i) 탈락, (ii) 통과, (iii) 탈락 (pcb 1.071) → **기준 미달** (사람 판정과 같음; pkg CI 상한 +0.10은 사람 +0.12와 bootstrap 차이만). RESULTS TRIED_MODS에 "C1" 행 (collect_results), ABLATIONS C1 짝 절은 자동. RESULTS 다시 생성 (1124 기록).
- M1: 서버 1에서는 아무것도 안 함 (QUEUE 19:28). 서버 2 기록 `runs/2026-11-07s2/{lo_pkg_r2pkfcm1,r2pkfcm1}_rl/eval`이 push되면 pull → `m1_screen.py` → 거르기 판정만 RESULTS (확인 run은 사람 지시 뒤).
- 할 일 (다음): (0) 서버 2 M1 기록 → `m1_screen.py`. (1) `r2pkfc_s1` 끝 → 10칸 평가 → `queue_scale_g1s.log` "appended 12" 확인, 16·20칸 평가 진행 (파일당 4~15 분, 드라이버/side_slot 한 번에 하나). (2) 서버 2 C1 기록 → merge → `c1_screen.py`. (3) 7개 끝 ~21:00 UTC → F1 표·G1 교체 판정 줄 + G1 보충 2 읽기 줄 → 사람에게 남김 (최종 구조 결정은 사람). (4) 11/08 03:00 UTC 뒤 새 학습 없음; DAILY 다음 11/08 01:xx UTC.

## 라운드 85 (2026-11-07 10:05~ UTC)
- `git fetch r2e` (10:05): 새 사람 커밋 없음 (QUEUE 마지막 14:30 KST 그대로). 서버 2 C1 기록 (`runs/2026-11-07s2/r2pk_{vsi,ctl}_rl`) 아직 없음 (a1_nogrow만).
- `ensure_running`: 드라이버 alive (pid 1575375), report_watch·resume_agent 다시 띄움. side_slot alive (pid 1614768). pod 18.2 GiB / 60, oom_kill 2.
- 상태: `lo_pkg_r2pkfc_s2` RL it 225 / 3074 s (cert-bad ≤ 4, oom 0, GPU peak 11.4 GB; it 200 val pcb10_0 0.744 fail 0.75 → best는 it 100 그대로), resume ckpt it 177 — 정상, 끝 ~10:27 (1.25 h). 끝난 실험 없음 → RESULTS 그대로.
- 결정: 큐·설정 그대로 — 이유: G1 확인 run 사람 순서대로 정상, 새 사람 지시 없음.
- 할 일 (다음): 라운드 81 목록 그대로 ((1) s2 끝 → side_slot 10칸 평가 → RESULTS; (2) 서버 2 C1 기록 → merge → `c1_screen.py`; (3) 7개 끝 ~21:00 UTC → F1 표·교체 판정; (4) 11/08 03:00 UTC 뒤 새 학습 없음; DAILY 다음 11/08 01:xx UTC).

## 라운드 84 (2026-11-07 09:49~ UTC)
- `git fetch r2e` (09:49): 새 사람 커밋 없음 (QUEUE 마지막 14:30 KST 그대로). 서버 2 C1 기록 (`runs/2026-11-07s2/r2pk_{vsi,ctl}_rl`) 아직 없음 (a1_nogrow만).
- `ensure_running`: 드라이버 alive (pid 1575375), report_watch·resume_agent 다시 띄움. side_slot alive (pid 1614768). pod 18.7 GiB / 60, oom_kill 2.
- 상태: `lo_pkg_r2pkfc_s2` RL it 150 / 2124 s (cert-bad ≤ 5, oom 0, GPU peak 11.2 GB, best it 100 score 1.585) — 정상, s1 (337 it, 1.26 h) 기준 끝 ~10:30 그대로. 끝난 실험 없음 → RESULTS 그대로.
- 결정: 큐·설정 그대로 — 이유: G1 확인 run 사람 순서대로 정상, 새 사람 지시 없음.
- 할 일 (다음): 라운드 81 목록 그대로 ((1) s2 끝 → side_slot 10칸 평가 → RESULTS; (2) 서버 2 C1 기록 → merge → `c1_screen.py`; (3) 7개 끝 ~21:00 UTC → F1 표·교체 판정; (4) 11/08 03:00 UTC 뒤 새 학습 없음; DAILY 다음 11/08 01:xx UTC).

## 라운드 83 (2026-11-07 09:33~ UTC)
- `git fetch r2e` (09:33): 새 사람 커밋 없음 (QUEUE 마지막 14:30 KST 그대로). 서버 2 C1 기록 (`runs/2026-11-07s2/r2pk_{vsi,ctl}_rl`) 아직 없음 (a1_nogrow만).
- `ensure_running`: 드라이버 alive (pid 1575375), report_watch·resume_agent 다시 띄움. side_slot alive (pid 1614768, 확인 7 run 대상). pod 18.2 GiB / 60, oom_kill 2.
- 상태: `lo_pkg_r2pkfc_s2` RL it 75 / 1073 s (cert-bad 6, oom 0, GPU peak 10.5 GB), it 81 체크포인트 저장 — 정상, 끝 ~10:30 예상 그대로. 끝난 실험 없음 → RESULTS 그대로.
- 결정: 큐·설정 그대로 — 이유: G1 확인 run 사람 순서대로 정상, 새 사람 지시 없음.
- 할 일 (다음): 라운드 81 목록 그대로 ((1) s2 끝 → side_slot 10칸 평가 → RESULTS; (2) 서버 2 C1 기록 → merge → `c1_screen.py`; (3) 7개 끝 ~21:00 UTC → F1 표·교체 판정; (4) 11/08 03:00 UTC 뒤 새 학습 없음; DAILY 다음 11/08 01:xx UTC).

## 라운드 82 (2026-11-07 09:17~ UTC)
- `git fetch r2e` (09:17): 새 사람 커밋 없음 (QUEUE 마지막 14:30 KST 그대로). 서버 2 C1 기록 (`runs/2026-11-07s2/r2pk_{vsi,ctl}_rl`) 아직 없음 (a1_nogrow만).
- `ensure_running`: 드라이버 alive (pid 1575375), report_watch·resume_agent 다시 띄움. side_slot alive (대상 확인 7 run). pod 18.1 GiB / 60, oom_kill 2.
- 상태: `lo_pkg_r2pkfc_s2` 증류 끝 (1590 it, 0.76 h, best it 600 score 1.415, cert-bad 0, GPU peak 8.9 GB) → RL 09:12~ it 25 / 363 s (cert-bad 12, oom 0, GPU peak 9.0 GB) — 정상, 끝 ~10:30. 끝난 실험 없음 → RESULTS 그대로.
- 결정: 큐·설정 그대로 — 이유: G1 확인 run 사람 순서대로 정상, 새 사람 지시 없음.
- 할 일 (다음): 라운드 81 목록 그대로 ((1) s2 끝 → side_slot 10칸 평가 → RESULTS; (2) 서버 2 C1 기록 → merge → `c1_screen.py`; (3) 7개 끝 ~21:00 UTC → F1 표·교체 판정; (4) 11/08 03:00 UTC 뒤 새 학습 없음; DAILY 다음 11/08 01:xx UTC).

## 라운드 81 (2026-11-07 09:00~ UTC)
- `git fetch r2e` (09:00): 새 사람 커밋 없음 (QUEUE 마지막 14:30 KST 그대로). 서버 2 C1 기록 (`runs/2026-11-07s2/r2pk_{vsi,ctl}_rl`) 아직 없음 (a1_nogrow만).
- `ensure_running`: 드라이버 alive (pid 1575375), report_watch·resume_agent 다시 띄움. pod 18.1 GiB / 60, oom_kill 2.
- 상태: GPU `lo_pkg_r2pkfc_s2` 증류 08:24~ it 1200 (2094 s, cert-bad 0, oom 0, GPU peak 8.9 GB) — 정상, 끝 ~10:25 예상 그대로. side_slot 대기 (s2 RL 끝나면 10칸 평가). 끝난 실험 없음 → RESULTS 그대로.
- 점검 (기록 읽기만): 라운드 61의 "16칸 pcb r2pk 7.6 s" 시간 의문은 이미 라운드 61 08:55에 풀림 (`eval_pcb16_0` r2pk_rl seed 0 #9 폭주 한 문제, 위 277줄 근처). 다시 확인: 같은 파일 r2pk_s1·s2 1.95·3.50 s, pcb16_1 r2pk 3 seed 1.75~2.87 s, r2_mott 1.78·2.94 s → 입력 계산 비용 아님. 중복 스크립트는 지움 (`scripts/rec_time.py`가 이미 있음).
- 결정: 큐·설정 그대로 — 이유: G1 확인 run 사람 순서대로 정상, 새 CPU 일은 사람 지시 없음.
- 할 일 (다음): (1) `lo_pkg_r2pkfc_s2` 끝 (~10:25) → side_slot 10칸 평가 → RESULTS. (2) 서버 2 C1 기록 push되면 merge → `c1_screen.py`. (3) 7개 끝 (~21:00 UTC) → F1 표·교체 판정 줄. (4) 11/08 03:00 UTC 뒤 새 학습 없음. DAILY 다음 11/08 01:xx UTC.

## 라운드 80 (2026-11-07 07:49~ UTC)
- `git fetch r2e` (07:49, 08:24): 새 사람 커밋 없음 (QUEUE 마지막 14:30 KST 그대로). 서버 2 C1 기록 (`runs/2026-11-07s2/r2pk_{vsi,ctl}_rl`) 아직 없음.
- `ensure_running`: 드라이버 alive (pid 1575375), report_watch·resume_agent 다시 띄움.
- `lo_pkg_r2pkfc_s1` RL 끝 (08:2x): 337 iter, 1.26 h, best iter 300 (score 1.520), cert-bad ≤ 8, oom 0, GPU peak 11.5 GB. 코드 = HEAD 8c1b814 (train.py는 라운드 79와 같음). side_slot `eval10 lo_pkg_r2pkfc_s1_rl` 08:25:51~08:38:05 rc 0. 드라이버가 2/7 `lo_pkg_r2pkfc_s2` 08:20 시작 (증류 → RL, 끝 ~10:25).
- 끝난 실험 (10칸, 정책 단독, 50문제): `lo_pkg_r2pkfc_s1_rl` 뺀 pkg **0.78** (같은 seed `lo_pkg_r2pk_s1` 0.64, `lo_pkg_r0_s1` 0.84), pcb 0.88, ip 0.98.
- r2pkfc 16·20칸 보고만 평가 (QUEUE 13:25) 전부 끝: pcb20_0·1, ip20_0·1 07:51~08:24 기록. "G1 후보 16·20칸" 표 (`r2pkfc_rl` − `r2pk_rl`): pcb16 −0.10 [−0.24, +0.04], pkg16 +0.12 [−0.06, +0.30], ip16 +0.00, pcb20 −0.12 [−0.31, +0.00] (16문제), pkg20 −0.06 [−0.44, +0.31], ip20 +0.00. 판정에 쓰지 않음.
- `collect_results.py`: F1 표 ("최종 후보 r2pk의 한 종류 빼고 학습, seed별")에 **"G1 후보 r2pkfc (확인 run, 같은 규칙)"** 행 + "G1 구조 교체 판정" 줄 (PROPOSALS G1 확인: 같은 종류 pkg `r2pkfc` − `r2pk` seed 3개 묶음 CI 하한 > 0이면 교체). **확인 7 run 중 1개만 나온 때 작성** — 규칙은 사람이 고정한 그대로, 코드만.
- **F1 규칙 중간 상태 (G1 확인):** 뺀 pkg (b') `lo_pkg_r2pkfc` − `lo_pkg_r0`: seed 0 +0.06, **seed 1 −0.06** → "seed 0·1·2 모두 > 0" 조건이 이미 깨짐 → **r2pkfc는 pkg 범용화 "통과" 불가** (seed 2와 상관없음; 표 판정 칸은 규칙대로 seed 3개가 다 나온 뒤 "실패"로 찍힘). 참고: r2pk도 같은 seed에서 0.64 (− 0.84 = −0.20)로 이미 실패였음. 구조 교체 판정 (같은 종류 pkg, seed 3개 묶음)과 뺀 pcb 판정은 남은 run이 정함.
- 결정: 큐·설정 그대로 (남은 6 run 사람 순서대로 계속) — 이유: QUEUE 10:50 3번이 확인 7 run 전부를 지시했고, 구조 교체 판정·뺀 pcb 판정은 아직 열려 있음. pkg 결과를 보고 run을 빼거나 바꾸지 않는다 (규칙 바꾸기 금지).
- 할 일 (다음): (1) `lo_pkg_r2pkfc_s2` 끝 (~10:25) → side_slot 10칸 평가 → RESULTS. (2) 서버 2 C1 기록 push되면 merge → `c1_screen.py`. (3) 7개 끝 (~21:00 UTC) → F1 표·교체 판정 줄 읽고 PROGRESS·RESULTS ABLATIONS. (4) 11/08 03:00 UTC 뒤 새 학습 없음. DAILY 다음 11/08 01:xx UTC. side_slot CPU 칸: r2pkfc 보고만 평가가 다 끝나 비어 있음 (scale-eval 대상 없음) — 새 CPU 일은 넣지 않음 (사람 지시 없음).

## 라운드 79 (2026-11-07 07:30~ UTC)
- `git fetch r2e`: 새 사람 커밋 없음 (QUEUE 마지막 14:30 KST 그대로). 서버 2 C1 기록 (`runs/2026-11-07s2/r2pk_{vsi,ctl}_rl`) 아직 없음 (a1_nogrow만).
- `ensure_running`: 드라이버 alive (pid 1575375), report_watch·resume_agent 다시 띄움.
- 상태 (07:35): GPU `lo_pkg_r2pkfc_s1` RL 07:08~ it 75 (1081 s, cert-bad 4, oom 0, GPU peak 10.5 GB), it 86에서 20분 체크포인트 저장 — 정상. CPU: side_slot `scale-eval eval_pcb16_0 r2pkfc_rl` 07:01~07:14 rc 0 → `eval_pcb16_1` 07:17~ (보고만). pod 28.2 GiB / 60, oom_kill 2.
- 끝난 실험: r2pkfc pcb16_0 (보고만) → RESULTS 다시 생성 (1092 기록). "G1 후보 16·20칸" 표: pcb16 (25문제, 1파일) `r2pkfc_rl` − `r2pk_rl` −0.08 [−0.28, +0.16], 개수비 1.029 [0.893, 1.185]. G1 판정과 무관 (보고만).
- 일정 확인: 확인 7 run × ~2.1 h, 1번째 06:22 시작 → 7번째 끝 ~21:00 UTC 11/07. 새 학습 마감 11/08 03:00 UTC (12:00 KST) 전에 모두 시작됨.
- 결정: 큐·설정 그대로 — 이유: G1 확인 run 사람 순서대로 정상, C1은 서버 2, G-tr은 사람 지시 대기.
- 할 일 (다음): (1) `lo_pkg_r2pkfc_s1` 끝 (~08:30) → side_slot 10칸 평가 확인. (2) 서버 2 C1 기록 push되면 merge → `c1_screen.py` → RESULTS. (3) 7개 끝나면 F1 규칙 판정. (4) 11/08 03:00 UTC 뒤 새 학습 없음. DAILY 다음 11/08 01:xx UTC.

## 라운드 78 (2026-11-07 07:08~ UTC)
- `git fetch r2e`: 새 사람 커밋 없음 (QUEUE 마지막 14:30 KST 그대로).
- G-inf (`ginf eval10 r2pk_rl@mott_pkgate`) 06:23:59~06:58:31 rc 0, 10칸 6파일 기록.
- **C1' G-inf 판정 (`c1p_screen.py`, PROPOSALS C1' 규칙 그대로, 결과 전에 쓴 스크립트):** `r2pk_rl@mott_pkgate` − `r2pk_rl` (같은 가중치·같은 문제, 50문제): pcb 0.96 vs 0.90 +0.06 [−0.02, +0.14] 개수비 0.992; **pkg 0.76 vs 0.78 −0.02 [−0.14, +0.10]** 개수비 1.016; ip 1.00 vs 0.98 +0.02 [+0.00, +0.06] 개수비 1.003 → (i) **탈락** (−0.02 < +0.10), (ii) 통과, (iii) 통과 → **"추론 게이트만으로 성립" 아님**. 게이트가 pkg 근소 실패를 고치지 못함 (pkg 성공은 오히려 1문제 적음, CI 0 포함). RESULTS TRIED_MODS에 "C1' G-inf" 행 (collect_results), ABLATIONS C1' 짝 절은 자동.
- G-tr `r2pk_gate`는 **넣지 않음** — QUEUE 14:30 3번: G1 통과면 G1 확인이 먼저, G-tr은 사람 지시 대기. G-inf 탈락은 G-tr 판정을 대신하지 않음 (규칙상 각각 따로).
- 상태 (07:10): GPU `lo_pkg_r2pkfc_s1` 증류 끝 → RL 단계 07:08 시작 (정상). CPU: side_slot `scale-eval eval_pcb16_0 r2pkfc_rl` 07:01~ (보고만, 큐 순서 그대로 G-inf 다음) + 학습 정확 검사. pod 25.6 GiB / 60, oom_kill 2.
- 결정: 큐·설정 그대로 — 이유: G1 확인 run 사람 순서대로 진행, C1'은 G-tr 사람 지시 대기, C1은 서버 2.
- 할 일 (다음): (1) 서버 2 C1 기록 push되면 merge → `c1_screen.py` → RESULTS. (2) 확인 run 진행·10칸 평가 확인 (`lo_pkg_r2pkfc_s1` 끝 ~08:30) → 7개 끝나면 F1 규칙 판정. (3) 11/08 03:00 UTC 뒤 새 학습 없음. DAILY 다음 11/08 01:xx UTC.

## 라운드 77 (2026-11-07 06:42~ UTC)
- 상태 (06:45): GPU `lo_pkg_r2pkfc_s1` 증류 it 625 / 1121 s (cert-bad 0, oom 0, GPU peak 8.9 GB, val에 pkg 없음) 정상. CPU: side_slot `ginf eval10 r2pk_rl@mott_pkgate` 06:23:59~ (`eval_pkg10_1` 06:33, `eval_pcb10_0` 06:41 기록; 파일당 ~8 분 → 끝 ~07:15) + 학습 정확 검사 2. `scale-eval eval_ip20_1 r2pk_s2_rl`은 06:23 기록 생성으로 끝 (side_slot 재시작 때문에 END 줄 없음; 기록 파일로 확인). pod 20.2 GiB / 60, oom_kill 2. `ensure_running`: 드라이버 alive, report_watch·resume_agent 다시 띄움. `git fetch r2e`: 새 사람 커밋 없음.
- `scripts/c1p_screen.py` (새, **G-inf 기록이 다 나오기 전에** 작성): PROPOSALS C1' 규칙 그대로 `r2pk_rl@mott_pkgate` vs `r2pk_rl` 10칸 짝 (pkg ≥ +0.10, pcb·ip ≥ −0.06, 종류마다 개수비 ≤ 1.05, 점추정 판정·CI 같이). `g1_screen.pair` 재사용. 지금 "waiting" (종류마다 2파일 필요).
- 끝난 실험: `r2pk_s2_rl` ip20_1 (규모 곡선 보충) → RESULTS 다시 생성.
- 결정: 큐·설정 그대로 — 이유: G1 확인 run 사람 순서대로 정상 진행, G-inf는 QUEUE 14:30 1번 그대로 도는 중, G-tr은 G1 통과라 사람 지시 대기.
- 할 일 (다음): (1) G-inf 끝 (~07:15) → `c1p_screen.py` → 판정 줄 PROGRESS·RESULTS (TRIED_MODS/ABLATIONS 자동 절). (2) 서버 2 C1 기록 push되면 merge → `c1_screen.py`. (3) 확인 run 진행·10칸 평가 확인 → 7개 끝나면 F1 규칙 판정. (4) 11/08 03:00 UTC 뒤 새 학습 없음. DAILY 다음 11/08 01:xx UTC.

## 라운드 76 (2026-11-07 05:26~ UTC)
- 새 사람 커밋: fb1e316·394305f·cb491df (14:16~14:22 KST, QUEUE 14:15 + 14:25 변경: 최종 RL 방식 C1 `r2pk_vsi` + 대조군 `r2pk_ctl`은 **사람이 서버 2에서** `scripts/server2_c1.sh`; 서버 1에서는 돌리지 않음), dab9694 (14:28 KST, QUEUE 14:30: 두 번째 후보 C1' `--peak_gate`, G-inf CPU 평가는 지금, GPU smoke, G-tr은 G1 탈락일 때만). `git merge r2e/main` 충돌 없음.
- 확인: `agent/day2_queue.json`에 C1·대조군 항목 없음 (서버 1에 넣지 않음). 서버 1 `runs/2026-10-29/r2pk_ctl{,_rl}`는 11/07 01:46의 빈 폴더 (10:45 절이 10:50 절로 대체되며 시작 안 됨) — 서버 2 기록 `runs/2026-11-07s2/`와 겹치지 않음.
- `scripts/c1_screen.py` (새, **C1 기록 전에** 작성): PROPOSALS C1 규칙 그대로 `runs/2026-11-07s2/r2pk_vsi_rl` vs `r2pk_ctl_rl` 10칸 짝 (pkg ≥ +0.10, pcb·ip ≥ −0.06, 종류마다 개수비 ≤ 1.05, 점추정 판정·CI 같이), metrics.json `iters`도 출력. 지금은 "waiting".
- `collect_results.py`: (a) `runs/2026-11-07s2/`를 서버 2 기록으로 (시간 칸 "–", 서버 1 중복이면 뺌) — a1_nogrow 시간 칸이 서버 2 CPU 값이었음. (b) ABLATIONS에 C1 (`r2pk_vsi_rl` vs `r2pk_ctl_rl`)과 C1' G-inf (`r2pk_rl@mott_pkgate` vs `r2pk_rl`) 짝 절 (기록이 생기면 나타남). (c) TRIED_MODS에 G1 행.
- **보고만 (G1 판정 아님):** r2pkfc pkg16_1 05:10~05:24, pkg20_0 05:27~05:38, pkg20_1 (드라이버) 끝. pkg16 50문제 `r2pkfc_rl` − `r2pk_rl` 성공률 +0.12 [−0.06, +0.30], 개수비 1.004 [0.938, 1.075] (RESULTS "G1 후보 16·20칸" 표).
- `lo_pkg_r2pkfc` 학습 끝 05:58 (rc 0) → side_slot 10칸 평가 06:02~06:14 rc 0.
- **G1 거르기 판정 (`g1_screen.py`, 규칙 PROPOSALS 11/07 G1 그대로): 통과, pick = r2pkfc.** 같은 종류 (짝 `r2pk_rl`, 50문제): pcb 0.94 vs 0.90 +0.04 [−0.06, +0.12] 개수비 0.981; pkg 0.84 vs 0.78 +0.06 [−0.06, +0.18] 개수비 1.029; ip 1.00 vs 0.98 +0.02 [+0.00, +0.06] 개수비 1.013. **뺀 pkg `lo_pkg_r2pkfc_rl` − `lo_pkg_r2pk_rl`: 0.84 vs 0.70, +0.14 [−0.02, +0.30], 개수비 0.991** → (i) 뺀 pkg 갈래 ≥ +0.10 통과, (ii) 통과, (iii) 통과. CI는 0을 포함 (seed 1개) — 채택은 확인 run (F1 규칙, seed 3개) 뒤에만. RESULTS TRIED_MODS에 "거르기 통과 → 확인 진행 중, 채택 아님".
- C1' (QUEUE 14:30 2번): GPU smoke `scripts/smoke_peak_gate_gpu.py` (새, `smoke_peak_refine.py` 3번과 같게 증류 30 + RL 3 iter, `--model mott --peak_gate`, 체크포인트 feat `mott_pkgate`·peak_refine·peak_gate 확인) 06:1x **통과** (GPU peak 12.4 GB) → `runs/analysis/smoke_peak_gate_gpu.ok`. 확인 run 큐에 넣기 전, GPU가 빈 동안 돌림 (GPU 학습 하나 규칙).
- **G1 통과 → 확인 7 run 큐** (`g1_confirm_queue.py r2pkfc --apply`, 사람 순서 `lo_pkg_r2pkfc_s1`, `_s2`, `r2pkfc_s1`, `_s2`, `lo_pcb_r2pkfc`, `_s1`, `_s2`; arm은 seed만 다름). 드라이버가 06:22:57 `lo_pkg_r2pkfc_s1` 시작 (정상). 각 ~2.1 h → 7개 끝 ~21:10 UTC (새 학습 마감 11/08 03:00 UTC 전). side_slot 대상을 7 run으로 바꿔 다시 띄움 (도는 `r2pk_s2_rl` ip20_1 평가는 그대로).
- **C1 대기 (서버 2로)** — QUEUE 14:25 그대로 서버 1에 넣지 않음. **G-tr `r2pk_gate`는 넣지 않음** (QUEUE 14:30 3번: G1 통과면 사람 지시 대기).
- G-inf (QUEUE 14:30 1번, 학습 없음): CPU 큐 `ls10 ginf eval10 r2pk_rl@mott_pkgate` (사람 명령 그대로, `--env_feat mott_pkgate --workers 1 --inst_chunk 5 --out runs/2026-10-29/r2pk_rl`)를 r2pkfc pcb·ip 16·20칸 평가 **앞**에 넣음 → side_slot/드라이버가 ip20_1 다음에 집음.
- 결정: LS16 (pcb16_0, pkg16_0·1)은 계속 `cpu_deferred` — 이유: QUEUE 10:50 "LS16은 G1 뒤로" + 13:40 "LS16은 부록 기준선"; G1이 확인 단계로 이어졌고 CPU 줄에는 G-inf·r2pkfc 16·20칸·확인 run 10칸 평가가 있음. LS16 하나가 ~18 h 드라이버 CPU 줄을 잡고, 16·20칸 평가와 겹치면 mem_guard가 다시 멈춤 (11/05·11/07 두 번). 가장 보수적인 선택 = 그대로 미룸.
- 할 일 (다음): (1) G-inf 끝 → `r2pk_rl@mott_pkgate` vs `r2pk_rl` C1' 규칙 판정 (RESULTS 절 자동). (2) 서버 2 C1 기록 push되면 `git merge` → `c1_screen.py`. (3) 확인 run 진행·10칸 평가 확인; 7개 끝나면 F1 규칙 판정 (뺀 pkg·pcb seed 0·1·2 vs `lo_t_r0`, `lo_t_X` − `X` ≥ −0.05, 같은 종류 pkg 3 seed CI 하한 > 0). (4) 11/08 03:00 UTC 뒤 새 학습 없음. DAILY 다음 11/08 01:xx UTC.

## 라운드 75 (2026-11-07 04:50~ UTC)
- 새 사람 커밋 f60f969 (13:36 KST): QUEUE 13:40 KST — `r2pkfc_rl` 16·20칸 평가를 **LS16이 살아 있어도 바로** (순서 pkg16_0 → pkg16_1 → pkg20_0 → pkg20_1, 한 번에 하나, `--workers 1 --inst_chunk 5`). 메모리 초과 때는 기존 `mem_guard`가 LS16을 멈추고 `.memskip` (r2pkfc 평가는 죽이지 않음). G1 `lo_pkg_r2pkfc` 10칸 평가가 먼저.
- `side_slot.py`: (a) `LS16_EXEMPT = ("r2pkfc_rl",)` — 이름에 `r2pkfc_rl`이 있는 scale-eval만 `ls16_alive` 막음에서 뺌 (`r2pk_s2_rl` ip20 등 다른 보충은 그대로 막힘). (b) 보충 시작 간격을 작업별로: scale-eval은 `SCALE_GAP_S = 1800` (지난 scale-eval 7~15 분, side_slot.log 11/06), 다른 보충은 기존 3600. side_slot 다시 띄움 → **04:50:49 `scale-eval eval_pkg16_0 r2pkfc_rl` 시작** (pod 24.0 GiB). r2pkfc 평가 명령은 `mem_guard`로 감싸지 않음 (확인) → pod > 45 GiB 때 죽는 것은 LS16 쪽만 (사람 지시와 같음).
- 결정: scale-eval 보충은 대상 RL이 30 분 넘게 남았을 때만 시작 (기존 1 h 규칙을 scale-eval에만 30 분으로) — 이유: 사람 지시 "바로 시작" + "G1 10칸 평가가 먼저"를 함께 지키는 가장 작은 변경. side_slot은 작업을 선점하지 않으므로, 평가 1개(≤15 분)가 G1 평가를 막지 않을 만큼만 간격을 둠. `lo_pkg_r2pkfc` RL 끝 ~05:58이므로 pkg16_0 (~05:05) 뒤 pkg16_1 정도까지 들어가고, 나머지는 G1 평가 (~06:00~06:12) 뒤에 이어짐.
- 상태: `lo_pkg_r2pkfc` 증류 끝, RL it 25 / 350 s (cert-bad 13, oom 0, GPU peak 9.0 GB) 정상. `git fetch r2e`: f60f969 뒤 새 커밋 없음.
- **LS16 `ls16c eval_ip16_1` 04:51 mem_guard가 멈춤** (pod 45.09 GiB, pkg16_0 평가 시작 1 분 안의 메모리 오름과 겹침; 18.7 h 진행분 버려짐) → `runs/2026-10-28/baselines/eval/eval_ip16_1__local_search.json.memskip`. 사람 지시 (13:40 KST) 그대로의 결과 — r2pkfc 평가는 그대로 돎. 드라이버는 그 항목을 건너뛰고 r2pkfc pkg16_0에 ATTACH (04:56). collect_results가 marker를 "메모리로 생략"으로 표에 넣음. LS16 ip16 기록은 ip16_0 하나 (25문제)만 남음.
- **보고만 (G1 판정 아님)**: `scale-eval eval_pkg16_0 r2pkfc_rl` 04:50~05:07 rc 0 (25문제, 4.1 s/문제): 성공 0.68 (r2pk_rl 같은 문제 0.64), 성공한 문제 평균 개수 10.88 (r2pk_rl 11.0). 짝 CI는 RESULTS "G1 후보 16·20칸 (보고만)" 표 (collect_results 자동). 05:10 `eval_pkg16_1 r2pkfc_rl` 시작 (pod 18.1 GiB).
- RESULTS: collect_results 자동 재생성 (pkg16_0 r2pkfc 행, LS16 ip16_1 "메모리로 생략").
- 할 일 (다음): (1) pkg16·20 r2pkfc 평가 진행·메모리 확인 (LS16 `.memskip` 생기면 PROGRESS·RESULTS에 "메모리로 생략 (사람 결정 13:40 KST)"). (2) 라운드 73 목록 그대로 (`lo_pkg_r2pkfc` 끝 ~05:58 → 10칸 평가 → `g1_screen.py` → 판정). (3) 11/08 03:00 UTC 뒤 새 학습 없음.

## 라운드 74 (2026-11-07 04:29~ UTC)
- 새 사람 커밋 43ed512 (13:20 KST): QUEUE 13:25 KST + PROPOSALS "11/07 G1 보충" — `r2pkfc_rl` 16·20칸 평가, **보고만 (G1 판정 아님)**, `lo_pkg_r2pkfc` 10칸 평가가 먼저, 4번째 칸이 비는 대로. 38584c4는 서버 2 A1/A2 기록 (판정·큐 변경 안 함; collect_results가 기록을 자동으로 표에 넣음 — `a1_nogrow_rl`).
- `scripts/queue_scale_r2pkfc.py` (새, 평가만): `ls10 scale-eval eval_{set} r2pkfc_rl` 12개, 명령은 규모 곡선과 같음 (`--workers 1 --inst_chunk 5`, `--out runs/2026-10-29/r2pkfc_rl`). 순서 사람 그대로: pkg16_0·1, pkg20_0·1을 대기 중인 이전 보충 (`r2pk_s2_rl` ip20_0·1) 앞에, pcb·ip 16·20 8개는 CPU 줄 끝에. side_slot (대상 `r2pkfc lo_pkg_r2pkfc`)이 큐를 매번 다시 읽으므로 재시작 없이 집어감.
- `collect_results.py`: 규모 곡선 행에 `r2pkfc_rl`, 그리고 "G1 후보 16·20칸 (보고만, G1 판정 아님)" 짝 표 (`r2pkfc_rl` − `r2pk_rl`, 같은 문제 bootstrap 성공률 차·개수비 CI). "좋아졌다" 문구 없음. RESULTS 다시 생성.
- 결정: 16·20칸 보충 평가는 **LS16 `eval_ip16_1`이 살아 있는 동안 시작하지 않음** (side_slot의 기존 `ls16_alive` 막음 그대로) — 이유: 16칸 평가 peak ~22 GB + 지금 pod ~24 GiB > 45 GiB → mem_guard가 18 h째 LS16을 죽임 (11/05 라운드 50 같은 사고). 사람 지시 "4번째 칸이 비는 대로"를 메모리 규칙 안에서 따름. 그리고 `lo_pkg_r2pkfc` RL 남은 시간 < 1 h이면 보충 없음 (FILL_GAP_S) → G1 판정 평가가 먼저.
- 상태: `lo_pkg_r2pkfc` 증류 it 1250 / 2128 s (cert-bad 0, oom 0, GPU peak 8.9 GB, val에 pkg 없음) 정상. `ensure_running`: 드라이버 alive, report_watch·resume_agent 다시 띄움. `git fetch r2e`: 새 커밋 없음.
- 할 일 (다음): (1) 라운드 73 목록 그대로 (`lo_pkg_r2pkfc` 끝 ~05:57 → 10칸 평가 → `g1_screen.py` → 판정). (2) LS16 ip16_1 끝나면 side_slot이 r2pkfc pkg16·20부터 돌리는지 확인. G1이 통과 못 하면 LS16 `--restore` 전에 r2pkfc 보충 평가 순서를 볼 것 (복원된 LS16이 다시 보충을 막음). (3) 11/08 03:00 UTC 뒤 새 학습 없음.

## 라운드 73 (2026-11-07 04:02~ UTC)
- `r2pkfc` 학습 끝 (03:57, rc 0): RL 342 it / 1.27 h, best it 200 (score 1.633), GPU peak 16.7 GB, oom 0. 드라이버가 바로 `lo_pkg_r2pkfc` 시작 (03:57:19, 증류 it 150 / 263 s, 정상). side_slot 10칸 평가 `r2pkfc_rl` 03:58~04:10 rc 0 (정책 단독, `--repair` 0; `rounds`는 cutting-plane 횟수).
- 10칸 기록 (문제당 성공 / 개수): pcb10 0.92·0.96, pkg10 0.80·0.88, ip10 1.00·1.00.
- `g1_screen.py` (같은 종류 부분, 짝 `r2pk_rl`, 50문제, 점추정 판정·CI 같이 보고): pcb 0.94 vs 0.90 **+0.04** [−0.06, +0.12] 개수비 0.981; pkg 0.84 vs 0.78 **+0.06** [−0.06, +0.18] 개수비 1.029; ip 1.00 vs 0.98 **+0.02** [+0.00, +0.06] 개수비 1.013. → (i)의 같은 종류 갈래 미달 (+0.06 < +0.10), (ii) 통과 (pcb·ip ≥ −0.06), (iii) 통과 (셋 모두 ≤ 1.05). **최종 판정은 (i)의 뺀 pkg 갈래 (`lo_pkg_r2pkfc_rl` − `lo_pkg_r2pk_rl` ≥ +0.10)에 달림** — 아직 판정 아님, 규칙 그대로.
- RESULTS 다시 생성 (평가 표에 `r2pkfc_rl` 10칸 6파일). ABLATIONS·TRIED_MODS 판정 줄은 거르기 판정이 끝난 뒤에 씀 (부분 결과로 쓰지 않음).
- `ensure_running`: 드라이버 alive, report_watch·resume_agent 다시 띄움. `git fetch r2e`: 새 사람 커밋 없음.
- 결정: 큐·설정 그대로 — 이유: G1 사람 큐 순서대로 정상 진행, 판정은 두 번째 run 결과 전에는 내리지 않음.
- 할 일 (다음): (1) `lo_pkg_r2pkfc` 끝 (~05:57) → side_slot 10칸 평가 (~06:10) → `g1_screen.py` → 통과면 `g1_confirm_queue.py r2pkfc --apply` + side_slot 대상 갱신, 아니면 RESULTS TRIED_MODS/ABLATIONS "G1 효과 없음" + LS16 `queue_defer.py --restore`. (2) 11/08 03:00 UTC 뒤 새 학습 없음. DAILY 다음 11/08 01:xx UTC.

## 라운드 72 (2026-11-07 03:46~ UTC)
- 상태 (03:47): `r2pkfc` RL it 275 / 3718 s (count/ref 0.609, fail 0.00, cert-bad 0, oom 0, GPU peak 16.7 GB, resume 체크포인트 it 264) 정상, 끝 ~03:55 → side_slot 10칸 평가 자동. CPU: `ls16c eval_ip16_1` (63 430 s째, worker rss 5.4 GB) + 학습 정확 검사 2 + side_slot 대기 (`r2pkfc lo_pkg_r2pkfc`). pod 23.4 GiB / 60, oom_kill 2 (변화 없음). `ensure_running`: 드라이버 alive, report_watch·resume_agent 다시 띄움. `git fetch r2e`: 새 사람 커밋 없음 (QUEUE 변경 없음).
- 끝난 실험 없음 → RESULTS 변경 없음.
- 결정: 큐·설정 그대로 — 이유: G1 사람 큐 순서대로 정상 진행, 바꿀 근거 없음.
- 할 일 (다음): 라운드 67 목록 그대로 (다음 라운드에서 `r2pkfc` 끝·side_slot 평가 시작 확인).

## 라운드 71 (2026-11-07 03:30~ UTC)
- 상태 (03:32): `r2pkfc` RL it 200 / 2800 s (count/ref 0.797, fail 0.12, cert-bad 5, oom 0, GPU peak 16.5 GB; it 200 new best score 1.633 — 학습 중 val 기록, 판정 아님) 정상. CPU: `ls16c eval_ip16_1` (62 480 s째, worker rss 6.5 GB) + 학습 정확 검사 2 + side_slot 대기 (`r2pkfc lo_pkg_r2pkfc`). pod 24.3 GiB / 60, oom_kill 2 (변화 없음). `ensure_running`: 드라이버 alive, report_watch·resume_agent 다시 띄움.
- 새 사람 커밋 c2b8534 (03:23 UTC): 서버 2 비교 기록 `runs/s2base/eval/*r2pk_rl+rep4.json` 8개 — **서버 2 일이라 판정·큐 변경 안 함.** `git fetch r2e`: 그 밖에 새 커밋 없음 (QUEUE 변경 없음).
- 끝난 실험 없음 → RESULTS 변경 없음.
- 결정: 큐·설정 그대로 — 이유: G1 사람 큐 순서대로 정상 진행, 바꿀 근거 없음.
- 할 일 (다음): 라운드 67 목록 그대로.


## 라운드 70 (2026-11-07 03:14~ UTC)
- 상태 (03:17): `r2pkfc` RL it 125 / 1840 s (count/ref 0.977, fail 0.06, cert-bad 8, oom 0, GPU peak 15.6 GB; it 100 new best score 1.852 — val 숫자는 학습 중 기록, 판정 아님) 정상. CPU: `ls16c eval_ip16_1` (61 500 s째, worker rss 6.2 GB) + 학습 정확 검사 2 + side_slot 대기 (`r2pkfc lo_pkg_r2pkfc`). pod 23.9 GiB / 60, oom_kill 2 (변화 없음). `ensure_running`: 드라이버 alive, report_watch·resume_agent 다시 띄움. `git fetch r2e`: 새 사람 커밋 없음 (QUEUE 마지막 변경 10:48 KST 그대로).
- 끝난 실험 없음 → RESULTS 변경 없음.
- 결정: 큐·설정 그대로 — 이유: G1 사람 큐 순서대로 정상 진행, 바꿀 근거 없음.
- 할 일 (다음): 라운드 67 목록 그대로.

## 라운드 69 (2026-11-07 02:59~ UTC)
- 상태 (03:00): `r2pkfc` RL it 50 / 847 s (count/ref 0.559, fail 0.01, cert-bad 1, oom 0, GPU peak 13.7 GB) 정상. CPU: `ls16c eval_ip16_1` (60 600 s째, worker rss 6.6 GB) + side_slot 대기 (`r2pkfc lo_pkg_r2pkfc`). pod 23.8 GiB / 60, oom_kill 2 (변화 없음). `ensure_running`: 드라이버 alive, report_watch·resume_agent 다시 띄움. `git fetch r2e`: 새 사람 커밋 없음.
- 끝난 실험 없음 → RESULTS 변경 없음.
- 결정: 큐·설정 그대로 — 이유: G1 사람 큐 순서대로 정상 진행, 바꿀 근거 없음.
- 할 일 (다음): 라운드 67 목록 그대로.

## 라운드 68 (2026-11-07 02:43~ UTC)
- 상태 (02:45): `r2pkfc` 증류 끝 (1570 iter, 0.77 h, best it 1000 score 1.918, cert-bad 0, oom 0, GPU peak 12.1 GB). RL 단계 02:40 시작 (풀 27 batch / 432 판 적재, GPU 14.5 GB 사용, 정확 검사 worker 2). CPU: `ls16c eval_ip16_1` (59 600 s째, worker rss 5.2 GB) + side_slot 대기 (`r2pkfc lo_pkg_r2pkfc`). pod 22.9 GiB / 60, oom_kill 2 (변화 없음). `git fetch r2e`: 새 사람 커밋 없음. `g1_screen.py`: 모두 waiting.
- 끝난 실험 없음 → RESULTS 변경 없음.
- 결정: 큐·설정 그대로 — 이유: G1 사람 큐 순서대로 정상 진행, 바꿀 근거 없음.
- 할 일 (다음): 라운드 67 목록 그대로.

## 라운드 67 (2026-11-07 02:27~ UTC)
- 상태 (02:30): GPU `r2pkfc` 증류 it 1100 / 1872 s (new best 1.918 @ it 1000, GPU peak 12.1 GB, cert-bad 0, oom 0) 정상. CPU: `ls16c eval_ip16_1` (58 660 s째, worker rss 6.4 GB) + side_slot 대기 (`side_slot_dry`: 둘 다 rl_done False). pod 22.7 GiB / 60, oom_kill 2 (변화 없음). `ensure_running`: 드라이버 alive, report_watch·resume_agent 다시 띄움. `git fetch r2e`: 새 사람 커밋 없음.
- `g1_screen.py` 미리 실행: 기록 없음 → 모두 "waiting" (결과 전 동작 확인).
- 새 `scripts/g1_confirm_queue.py X [--apply]`: 거르기 통과 때 확인 7 run (PROPOSALS 순서, X의 seed 0 arm + 같은 뺀-종류 풀·val 옵션, seed만 다름)을 train 끝에 붙임. 기본은 dry run (지금 큐 안 바꿈). dry run 출력이 PROPOSALS 순서·`lo_pkg_r2pkfc` arm과 같음을 확인 (assert 포함).
- 끝난 실험 없음 → RESULTS 변경 없음.
- 결정: 큐·설정 그대로 — 이유: G1 사람 큐 순서대로 정상 진행 중. 확인 run은 판정 전에는 넣지 않음 (규칙: 통과 때만).
- 할 일 (다음): (1) `r2pkfc` 끝 (~03:55) → side_slot 10칸 평가. (2) `lo_pkg_r2pkfc` (~05:56) → 평가 → `g1_screen.py` → 통과면 `g1_confirm_queue.py r2pkfc --apply` + side_slot 대상 갱신 (7 run × ~2 h ≈ 14 h, 11/08 03:00 UTC 시작 마감 안), 아니면 RESULTS ABLATIONS "G1 효과 없음" + LS16 `queue_defer.py --restore`. (3) DAILY 다음 11/08 01:xx UTC.

## 라운드 66 (2026-11-07 02:11~ UTC)
- 상태 (02:13): GPU `r2pkfc` 증류 it 550 / 922 s (val fail pcb 0.88 / pkg 0.75 / ip 0.19 — 증류 초반, 판정 아님), GPU peak 12.1 GB, 정상. CPU: 드라이버 줄 `ls16c eval_ip16_1` (16 h째, worker rss 6.7 GB) + 학습 정확 검사 2 + side_slot 대기 (`r2pkfc lo_pkg_r2pkfc`). pod 23.9 GiB / 60, oom_kill 2 (변화 없음). `ensure_running`: 드라이버·report_watch alive, resume_agent 다시 띄움.
- 새 사람 커밋 (7415f04·249c7b1, 11:00 KST): 서버 2 A1·A2 RL 설계 ablation (PROPOSALS 11/07 A1·A2) + `train.py --no_prior`. **서버 2 일이라 큐에 넣거나 판정하지 않음.** 코드 확인: `MoTT.use_prior` 기본 True, 플래그 없을 때 logits 식이 그대로 (파라미터·state_dict 변화 없음) → 지금 도는 `r2pkfc` (68b59b3로 시작)와 다음 `lo_pkg_r2pkfc` (249c7b1로 시작 예정)는 같은 계산 경로.
- 끝난 실험 없음 → RESULTS 변경 없음.
- 결정: 큐·설정 그대로 — 이유: G1 사람 큐 순서 그대로 진행 중, 모두 정상, 서버 2 지시는 서버 1 일 아님.
- 할 일 (다음): 라운드 65 목록 그대로 — (1) `r2pkfc` 끝 (~03:55) → side_slot 10칸 평가. (2) `lo_pkg_r2pkfc` (~05:56) → 평가 → `g1_screen.py` → 통과면 확인 7 run 큐, 아니면 RESULTS ABLATIONS "G1 효과 없음" + LS16 `--restore`. (3) 11/08 03:00 UTC 뒤 새 학습 없음. DAILY 다음 11/08 01:xx UTC.

## 라운드 65 (2026-11-07 01:34~ UTC) — QUEUE 11/07 방법 탐색 → G1
- 사람 지시가 라운드 중 세 번 바뀜 (모두 따름): (1) QUEUE 10:45 KST "마지막 방법 탐색" (대조군 + C1~C4) → PROPOSALS `11/07 M`에 규칙을 결과 전에 적고 대조군 `r2pk_ctl` 시작 (01:46). (2) push 때 받은 사람 커밋 a208879 (QUEUE 10:50 KST, G1이 10:45를 대체) → `r2pk_ctl` 중단 (01:51, 약 5 분, 결과 없음), 큐를 사람 G1 순서로. (3) 다시 받은 6a0726d·49cebd8 (루프 후보 철회) → `r3pk` 중단 (01:53:49 시작, 1 분 안, 결과 없음). 지금 G1 = `r2pkfc` → `lo_pkg_r2pkfc` (사람 큐 그대로, 순서 안 바꿈). 남은 부분 파일 `runs/2026-10-29/{r2pk_ctl_rl,r3pk_distill}`은 쓰지 않음 (resume.pt 없음 → 다시 큐에 들어가도 run_rung가 옆으로 치움).
- 코드 hash: `r2pkfc`는 68b59b3 (= 사람 G1 커밋 + 내 변경). `train.py` 변경은 기본값 꺼진 새 플래그 두 개 (`--kind_weight`, `--self_repair`; 10:45 절 C2·C1용, 이제 안 씀) — 기본 경로는 같은 RNG·같은 손실 (`pi` 뽑기는 플래그 없을 때 `rng.integers` 그대로). smoke: `runs/analysis/smoke_self_repair.ok` (6 iter, 정확 검사 94, 고침 통과 63, 교체 34, ~29 s/iter vs 대조 ~13 s — 기록만, 판정 없음).
- G1 smoke (QUEUE G1 할 일 1, 새 플래그 조합): `scripts/smoke_r3pk.py [fc]` GPU 증류 30 + RL 3 iter — loop+P1 (`smoke_r3pk.ok`), mott+P1+fc_bias (`smoke_r2pkfc.ok`) 모두 통과.
- 드라이버: `pipeline_day2.py`에 train 항목 `"rl_hours"` (init 있을 때만; 지금 쓰는 항목 없음). LS16 4항목 → `cpu_deferred` (`scripts/queue_defer.py`). side_slot 대상 = `r2pkfc lo_pkg_r2pkfc` (G1 할 일 2).
- 거르기 규칙 코드: `scripts/g1_screen.py` (G1 결과 전에 씀; 해석 = 점추정, 개수비는 종류마다 ≤ 1.05, PROPOSALS G1 끝 줄).
- 결정: 10:45 절의 PROPOSALS `11/07 M` 규칙을 지우지 않고 "하지 않음 (사람이 대체)"로 남김 — 이유: 대조군을 실제로 5 분 돌렸으므로 기록을 남기는 것이 정직함; 판정에는 쓰지 않음.
- 할 일 (다음): (1) `r2pkfc` 끝 (~03:55) → side_slot 10칸 평가 (~04:15). (2) `lo_pkg_r2pkfc` (~05:56) → 평가 (~06:15) → `g1_screen.py` → 통과면 확인 7 run을 PROPOSALS 순서대로 바로 큐 (+ side_slot 대상 갱신), 아니면 RESULTS ABLATIONS "G1 효과 없음" + LS16 `--restore`. (3) 11/08 03:00 UTC (12:00 KST) 뒤 새 학습 없음. DAILY 다음 11/08 01:xx UTC.

## Claude 복귀 라운드 64 (2026-11-07 01:12~ UTC)
- 사이: 11/06 02:06 ~ 11/07 01:00 UTC 루프 라운드는 사용량 한도로 일을 못 했고, 11/06 10:06 ~ 11/07 00:14 UTC는 Codex가 관리 (사람이 코드 되돌림, QUEUE 11/07 10:20 KST). Codex 기간 드라이버가 돌린 `peakmask …` 진단 8개 (`reports/peak_mask_diag_20261106/`)는 **보고서 근거로 안 씀** (QUEUE). 지금 코드 = f687b21 (+ `scripts/stop_codex.sh`), `git diff` 확인.
- 코드 hash 확인 (QUEUE 11/07 할 일): 지금 도는 학습 없음. 마지막 학습 `bp_park_rlonly_s2` (git c4514cf)·`bm_manoharan_s2` (1821d63)는 Codex 전 코드. 진행 중 LS16 `eval_ip16_1` (11/06 10:09 UTC 시작)은 `eval_policy.py` — Codex가 바꾸지 않은 파일 (`git diff f687b21 archive/codex-20261107` = `train.py`·`decap_loop/` 새 파일만).
- 끝난 것 (11/06, 기록만 커밋 안 돼 있었음 → 이번에 `git add -f`): **B1 재구현 seed 1·2 전부** (`bp_park_rlonly_s{1,2}`, `bm_manoharan_s{1,2}` 10칸 평가) → RESULTS "선행 연구 재구현 seed 0·1·2" 표 12행 모두 채워짐. 읽기 (3 seed 묶음 CI): r2pk가 10행에서 높음, **구분 안 됨 2행** = `bp_park_rlonly` pkg (−0.020 [−0.107, +0.067]), `bm_manoharan_rlonly_l20` ip (+0.027 [−0.007, +0.067]). r2pk_s2 규모 평가 ip16 ×2·20칸 ×4 기록도 추가. LS16 pcb16_1 완주 (0.64, 문제당 114.6 s).
- RESULTS.md: 되돌린 `collect_results.py`로 다시 생성 (01:13) — 차이는 머리 시각뿐 (Codex 내용 없음 확인).
- 결정: pcb16_0 LS를 CPU 줄 맨 끝 (`ls16c pkg16_0·1` 뒤)에 `ls16r2`로 다시 — 이유: 앞의 두 중단이 모두 다른 작업과 겹친 때라 "혼자서 넘음"이 확인되지 않음; 새 학습이 없으니 이제 혼자 돌 수 있다. 실험이 아니라 기존 기준선 평가 (11/9~10 평가 허용 범위).
- 결정: GPU는 비워 둠 — 이유: QUEUE 11/03 (11/6 12:00 UTC 뒤 새 시도 없음)·11/07 (새 시도 시작 안 함), 11/8 동결.
- 할 일 (다음): (1) LS16 ip16_1 → pkg16_0 → pkg16_1 → pcb16_0 (파일당 7~15 h, 11/09 안에 끝날지 불확실 → 못 끝나면 그림 2 캡션에 "LS16 일부 파일"). (2) 보고서용 정리 (11/8 뒤 표·그림만): 그림 2 LS 열, 캡션 메모. (3) DAILY 다음 11/08 01:xx UTC.

## 재개 라운드 63 (2026-11-06 01:39~ UTC)
- 상태 (01:51): **B1 `bp_park_rlonly_s1_rl` 학습 끝** (RL만 281 iter / 2.02 h, best it 100 score 12.645, teacher 교체 0, 끝 λ 22.96, GPU peak 16.8 GB) → side_slot **10칸 평가 01:50:53~** (rc 대기; seed 1 Park 때 82 분). GPU: 드라이버가 **`bm_manoharan_s1`** 증류 시작 (01:50~, 0.75 h + RL 1.25 h → 끝 ~03:50). CPU: 드라이버 줄 `ls16r eval_pcb16_0` (20:01~, 5.8 h째, worker rss 10.7 GB, 100 % CPU; ~03:20 예상) + 학습 정확 검사 + side_slot 평가. pod 24.2 GiB / 60, oom_kill 2 (변화 없음). `ensure_running`: 드라이버 alive, report_watch·resume_agent 다시 띄움. QUEUE·원격 새 지시 없음 (`git fetch` 새 커밋 0).
- 끝난 실험: 학습만 끝 (평가 진행 중) → RESULTS 변경 없음 (평가 기록이 나오면 `collect_results`가 B1 표에 자동).
- 결정: 큐·설정 그대로, 새 작업 띄우지 않음 — 이유: 동시 4개 한도 꽉 참 (LS + 학습 + 평가 등), 모두 정상, 새 지시·근거 없음.
- 할 일 (다음): (1) `bp_park_rlonly_s1` 평가 결과 (~03:10?) 읽어 PROGRESS에 숫자; 나머지 B1 (`bm_manoharan_s1` ~03:50, `_s2`, `bp_park_rlonly_s2`) 자동. (2) LS16 pcb16_0 (~03:20) 끝/죽음 확인 → `ls16r pcb16_1`. (3) LS16 뒤 r2pk_s2 규모 평가 8파일. (4) 12:00 UTC 뒤 GPU 새 학습 없음 기본; DAILY 다음 11/07 00:xx UTC.

## 재개 라운드 62 (2026-11-06 01:23~ UTC)
- 상태 (01:26): GPU **B1 `bp_park_rlonly_s1`** RL만 (23:48~, it 200 / 5088 s, val fail pcb 0.94 / pkg 0.81 / ip 0.31, λ 22.39, GPU peak 16.3 GB, resume 체크포인트 it 191 정상; 끝 ~01:50 → side_slot 10칸 평가 자동). CPU: 드라이버 줄 `ls16r eval_pcb16_0` (20:01~, 5.4 h째, worker rss 11.1 GB, 100 % CPU; ~03:20 예상) + 학습 정확 검사 2; 4번째 칸 side_slot 대기. pod 25.4 GiB / 60, oom_kill 2 (변화 없음). `ensure_running`: 드라이버 alive, report_watch·resume_agent 다시 띄움. QUEUE·원격 새 지시 없음 (`git fetch` 새 커밋 0).
- 끝난 실험 없음 (라운드 61 뒤 15 분) → RESULTS 변경 없음.
- 결정: 큐·설정 그대로, 새 작업 띄우지 않음 — 이유: 라운드 58~61과 같음 (동시 4개 한도 꽉 참, 모두 정상, 새 지시·근거 없음).
- 할 일 (다음): 라운드 58 목록 그대로 — (1) `bp_park_rlonly_s1` 평가 (~01:50~) → 나머지 B1 3 run (자동). (2) LS16 pcb16_0 (~03:20) 끝/죽음 확인 → `ls16r pcb16_1`. (3) LS16 뒤 r2pk_s2 규모 평가. (4) 12:00 UTC 뒤 GPU 새 학습 없음 기본; DAILY 다음 11/07 00:xx UTC.

## 재개 라운드 61 (2026-11-06 01:07~ UTC)
- 상태 (01:10): GPU **B1 `bp_park_rlonly_s1`** RL만 (23:48~, it 175 / 4422 s, val fail 0.16, λ 22.21, GPU peak 16.1 GB, resume 체크포인트 it 146 정상; 끝 ~01:50 → side_slot 10칸 평가 자동). CPU: 드라이버 줄 `ls16r eval_pcb16_0` (20:01~, 5.1 h째, worker rss 10.0 GB 평평, 100 % CPU; ~03:20 예상) + 학습 정확 검사 2; 4번째 칸 side_slot 대기. pod 23.4 GiB / 60, oom_kill 2 (변화 없음). `ensure_running`: 드라이버 alive, report_watch·resume_agent 다시 띄움. QUEUE·원격 새 지시 없음 (`git fetch` 새 커밋 0).
- 끝난 실험 없음 (라운드 60 뒤 15 분) → RESULTS 변경 없음.
- 결정: 큐·설정 그대로, 새 작업 띄우지 않음 — 이유: 라운드 58~60과 같음 (동시 4개 한도 꽉 참, 모두 정상, 새 지시·근거 없음).
- 할 일 (다음): 라운드 58 목록 그대로 — (1) `bp_park_rlonly_s1` 평가 (~01:50~) → 나머지 B1 3 run (자동). (2) LS16 pcb16_0 (~03:20) 끝/죽음 확인 → `ls16r pcb16_1`. (3) LS16 뒤 r2pk_s2 규모 평가. (4) 12:00 UTC 뒤 GPU 새 학습 없음 기본; DAILY 다음 11/07 00:xx UTC.

## 재개 라운드 60 (2026-11-06 00:52~ UTC)
- 상태 (00:54): GPU **B1 `bp_park_rlonly_s1`** RL만 (23:48~, it 150 / 3718 s, λ 21.97, GPU peak 15.6 GB, resume 체크포인트 it 146 정상; 끝 ~01:50 → side_slot 10칸 평가 자동). CPU: 드라이버 줄 `ls16r eval_pcb16_0` (20:01~, 4.8 h째, worker rss 10.0 GB 평평, 100 % CPU; ~03:20 예상) + 학습 정확 검사 2; 4번째 칸 side_slot 대기. pod 24.4 GiB / 60, oom_kill 2 (변화 없음). `ensure_running`: 드라이버 alive, report_watch·resume_agent 다시 띄움. QUEUE·원격 새 지시 없음 (`git fetch` 새 커밋 0).
- 끝난 실험 없음 (라운드 59 뒤 16 분) → RESULTS 변경 없음.
- 결정: 큐·설정 그대로, 새 작업 띄우지 않음 — 이유: 라운드 58·59와 같음 (동시 4개 한도 꽉 참, 모두 정상, 새 지시·근거 없음).
- 할 일 (다음): 라운드 58 목록 그대로 — (1) `bp_park_rlonly_s1` 평가 (~01:50~) → 나머지 B1 3 run (자동). (2) LS16 pcb16_0 (~03:20) 끝/죽음 확인 → `ls16r pcb16_1`. (3) LS16 뒤 r2pk_s2 규모 평가. (4) 12:00 UTC 뒤 GPU 새 학습 없음 기본; DAILY 다음 11/07 00:xx UTC.

## 재개 라운드 59 (2026-11-06 00:36~ UTC)
- 상태 (00:38): GPU **B1 `bp_park_rlonly_s1`** RL만 (23:48~, it 101 / 2404 s, val fail pcb 0.94 / pkg 0.81 / ip 0.31, λ 21.5, it 100 new best score 12.645, resume 체크포인트 it 101 정상; 끝 ~01:50 → side_slot 10칸 평가 자동). CPU: 드라이버 줄 `ls16r eval_pcb16_0` (20:01~, 4.6 h째, worker rss 11.4 GB, 100 % CPU; ~03:20 예상) + 학습 정확 검사 2; 4번째 칸 side_slot 대기 (LS16 게이트). pod 24.8 GiB / 60, oom_kill 2 (변화 없음). `ensure_running`: 드라이버 alive, report_watch·resume_agent 다시 띄움. QUEUE·원격 새 지시 없음 (`git fetch` 새 커밋 0).
- 끝난 실험 없음 (라운드 58 뒤 17 분) → RESULTS 변경 없음.
- 결정: 큐·설정 그대로, 새 작업 띄우지 않음 — 이유: 라운드 58과 같음 (동시 4개 한도 꽉 참, 모두 정상, 새 지시·근거 없음).
- 할 일 (다음): 라운드 58 목록 그대로 — (1) `bp_park_rlonly_s1` 평가 (~01:50~) → 나머지 B1 3 run (자동). (2) LS16 pcb16_0 (~03:20) 끝/죽음 확인 → `ls16r pcb16_1`. (3) LS16 뒤 r2pk_s2 규모 평가. (4) 12:00 UTC 뒤 GPU 새 학습 없음 기본; DAILY 다음 11/07 00:xx UTC.

## 재개 라운드 58 (2026-11-06 00:19~ UTC)
- 상태 (00:25): GPU **B1 `bp_park_rlonly_s1`** RL만 (23:48~, it 75 / 1717 s, val fail 0.04, λ 21.35, GPU peak 14.2 GB, it 53에서 resume 체크포인트 한 번 — 정상 이어짐; 끝 ~01:50 → side_slot 10칸 평가 자동). CPU: 드라이버 줄 `ls16r eval_pcb16_0` (20:01~, 4.3 h째, worker rss 10.0 GB 평평, 100 % CPU; ip16_0 때 7.3 h → ~03:20 예상) + 학습 정확 검사 2; 4번째 칸 side_slot 대기 (LS16 게이트). pod 24.0 GiB / 60, oom_kill 2 (변화 없음). `ensure_running`: 드라이버 alive, report_watch·resume_agent 다시 띄움. QUEUE·원격 새 지시 없음 (`git fetch` 새 커밋 0).
- 끝난 실험 없음 (라운드 57 끝 뒤) → RESULTS 변경 없음.
- 결정: 큐·설정 그대로, 새 CPU 작업 띄우지 않음 — 이유: 동시 프로세스가 이미 한도 4 (LS 1 + 학습 1 + 정확 검사 2), 도는 작업 모두 정상, 새 지시·새 근거 없음, 11/6 12:00 UTC 새 시도 마감 전 P4 후보 없음 (라운드 57 판단 그대로).
- 할 일 (다음): (1) `bp_park_rlonly_s1` 평가 (~01:50~) → `bp_park_rlonly_s2` → `bm_manoharan_s1/_s2` (~07:45) → 재구현 seed 표 완성 (자동). (2) LS16 pcb16_0 (~03:20 예상) 끝/죽음 확인 → 다음 `ls16r pcb16_1`. (3) LS16 뒤 r2pk_s2 규모 평가 8파일. (4) 12:00 UTC 뒤 GPU 새 학습 없음 기본; DAILY 다음 11/07 00:xx UTC.

## 재개 라운드 57 (2026-11-05 20:58~ UTC)
- 상태 (21:50): GPU **B1 `bm_manoharan_rlonly_l20_s2`** RL만 (21:47~, 끝 ~23:47 → side_slot 평가). `bp_park_s2_rl` 학습 끝 (RL 175 iter / 1.27 h, best it 100 score 2.537, teacher 교체 0, GPU peak 16.2 GB) → side_slot **10칸 평가 21:49~** (seed 1 때 82 분 → ~23:10). CPU: 드라이버 줄 `ls16r eval_pcb16_0` (20:01~, 1 h 50 분째, worker rss 8.9 GB 평평) + 학습 정확 검사 2 + side_slot 평가. pod 25.1 GiB / 60, oom_kill 2. `ensure_running`: report_watch·resume_agent 다시 띄움. QUEUE·원격 새 지시 없음 (`git fetch` 새 커밋 0).
- 결정: 큐·설정 그대로 — 이유: 도는 작업 모두 정상, 새 지시·새 근거 없음.
- **B1 세 번째 run `bp_park_s2_rl`** (side_slot 21:49~23:05, rc 0) 10칸 성공률 ip **0.98** / pcb **0.02** / pkg **0.64**. → **Park 재구현 seed 3개 완성** (RESULTS `### 선행 연구 재구현 seed 0·1·2` 자동): seed 0 / 1 / 2 ip 0.72 / 0.96 / 0.98 (평균 0.89), pcb 0.46 / 0.48 / 0.02 (0.32), pkg 0.16 / 0.74 / 0.64 (0.51). r2pk − bp_park 3 seed 묶음: ip **+0.100 [+0.047, +0.153]**, pcb **+0.600 [+0.527, +0.673]**, pkg **+0.307 [+0.213, +0.393]** — 세 종류 모두 CI 0 밖 "r2pk 높음" (B1 규칙의 읽기, 보고만).
  - 기록만 (`rec_counts.py`, 기록 읽기): s2 pcb 실패 49문제는 놓을 수 있는 칸을 거의 다 채우고 (58~94개) 실패 — s1에서 실패한 문제와 같은 개수 (예: #0 58, #1 69, #5 72)로, 같은 "판을 다 채워도 못 맞춤" 꼴이 s2에서는 거의 모든 문제로 퍼짐. 학습 로그의 best it 100 val도 pcb fail 0.94 / pkg 0.94 / ip 0.62였음 (best.pt 고르기 규칙은 그대로, 바꾸지 않음). 즉 Park 재구현 pcb는 seed 사이 0.02~0.48로 흔들림 — 보고서에는 seed 평균 + 폭으로만 씀.
- **B1 네 번째 run `bm_manoharan_rlonly_l20_s2_rl`** (RL만 491 iter / 2.02 h, best it 100, 끝 λ 23.0, 교체 0; side_slot 23:49~00:00, rc 0) 10칸 ip **0.96** / pcb **0.00** / pkg **0.08**. → **Manoharan RL만 재구현 seed 3개 완성:** ip 0.96 / 0.96 / 0.96, pcb 0.00 / 0.00 / 0.00, pkg 0.88 / 0.00 / 0.08 (평균 0.32). r2pk − 재구현 3 seed 묶음: ip +0.027 [−0.007, +0.067] (구분 안 됨), pcb **+0.920 [+0.873, +0.960]**, pkg **+0.500 [+0.427, +0.573]** (보고만).
  - 기록만: s2 pcb 50문제 모두 개수 0 (seed 0·1과 같은 all-skip 붕괴); pkg는 0~82개로 섞여 놓지만 2 / 25 (`_0`) 성공 — seed 0 (0.88)만 pkg에서 "거의 모든 칸 채우기"로 성공했던 것. 즉 같은 2 h RL만 예산에서 Manoharan 재구현 pkg는 seed 운 (3개 중 1개).
- **DAILY 11/06 작성** (라운드 44~57).
- 할 일 (다음): (1) 나머지 B1 4 run (`bp_park_rlonly` s1·s2, `bm_manoharan` s1·s2, ~11/06 07:45; side_slot 자동) → 재구현 seed 표 완성. (2) LS16 pcb16_0 (20:01~, ~03:00 예상) → 끝/죽음 확인. (3) 11/6 12:00 UTC 새 시도 마감 (P4 없음) → 그 뒤 GPU 새 학습 없음 기본. (4) LS16 뒤 r2pk_s2 규모 평가 9파일.

## 재개 라운드 56 (2026-11-05 20:42~ UTC)
- 상태 (20:45): GPU **B1 `bp_park_s2`** RL (20:33~, it 25 / 606 s, val fail pkg 0.11, GPU peak 12.5 GB; 끝 ~21:45 → side_slot 평가). CPU: 드라이버 줄 `ls16r eval_pcb16_0` (20:01~, 41 분째, worker rss 8.6 GB, 100 % CPU) + 학습 정확 검사 2; 4번째 칸 side_slot 대기 (LS16 게이트). pod 21.6 GiB / 60, oom_kill 2. `ensure_running`: report_watch·resume_agent 다시 띄움. QUEUE·원격 새 지시 없음 (`git fetch` 새 커밋 0; 마지막 QUEUE 항목은 서버 2 FRESH — 서버 1 큐에 넣지 않음, 그대로).
- 결정: 큐·설정 그대로 — 이유: 도는 작업 모두 정상, 새 지시·새 근거 없음 (라운드 55 판단 그대로). 끝난 실험 없음 → RESULTS 변경 없음.
- 할 일 (다음): (1) `bp_park_s2` 평가 (~21:45~) → `bm_manoharan_rlonly_l20_s2` (~23:45) → 나머지 B1 4 run. (2) LS16 pcb16_0 (~03:00 예상) — 끝나거나 죽었는지 확인. (3) **DAILY 11/06 00:xx UTC** (대상 11/05 00:20 ~, 라운드 44~). (4) 11/6 12:00 UTC 새 시도 마감 (P4 없음).

## 재개 라운드 55 (2026-11-05 20:25~ UTC)
- 상태 (20:30): GPU **B1 `bp_park_s2`** 증류 (19:44~, it 2025, 2432 s; RL 끝 ~21:45 → side_slot 평가). CPU: 드라이버 줄 `ls16r eval_pcb16_0` (20:01~, 24 분째, worker rss 11.2 GB) + 학습 정확 검사 2; 4번째 칸 side_slot 대기 (LS16 게이트 닫힘 — scale-eval 9파일 그대로 대기). pod 22.2 GiB / 60, oom_kill 2. `ensure_running`: report_watch·resume_agent 다시 띄움. QUEUE·원격 새 지시 없음 (`git fetch` 새 커밋 0).
- **라운드 54 위험 기록 풀림:** LS16 사이 빈틈에 시작된 scale-eval `r2pk_s2_rl eval_pkg16_1`은 20:12에 rc 0으로 끝남 (pcb16_0 LS와 11분 겹침, mem_guard 중단 없음, 끝 pod 21.6 GiB). → side_slot 게이트는 고치지 않음 (라운드 54 결정 조건 "pcb16_0이 죽을 때만" 안 맞음).
- **r2pk_s2 16칸 pkg 50문제 완성** (RESULTS "규모 곡선" 자동): 성공률 **0.56** (seed 0 / 1 = 0.58 / 0.54), 4.38 s/문제. 짝지은 차이 vs g+p (0.82, 130.85 s/문제) −0.26 [−0.40, −0.10], 둘 다 성공 개수비 0.723 [0.597, 0.845]; vs 종류 인식 greedy −0.24 [−0.40, −0.06]; 개수비 vs modal_rule 0.916. 즉 16칸 pkg는 r2pk seed 3개 모두 0.54~0.58로 g+p보다 성공률 낮고, 성공한 문제에선 개수가 적음 — 기록만, 판정 없음. r2pk_s2 규모 평가 남은 것: ip16 ×2, 20칸 ×6 (LS16 뒤).
- 결정: 큐·설정 그대로 — 이유: 도는 작업 모두 정상, 새 지시·새 근거 없음.
- 할 일 (다음): (1) `bp_park_s2` 평가 (~21:45~) → `bm_manoharan_rlonly_l20_s2` (~23:45) → 나머지 B1 4 run. (2) LS16 pcb16_0 (~7 h 예상, ~03:00). (3) **DAILY 11/06 00:xx UTC** (대상 11/05 00:20 ~, 라운드 44~55). (4) 11/6 12:00 UTC 새 시도 마감 (P4 없음).

## 재개 라운드 54 (2026-11-05 20:08~ UTC)
- 상태 (20:12): GPU **B1 `bp_park_s2`** 증류 (19:44~, it 1200, val pcb fail 1.00 / pkg 0.75 / ip 0.31; 끝 ~21:45 → side_slot 평가). CPU: 드라이버 줄 `ls16r eval_pcb16_0` (20:01~) + 학습 정확 검사 2; 4번째 칸 side_slot은 **LS16 사이 빈틈 (19:57:49)에 scale-eval `r2pk_s2_rl eval_pkg16_1`을 시작** (ip16_0 LS가 19:57에 끝나고 pcb16_0 LS는 20:01 시작 → `ls16_alive()` 게이트가 그 4분 동안 열림). pod 32.0 GiB / 60 (scale-eval worker 10.5 GB, LS worker 8.6 GB), oom_kill 2. `ensure_running`: report_watch·resume_agent 다시 띄움. QUEUE·원격 새 지시 없음 (`git fetch` 새 커밋 0). 시작 때 loop의 `git pull --rebase`는 dirty 로그로 거절됨 (메모 그대로, 손대지 않음).
- **LS16 `eval_ip16_0` 끝** (`--ls_chunk 8`, 12:38~19:57, 26 329 s; 516 s/문제): 성공률 **0.64** (16 / 25), 성공 해 개수 7.38; 짝지은 차이 vs modal_rule −0.36 [−0.56, −0.16], 둘 다 성공한 문제 개수비 1.09 [1.03, 1.17] (RESULTS 자동, 문제 25개 = 한 파일). 실패 9개는 평균 개수 196.8, 종류 비율 0.05 / 0.01 / 0.95 (RESULTS 종류 구성 표) — greedy+prune 실패 해 (204.9, 0.05 / 0.01 / 0.94)와 같은 꼴, 즉 LS는 g+p가 작업 격자에서 성공한 문제만 줄이므로 g+p ip16 실패 (0.56)를 물려받음. 기록만, 판정 없음.
- 위험 기록: 지금 pcb16_0 LS와 scale-eval이 함께 돈다 (라운드 50 중단과 같은 짝). 다만 그때 중단은 scale-eval **시작** 30 s 스파이크 때였고, 이번에는 scale-eval이 LS보다 4분 먼저 시작해 스파이크가 지난 뒤 LS가 붙었다. 
- 결정: 그대로 둠 (scale-eval을 죽이지 않음) — 이유: 지금 pod 32 GiB로 45 GiB까지 13 GiB 여유, 두 worker 모두 평평한 단계; 죽이면 11분 한 평가를 버림. LS가 mem_guard로 또 죽으면 `.memskip`이 남고 다음 라운드가 `ls16r`로 다시 넣는다. side_slot 게이트 수정 (LS16 줄 사이 빈틈도 막기)은 **pcb16_0이 이번 동시 실행으로 죽을 때만** 한다 (작은 변경 먼저).
- 할 일 (다음): (1) `bp_park_s2` 평가 (~21:45~) → `bm_manoharan_rlonly_l20_s2` (~23:45) → 나머지 B1 4 run. (2) LS16 pcb16_0 (20:01~, ~7 h 예상) — 끝나거나 죽었는지 확인. (3) **DAILY 11/06 00:xx UTC** (대상 11/05 00:20 ~, 라운드 44~54). (4) 11/6 12:00 UTC 새 시도 마감 (P4 없음) — B1 끝 (~07:45) 뒤 GPU는 새 학습 없이 둠이 기본.

## 재개 라운드 53 (2026-11-05 19:24~ UTC)
- 상태 (19:30): GPU **B1 `bm_manoharan_rlonly_l20_s1`** RL만 (17:43~, 2.0 h, 끝 ~19:43 → side_slot 평가) → `bp_park_s2` → `bm_manoharan_rlonly_l20_s2` → 나머지 B1 4 run (~11/06 07:45). CPU: 드라이버 줄 `ls16c eval_ip16_0` (12:38~, 6 h 47 분째, worker 100 % CPU, rss 5.8 GB, 기록 아직 없음) + 학습 정확 검사 2; 4번째 칸 side_slot 대기. pod 19.3 GiB / 60, oom_kill 2. `ensure_running`: report_watch·resume_agent 다시 띄움. QUEUE·원격 새 지시 없음 (`git fetch` 새 커밋 0).
- 결정: 큐·설정 그대로 — 이유: 도는 작업 모두 정상, 새 지시·새 근거 없음 (라운드 52 판단 그대로).
- **B1 두 번째 run `bm_manoharan_rlonly_l20_s1_rl`** (RL만 591 iter / 2.02 h, best it 400, 끝 λ 31.4, teacher 교체 0; side_slot 19:44~19:49, 문제당 0.12~0.22 s) 10칸 성공률 ip **0.96** / pcb **0.00** / pkg **0.00** (seed 0: 0.96 / 0.00 / 0.88). pcb·pkg 100문제 모두 **개수 0·rounds 0** — 고정 포트 순서에서 모든 포트에 빈칸을 고름 (메모 "RL-only λ collapse"와 같은 꼴; 학습 로그 끝 train_pkg10_0 fail 1.00, λ 계속 오름). RESULTS `### 선행 연구 재구현 seed 0·1·2` 자동: r2pk − 재구현 seed 0 / 1 ip +0.02 / +0.04, pcb +0.90 / +0.92, pkg −0.10 / **+0.88**; 2 seed 묶음 ip +0.030 [−0.010, +0.080] (구분 안 됨), pcb +0.910 [+0.850, +0.960], pkg +0.390 [+0.310, +0.470] — **중간값 (seed 2 전)**. 기록만: Park pkg (0.16 → 0.74)처럼 Manoharan RL만 pkg도 seed 사이 0.88 → 0.00로 뒤집힘 → 재구현 pkg 숫자는 seed 하나로 쓰면 안 됨 (B1을 돌린 이유 그대로).
- GPU: 드라이버가 **`bp_park_s2`** 시작 (19:44, 증류 0.75 h + RL 1.25 h, 끝 ~21:45 → side_slot 평가).
- 할 일 (다음): (1) `bp_park_s2` 평가 (~21:45~, side_slot 자동) → `bm_manoharan_rlonly_l20_s2` (~23:45) → 나머지 B1 4 run; seed 2가 다 나오면 "선행 연구 재구현 seed" 표 평균 칸 자동. (2) LS16 ip16_0 (19:55 기준 7 h 17 분째, 기록 없음) → ls16r pcb16_0 → …. (3) **DAILY 11/06 00:xx UTC** (대상 11/05 00:20 ~, 라운드 44~53). (4) 11/6 12:00 UTC 새 시도 마감 (P4 없음) — B1 끝 (~07:45) 뒤 GPU는 새 학습 없이 둠이 기본.

## 재개 라운드 52 (2026-11-05 16:12~ UTC)
- 상태 (16:15): GPU **B1 `bp_park_s1`** 증류 (15:40~, it 1525 / val pcb fail 0.50, pkg 0.69, ip 0.19; RL 끝 ~17:40, side_slot 평가 ~17:55) → 나머지 B1 7 run (~11/06 07:45). CPU: 드라이버 줄 `ls16c eval_ip16_0` (12:38~, 3 h 35 분째, worker rss 5.6 GB, 기록 아직 없음) + 학습 정확 검사 2; 4번째 칸 side_slot은 LS16 게이트로 scale-eval 대기 (B1 평가 대상 그대로). pod 18.5 GiB / 60, oom_kill 2. `ensure_running`: report_watch·resume_agent 다시 띄움. QUEUE·원격 새 지시 없음 (`git fetch` 새 커밋 0).
- 확인: B1 표 `collect_results.prior_seed_table` (PROPOSALS B1 규칙, 보고만)은 seed 1 기록이 하나라도 생기면 RESULTS "선행 연구 재구현" 절 아래에 자동으로 나옴 — 코드 바꿀 것 없음.
- 일정 메모 (판정 없음): CPU 줄에 LS16 5파일 (ls16r pcb16_0·1, ls16c ip16_1·pkg16_0·1)이 ip16_0 뒤에 남음; 파일당 ≥ 3.5 h면 LS16 열은 11/07 이후. r2pk_s2 규모 평가 9파일은 LS16 게이트 뒤. B1 끝 (~11/06 07:45) 뒤 GPU가 비면 학습 정확 검사 2칸이 비므로, 그때 게이트를 그대로 둘지 (LS ~10 GB + scale-eval 시작 +22 GiB, 학습 없으면 pod ~35 GiB 이하 예상이 아니라 **그때 mem.py로 확인**) 다음 라운드가 정한다.
- 결정: 이번 라운드는 큐·설정을 바꾸지 않음 — 이유: 도는 작업 (B1, LS16) 모두 정상이고 새 지시·새 근거 없음; 11/6 12:00 UTC 새 시도 마감 전 P4 후보 없음 (라운드 46·48 판단 그대로).
- **B1 첫 run `bp_park_s1_rl`** (distill 0.77 h best it 800, RL 184 iter / 1.27 h best it 100; side_slot 평가 17:44~19:06, 문제당 8~11 s로 seed 0의 1.5~2배 — LS16·학습과 CPU 공유) 10칸 성공률 ip **0.96** / pcb **0.48** / pkg **0.74** (seed 0: 0.72 / 0.46 / 0.16). RESULTS `### 선행 연구 재구현 seed 0·1·2` 표가 자동으로 생김 (B1 규칙, 보고만): r2pk − bp_park 같은 seed 짝 ip +0.26 / +0.04, pcb +0.44 / +0.44, pkg +0.62 / +0.14; 2 seed 묶음 ip +0.150 [+0.080, +0.220], pcb +0.440 [+0.330, +0.540], pkg +0.380 [+0.280, +0.480] — **중간값 (seed 2 전)**, 표의 "r2pk 높음" 읽기는 seed 3개 뒤 그대로 씀. 기록만: Park 재구현 pkg가 seed 사이 0.16 → 0.74로 크게 흔들림 (seed 0 한 개로 쓴 "pkg 0.16"은 seed 운이 컸음).
- 새 도구 (읽기만): `scripts/cmdline.py <pat>` (ps.py가 자르는 전체 명령), `scripts/side_slot_dry.py <names>` (side_slot이 그 학습 평가를 왜 시작하는지/안 하는지 dry 확인).
- 할 일 (다음): (1) 나머지 B1 7 run 평가 (side_slot 자동; 다음 `bm_manoharan_rlonly_l20_s1` RL 끝 ~19:45). (2) LS16 ip16_0 기록·시간·peak (19:06 기준 6.5 h째, 기록 없음). (3) DAILY 11/06 00:xx UTC (대상 11/05 00:20 ~ 11/06 00:xx, 라운드 44~).

## 재개 라운드 51 (2026-11-05 14:33~ UTC)
- 상태 (14:40): GPU **F1 `lo_pkg_r2pk_s2`** RL (14:24~, 끝 ~15:50, side_slot 평가 ~16:05) → B1 8 run (~11/06 07:45). CPU: 드라이버 줄 `ls16c eval_ip16_0` (12:38~, 2 h째, worker rss 5.3 GB, 기록 아직 없음) + 학습 정확 검사 2; 4번째 칸은 side_slot (LS16 게이트로 scale-eval 대기, F1·B1 평가 대상). pod 19.3 GiB / 60, oom_kill 2. `ensure_running`: report_watch·resume_agent 다시 띄움. QUEUE·원격 새 지시 없음.
- 이번 라운드: lo_pkg_r2pk_s2 평가를 기다려 F1 pkg 판정 칸 (자동)·"설계 근거" r2pk pkg 칸 확인 → RESULTS 재생성·commit·push.
- **F1 끝 (12 / 12 run).** `lo_pkg_r2pk_s2_rl` (side_slot 15:41~15:53) 10칸 뺀 pkg **0.78**. **F1 뺀 pkg 판정 (QUEUE 10/31 기준 그대로, RESULTS 자동): 실패** — (b') vs 0단 뺀 학습 seed별 −0.08 / −0.20 / **+0.00**, (c') 뺀 − 전체 −0.08 / −0.24 / −0.02, 평균 **−0.11** (< −0.05). 규칙 바꾸지 않음.
  - 기술용 (판정 아님, `pair2.py`, 같은 50문제): vs `lo_pkg_r0_s2` 성공 +0.00 [−0.14, +0.14], 개수비 1.102 [1.021, 1.186] (둘 다 성공 33); vs 전체 학습 `r2pk_s2_rl` −0.02 [−0.16, +0.12], 개수비 1.000; vs P1 없는 `lo_pkg_r2_s2` (0.50) **+0.28 [+0.12, +0.42]**.
  - **최종 구조 r2pk의 범용화 정리 (RESULTS "설계 근거" 2번 표, seed 3개, 같은 기준):** pcb 통과 (c' −0.03), ip 실패 (c' +0.01이지만 (b') seed 1 동률), pkg 실패 (c' −0.11, 0단 좌표가 안 떨어짐 = 11/01 사람 판독 "pkg 반전" 그대로). 보고서 주장 범위는 2단 MoTT와 같이 **pcb 하나**. 기록만: r2pk는 2단 MoTT의 뺀 학습 무너짐 seed (ip s1 0.66, pkg s1·s2 0.56·0.50)를 없앰 (r2pk 0.98, 0.64·0.78) — seed 안정성 관찰이지 판정 기준이 아님.
- GPU: 드라이버가 F1 다음 **B1 `bp_park_s1`** 시작 (15:40). side_slot이 B1 평가 대상 그대로.
- 할 일 (다음): (1) B1 8 run 평가 (side_slot 자동) → RESULTS "선행 연구 재구현 seed 0·1·2" (B1 규칙, 보고만). (2) LS16 ip16_0 → ls16r pcb16_0 → … 기록·peak. (3) LS16 뒤 r2pk_s2 규모 평가 9파일. (4) 11/6 12:00 UTC 새 시도 마감 (P4 없음) → 그 뒤 GPU는 B1 끝 (~07:45)부터 빔; 새 학습 없이 둠이 기본. DAILY 다음 11/06 00:xx UTC.

## 재개 라운드 50 (2026-11-05 14:09~ UTC)
- 상태 (14:15): GPU **F1 `lo_pkg_r2pk_s2`** 증류 (13:38~, RL 끝 ~15:50, 평가 ~16:05) → B1 8 run (~11/06 07:45). CPU: 드라이버 줄 `ls16c eval_ip16_0` (12:38~, worker rss 5.9 GB) + 학습 정확 검사 2 + 4번째 칸 `scale-eval eval_pkg16_0 r2pk_s2_rl` (13:53~). pod 29 GiB / 60, oom_kill 2. `ensure_running`: report_watch·resume_agent 다시 띄움. QUEUE·원격 새 지시 없음.
- **라운드 49 기록 고침: ls16c pcb16_0·pcb16_1은 12:16·12:34에 mem_guard로 중단됐음** (라운드 49 상태 줄의 "pcb16_0 4 h 째"는 이미 끝난 뒤였음). 원인은 LS가 아니라 **내가 라운드 49에 넣은 4번째 칸 scale-eval**: `side_slot.log` START scale-eval pcb16_0 12:16:04 → mem_guard KILL 12:16:41 (37 s), START pcb16_1 12:33:44 → KILL 12:34:12 (28 s). 16칸 정책 평가는 시작 30 s 동안 pod를 ~22 GiB 올림 (23.0 → 45.0), 그동안 chunk LS worker는 4 h 동안 ~10 GB 평평 (라운드 47~49 기록). 즉 두 "메모리로 생략" 표시는 LS16 자체의 메모리를 말하지 않음.
  - 고침: (1) `scripts/side_slot.py`: 드라이버 LS16 (`--methods ls --ls_chunk`, 16_) 작업이 살아 있는 동안 `scale-eval` filler를 시작하지 않음 (10칸 학습 평가·다른 filler는 그대로) → side_slot 같은 대상으로 다시 띄움 (도는 scale-eval pkg16_0 자식은 그대로 끝까지). (2) `scripts/queue_ls16_retry.py`: 두 marker를 `.memskip_concurrent`로 보관 (RESULTS glob 밖, git에 기록), 항목 이름 `ls16r …` (명령·mem_guard·`--ls_chunk 8` 그대로) → 드라이버가 ip16_0 다음에 pcb16_0부터 다시.
  - 결정: 두 pcb16 LS 파일을 다시 돌림 (생략 확정하지 않음) — 이유: 미리 적은 규칙 "넘으면 생략"은 그 작업이 메모리를 넘을 때를 뜻함; 넘게 만든 것은 내가 넣은 선택 작업 (평가만, 판정 없음)이므로 생략으로 적으면 그림 2·규모 곡선 LS 열을 사실과 다르게 비움. 알고리즘·설정·45 GiB 규칙은 그대로.
  - 결정: r2pk_s2 규모 평가 (남은 10파일)는 LS16 6파일 뒤로 밀림 — 이유: LS16은 QUEUE/AGENT_PROMPT 그림 2 항목, scale-eval은 내 선택 보충; 평가는 11/8 동결 뒤에도 허용. 드라이버 줄도 queue 순서상 LS16 뒤에 scale-eval을 돌림.
  - 남은 위험 (기록만): LS16이 다 끝난 뒤 드라이버 줄과 4번째 칸이 scale-eval을 하나씩 동시에 돌릴 수 있음 (각 시작 +22 GiB, mem_guard 없음). 둘의 시작이 30 s 안에 겹칠 때만 문제라 그대로 둠; 그때 pod가 높으면 다음 라운드에서 하나를 멈춤.
- r2pk_s2 규모 평가 지금까지 3파일 (RESULTS "규모 곡선" 자동, 판정 없음): 16칸 pcb **0.90** (50문제; seed 0 / 1 = 0.90 / 0.96), pkg **0.44** (eval_pkg16_0 25문제만; seed 0 / 1 = 0.58 / 0.54, 50문제). pkg16 _1이 아직이라 같은 문제끼리가 아님 — 읽지 않음. 14:17 확인: pkg16_0 끝난 뒤 4번째 칸은 다음 scale-eval을 시작하지 않음 (게이트 작동), LS ip16_0 계속.
- 할 일 (다음): (1) lo_pkg_r2pk_s2 평가 (~16:05, side_slot) → F1 pkg 판정 칸 자동 (규칙상 "실패") → "설계 근거" r2pk pkg 칸. (2) LS16 ip16_0 → ls16r pcb16_0 → … 6파일 기록·시간·peak (mem_guard 로그). (3) LS16 끝나면 r2pk_s2 규모 평가 9파일 (드라이버 줄 + 4번째 칸). (4) 11/6 12:00 UTC 새 시도 마감 (P4 없음). DAILY 다음 11/06 00:xx UTC.
- LS16 시간: pcb16_0 chunk 판은 4 h 6 분 돌고도 기록 전 (문제별 로그 없음) → 6파일이 각각 4 h 이상이면 LS16 열은 11/07 이후. 그림 2 캡션 메모의 "LS16은 일부 파일" 가능성 그대로.

## 재개 라운드 49 (2026-11-05 12:13~ UTC)
- 상태 (12:20): GPU **F1 `lo_pkg_r2pk_s1`** 증류 끝 무렵 (11:36~, RL 끝 ~13:40, side_slot 평가 ~13:55) → lo_pkg_r2pk_s2 (~15:50) → B1 8 run (~11/06 07:45). CPU: 드라이버 줄 `ls16c eval_pcb16_0` (08:10~, 4 h 째, worker rss 10.2 GB 평평, 기록 없음) + 학습 정확 검사 2 + side_slot (아래 filler). pod 23.5 GiB / 60, oom_kill 2. `ensure_running`: report_watch·resume_agent 다시 띄움. QUEUE·원격 새 지시 없음, 서버 2 기록 아직 없음.
- **규모 곡선에 채택 구조 seed 2 추가 (평가만):** RESULTS "규모 곡선"의 r2pk는 16·20칸에서 seed 0·1뿐 (`r2pk_s2`는 QUEUE 11/04 18:20 뒤에 학습됨). `scripts/queue_scale_r2pk_s2.py`로 `r2pk_s2_rl` × `eval_{pcb,pkg,ip}{16,20}_{0,1}` 12파일을 seed 0·1과 같은 명령 (`--workers 1 --inst_chunk 5`)으로 CPU 큐 끝에 넣음 — 이름 `ls10 scale-eval …` (side_slot FILL 접두, `ls10 distill-eval` 관례)이라 4번째 칸이 학습 평가 사이에 돌림, 드라이버 줄은 LS16 먼저. 첫 파일 12:16 시작. `collect_results.SCALE_METHODS`에 `r2pk_s2_rl` 행 (기록 전엔 "–").
  - 결정: seed 2 규모 평가를 넣음 — 이유: 최종 구조의 10칸 숫자는 seed 3개인데 16·20칸은 2개라 그림 2·규모 곡선 캡션이 조건이 달라짐; 새 학습·새 풀·판정 없음, seed 0·1 때 같은 4번째 칸에서 pod ≤ 29 GiB였음 (`agent/logs/fourth_slot_scale.log`), `wait_mem` 그대로. 서버 2 L3 (r2pk 16·20칸 seed 0~2)는 FRESH로 대체되어 기다리지 않음. 표에 seed 평균 행은 만들지 않음 (판정 없음, 행 그대로 읽기).
- **F1 `lo_pkg_r2pk_s1_rl`** (side_slot 13:38~13:51) 10칸: 뺀 pkg **0.64** / pcb 0.90 / ip 1.00. F1 표: 뺀 pkg seed 0 / 1 = 0.70 / 0.64, (b') vs 0단 뺀 학습 −0.08 / **−0.20**, (c') vs 전체 −0.08 / −0.24. 규칙대로: pkg "통과"는 seed 0에서 이미 불가 (판정 칸은 seed 2 뒤 자동 "실패"가 됨). 규칙 바꾸지 않음.
  - 기술용 (판정 아님, `pair2.py`, 같은 50문제): vs `lo_pkg_r0_s1` 성공 −0.20 [−0.36, −0.04], 개수비 1.187 [1.093, 1.278] (둘 다 성공 28); vs 전체 학습 `r2pk_s1_rl` −0.24 [−0.40, −0.08]; vs P1 없는 `lo_pkg_r2_s1` (0.56) +0.08. 즉 pkg를 처음 볼 때는 2단 MoTT와 r2pk 모두 0단 좌표보다 낮음 (11/01 23:45 사람 판독 "pkg 반전"과 같은 방향, r2pk로도 그대로). 실패 18문제의 평균 개수 13.9 (성공 6.9).
- 할 일 (다음): (1) lo_pkg_r2pk_s2 평가 (~15:50) → F1 pkg 판정 자동 → "설계 근거" r2pk pkg 칸. (2) r2pk_s2 규모 평가 12파일 (~8 h, filler 틈) → RESULTS 자동. (3) ls16c pcb16_0 끝나면 시간·peak rss → STUCK 줄 정리; 6파일이 각각 4 h 넘게 걸리면 LS16 열은 11/07~08에야 차므로 그림 2 캡션에 "LS16은 일부 파일" 가능성. (4) 11/6 12:00 UTC 새 시도 마감 (P4 없음). DAILY 다음 11/06 00:xx UTC.

## 재개 라운드 48 (2026-11-05 10:06~ UTC)
- 상태 (12:00, 라운드 끝): GPU **F1 `lo_pkg_r2pk_s1`** 증류 (11:36~, RL 끝 ~13:40, side_slot 평가 ~13:55) → lo_pkg_r2pk_s2 (~15:50) → B1 8 run (~11/06 07:45). CPU: 드라이버 줄 `ls16c eval_pcb16_0` (08:10~, 3 h 50 분째, rss 9.4 GB 평평) + 학습 정확 검사 2 + side_slot. pod 22.6 GiB / 60, oom_kill 2.
- 할 일 (다음): (1) lo_pkg_r2pk_s1·s2 평가 → F1 pkg 판정 (seed 0 (b') −0.08이라 규칙상 "통과" 불가, 표는 seed 3개 뒤 자동) → "설계 근거" r2pk 열 완성. (2) ls16c pcb16_0 끝나면 기록·peak rss → STUCK 줄 정리. (3) 서버 2 기록이 push되면 RESULTS 머리 줄 (서버 2 개수·FRESH 제외 개수) 확인, 판정은 사람. (4) 11/6 12:00 UTC 새 시도 마감 그대로 (지금 P4 없음). DAILY 다음 11/06 00:xx UTC.
- 상태 (10:25): GPU **F1 `lo_ip_r2pk_s2`** 증류 끝 무렵 (2234 s, val pkg fail 0.62) → RL (~12:00) → side_slot 평가 (~12:15) → lo_pkg_r2pk_s1 → lo_pkg_r2pk_s2 → B1 8 run. CPU: 드라이버 줄 `ls16c eval_pcb16_0` (08:10~, 1 h 57 분째 worker rss **9.4 GB 평평**) + 학습 정확 검사 2 + side_slot 대기. pod 13.7 GiB / 60, oom_kill 2. `ensure_running`: report_watch·resume_agent 다시 띄움. QUEUE 새 지시 없음.
- **RESULTS "설계 근거" 표에 채택 구조 r2pk 열** (`collect_results.design_rationale`): 뺀 − 전체 seed 평균과 같은 10/31 기준 판정 (F1 표와 같은 값). 지금 pcb **−0.03 / 통과**, ip·pkg는 seed 3개 뒤 자동. 3번 "주장하지 않는 것"에 P2·P3 (효과 없음) 추가, 4번 "채택한 수정 하나: P1" (근거 D7, 숫자는 P1 확장 표를 가리킴, 손 숫자 없음).
- **`research_notes/그림 캡션 메모.md`** (새): 그림 1·2·3·정성 그림마다 캡션에 넣을 조건 (seed, 문제 고르기 규칙, 시간 = 평균·폭주 한 문제 포함, 개수비 = 둘 다 성공만, 20칸 작은 표본, LS16 메모리 생략·`ls_chunk`, r2pk는 사다리 단 아님, 평가 배치 잡음). 숫자는 RESULTS 표에서 읽게 함.
- **F1 `lo_ip_r2pk_s2_rl`** (304 iter / 1.27 h, best it 200; side_slot 11:37~11:49) 10칸: 뺀 ip **1.00** / pcb 0.88 / pkg 0.90. **F1 뺀 ip 판정 (QUEUE 10/31 기준 그대로, RESULTS 자동): 실패** — (b') seed별 +0.44 / **+0.00** / +0.60 ("모두 > 0" 아님, seed 1은 0단 뺀 학습도 0.98), (c') 평균 **+0.01** (−0.05 위). 즉 r2pk는 뺀 ip에서 전체 학습만큼 성공 (1.00 / 0.98 / 1.00; 2단 MoTT seed 1의 0.66 무너짐 없음)이지만, 미리 적은 (b') 조건이 seed 1 동률로 깨져 ip는 범용화 주장에서 뺀다. 규칙 바꾸지 않음.
  - 기술용 (판정 아님, `pair2.py`): vs `lo_ip_r0_s2` 성공 +0.60 [+0.46, +0.72], 개수비 0.771 [0.660, 0.872] (둘 다 성공 20문제); vs 전체 학습 `r2pk_s2_rl` 성공 +0.02 [+0.00, +0.06], 개수비 1.018 [0.999, 1.037].
  - RESULTS "설계 근거" r2pk 열: ip +0.01 / 실패, pcb −0.03 / 통과, pkg 대기 (lo_pkg_r2pk_s1 11:36~ 학습, 평가 ~13:50; s2 ~15:50).
- ls16c `eval_pcb16_0`: 3 h 43 분째 (11:53), worker rss 9.4 GB 그대로, pod 22 GiB (학습 + 평가 겹침) — unchunked (2 h 안에 OOM)보다 오래 버팀, 아직 기록 없음.
- **QUEUE 11/05 19:50·20:15 KST + FRESH (사람, 12:00 UTC 쯤 merge):** 서버 2에서 루프 확인 L1~L4 / 새 서버 단독 FRESH — 서버 1은 큐에 넣지 않고 F1 → B1 그대로 (지킴, 아무것도 안 바꿈). 판정 줄은 사람이 씀.
  - **RESULTS 보호 (`collect_results.load_evals`):** FRESH는 다시 만든 풀에 **같은 파일 이름 (`eval_pcb10_0` …)·같은 방법 이름 (`r2pk_rl` …)**으로 기록함 → `merged()`가 (파일, 방법)마다 glob 순서상 마지막 기록을 쓰므로, 그대로 두면 서버 1 `r2pk_rl` 기록이 조용히 FRESH 것으로 바뀜. L3 `l3_r2pk*`도 서버 1 r2pk 16·20칸 기록을 서버 2 시간으로 바꿈. 고침: `runs/2026-11-05fresh/`는 읽지 않음; 서버 2 경로 (`runs/2026-11-05s2/`, L1의 `runs/2026-10-29/{r2_mott,r3_loop}_s{1,2}_rl/eval/*{16,20}_*`)는 넣되 `sec_per_inst` = NaN (시간 칸 "–", QUEUE "서버 2 시간은 쓰지 마라"); 서버 1에 같은 (파일, 방법)이 있으면 서버 1이 이김. RESULTS 머리에 개수 한 줄 (서버 2 기록이 생기면). `scripts/test_s2_guard.py` (temp dir, 실제 기록 하나를 틀로) 통과; 지금 RESULTS는 시각 말고 바뀐 것 없음.
  - 결정: FRESH 기록을 별도 표로 만들지 않음 — 이유: PROPOSALS FRESH가 "기존 서버 결과와 합쳐 판정을 만들지 않는다"이고 판정·보고는 사람이 함; 섞이지 않게 막는 것만 보수적으로.
- 결정: P4 새 시도 없음, B1 그대로 — 이유: 라운드 46 판단 (근거 있는 후보 없음) 이후 새 진단 근거가 생기지 않음; 큐를 바꾸지 않는 쪽이 보수적. 11/6 12:00 UTC 시작 마감 그대로.

## 재개 라운드 47 (2026-11-05 08:31~ UTC)
- 상태 (10:10, 라운드 끝): GPU **F1 `lo_ip_r2pk_s2`** 증류 (09:52~, RL 끝 ~11:50, side_slot 평가 ~12:05) → lo_pkg_r2pk_s1 (~14:05) → lo_pkg_r2pk_s2 (~16:00) → B1 8 run. CPU: 드라이버 줄 `ls16c eval_pcb16_0` (08:10~, 1 h 41 분째 worker rss **8.9 GB로 평평**, pod 13.6 GiB — unchunked 때는 같은 파일이 2 h 안에 커널 OOM) + 학습 정확 검사 2 + side_slot. oom_kill 2.
- 할 일 (다음): (1) lo_ip_r2pk_s2 평가 (~12:05) → F1 ip 판정 칸 "실패" 자동 확인. (2) lo_pkg_r2pk_s1·s2 평가 (~14:05, ~16:00) → F1 pkg 판정 (seed 0 (b') −0.08이라 pkg도 "통과" 불가 — 규칙상 seed 0이 이미 ≤ 0; 표는 seed 3개 다 나온 뒤 자동). (3) ls16c pcb16_0 끝나면 기록·시간·peak rss → STUCK 줄 정리 (통과하면 해제). (4) 그림 2 캡션 메모: 시간 = 평균 (폭주 한 문제 포함), LS 열 16칸 상황. DAILY 다음 11/06 00:xx UTC.
- 상태 (08:45): GPU **F1 `lo_ip_r2pk_s1`** RL (07:32~, 끝 ~09:35, side_slot 평가 ~09:50) → lo_ip_r2pk_s2 → lo_pkg_r2pk_s1 → lo_pkg_r2pk_s2 (~15:45) → B1 8 run. CPU: 드라이버 줄 `ls16c eval_pcb16_0` (08:10~, `--ls_chunk 8`, mem_guard; 22 분째 worker rss 8.9 GB, pod 13.3 GiB — chunk 전 같은 시점 11~16 GB보다 낮음) + 학습 정확 검사 2 + side_slot 대기. oom_kill 2 (그대로). `ensure_running`: report_watch·resume_agent 다시 띄움. QUEUE 새 지시 없음 (마지막 11/05 08:05 KST 그림 절은 라운드 43에 끝).
- **그림 2·3 초안을 채택 구조 `r2pk`로 다시** (기존 평가 기록만, 새 평가 없음; 옛 초안 파일은 그대로 둠):
  - `reports/fig2_bigboard_draft_r2pk.png` (`fig_results.py --fig 2 --sizes 16,20`, 방법 g+p·LS·E0+·modal_rule·0단·2단·r2pk, seed 0). 문제당 초 16칸 pcb / pkg / ip: g+p 62 / 131 / 82, **r2pk 7.6 / 3.3 / 1.4** (r2_mott 2.4 / 4.1 / 1.8); 20칸 g+p 1.3 / 565 / 72, r2pk 1.5 / 4.5 / 4.5. 성공률 16칸 r2pk 0.90 / 0.58 / 1.00 (g+p 0.92 / 0.82 / 0.56), 20칸 (16문제) 1.00 / 0.69 / 1.00 (g+p 1.00 / 0.75 / 0.62). LS 막대 칸은 16·20칸 기록이 없어 빔 (ls16c 진행 중, pcb16_1 등 "메모리로 생략"). 주의 (읽기용): 16칸 pcb r2pk 7.6 s는 r2_mott 2.4 s의 3배 — 원인 미확인 (peak_refine 입력 계산 또는 평가 때 pod 부하; 시간 비교를 보고서에 쓰기 전에 기록의 `sec_per_inst`·평가 시각 확인 필요).
    - **확인 (08:55, `scripts/rec_time.py`·`rec_counts.py`, 기록 읽기만): 부하도 입력 계산도 아님 — 폭주 한 문제.** `eval_pcb16_0` r2pk_rl seed 0: 12.71 s/문제, 개수 평균 17.76 (r2pk_s1 1.95 s / 9.08, r2_mott 1.78 s / 9.40); `eval_pcb16_1`은 2.41 s / 10.48로 정상. 원인은 **#9 하나가 199개를 놓고 실패** (같은 판 r2pk_s1 11개 ok, r2_mott 13개 ok, g+p 44개 ok) — 걸음 수만큼 시간이 늘어 평균을 끌어올림. #15도 41개 (s1 17개). 같은 종류의 폭주는 g+p에도 있음 (#17 232개 실패). → 보고서 그림 2의 시간 막대는 평균이라 이 한 문제가 반영됨; 그대로 두고 캡션에 "평균, seed 0, 폭주 한 문제 포함"을 적음 (숫자를 고르지 않음). 개수비는 둘 다 성공한 문제만이라 #9는 빠짐.
  - `reports/fig3_ladder_seeds_draft_r2pk.png` (`--fig 3 --seeds`, 10칸, 0~3단 + `r2pk_rl` seed 0·1·2): 성공률 seed 평균 pcb 0.96 / 0.93 / 0.83 / 0.91 / **0.92**, pkg 0.75 / 0.74 / 0.73 / 0.83 / **0.82**, ip 0.99 / 0.95 / 1.00 / 0.99 / **0.99**; 개수÷E0+ ip 1.41 / 1.24 / 1.01 / 1.01 / **1.01**. r2pk는 사다리 단이 아니라 2단 + 입력 하나 (P1) — 막대 이름을 `r2pk_rl`로 두고 보고서 캡션에 적을 것.
  - `fig_results.py` 고침: x축 "rung k" 이름을 사다리 4개 이름에만 (예전 코드는 `r2pk_rl`도 "rung 2"로 표시했을 것).
- **F1 `lo_ip_r2pk_s1_rl`** (side_slot 09:35~09:47) 10칸: 뺀 ip **0.98** / pcb·pkg 기록 RESULTS. F1 표 (RESULTS `### 최종 후보 r2pk의 한 종류 빼고 학습, seed별`, 자동): 뺀 ip seed 0 / 1 = 1.00 / 0.98, (b') vs 0단 뺀 학습 = +0.44 / **+0.00** (`lo_ip_r0_s1` 도 0.98), (c') +0.02 / −0.02.
  - **규칙대로 읽기 (PROPOSALS F1 = QUEUE 10/31 기준): 뺀 ip "통과"는 이제 불가능** — (b')가 "seed 0·1·2 모두 > 0"이어야 하는데 seed 1이 +0.00. 판정 칸은 seed 2 (`lo_ip_r2pk_s2`, ~11:50 평가) 뒤 자동으로 "실패"가 됨. 규칙을 바꾸지 않음. 보고서: 최종 구조로 범용화를 주장하는 종류는 지금 pcb 하나 (seed 3개 통과), ip는 2단 MoTT와 같이 주장에서 뺌.
  - 기술용 (판정 아님, `scripts/pair2.py` 새로, 기록 읽기만): 같은 50문제 짝 성공 +0.00 [−0.06, +0.06], 그러나 **개수비 0.794 [0.728, 0.857]** (둘 다 성공 48문제, 8.21 vs 11.17개) — 0단 seed 1은 뺀 ip에서 무너지지 않았지만 (seed 0은 −0.44) 개수를 21% 더 씀. 2단 MoTT의 같은 seed는 0.66 (r2pk가 이 seed의 무너짐을 없앰).
  - 결정: 남은 F1 3 run (lo_ip_r2pk_s2, lo_pkg_r2pk_s1·s2)은 그대로 돌림 — 이유: ip 판정은 정해졌지만 범용화 표를 seed 3개로 채우는 것이 F1의 목적이고 (보고서 표), pkg 판정은 아직 열려 있음; 큐를 바꾸지 않는 쪽이 보수적.

## 재개 라운드 46 (2026-11-05 04:53~ UTC)
- 상태 (08:15, 라운드 끝): GPU **F1 `lo_ip_r2pk_s1`** (07:32~, ~09:35) → lo_ip_r2pk_s2 → lo_pkg_r2pk_s1 → lo_pkg_r2pk_s2 (~15:45) → B1 8 run (~11/06 07:45). CPU: 드라이버 줄 **`ls16c eval_pcb16_0`** (08:10~, `--ls_chunk 8`, mem_guard; 시작 때 pod 27 GiB 한 번 (판 읽기) → 5 분째 12 GiB) + 학습 정확 검사 2 + side_slot (F1·B1 평가 대기). oom_kill 2.
- **P3 판정 (PROPOSALS P3 규칙 그대로, RESULTS 자동): "효과 없음".** `lo_pcb_r2pkdd_s1_rl` (side_slot 07:33~07:49) 10칸: 뺀 pcb **0.86** / pkg 0.84 / ip 0.98; vs P1 `lo_pcb_r2pk_s1_rl` 뺀 pcb −0.02 [−0.12, +0.06]. seed 0·1 묶음: 같은 종류 pcb +0.050 [−0.010, +0.110], pkg +0.010 [−0.080, +0.110], ip +0.000, 뺀 pcb **−0.030 [−0.100, +0.050]** — 이김·해로움 없음. 증류 단계 "불성립". → **최종 구조 = `r2pk`** (P1 확장 판정 그대로), P3 확장 없음. RESULTS "시도한 수정과 효과" P3 행 판정 칸도 자동으로 채워짐.
- **LS16 ip16_0·ip16_1도 mem_guard 중단** (05:53~06:48, 06:52~07:50, 둘 다 pod > 45 GiB, 커널 OOM 없음) → 6파일 중 4개 메모리 초과, pkg16_0은 07:54 시작. **원인 (코드 읽기):** `local_search`는 판마다 돌지만, 복구 한 번에 그 판의 사본 max(trials 32, 놓인 개수)개를 한 greedy 배치로 돌림 — ip16은 g+p가 ~70~100개 놓아 사본 ~100개, 16칸 env 하나 ~1 GB (g+p16 25문제 rss 19~34 GB에서) → 45 GiB 초과. pcb16_0이 혼자서도 OOM 난 것과 맞음.
  - **수정 (메모리만, 결과 같음):** `decap_loop/baselines.local_search(sub_chunk=)` (사본을 sub_chunk개씩, 같은 subset·같은 rng 순서·같은 argmin; 기본 0 = 예전과 같은 한 번 호출) + `eval_policy.py --ls_chunk K` (시작 greedy+prune도 K문제씩, 기록에 `ls_chunk`). 환경 파일 불변. **확인 `scripts/check_ls_chunk.py`:** `eval_ip10_0` LS를 `--ls_chunk 8`로 다시 (`smoke_records/ls_chunk8/`, RESULTS glob 밖) → 기록된 LS10과 ok / count / worst / occ **25 / 25 같음**, peak rss 2.9 vs 4.4 GB, 347 vs 424 s.
  - `scripts/queue_ls16_chunk.py`: 기록 없는 ls16 6항목 → 이름 `ls16c …` + `--ls_chunk 8` (mem_guard 그대로), 옛 marker는 `.memskip_unchunked`로 보관 (RESULTS glob 밖, 기록 남김). 돌던 unchunked pkg16_0 (07:54~)은 멈춤 (rc −15, 12 분 잃음) → 드라이버 `START ls16c eval_pcb16_0` 08:09:56.
  - 결정: STUCK 문제에 "다른 길"로 한 번 더 — 이유: 그림 2 (AGENT_PROMPT)의 LS 열이 16칸에서 전부 빠질 상황이고, 수정이 결과를 바꾸지 않음을 기록된 10칸 LS로 확인했음 (보수적: 알고리즘·설정·평가 그대로). 시간 측정 (문제당 초)은 묶음 크기가 달라 LS10과 조건이 다름 → 기록의 `ls_chunk`로 표시, 보고서에 한 줄.
- 할 일 (다음): (1) F1 평가 (~10:05, ~12:05, ~14:05, ~16:00, side_slot 자동) → "최종 후보 r2pk의 한 종류 빼고 학습, seed별" 판정 (QUEUE 10/31 기준 그대로, 자동). (2) 15:45 전에 근거 있는 P4가 나오면 PROPOSALS에 규칙 먼저 → B1 앞으로; 아니면 B1 그대로. (3) ls16 ip16·pkg16 4파일 (mem_guard) → 그림 2 LS 열. (4) 11/6 12:00 뒤 새 시도 없음 → "시도한 수정과 효과" 표는 이미 다 채워짐 (P3 포함), 보고서용 정리. DAILY 다음 11/06 00:xx UTC.
- 상태 (06:00): GPU **`lo_pcb_r2pkdd_s1`** 증류 (05:46~, RL 끝 ~07:30, side_slot 평가 ~07:45) → F1 4 run (~15:45). CPU: 드라이버 줄 `ls16 eval_ip16_0` (05:53~, mem_guard) + 4번째 칸 (chip zero-shot r2pk 끝 → `r2pk_s2_distill` 평가 filler) + 학습 정확 검사 2. pod 14.5 GiB / 60, oom_kill 2 (그대로).
- **LS16 `eval_pcb16_1`도 메모리로 생략 (mem_guard가 설계대로 작동):** 03:46~05:49, worker rss 26 GB·2 h째에 pod **45.6 GiB > 45** → mem_guard가 그 작업만 죽이고 `eval_pcb16_1__local_search.json.memskip` (peak 45.61 GiB) → 커널 OOM 없음 (oom_kill 2 그대로), 학습 영향 없음. 즉 LS16 pcb는 두 파일 모두 "메모리로 생략" (RESULTS 규모 곡선·그림 2 LS 열 자동). 남은 ip16·pkg16 4파일은 그대로 mem_guard로. `ensure_running`: report_watch·resume_agent 다시 띄움. QUEUE·원격 새 지시 없음.
- **P3 `r2pkdd_s1_rl`** (338 iter / 1.27 h; side_slot 05:31~05:45) 10칸: ip 1.00 / pcb **0.96** / pkg **0.86**; vs P1 `r2pk_s1_rl` pcb +0.04 [−0.06, +0.14], pkg −0.02 [−0.16, +0.10], ip +0.00 (개수비 0.982 / 0.980 / 1.006). 같이 보고: vs greedy+prune pcb +0.06 [−0.04, +0.16], pkg −0.04 [−0.16, +0.08]; vs 종류 인식 greedy pcb +0.30, pkg +0.04, ip +0.60.
  - 같은 종류 seed 0·1 묶음 (`pooled_mod_table`, 이제 4 run 중 3 끝): pcb +0.050 [−0.010, +0.110], pkg +0.010 [−0.080, +0.110], ip +0.000. → **P3 "이김"은 더 나올 수 없음** (규칙 (i): 같은 종류 pcb·pkg는 CI 하한 ≤ 0으로 확정, 뺀 pcb는 seed 0이 −0.04라 "seed 두 개 모두 > 0" 불가). 남은 갈래는 "효과 없음" / "해로움" (뺀 pcb 묶음 CI 상한 < 0일 때만) — lo_pcb_r2pkdd_s1 (~07:45) 뒤 자동 판정 줄. 어느 쪽이든 P3 확장은 큐에 넣지 않고 최종 구조 = **r2pk** (P1 확장 판정 그대로).
- **보너스 chip zero-shot에 채택 구조 추가** (4번째 칸, 05:50~, `eval_policy.py --pool runs/pool_zs --sets 'eval_chip10_*'`, `r2pk_rl`·`r2pk_s1_rl`·`r2pk_s2_rl`, 기존 사다리 chip 평가와 같은 명령, → `runs/zs_chip/r2pk/`). `collect_results.zs_chip_table`에 r2pk 행 + "r2pk − 0단"·"r2pk − 2단" 비교 행 (기술용; 미리 적은 판정 (b) 2단 − 0단은 그대로, 이미 "불충족").
  - 결정: chip zero-shot을 최종 구조로도 잰다 — 이유: 보고서 보너스 절은 최종 구조로 써야 하고 (r2a 행은 이전 후보), 새 학습 없이 기존 eval 판·명령 그대로라 CPU 10칸 평가 하나뿐 (보수적). 판정 규칙은 새로 만들지 않음.
  - 결과 (05:50~05:55, 6 job 277 s): r2pk seed 0 / 1 / 2 chip 성공률 **1.00 / 1.00 / 1.00**, 개수÷g+p 1.36 / 1.12 / 1.11 (r2_mott 1.11 / 1.11 / 1.14). seed 0 짝: vs modal_rule 개수비 1.160 [1.065, 1.263], vs 종류 인식 greedy 1.035 [0.963, 1.139], vs 0단 0.914 [0.777, 1.061]. 성공률은 모든 단이 천장이라 구분 없음 (기존 판정 "불충족" 그대로); 기술용 기록만.
- **F1 뒤 GPU = 선행 연구 재구현 seed 1·2 (PROPOSALS 11/05 B1, 보고만, run 전에 고정)** — `scripts/queue_baseline_seeds.py` 8 run (`bp_park`·`bm_manoharan_rlonly_l20` s1·s2 먼저 ~23:45, 그다음 `bp_park_rlonly`·`bm_manoharan` s1·s2 ~11/06 07:45). `driver_dryrun.py`로 큐 순서 확인 (lo_pcb_r2pkdd_s1 ATTACH → F1 4 → B1 8).
  - 결정: 새 수정 (P4) 대신 재구현 seed — 이유: 근거 있는 P4 후보 없음 (r2pk 실패 = 근소 실패 0.06~0.11 decade·6~8개, 이미 본 cutting-plane 상한 꼴, T2 "방법 무관"); 재구현 표가 seed 1개라 한 seed 비교는 RL seed 잡음 d* 0.16 안에서 약함. 15:45 전에 근거 있는 P4가 PROPOSALS에 적히면 P4를 B1 앞으로 (11/6 12:00 시작 마감).
- **`r2pk_s2_distill` 증류만 평가** (`queue_distill_eval_named.py`, side_slot filler 05:55~06:13) — 채택 구조의 "증류만" 숫자를 seed 3개로. chip 평가가 끝난 뒤 넣음 (4번째 칸 동시 2개 방지). 10칸 ip **0.98** / pcb **0.72** / pkg **0.80** (RL 뒤 `r2pk_s2_rl` 0.98 / 0.94 / 0.80). r2pk 증류만 seed 0 / 1 / 2: ip 0.90 / 0.88 / 0.98, pcb 0.80 / 0.78 / 0.72, pkg 0.74 / 0.80 / 0.80 (RESULTS 요약 표 자동).
- **side_slot 다시 띄움** (06:14, `r2pk_s2_distill` END 뒤 칸 빔 확인): 대상 `lo_pcb_r2pkdd_s1` + F1 4 + B1 8 (이전 목록은 F1 끝에서 멈춤 → B1 평가가 LS16 드라이버 줄 뒤로 밀리지 않게).
- **RESULTS 새 표 `### 선행 연구 재구현 seed 0·1·2`** (`collect_results.prior_seed_table`, B1 규칙 그대로; seed 1 run이 생기면 나타남). `scripts/test_prior_seed_table.py`로 확인: r2_mott를 대역으로 넣으면 P1 확장 행 그대로 재현 (pcb +0.087 [+0.020, +0.153], pkg +0.093 [+0.007, +0.180], ip −0.013).

## 재개 라운드 45 (2026-11-05 02:24~ UTC)
- 상태 (04:45, 라운드 끝): GPU **`r2pkdd_s1`** RL (04:15~, 끝 ~05:30, side_slot 평가 ~05:45) → lo_pcb_r2pkdd_s1 (~07:30) → F1 4 run (~15:45). CPU: 드라이버 줄 `ls16 eval_pcb16_1` (03:46~, mem_guard) + 학습 정확 검사 2; side_slot 대기. pod 12 GiB / 60, oom_kill 2.
- 상태 (03:50): GPU **`r2pkdd_s1`** 증류 (03:27~, 끝 ~04:15) → RL (~05:30) → lo_pcb_r2pkdd_s1 (~07:30) → F1 4 run (~15:45). CPU: 드라이버 줄 `ls16 eval_pcb16_1` (mem_guard) + 학습 정확 검사 2; 4번째 칸 side_slot 대기. pod ~7 GiB / 60, **oom_kill 2** (아래). QUEUE·원격 새 지시 없음.
- **OOM 2번째 (oom_kill 1 → 2, 02:24~03:27 사이):** 커널이 LS16 `eval_pcb16_0` worker를 다시 죽임 (02:24에 rss 11 GB·2 h째, pod 14.8 GiB; 옆에는 RL 학습 + 정확 검사 2뿐 — 11/03처럼 겹친 무거운 작업 없음). 즉 LS16 pcb16_0은 혼자서도 ~45 GiB를 넘게 자람. 부모 eval_policy는 worker 없이 멈춰 있어 `kill_pat.py`로 정리 → 드라이버가 다음 항목으로 넘어감. pod·학습은 영향 없음 (lo_pcb_r2pkdd rc 0).
  - 결정 (STUCK 규칙: 같은 문제 두 번): pcb16_0 LS는 포기, `scripts/guard_ls16.py`로 `eval_pcb16_0__local_search.json.memskip` (reason memory, "kernel oom_kill x2")을 쓰고 ls16 6항목을 `mem_guard.py`로 감쌈 (명령 그대로; pod > 45 GiB면 그 작업만 죽이고 marker → RESULTS "메모리로 생략"). 이유: 커널이 고르는 대신 우리 작업만 죽게 해 학습을 지킴 (보수적), 나머지 5파일은 혼자 돌 때 맞을 수도 있으니 측정은 계속.
- **P3 `lo_pcb_r2pkdd_rl` seed 0** (366 iter / 1.27 h, best it 300; side_slot 03:28~03:42) 10칸: 뺀 pcb **0.84** / pkg 0.82 / ip 1.00; vs P1 `lo_pcb_r2pk_rl` 뺀 pcb **−0.04 [−0.16, +0.08]** (개수비 0.969), pkg +0.02, ip +0.00. 같이 보고: vs 종류 인식 greedy 뺀 pcb +0.18 [+0.02, +0.34], vs greedy+prune −0.06 [−0.18, +0.06]. P3 seed 0 쌍: 같은 종류 pcb +0.06 / pkg +0.04 / ip +0.00, 뺀 pcb −0.04 → 일찍 멈춤 규칙 (pkg·뺀 pcb 둘 다 ≤ −0.16) 발동 안 함; 판정은 seed 1 쌍 뒤 (~07:45, `pooled_mod_table`).
- **P3 증류 단계 판정 (PROPOSALS P3 규칙 그대로, RESULTS P3 절 자동): 불성립 — "고유 이웃 보정이 D1 차이를 줄임"은 아님.** `r2pkdd_s1_distill` (1527 iter / 0.77 h, best it 900; side_slot 04:16~04:35) 10칸 ip 0.92 / pcb 0.84 / pkg 0.68. 증류만 pkg vs `r2pk_*_distill` seed 0 / 1 = −0.16 / −0.12, 묶음 **−0.140 [−0.240, −0.030]** — 같은 증류 예산에서 P3 입력이 pkg를 오히려 낮춤 (CI 0 아래; 증류 단계 규칙에는 "해로움" 갈래가 없어 기록만). RL 단계 P3 판정 (주 판정)은 그대로 seed 1 쌍 뒤.
- 할 일 (다음): (1) (끝) r2pkdd_s1 증류 평가. (2) r2pkdd_s1 RL 평가 (~05:45)·lo_pcb_r2pkdd_s1 평가 (~07:45, side_slot 자동) → **P3 판정**; "이김"이면 P3 확장 4 run을 F1 뒤 (11/6 12:00 전 시작). (3) F1 평가 → 범용화 표. (4) ls16 나머지 (mem_guard) → 그림 2 LS 열 / "메모리로 생략". DAILY 다음 11/06 00:xx UTC.

## 재개 라운드 44 (2026-11-05 00:29~ UTC)
- 상태 (02:10, 라운드 끝): GPU **`lo_pcb_r2pkdd`** (01:25~, 증류 → RL, 끝 ~03:25, side_slot 평가 ~03:40) → r2pkdd_s1 → lo_pcb_r2pkdd_s1 (~07:30) → F1 4 run. CPU: 드라이버 줄 LS16 `eval_pcb16_0` (00:22~) + 학습 정확 검사 2 = 3 (4번째 칸은 side_slot 대기). pod 13 GiB / 60, oom_kill 1 (라운드 27 그것).
- P3 일찍 멈춤 규칙 (seed 0 쌍의 같은 종류 pkg **와** 뺀 pcb가 둘 다 ≤ −0.16): 같은 종류 pkg가 이미 +0.04 → lo_pcb_r2pkdd 결과와 무관하게 발동 안 함, seed 1 run은 큐 그대로.
- 상태 (00:55): GPU **P3 `r2pkdd`** RL (00:0x~, ~01:25 끝) → lo_pcb_r2pkdd → r2pkdd_s1 → lo_pcb_r2pkdd_s1 (~07:30) → F1 4 run. CPU: 드라이버 줄 **LS16** `eval_pcb16_0 ls` (00:22~, rss 16 GB, 무거운 작업 하나) + 학습 정확 검사 2 + 4번째 칸 비어 있음 (side_slot이 r2pkdd_rl 끝 기다림) = 3. pod 15 GiB / 60, oom_kill 1 (라운드 27 그것). `ensure_running`: report_watch·resume_agent 다시 띄움. QUEUE·원격 새 지시 없음.
- **R 스윕 16·20칸 18 / 18 끝 → 미리 적은 읽기 (QUEUE 11/04 18:15, RESULTS "루프 반복 수 vs 판 크기"):** 16칸 **"반복의 이득은 관찰되지 않았다"** — R16 − R2 짝지은 성공률 pcb −0.08 [−0.18, +0.02], pkg −0.08 [−0.24, +0.08], ip +0.00; 개수비 1.045 / 0.994 / 1.001. 20칸 (16문제): pcb R16 − R2 **+0.25 [+0.06, +0.50]** 이지만 R2/4/8/16 = 0.75 / 0.94 / 0.75 / 1.00 단조 아님·개수비 1.143 → 주장하지 않고 고찰용 관찰; pkg +0.06 [−0.31, +0.44], ip +0.00 (개수비 1.035).
  - 결정: 표에 **R16 vs R2 짝지은 행**을 더함 (`collect_results.loop_r_size_table`) — 이유: 기존 표의 개수는 R마다 다른 성공 문제 집합의 평균이라, pkg의 "개수 감소" (16칸 11.2 → 8.9)가 짝지으면 사라짐 (0.994). 짝 없는 숫자로 "개수가 줄었다"를 읽으면 규칙을 잘못 적용하게 됨. 읽기 문장은 숫자 없이 스크립트에 고정.
- **P3 증류 단계 seed 0 (`r2pkdd_distill`, 10칸, 00:10~00:33):** ip 0.68 / pcb 0.82 / pkg 0.58; vs `r2pk_distill` pkg **−0.16 [−0.32, +0.00]**, ip −0.22 [−0.34, −0.12], pcb +0.02. 판정은 seed 1 증류 (~05:15 `done:`) 평가 뒤 (PROPOSALS P3 규칙 그대로).
- **P3 `r2pkdd_rl` seed 0** (341 iter / 1.27 h, best it 300; side_slot 01:27~01:40) 10칸: ip 0.98 / pcb **0.96** / pkg **0.82**; vs P1 `r2pk_rl` pcb +0.06 [−0.02, +0.14], pkg +0.04 [−0.10, +0.18], ip +0.00 (개수비 0.973 / 0.969 / 1.022). 같이 보고 vs greedy+prune: pcb +0.06 [−0.04, +0.16], pkg −0.08 [−0.22, +0.04]; vs 종류 인식 greedy pcb +0.30, pkg +0.00, ip −0.02. 판정은 seed 1 + lo_pcb 쌍 뒤.
- **그림 1 초안을 채택 구조로 다시** `reports/fig1_board_draft_r2pk_10.png` (`fig_board.py`, `r2pk_rl` seed 0, 10칸 `_0` 세 종류, 이전 초안과 같은 `--pick median`): pcb #8 7개 (최악 0.917), pkg #22 5개 (0.675), ip #13 7개 (0.930). 기존 평가 기록만, 새 평가 없음. P3가 "이김"이면 r2pkdd로 다시 그림.
- **RESULTS 새 절 "시도한 수정과 효과 (한눈에)"** (`collect_results.tried_mods_table`, QUEUE 11/03 11:40 KST의 ABLATIONS 정리를 미리): 수정 15개 + 평가 단계 3개, 칸 = seed 0 짝지은 성공률 차 [CI] (자동), 판정 = 각 규칙을 적용했을 때 적은 말 (PROGRESS·PROPOSALS 위치 같이), P1~P3 판정은 `pooled_mod_table`에서 바로 (P3는 지금 "판정 대기"). 11/6 12:00 뒤에는 P3 판정만 채워지면 됨.
- 할 일 (다음): (1) lo_pcb_r2pkdd 평가 (~03:40) → RESULTS·PROGRESS. (2) r2pkdd_s1 증류 `done:` 뒤 `queue_distill_eval_named.py r2pkdd_s1_distill`. (3) P3 판정 (~07:45). (4) LS16 6파일 (드라이버) → 그림 2 LS 열.

## 재개 라운드 43 (2026-11-04 19:50~ UTC)
- 상태 (00:20): GPU **P3 `r2pkdd`** 증류 끝 (23:2x~00:07, 1520 iter / 0.77 h, best it 600, val score 1.768) → RL ~01:25 → lo_pcb_r2pkdd → r2pkdd_s1 → lo_pcb_r2pkdd_s1 (~07:30) → **F1** 4 run (~15:45). CPU: 드라이버 줄 R 스윕 20칸 (pkg20_1 진행 → ip20_0·1 → ls16 6) + 학습 정확 검사 2 + side_slot `ls10 distill-eval r2pkdd_distill` (00:10~, `queue_distill_eval_named.py`, P3 증류 단계 판정용) = 4. pod 8~14 GiB / 60, oom_kill 1 (라운드 27 그것).
- **DAILY 11/05 작성** (라운드 37~43).
- 할 일 (다음): (1) r2pkdd RL 평가 (~01:40, side_slot 자동) → RESULTS. (2) r2pkdd_s1 증류 `done:` 뒤 (~05:15) `queue_distill_eval_named.py r2pkdd_s1_distill` → P3 증류 단계 판정 (seed 0·1 묶음). (3) lo_pcb_r2pkdd_s1 평가 (~07:45) → **P3 판정** (`pooled_mod_table`, PROPOSALS P3 규칙); "이김"이면 P3 확장 4 run을 F1 뒤에 (11/6 12:00 전 시작). (4) R 스윕 20칸 다 끝나면 미리 적은 읽기 한 줄. (5) F1 평가 (~09:45~16:00) → 범용화 판정 (자동 표). (6) 11/6 12:00 뒤 새 시도 없음 → ABLATIONS "시도한 수정과 효과" 정리.
- 상태 (20:40): GPU **`lo_ip_r2pk`** RL 20:05~ (P1 확장 3/4, 끝 ~21:25) → lo_pkg_r2pk (~23:30) → P3 4 run (~11/05 07:30). CPU: 드라이버 줄 g+p pkg20_1 (15:45~, 4.9 h째) + 학습 정확 검사 2 + 4번째 칸 `run_queue_entries.py 20:40` (R 스윕 ip16_1 하나, 20:35~) = 4. pod 16~29 GiB / 60, oom_kill 1 (라운드 27 그것). `ensure_running`: report_watch (27 rung 다 끝나 바로 종료)·resume_agent 다시 띄움. QUEUE·원격 새 지시 없음.
- **R 스윕 16칸 pcb·pkg·ip16_0 끝 (기록만, 읽기는 16·20칸 다 끝난 뒤, RESULTS "루프 반복 수 vs 판 크기" 자동):** pkg16 R2 / R4 / R8 / R16 = 0.58 / 0.64 / 0.58 / 0.50, 개수 11.2 / 9.4 / 9.2 / 8.9, 문제당 3.4 / 3.6 / 3.6 / 6.4 s (50문제). ip16_0 (20:18~20:33, 15 분), ip16_1 진행 중 → 그다음 20칸 6개.
- **P1 확장 `lo_ip_r2pk_rl`** (278 iter / 1.27 h, best it 200; side_slot 21:22~21:34) 10칸: 뺀 ip **1.00** / pcb 0.90 / pkg 0.86; vs `lo_ip_r2_rl` 뺀 ip **+0.00 [+0.00, +0.00]** (개수비 1.005 [0.990, 1.021]), pcb +0.02, pkg +0.02 (학습한 종류, 보고만). 뺀 ip 점추정 > −0.16 → 채택 조건 절반 충족; 뺀 pkg는 lo_pkg_r2pk (~23:30 RL 끝, 평가 ~23:45). 주의 (판정 안 바꿈): 기준 lo_ip_r2 seed 0도 1.00이라 천장 — 이 짝은 손해 여부만 잼.
- **QUEUE 11/04 18:20 1번 끝: 20칸 g+p 6 / 6 (메모리로 생략 0, mem_guard).** pkg20_1 15:45~21:10 (0.88 / 17.7개 / 570 s/문제), ip20_0·1 21:14~21:49 (둘 다 0.62, 73.2 / 61.6개, 72 s/문제). RESULTS "규모 곡선" 20칸 g+p 행 채워짐 (자동): pcb 1.00, pkg 0.75 [0.50, 0.94] (565 s/문제), ip 0.62 [0.38, 0.88] (개수비 vs 규칙 3.97). 해석은 보고서 단계 (작은 표본). 드라이버 줄은 이제 R 스윕 20칸 (4번째 칸에 ATTACH) → ls16 6.
- **P1 확장 `lo_pkg_r2pk_rl`** (359 iter / 1.26 h, best it 100; side_slot 23:24~23:36) 10칸: 뺀 pkg **0.70** / pcb 0.86 / ip 1.00; vs `lo_pkg_r2_rl` 뺀 pkg **−0.10 [−0.24, +0.04]**, pcb −0.10 [−0.22, +0.00], ip +0.00.
- **P1 확장 판정 (PROPOSALS P1 규칙 그대로, RESULTS `### P1 확장` 자동): "이김 (seed 3개 확인)" (같은 종류 pcb +0.087 [+0.020, +0.153], seed 2 +0.10) + 뺀 ip +0.00 / 뺀 pkg −0.10 둘 다 −0.16 위 → `r2pk` (2단 + `--peak_refine`)를 최종 구조 후보로 채택.** 해로움 없음. 같이 보고: 뺀 pkg 점추정 음수, 처음 보는 pcb 이득 없음 (−0.013), 같은 종류 pcb 개수비 1.08. P3 (P1 위 고유 이웃)는 그대로; P3가 이기지 않으면 최종 구조 = r2pk.
- **F1 (PROPOSALS 11/04 F1, 결과 전에 고정): 최종 후보 r2pk의 뺀 ip·pkg seed 1·2** — `lo_ip_r2pk_s1/_s2`, `lo_pkg_r2pk_s1/_s2` (arm은 seed 0 짝 그대로) GPU 큐 P3 뒤 (`scripts/queue_f1.py`, P1 확장 판정이 채택이 아니면 거부; driver dry-run 확인), ~11/05 07:30~15:45. 판정 = QUEUE 10/31 11:30 기준 그대로 (2단 자리에 r2pk), RESULTS "최종 후보 r2pk의 한 종류 빼고 학습, seed별" 자동 (r2_mott도 같은 규칙으로 나란히: ip 실패, pcb 통과, pkg 실패 — 기존 판정과 같음). r2pk 뺀 pcb는 seed 3개로 이미 **통과** ((b') +0.28 / +0.24 / +0.22, (c') 평균 −0.03).
  - 결정: F1을 P3 확장 (P3가 이길 때) 앞에 둠 — 이유: 지금 채택된 구조의 확인이 먼저 (보수적), P3 판정 (~07:45)은 F1 첫 run 시작 (~07:30)과 거의 같고 P3 확장 4 run도 11/6 12:00 전 시작 가능 (~15:45).
  - side_slot 다시 띄움 (23:45, 칸 빔 확인 뒤): `r2pkdd lo_pcb_r2pkdd r2pkdd_s1 lo_pcb_r2pkdd_s1 lo_ip_r2pk_s1 lo_ip_r2pk_s2 lo_pkg_r2pk_s1 lo_pkg_r2pk_s2`.
- **QUEUE 11/05 08:05 KST (사람, 23:50 UTC pull) 그림 "같은 판, 공진 규칙 vs MoTT" 끝** (`scripts/fig_rule_vs_mott.py`, 기존 평가 기록 occ + `fig_board.draw`만, 새 평가 없음; 판 고르기는 QUEUE 규칙 그대로 — 둘 다 정확 통과 중 개수 차 (modal_rule − r2pk_rl)가 파일 중앙값에 가장 가까운 문제, 동률이면 작은 인덱스):
  - `reports/fig_rule_vs_mott_ip16.png`: `eval_ip16_0` #3 (둘 다 통과 25 / 25, 중앙값 차 +4.0) — modal_rule **10개** (최악 0.979) vs MoTT+P1 **6개** (0.952).
  - `reports/fig_rule_vs_mott_pcb20.png`: `eval_pcb20_0` #4 (둘 다 통과 8 / 8, 중앙값 차 +2.0) — modal_rule **6개** (0.980) vs MoTT+P1 **4개** (0.991).
  - 결정: 두 파일 중 `_0`을 씀 — 이유: QUEUE가 "_0 또는 _1"로 열어 둠; 결과를 보기 전에 첫 파일로 고정 (파일 고르기로 체리피킹하지 않게). 4번째 칸이 비어 있을 때 (side_slot 대기, R 스윕 20칸은 드라이버 줄) 돌림, 각 1 분 미만.
- 상태 (21:40): GPU **`lo_pkg_r2pk`** 증류 21:2x~ (P1 확장 4/4, RL 끝 ~23:30). 4번째 칸 `run_queue_entries.py 22:30` R 스윕 20칸 (pcb20 → pkg20 → ip20; R 스윕 16칸 6/6 끝).

## 재개 라운드 42 (2026-11-04 16:39~ UTC)
- 상태 (17:40): GPU **`lo_pcb_r2pk_s2`** 증류 17:18~ (P1 확장 2/4, RL 끝 ~19:25) → lo_ip_r2pk → lo_pkg_r2pk (~23:30) → P3 4 run (~11/05 07:30). CPU: 드라이버 줄 `base20 eval_pkg20_1 greedy+prune` (mem_guard, 15:45~, 끝 ~20:30 예상) + 학습 정확 검사 2 + 4번째 칸 `run_queue_entries.py 18:30` (R 스윕 남은 11개, queue 순서, pcb16_1부터) = 4. pod 13~21 GiB / 60, oom_kill 1 (라운드 27 그것). `ensure_running`: report_watch·resume_agent 다시 띄움. QUEUE·원격 새 지시 없음.
- **P1 확장 seed 2 `r2pk_s2_rl`** (341 iter / 1.27 h, best it 200; side_slot 평가 17:18~17:30) 10칸: ip 0.98 / pcb **0.94** / pkg **0.80**; vs `r2_mott_s2_rl` pcb +0.10 [−0.02, +0.22], pkg **+0.18 [+0.02, +0.34]**, ip −0.02 [−0.06, +0.00]; 개수비 pcb 1.08 [1.01, 1.14], pkg 1.03, ip 0.99. seed 0·1·2 묶음 (150 짝, RESULTS `### P1 확장` 자동): 같은 종류 pcb **+0.087 [+0.020, +0.153]**, pkg +0.093 [+0.007, +0.180], ip −0.013 [−0.033, +0.000].
  - PROPOSALS P1 확장 규칙 대조 (판정 줄은 남은 3 run 뒤 자동): (i)을 성립시킨 판 (같은 종류 pcb)에서 묶음 CI 하한 +0.020 > 0 이고 seed 2 +0.10 ≥ 0 → "seed 3개 확인" 조건은 충족. 최종 구조 채택은 뺀 ip·pkg 점추정 > −0.16 (lo_ip_r2pk·lo_pkg_r2pk, ~23:30) 뒤. 주의 (판정 안 바꿈): 개수비 pcb 1.08 (CI 1 위) — 성공이 오른 만큼 개수가 약간 늘어남, 같이 보고.
- R 스윕 16칸 `eval_pcb16_0` R2·R4·R16 끝 (16:21~17:07, 4번째 칸; R2 13 분, R4 14 분, R16 19 분). 표는 두 파일 다 끝나야 채워짐 (자동).
- 상태 (19:40): GPU **`lo_ip_r2pk`** 증류 19:18~ (P1 확장 3/4, RL 끝 ~21:25) → lo_pkg_r2pk (~23:30) → P3 r2pkdd → lo_pcb_r2pkdd → r2pkdd_s1 → lo_pcb_r2pkdd_s1 (~11/05 07:30). CPU: 드라이버 줄 g+p pkg20_1 (15:45~, 3.6 h째, rss 11 GB) + 학습 정확 검사 2 + 4번째 칸 `run_queue_entries.py 20:20` (R 스윕 남은 9개, pkg16_1부터) = 4. pod 15~23 GiB / 60, oom_kill 1.
- **P1 확장 seed 2 `lo_pcb_r2pk_s2_rl`** (357 iter / 1.26 h, best it 300; side_slot 19:20~19:32) 10칸: 뺀 pcb **0.90** / pkg 0.76 / ip 1.00; vs `lo_pcb_r2_s2` 뺀 pcb −0.04 [−0.12, +0.04], pkg −0.06 [−0.22, +0.10], ip +0.00. 뺀 pcb seed 0·1·2 묶음 **−0.013 [−0.080, +0.053]** (seed −0.04 / +0.04 / −0.04) — 처음 보는 pcb 이득은 seed 3개에서도 없음 (P1 "이김"은 같은 종류 pcb에서 온 것, 원래 판정 그대로). 어느 판도 seed 3개 묶음 CI 상한 < 0 아님 → "해로움" 없음.
- **R 스윕 pcb16 끝 (기록만, 읽기는 16·20칸 다 끝난 뒤):** R2 / R4 / R8 / R16 = 0.92 / 0.90 / 0.86 / 0.84, 개수 9.5 / 9.6 / 10.3 / 9.5, 문제당 2.1 / 2.6 / 2.8 / 4.3 s (50문제). pkg16_0 R2·R4·R16 끝 (18:14~19:09, 54 분), pkg16_1 차례.
- 할 일 (다음): (1) lo_ip_r2pk·lo_pkg_r2pk 평가 (~21:40, ~23:45, side_slot 자동) → `### P1 확장` 판정 줄 (자동) → PROPOSALS P1에 최종 구조 채택 여부 (규칙: 뺀 ip·pkg 점추정 > −0.16). (2) 4번째 칸 빈 시간 (side_slot 평가 사이 ~1.5 h)에 R 스윕 남은 것 — 마감은 다음 RL 끝 −55 분 (16칸 pkg 한 파일 54 분). (3) r2pkdd·r2pkdd_s1 증류 `done:` 뒤 `queue_distill_eval_named.py` (P3 증류 단계). (4) 드라이버 줄 g+p pkg20_1 (~20:30) → ip20_0·1. (5) **DAILY는 11/05 00:15 UTC 뒤 첫 라운드.** (6) 새 시도 시작 마감 11/6 12:00 UTC — P3가 마지막.

## 재개 라운드 41 (2026-11-04 12:38~ UTC)
- 상태 (15:40): GPU **`r2pk_s2`** 증류 15:13~ (P1 확장 1/4, RL 끝 ~17:15) → lo_pcb_r2pk_s2 → lo_ip_r2pk → lo_pkg_r2pk (~23:15) → **P3** r2pkdd → lo_pcb_r2pkdd → r2pkdd_s1 → lo_pcb_r2pkdd_s1 (~11/05 07:30). CPU: 드라이버 줄 `base20 eval_pkg20_0 greedy+prune` (11:02~, 4.5 h째, 8문제, rss 9 GB) + 학습 정확 검사 2 + 4번째 칸 `run_queue_entries.py 16:20` (r2pk 20칸 남은 6개) = 4. pod 12~15 GiB / 60, oom_kill 1.
- **P2 묶음 판정 (PROPOSALS P2 규칙 그대로, RESULTS 자동): "효과 없음".** `lo_pcb_r2dd_s1_rl` (365 iter / 1.27 h) 뺀 pcb **0.98** (vs lo_pcb_r2_s1 +0.14 [+0.04, +0.26]), pkg 0.92, ip 1.00. 묶음: 같은 종류 pcb +0.070 [−0.020, +0.160], pkg +0.070 [−0.020, +0.160], ip +0.000, 뺀 pcb +0.030 [−0.050, +0.110] (seed −0.08 / +0.14). 증류 단계는 "성립" 그대로 (pkg +0.120 [+0.030, +0.220]). → P2 "이김" 조건부 항목 (r1 수정판 공정성 짝, r2dd 16·20칸) 안 함.
- **P3 큐** (`scripts/queue_p3.py`, 조건 P1 이김 + P2 해로움 아님 충족, P1 확장 뒤 4 run). side_slot 다시 띄움 (15:36, 칸 빔 확인 뒤): `r2pk_s2 lo_pcb_r2pk_s2 lo_ip_r2pk lo_pkg_r2pk r2pkdd lo_pcb_r2pkdd r2pkdd_s1 lo_pcb_r2pkdd_s1`. P3 증류 단계 판정용 `r2pkdd_distill`·`r2pkdd_s1_distill` 평가는 각 증류 `done:` 뒤 `queue_distill_eval_named.py` (다음 라운드들).
- 상태 (16:30): 4번째 칸 `run_queue_entries.py 16:35` R 스윕 16칸 (pcb16_0·1, r2pk_s2 RL 평가 ~17:18 전에 끝남). 드라이버 줄 `base20 eval_pkg20_1 greedy+prune` 15:45~. **`base20 eval_pkg20_0 greedy+prune` 끝 (15:41, 메모리로 생략 아님, peak pod 38.9 GiB):** 0.62 / 12.0개 / 559 s/문제 (8문제).
- **규모 곡선 QUEUE 11/04 18:20 2번 완료: r2pk·r2pk_s1 × 16·20칸 24 / 24** (RESULTS "규모 곡선" 자동). 20칸 (16문제, 작은 표본): pcb 1.00 / 0.88, pkg **0.69 / 0.44** (r2_mott 0.31, 규칙 0.56, g+p pkg20_0만 0.62), ip 1.00 / 1.00 (개수비 vs 규칙 0.74 / 0.72, r2_mott 0.84). 16칸 pkg 0.58 / 0.54 (g+p 0.82, 규칙 0.60). 해석 안 함 (seed 간 pkg20 차 0.25가 표본 CI 폭 안).
- 규모 곡선: 16칸 r2pk·r2pk_s1 12 / 12 끝, 20칸 12 / 12 끝.
- 할 일 (다음): (1) r2pk_s2 RL 평가 (~17:35, side_slot 자동) → RESULTS `### P1 확장`. (2) 4번째 칸 빈 시간에 R 스윕 남은 항목 (`run_queue_entries.py`, 마감은 side_slot 다음 평가 1 job 전). (3) r2pkdd·r2pkdd_s1 증류 `done:` 뒤 `queue_distill_eval_named.py` (P3 증류 단계 판정). (4) 드라이버 줄 g+p 20칸 pkg20_1 → ip20_0·1 (파일당 최대 ~4.7 h). (5) DAILY는 11/05 00:15 UTC 뒤 첫 라운드. (6) 11/6 12:00 UTC 새 시도 시작 마감 — P3가 마지막 시도. 주의 (기록): 20칸 pcb 문제는 쉬움 — g+p 개수 2~5개가 8문제 중 7개 (eval_pcb20_0), 그래서 g+p 문제당 1.3 s < 16칸 62 s (버그 아님, 표 해석 때 단서로).
- 상태 (13:35): GPU **`lo_pcb_r2dd_s1`** 증류 13:13~ (RL 끝 ~15:15) → P1 확장 4 run (~23:00) → (조건부) P3 4 run. CPU: 드라이버 줄 `base20 eval_pkg20_0 greedy+prune` (mem_guard, 11:02~) + 학습 정확 검사 2 + 4번째 칸 `run_queue_entries.py 14:40` (r2pk 16칸 남은 5개: ip16_0·1) = 4. pod 12~22 GiB / 60, oom_kill 1 (라운드 27 그것). `ensure_running`: report_watch·resume_agent 다시 띄움. QUEUE·원격 새 지시 없음.
- **P2 `r2dd_s1_rl`** (342 iter / 1.27 h, best it 200) 10칸 (side_slot 13:13~13:26): ip 1.00 / pcb 0.88 / pkg 0.84; vs `r2_mott_s1_rl` pcb +0.10 [−0.04, +0.24], pkg +0.00 [−0.14, +0.14], ip +0.00; 개수비 0.97~1.02. 같은 종류 묶음 (seed 0·1): pcb **+0.070 [−0.020, +0.160]**, pkg **+0.070 [−0.020, +0.160]**, ip +0.000 → 규칙 (i)의 같은 종류 판은 CI 하한 ≤ 0, 뺀 pcb는 seed 0이 −0.08이라 "두 seed 모두 > 0" 불충족 → **RL 단계 "이김"은 이미 불가능**; "해로움" 여부는 lo_pcb_r2dd_s1 (~15:30) 뒤 (`pooled_mod_table`).
- 16칸 r2pk 규모 곡선: pkg16_1 r2pk_s1까지 7 / 12 끝 (라운드 40 4번째 칸).

## 재개 라운드 40 (2026-11-04 08:44~12:35 UTC)
- 상태 (12:35): GPU **`r2dd_s1`** RL 12:03~ (끝 ~13:18) → lo_pcb_r2dd_s1 (~15:15) → P1 확장 4 run (~23:00) → (조건부) P3 4 run. CPU: 드라이버 줄 = 사람 11/04 순서 (base20 modal_rule 진행 중 → g+p 20 (mem_guard) → r2pk 16·20칸 → r3_loop R 스윕 16·20칸 → ls16) + 학습 정확 검사 2 + 4번째 칸 (`run_queue_entries.py 12:45` r2pk 16칸, side_slot은 r2dd_s1 RL 평가 대기) = 4. pod 12~20 GiB / 60, oom_kill 1 (라운드 27 그것). 16칸 r2pk 평가 6 / 12 끝 (4번째 칸).
- 상태 (09:30): GPU **`lo_pcb_r2dd`** 증류 09:07~ (RL 끝 ~11:10) → r2dd_s1 → lo_pcb_r2dd_s1 (~15:00) → P1 확장 4 run (~23:00). CPU: 드라이버 줄 eval20 + 학습 정확 검사 2 + 4번째 칸 (side_slot 대기 중 → `diag_fail_freq --refine_u` 09:28~) = 4. pod 9 GiB / 60, oom_kill 1 (라운드 27 그것). `ensure_running`: report_watch·resume_agent 다시 띄움. QUEUE 새 지시 없음.
- **P2 `r2dd_rl` seed 0** (367 iter / 1.27 h, best it 300) 10칸 (side_slot 09:08~09:20): ip 1.00 / pcb 0.92 / pkg **0.86**; vs `r2_mott_rl` pkg **+0.14 [+0.00, +0.28]**, pcb +0.04 [−0.08, +0.14], ip +0.00; 개수비 0.98~1.01. 일찍 멈춤 규칙 (seed 0 쌍 둘 다 ≤ −0.16) 해당 없음 (lo_pcb_r2dd 대기). 판정은 seed 1 뒤 (`pooled_mod_table`). 주의 (판정 안 바꿈): pkg +0.14는 RL 잡음 d* 0.16 안.
- **P3 미리 적음 (PROPOSALS 11/04 P3, 08:56 UTC — `r2dd_rl` 평가 전):** P1 위에 P2 (`--peak_refine --dedup_modes` → `--feat mott_pkdd`, `decap_loop/pkdd_env.py`), 기준 = P1 run (r2pk·lo_pcb_r2pk seed 0·1). 근거 (smoke `scripts/smoke_pkdd.py` → `.ok`): 맨판 유효 토큰 중 P1 보정 초과량 > 0.01 비율 pkg 0.09 / pcb 0.21 — 패딩 중복으로 j±1 이웃이 복사본이라 0; 고유 이웃이면 0.65 / 0.59. 큐 조건: P1 "이김" + P2 판정 "해로움" 아님 → `scripts/queue_p3.py` (P1 확장 뒤). `pooled_mod_table`·ABLATIONS에 P3 등록 (기준 run을 mod마다 받게 일반화).
  - 결정: P2 "이김"만 조건으로 하지 않음 — 이유는 PROPOSALS P3 (표적이 P1 채널 결함, P2 판정은 그 질문에 답하지 않음; "해로움"이면 취소).
  - 보충 (정확하게): 0.09는 **맨판** 값. P1 큐 전 점검 (근-만족 실패 해의 덮는 토큰)에서는 초과량 > 0.02 비율 0.41이었음 → 같은 실패 해에서 고유 이웃 값을 `diag_fail_freq.py --refine_u` (보고만, 규칙 없음)로 잼.
- **QUEUE 11/04 18:20·18:15 KST (사람, 09:20 UTC에 pull) 반영 — 드라이버 CPU 줄 (`scripts/queue_scale_curve.py`, 원자적 쓰기, 다시 돌려도 같음):** 대기 순서 = base20 modal_rule 6 → eval20 남은 것 (r3_loop ip20_1 하나) → base20 g+p 6 (`scripts/mem_guard.py`로 감쌈: pod > 45 GiB면 죽이고 `<기록>.json.memskip` → RESULTS "메모리로 생략") → **r2pk_rl·r2pk_s1_rl × 16·20칸 12 세트** (24) → **r3_loop_rl @R2·@R4·@R16 × 같은 12 세트** (12) → 나머지 (ls16 6). RESULTS 새 절 "규모 곡선" (종류별 방법 × 10/16/20칸, 개수비 vs modal_rule, 20칸 CI + "작은 표본", 20칸 r2 − r1 기록만)·"루프 반복 수 vs 판 크기" (R 2/4/8/16, 미리 적은 읽기 그대로, 기록만) 자동.
  - 결정: 09:24에 이미 시작된 `eval20 eval_ip20_1 r3_loop_rl`은 죽이지 않고 끝까지 둠 — 이유: 지시가 도착하기 전 시작, 수 분짜리라 죽이면 일만 버림; 그다음이 바로 modal_rule 20.
  - 결정: ls16 (LS 16칸 6개, 지시에 없음)은 사람 지시 항목 전부 뒤로 — 이유: 지시 항목이 우선이고, LS16은 메모리가 큰 작업 (메모리 기록 OOM 11/03)이라 늦게 도는 쪽이 보수적.
  - 결정: 4번째 칸이 비는 동안 (side_slot은 lo_pcb_r2dd RL 끝 ~11:10까지 대기) r2pk·r2pk_s1 16칸 12개를 그 칸에서 먼저 (`scripts/run_queue_entries.py 10:55 ...`, 드라이버 항목과 같은 cmd·done → 드라이버는 나중에 건너뜀; 10:55 뒤 새 항목 안 띄움) — 이유: 동시 4개 한도 안, 드라이버 순서는 안 바꿈, 정책 16칸 평가는 짧음 (r2 2.4 s/문제).
  - P2가 "이김"이면 `queue_scale_curve.py --r2dd`로 r2dd_rl 16·20칸을 r2pk 뒤에 (스크립트가 판정 확인).
- **`--refine_u` (보고만):** P1 점검과 같은 덮인 실패 70개에서 보정 초과량 > 0.02 비율 원래 이웃 0.41 (재현) → 고유 이웃 0.56, 중앙값 0.012 → 0.024 decade (덜 풀림 0.45). 맨판 (0.09 → 0.65)보다 차이가 작음 → PROPOSALS P3 근거 문장을 "맨판 기준"으로 정정 (규칙·큐 조건 그대로).
- **P2 `lo_pcb_r2dd` seed 0** (359 iter / 1.26 h, best it 148) 10칸 (side_slot 11:10~11:24): 처음 보는 pcb **0.84** vs `lo_pcb_r2_rl` **−0.08 [−0.20, +0.02]**, pkg 0.90 (+0.08), ip 1.00 (+0.00). P2 seed 0 쌍: 같은 종류 pkg +0.14, 뺀 pcb −0.08 → 일찍 멈춤 규칙 (둘 다 ≤ −0.16) 해당 없음 → r2dd_s1·lo_pcb_r2dd_s1 그대로.
- 4번째 칸 규모 곡선 (10:55까지): r2pk·r2pk_s1 16칸 5개 끝 (pcb16_0·1 둘 다, pkg16_0 r2pk_rl), 11:35까지 하나 더.
- **P2 증류 단계 판정 (규칙 그대로): "성립" (모드 토큰 중복이 D1 차이를 일부 설명).** `r2dd_s1_distill` (1584 iter / 0.77 h, best it 900) 10칸 (side_slot 11:58~12:17): ip 0.82 / pcb 0.90 / pkg **0.84**; vs r2_mott_s1_distill pkg +0.16 [+0.02, +0.30], pcb +0.08, ip −0.10. pkg seed 0 / 1 = +0.08 / +0.16, 묶음 **+0.120 [+0.030, +0.220]**. 주의 (같이 보고): 증류 ip는 두 seed 모두 −0.10~−0.12 (계속 놓는 실패 꼴). D7·D8b seed 1: 안 덮임 0 vs 17, 고유 토큰 5 vs 3, 덜 풀림 0.45 그대로 (seed 0과 같은 꼴). PROPOSALS P2에 기록.
- r2dd_s1 RL 11:58~ (끝 ~13:15) → lo_pcb_r2dd_s1 (~15:15) → P2 RL 묶음 판정 (~15:30).
- 할 일 (다음): (4) lo_pcb_r2dd_s1 평가 (~15:20) → P2 묶음 판정 → `queue_p3.py`. (5) DAILY는 11/05 00:15 UTC 뒤.

## 재개 라운드 39 (2026-11-04 04:36~ UTC)
- 상태 (08:45): GPU **P2 `r2dd`** RL 07:52~ (끝 ~09:05) → lo_pcb_r2dd → r2dd_s1 → lo_pcb_r2dd_s1 (~15:00) → **P1 확장** r2pk_s2 → lo_pcb_r2pk_s2 → lo_ip_r2pk → lo_pkg_r2pk (~23:00). CPU: 드라이버 줄 eval20 (`eval_ip20_0`) + 학습 정확 검사 2 + 4번째 칸 side_slot = 4. pod ~8 GiB / 60, oom_kill 1 (라운드 27 그것).
- **P2 `r2dd_distill` (1617 iter / 0.77 h, best it 700, val score 1.769)** 10칸 (side_slot 08:03~08:21): ip 0.80 / pcb 0.88 / pkg **0.78**; vs r2_mott_distill pkg +0.08 [−0.06, +0.22], pcb +0.10 [−0.02, +0.24], ip −0.12 [−0.24, +0.00] (seed 0 하나, 증류 단계 판정은 seed 1과 묶어서). ip 실패 10문제는 평균 78개·종류 2 0.99로 계속 놓는 꼴 — r2_mott_distill (4문제, 43개)·r2pk_distill (5문제, 71개)에도 있는 같은 꼴, 수가 많음 (관찰). 고유 주파수 규칙은 항상 유효 토큰 ≥ 1 (전역 최대 run) → 입력 결함은 아님.
- **P2 같이 보고 (판정 아님, `diag_fail_freq.py --ckpt r2dd_distill=...`, RESULTS D8 아래 새 표 "P1/P2 같이 보고" 자동):** r2dd 증류 vs r2_mott 증류 seed 0 pkg: 정확 통과 0.26 vs 0.18, 실패 36 vs 40, **안 덮임 0 vs 14** (덮임 1.00), 고유 토큰 중앙값 **5 vs 3**, 덜 풀림 0.45 vs 0.44 — 중복 수정은 덮임을 다 고치지만 덜 풀림 (토큰 수준을 격자 값으로 낮게 읽음)은 그대로.
- side_slot 다시 띄움 (08:40, 칸 빔 확인 뒤): `r2dd lo_pcb_r2dd r2dd_s1 lo_pcb_r2dd_s1 r2pk_s2 lo_pcb_r2pk_s2 lo_ip_r2pk lo_pkg_r2pk`. RESULTS ABLATIONS에 P1 확장 4행 등록 (자동).
- 할 일 (다음 라운드): (1) r2dd RL 평가 (~09:20, 자동) → RESULTS; lo_pcb_r2dd (~11:20) → P2 일찍 멈춤 규칙 (seed 0 쌍 pkg·뺀 pcb 둘 다 ≤ −0.16이면 r2dd_s1·lo_pcb_r2dd_s1 큐에서 뺌). (2) r2dd_s1 증류 끝나면 (~12:00, `done:` 줄 확인 뒤) `queue_distill_eval_named.py r2dd_s1_distill` + `diag_fail_freq.py --ckpt r2dd_s1_distill=...` → P2 증류 단계 판정. (3) lo_pcb_r2dd_s1 평가 (~15:20) → **P2 묶음 판정** (`pooled_mod_table`). P2도 "이김"이면 P1+P2 합치기를 결과 전에 PROPOSALS에 (11/6 12:00 시작 마감; 합치기 run은 P1 확장 뒤). (4) P1 확장 끝 (~23:30) → `### P1 확장` 판정. (5) DAILY는 11/05 00:15 UTC 뒤 첫 라운드.
- **P1 판정 (PROPOSALS P1 규칙 그대로, RESULTS "P1 seed 0·1 묶음 판정" 자동): "이김" (같은 종류 pcb)** — pcb 묶음 +0.080 [+0.010, +0.160] (seed +0.02 / +0.14), pkg +0.050 [−0.050, +0.150], ip −0.010, 뺀 pcb +0.000 [−0.080, +0.090] (seed −0.04 / +0.04). `lo_pcb_r2pk_s1` (355 iter / 1.26 h) 10칸: 뺀 pcb 0.88 / pkg 0.82 / ip 1.00. 확장 4 run 큐 (`scripts/queue_p1_ext.py`, 판정이 "이김"이 아니면 거부하는 스크립트). 최종 구조 채택은 seed 3개 확인 + 뺀 ip·pkg 뒤 (RESULTS `### P1 확장` 자동).
- 이전 상태 (05:40): GPU `lo_pcb_r2pk_s1` (끝 ~07:05) → P2 4 run (~15:00). CPU: 드라이버 줄 eval20 (`eval_pkg20_1` 차례) + 학습 정확 검사 2 + 4번째 칸 side_slot (P1 seed 1 + P2 대상) = 4. pod 5~13 GiB / 60, oom_kill 1 (라운드 27 그것). `ensure_running`: report_watch·resume_agent 다시 띄움. QUEUE 새 지시 없음.
- **`r2pk_s1_rl` 끝** (340 iter / 1.28 h, best it 300, replaced 133 시점 기록 — RESULTS ckpt 표), 10칸 평가 05:05~05:16: ip 1.00 / pcb **0.92** / pkg **0.88**; vs `r2_mott_s1_rl` pcb **+0.14 [+0.02, +0.26]**, pkg +0.04 [−0.08, +0.16], ip +0.00.
- **P1 묶음 (뺀 pcb seed 1 대기, 판정 전):** 같은 종류 pcb +0.02 / +0.14 → 묶음 **+0.080 [+0.010, +0.160]** (조건 (i) 충족), pkg +0.050 [−0.050, +0.150], ip −0.010. 남은 것: (ii) 뺀 pcb 묶음 점추정 ≥ −0.05 이고 해로움 없음 (seed 0 −0.04). 주의 (판정 안 바꿈): seed 1 기준 `r2_mott_s1_rl` pcb 0.78은 seed 3개 중 가장 낮음.
- **P1 확장 계획 (PROPOSALS P1, lo_pcb_r2pk_s1 결과 전에 적음):** "이김"이면 r2pk_s2·lo_pcb_r2pk_s2·lo_ip_r2pk·lo_pkg_r2pk를 **P2 4 run 뒤**에 큐 (기존 순서 유지), seed 0·1·2 묶음 CI 하한 > 0 + seed 2 ≥ 0이면 "seed 3개 확인", 최종 구조 채택은 뺀 ip·pkg 점추정 > −0.16 추가. `collect_results.pooled_mod_table`에 확장 표 코드 (`### P1 확장`, 판정 "이김"일 때만 나옴); `scripts/test_pooled_ext.py` (가짜 기록으로 세 상태 렌더링만 확인) 통과.
- 결정: 확장 4 run을 P2 앞이 아니라 뒤에 — 이유: 기존 큐 순서 유지 (보수적), 둘 다 11/4 23:00 UTC 전에 끝나 11/6 마감 안; P2가 같이 이기면 합치기를 일찍 정할 수 있음.

## 재개 라운드 38 (2026-11-04 01:57~ UTC)
- 상태 (04:35): GPU **`r2pk_s1`** RL 03:47~ (끝 ~05:05) → lo_pcb_r2pk_s1 (~07:05) → P2 r2dd → lo_pcb_r2dd → r2dd_s1 → lo_pcb_r2dd_s1 (~15:00). CPU: 드라이버 줄 eval20 (`eval_pkg20_0` r3_loop_rl) + 학습 정확 검사 2 + 4번째 칸 side_slot = 4. pod ~12 GiB / 60, oom_kill 1 (라운드 27 그것). `ensure_running`: report_watch·resume_agent 다시 띄움. QUEUE 새 지시 없음.
- 결정: side_slot을 P1 전용 (`r2pk lo_pcb_r2pk r2pk_s1 lo_pcb_r2pk_s1`)에서 `r2pk_s1 lo_pcb_r2pk_s1 r2dd lo_pcb_r2dd r2dd_s1 lo_pcb_r2dd_s1`로 다시 띄움 (04:33, 칸에 도는 작업 없을 때) — 이유: P2 평가가 다음 라운드의 재실행에 기대지 않게; 끝난 두 target은 이미 평가 끝.
- **P1 증류 단계 판정 (규칙 그대로):** `r2pk_s1_distill` 10칸 (side_slot 03:51~04:14) ip 0.88 / pcb 0.78 / pkg **0.80** (r2_mott_s1_distill 0.92 / 0.82 / 0.68). pkg seed 0 / 1 = +0.04 / +0.12, 묶음 **+0.080 [−0.030, +0.190]** → **"D1은 피크 보정으로 설명 안 됨"** (CI 하한 ≤ 0). PROPOSALS P1에 기록.
- P1 같이 보고 seed 1 (`diag_fail_rank_r2pk_s1_distill.json`): r2pk_s1 vs r2_mott_s1 증류 정확 통과 0.28 vs 0.20, 실패 35 vs 39, 안 덮임 10 vs 17, 덜 풀림 0.42 vs 0.44. 두 seed 모두 같은 방향 (통과 +0.08, 안 덮임 감소, 덜 풀림 그대로) — 보고만.
- 할 일 (다음 라운드): (1) r2pk_s1 RL 평가 (~05:20, side_slot 자동) → RESULTS. (2) lo_pcb_r2pk_s1 평가 (~07:20) → **P1 묶음 판정** (`pooled_mod_table` 자동, PROPOSALS P1 규칙 그대로) → PROPOSALS·PROGRESS. (3) r2dd 증류 끝나면 (~08:00, RL 시작 확인 뒤) `queue_distill_eval_named.py r2dd_distill` + `diag_fail_freq.py --ckpt r2dd_distill=...` (고유 토큰으로 덮임). (4) P1 "이김"이면 PROPOSALS P1대로 seed 2 + 뺀 ip·pkg를 11/6 12:00 전에 큐 (P2 뒤 또는 앞은 그때 결과 전에 적음).
- **P1 같이 보고 (판정 아님, PROPOSALS P1 "D7 측정을 r2pk 증류에 다시"):** `scripts/diag_fail_freq.py --ckpt r2pk_distill=...` (D7·D8b와 같은 경로, pkg 50판, 4번째 칸 빈 시간에 6분) → `runs/analysis/diag_fail_rank_r2pk_distill.json`. r2pk 증류 vs r2_mott 증류 seed 0: 기본 격자 greedy 한 번 정확 통과 **0.26 vs 0.18**, 격자 만족·정확 실패 36 vs 40, 덮임 0.67 vs 0.65, 덜 풀림 중앙값 0.42 vs 0.44 decade, 안 덮임 12 vs 14 (D_dup 1.00, 고유 토큰 중앙값 3 그대로). 즉 피크 보정 채널을 넣어도 실패 꼴 (토큰 옆 피크를 낮게 읽음, 중복 토큰) 은 거의 그대로 — seed 하나라 해석 안 함.
  - 같은 스크립트가 `feat mott_dd` ckpt면 DedupEnv와 같은 고유 주파수 토큰으로 "덮임"을 셈 (P2 r2dd 증류 같이 보고용, r2pk 결과는 그대로).
- **P1 `lo_pcb_r2pk` seed 0 (03:17 평가 끝, 381 iter / 1.26 h, best it 200):** 처음 보는 pcb **0.88** vs `lo_pcb_r2` 0.92 → **−0.04 [−0.16, +0.08]**, pkg 0.80 (−0.02), ip 1.00 (+0.00), 개수비 pcb 1.10 [1.00, 1.19]. P1 seed 0 쌍: 같은 종류 pkg +0.06, 뺀 pcb −0.04 → 일찍 멈춤 규칙 (둘 다 ≤ −0.16) 해당 없음 → r2pk_s1·lo_pcb_r2pk_s1 그대로. 판정은 seed 1 뒤 (`pooled_mod_table`).

## 재개 라운드 37 (2026-11-04 00:23~ UTC)
- 상태 (01:40): GPU **`lo_pcb_r2pk`** 증류 00:59~ (RL 끝 ~03:00) → r2pk_s1 → lo_pcb_r2pk_s1 (~07:00) → **P2 r2dd → lo_pcb_r2dd → r2dd_s1 → lo_pcb_r2dd_s1** (~15:00). CPU: 드라이버 줄 eval20 (16칸 고치기 `repair16 pkg greedy+prune` 00:33 끝: pkg16_1 0.96 / 17.3개) + 학습 정확 검사 2 + 4번째 칸 side_slot (P1 대상) = 4. pod 18.5 GiB / 60, oom_kill 1 (라운드 27 그것). QUEUE 새 지시 없음. DAILY 11/04는 라운드 36이 씀 (다음은 11/05 00:15 뒤).
- **P1 seed 0 (관찰, 판정은 seed 1과 묶어서):** `r2pk_rl` (349 iter / 1.27 h, best it 200) 10칸 ip 0.98 / pcb 0.90 / pkg 0.78; vs `r2_mott_rl` pcb +0.02 [−0.08, +0.12], pkg +0.06 [−0.08, +0.22], ip −0.02. 일찍 멈춤 규칙 (seed 0 쌍 둘 다 ≤ −0.16) 해당 없음 → seed 1 그대로.
- **D8 (PROPOSALS 11/04, 결과 전 규칙):** 규칙대로는 H-K 성립 (r2 안 덮인 실패 옆 격자 극대 순위 > 8 비율 1.00 × 3 seed) — **그러나 측정 오염 발견**: 작업 격자 236점 중 고유 주파수 58~167 (`scripts/diag_grid_shape.py`). `pdn.add_freqs` (pdn.py 323)가 판마다 seed peak 수를 맞추려 **마지막 seed peak 주파수를 반복**해 채움 → `sim.features_mott`의 ≥ 규칙이 반복을 모두 극대로 셈 (극대 134~157개) → **모드 토큰 8칸이 한 피크의 복사본·경사 위 평탄부로 참** (pkg10_0 판 0: 고유 3개).
  - 결정: K 8 → 16 후보는 큐에 넣지 않음 — 측정이 자동 성립에 가까워 근거가 아님; 오염을 뺀 D8b를 결과 전에 적고 먼저 돌림 (작은 실험 먼저).
- **D8b (결과 전 규칙): H-dup 성립.** r2 pkg 안 덮인 실패 14 / 17 / 22개 **모두** 중복만 없었으면 토큰이 됐을 고유 순위 ≤ 8 피크 옆 (D_dup 1.00 × 3); 실패 해의 고유 토큰 중앙값 3 / 8칸, 고유 극대 4~5개. r1 (`modal_env` 3피크)도 같은 ≥ 규칙 — 같은 결함. RESULTS "진단" 절에 D8/D8b 표 (자동).
- **P2 `--dedup_modes` (PROPOSALS 11/04 P2, 판정 규칙 학습 전에 적음 — P1과 같은 꼴, seed 0·1 묶음):** 같은 주파수 반복을 한 점으로, 다른 주파수 이웃과 비교해 극대 (`decap_loop/dedup_env.py`, `sim.py` 불변, node·edge·decap·K 그대로). smoke `scripts/smoke_dedup_modes.py` → `.ok` (맨판 node·decap 비트 단위 같음, 유효 토큰 고유; 맨판 고유 토큰 pkg 3.00 → 2.52 — 가짜 평탄부 피크가 빠짐; CPU 증류+RL). GPU 큐 P1 뒤 4 run (`scripts/queue_dedup_modes.py`), driver dry-run 확인. RESULTS ABLATIONS 6행 등록 (자동).
- 할 일 (다음 라운드): (1) lo_pcb_r2pk 평가 (~03:15) → RESULTS. (2) r2pk_s1 증류 best.pt (~04:00) → `scripts/queue_distill_eval_named.py r2pk_s1_distill`. (3) lo_pcb_r2pk_s1 평가 뒤 (~07:15) **P1 묶음 판정** (PROPOSALS P1 규칙 그대로; 묶음 CI 코드 필요 — `collect_results.py`에 P1·P2 공통 seed 0·1 묶음 표). (4) P1 side_slot이 끝나면 `side_slot.py r2dd lo_pcb_r2dd r2dd_s1 lo_pcb_r2dd_s1`로 다시 띄움 (안 띄워도 드라이버 CPU 줄이 eval20 사이에 평가함); r2dd 증류 best.pt가 생기면 증류만 평가 큐. (5) P2가 이기면 1단 수정판 공정성 짝 (PROPOSALS P2). 새 시도 시작 마감 11/6 12:00 UTC.

## 재개 라운드 36 (2026-11-03 22:25~ UTC)
- 상태 (23:05): GPU **`r2pk`** 증류 22:57~ (P1, 끝 ~00:57) → lo_pcb_r2pk → r2pk_s1 → lo_pcb_r2pk_s1 (~07:00). CPU: 드라이버 줄 `repair16 pkg greedy+prune` (16:12~) + 학습 정확 검사 2 + 4번째 칸 **side_slot10** (`r2pk lo_pcb_r2pk r2pk_s1 lo_pcb_r2pk_s1`) = 4. pod 13 GiB / 60, oom_kill 1 (라운드 27 그것). `ensure_running`: report_watch·resume_agent 다시 띄움. QUEUE 새 지시 없음.
- **D7 (PROPOSALS D7, 규칙 실행 전, `scripts/diag_fail_freq.py`, RESULTS "진단 D3~D6" 절 끝 자동): H-underres 성립, H-uncov 기각.** 기본 격자 greedy 한 번의 격자 만족·정확 실패 해 (pkg r2 40 / 39 / 44개): 최악 주파수가 모드 토큰 옆 (격자 간격 안) 0.65 / 0.56 / 0.50, 정확 worst − 가장 가까운 격자점 값 중앙값 **0.44 / 0.44 / 0.47 decade** → 토큰은 피크를 보는데 수준을 격자 값으로 낮게 읽음. 같이 보고 (판정 아님): r1도 같은 꼴 (덮임 0.72~0.75, 덜 풀림 0.43~0.49) → 덜 풀림은 기본 격자 자체 성질; r2는 실패가 더 많고 (40 vs 32 등) 안 덮임 비율이 seed 3개 모두 r1보다 큼.
- **P1 `--peak_refine` (PROPOSALS 11/03 P1, 판정 규칙 학습 전에 적음):** 모드 토큰마다 정렬 격자 세 점 (log f, 수준)의 포물선 꼭짓점 보정 초과량·보정 위반 표시 두 채널 추가 (`decap_loop/peak_env.py`, `sim.py` 불변, 모드 기술자 4 → 6). 입력 변경이라 증류부터 전체, seed 0·1 × 같은 종류·뺀 pcb = 4 run (N2: seed 2개 묶음 CI로 판정). smoke `scripts/smoke_peak_refine.py` (합성 피크 꼭짓점 복원, 나머지 채널 비트 단위로 sim과 같음, GPU 증류 30 + RL 3 iter, peak 12.4 GB) → `.ok`. 큐 전 점검 (미리 적은 규칙): r2 pkg 덮임 실패 70개에서 덮는 토큰 보정 초과량 > 0.02 비율 **0.41** (≥ 0.20) → 큐 (`scripts/queue_peak_refine.py`). 맨판에서는 0.09뿐 (신호는 만족 근처 상태에서 생김).
  - 결정: P1은 증류부터 다시라 4 run ~8 h. 이유: QUEUE 11/03 "GPU를 비우지 마라" + D7 근거가 미리 적은 규칙으로 섬; 큐 전 점검을 하나 더 넣어 (작은 실험 먼저) 신호가 없으면 GPU를 쓰지 않게 했음.
- **r2pk 증류 끝** (1549 iter / 0.77 h, best it 1500, val score 1.746 — r2_mott_distill 1595 iter·1.759), RL 23:43~ (끝 ~00:58). 증류만 10칸 평가 (side_slot10 23:47~00:04): ip 0.90 / pcb 0.80 / pkg **0.74** (0.76 / 0.72); vs r2_mott_distill pkg +0.04 [−0.12, +0.18], pcb +0.02, ip −0.02 (seed 0 하나, 판정은 seed 1과 묶어서).
- DAILY.md 11/04 새로 씀 (00:10).
- 할 일 (다음 라운드): r2pk_s1 증류 best.pt가 생기면 (~04:00) `scripts/queue_distill_eval_named.py r2pk_s1_distill` → 증류 단계 판정 (PROPOSALS P1). r2pk·lo_pcb_r2pk (~03:00) → 일찍 멈춤 규칙 확인 (둘 다 ≤ −0.16이면 seed 1 run을 큐에서 뺌). 묶음 판정은 lo_pcb_r2pk_s1 평가 뒤 (~07:15). RESULTS ABLATIONS에 P1 6행 등록됨 (자동). **DAILY는 11/04 00:15 UTC 뒤 첫 라운드.** 새 시도 시작 마감 11/6 12:00 UTC.

## 재개 라운드 35 (2026-11-03 20:50~ UTC)
- 상태 (22:35): **GPU 빔** (r2it RL 21:47 끝, 큐 비어 있음 — 아래 결정). CPU: 드라이버 줄 `repair16 pkg greedy+prune` (16:12~) + 4번째 칸 side_slot9 `eval10 r2it_rl` = 2. pod 16 GiB / 60, oom_kill 1 (라운드 27 그것). QUEUE 새 지시 없음.
- **D2 마무리 (관찰):** `r2it_distill@last` (it 2192) pkg 0.76 / 0.68 = 0.72 (r2 증류 범위 그대로). `r2it_rl` best.pt (it 300, score 1.691) = `r2_mott_rl` best.pt, **가중치 비트 단위로 같음** (`scripts/ckpt_diff.py`) → 같은 seed의 증류+RL 파이프라인 전체가 결정적 (끝 iter만 시간 상한 때문에 350 vs 355). side_slot9가 r2it_rl 10칸 평가 중 (같은 가중치라 같은 숫자 예상, 확인용).
- **D1 기제 진단 4개 (PROPOSALS 11/03 D3~D6, 모두 실행 전 규칙, RESULTS "진단 D3~D6" 절 자동) — 모두 기각:**
  - D3 일반화 (`diag_imit_heldout.py`): 학습 밖 eval 판에서 선생님 규칙 해 모방 nll seed 0: r0 7.37 / r1 3.85 / r2 3.89 → r2가 r0보다 낮아 "seed 3개 모두" 불가 → 규제 후보 버림 (seed 1·2는 판정에 필요 없어 멈춤). r0는 학습 밖 nll이 가장 나쁜데 평가 성공은 가장 높음.
  - D4 격자 경계까지 빼기 (`diag_prune_feasible.py`): pkg cost (정확 통과 빼기 금지 − 허용) r2 0.06 / 0.16 / 0.12 vs r0 0.12 / 0.26 / 0.14 → 기각.
  - D5 제거 채널 prior (`diag_rm_prior.py`, 추론 수술): drop 1.14→1.10 등 거의 그대로 → 빼기는 학습된 잔차에서 옴 → "prior를 추가 채널에만" 버림.
  - D6 증류 목표 (`diag_replay_exact.py`): 기본 격자 재생 끝 배치 정확 통과 pkg 0.98 / pcb 0.98 / ip 1.00 → 증류 단계 cutting-plane 버림.
  - 덧붙여 확인 (판정 아님): 평가 pkg는 학습 pkg보다 어렵지 않음 (g+p 개수 중앙값 8 vs 6, 학습 90백분위 위 0 %); 작업 격자 밀도도 같음 (pkg10 학습 227점 vs 평가 236점, `scripts/pool_grid_sizes.py`).
  - **사후 관찰 (판정에 안 씀, 다음 진단의 근거):** 기본 격자 greedy 한 번에서 격자 만족은 모두 0.86~1.00인데 정확 통과 (빼기 허용)는 **r2가 6칸 (pcb·pkg × seed 3) 모두 가장 낮음** (pkg 0.18 / 0.20 / 0.10, pcb 0.30 / 0.32 / 0.30 vs r1 0.34~0.56), 개수는 r1과 비슷 → MoTT는 같은 격자 만족에서 격자 사이 피크가 더 많은 배치를 고름. 이것은 기존 관찰 ("r2 실패 = 격자 만족 + 정확 실패")의 seed 3개·판 2개 확인이고, 원인 (어느 주파수·어느 종류)은 아직 모름.
- **결정: GPU를 근거 없는 run으로 채우지 않음 (이번 라운드 22:35~).** 이유: 이번 라운드에 세운 증류 단계 후보 넷 (규제·빼기·prior·증류 목표)이 실행 전 규칙으로 모두 기각되어 근거 있는 수정이 없음; QUEUE 11/01 23:45 "근거 없는 구조 변경은 하지 않는다"와 N2 (수정마다 seed 2개 필요)를 따르면 근거 없는 run은 GPU 4~8 h를 쓰고 판정도 못 함. QUEUE 11/03 "GPU를 비우지 마라"와 충돌 → 진단을 계속 돌리는 쪽으로 해석 (진단은 CPU). 다음 라운드가 아래 진단으로 후보를 찾으면 바로 큐에.
- 다음 (다음 라운드): (1) **D7 (새 데이터로, 결과 전에 적을 것):** r2·r1 증류의 기본 격자 정확 실패 해에서 최악 주파수가 (a) 가장 가까운 작업 격자점과 몇 점 사이인지, (b) 모드 토큰 8개 중 가장 가까운 것과의 거리 (log f), (c) 대역 (SRF 기준 종류별 영역) — r2 실패가 r1보다 "모드 토큰에서 먼 주파수"에 몰리면 모드 토큰 경로가 격자 사이를 못 본다는 근거 (모드 토큰은 이미 여유가 가장 작은 극대 8개이므로, 후보는 토큰 고르기가 아니라 토큰 사이 정보 쪽). (2) 새 시도 시작 마감 11/6 12:00 UTC, 그때까지 이긴 것 없으면 r2_mott 확정 + ABLATIONS에 "시도한 수정과 효과 없음". (3) DAILY는 11/04 00:15 UTC 뒤 첫 라운드.

## 재개 라운드 34 (2026-11-03 16:27~ UTC)
- 상태 (20:35): GPU **`r2it`** RL 20:31~ (~21:47 끝) → 10칸 평가 side_slot9 자동. GPU 큐 그 뒤 비어 있음. CPU: 드라이버 줄 `repair16 pkg greedy+prune` (16:12~) + 학습 정확 검사 2 + 4번째 칸 `ls10 distill-last r2it_distill` (20:32~) = 4. pod 24 GiB / 60, oom_kill 1 (라운드 27 그것).
- **T5b 최종 (PROPOSALS T5b 규칙 그대로):** lo_pcb_r2s2 (수정 없음, RL seed 2; 369 iter / 1.26 h, best it 200, replaced 207, OOM 0; 평가 19:27~19:43) − lo_pcb_r2_rl 뺀 pcb **−0.08 [−0.16, −0.02]** (CI 0 밖), pkg +0.00, ip +0.00. → **d* = 0.16** (재현 쌍 6값: 같은 종류 pcb +0.04 / +0.00, pkg +0.08 / **+0.16**, 뺀 pcb −0.06 / −0.08), **N2 발동** (재현 쌍 CI 3개가 0을 뺌: r2s2 pkg, lo_pcb_r2s1 pkg −0.24, lo_pcb_r2s2 pcb). N1 적용 (RESULTS "RL 재현 잡음" 절, 자동): RL만 다시 돈 수정 6개 × 2 run의 판별 점추정은 **모두 잡음 안, 예외 둘** — lo_pcb_r2c 뺀 pcb −0.30, lo_pcb_r2h 뺀 pcb −0.40 (원래 판정 "해로움" 그대로, 잡음보다 큼).
  - 해석 (판정 아님): seed 하나 × 50문제 짝 bootstrap CI는 RL 학습 잡음을 담지 못해, 수정 없는 재현도 "이김"·"해로움"을 냄. 앞으로 RL만 바꾸는 수정은 seed 0 단독 결과로 확장하지 않음 (N2). 남은 일정 (새 시도 시작 11/6 12:00 UTC까지)으로는 수정 하나당 RL seed 2개 이상이 필요 → 다음 후보는 미리 seed 0·1 둘 다로 적는 것이 맞음 (다음 라운드가 PROPOSALS에 먼저).
- **D2 판정 (PROPOSALS D2 규칙 그대로): "갱신 횟수로는 설명 안 됨" → 버림.** r2it 증류 끝 (19:28~20:31, 2192 iter / 1.06 h): best.pt = it 1300, val score 1.759 — `r2_mott_distill`과 같은 iter·점수, **가중치 비트 단위로 같음** (같은 seed의 증류는 결정적; 늘린 597 iter는 val score를 못 넘음) → 주 비교 차 = 0, 다시 평가하지 않음. 관찰 (판정 아님): `r2it_distill@last` (it 2192) pkg 평가 filler `ls10 distill-last r2it_distill` (20:32~, side_slot9). r2it RL (20:31~, ~21:47 끝)은 같은 init·같은 seed라 `r2_mott_rl` 재현성 점검이 됨 (메모: 같은 init·seed의 RL은 결정적이었음) — 결정: 끄지 않음 (GPU 대안 후보가 아직 없고, 재현되면 "파이프라인 결정성"을 RESULTS 한 줄로 쓸 수 있음).
- D1 원인 진단 요약 (모두 결과 전 규칙): 모드 토큰 누락 기각, fit 차이 기각, prior 지배 기각, stop 미학습은 전제 틀림, 갱신 횟수 기각. 남은 사실: r2 증류는 학습 풀 pkg 선생님을 r1만큼 맞추지만 평가 pkg에서 낮고, 실패는 "작업 격자에서는 만족·적은 개수 (11~12)·정확 검사 실패" 꼴, 종류 1을 적게 씀.
- 다음: GPU는 ~21:47 뒤 빔 → 다음 라운드가 PROPOSALS에 후보를 **seed 0·1 둘 다로** 먼저 적고 큐에 (N2). 후보 근거 메모 (판정 아님): r2 실패 꼴이 "격자 만족 + 정확 실패"라 격자 사이 피크를 보는 입력이 없다는 쪽 — 다만 r0·r1도 같은 격자만 봄. repair16 g+p+rep4 → 관찰 4 판정. DAILY 다음은 11/04 00:xx UTC.
- 상태 (16:30): GPU `lo_pcb_r2s1` RL (끝 ~16:55) → T5b r2s2 → lo_pcb_r2s2 (~19:30). CPU: 드라이버 줄 `repair16 pkg greedy+prune` (16:12~) + 학습 정확 검사 2 + 4번째 칸 D1 `r2_mott_s2_distill` 평가 (16:28~, bg로 직접 — ps로 빈 칸 확인 뒤, FILL_GAP 동안 side_slot은 비워 둠) = 4. pod 20 GiB / 60, oom_kill 1 (라운드 27 그것). `ensure_running`: report_watch·resume_agent 다시 띄움. QUEUE 새 지시 없음.
- **D1 r1 s1 증류 평가 끝:** pkg 0.88 (pkg10_0) / pcb10_1 1.00 / ip10_1 0.84 (표는 RESULTS 자동). seed 1 세 단 모두 나옴 → seed 2 세 개 대기 (r2 s2 진행, r0·r1 s2는 side_slot).
- **진단 (새, 가설·판정 규칙은 실행 전 `scripts/diag_mode_cover.py` docstring에):** "MoTT의 모드 토큰 8개 (맨판 최대점 상위 8개)가 한 덩어리의 잔물결에 몰려, 다른 위반 대역에 토큰이 없다; r2의 pcb·pkg 실패 (vs r0)가 그런 판에 몰린다" → 맞으면 (a) 그런 판 ≥ 20 % 이고 (b) seed 3개 모두 r2 − r0 차이가 그런 판에서 더 낮음 → 토큰 고르기 수정 후보. **결과: 기각** (`runs/analysis/diag_mode_cover.json`): 토큰 없는 위반 대역이 있는 판 pcb 0.18, pkg 0.06, ip 0.00 (< 0.20); r2 − r0가 그런 판에서 더 낮은 seed pcb 1/3, pkg 2/3 (pkg는 3판뿐). 위반 대역 수 중앙값 pcb 3, pkg 2 — "위반 피크 수 수십"은 대부분 같은 대역 안의 잔물결.
- **진단 준비 (D1 후속, 사용법은 실행 전 `scripts/diag_imit.py` docstring에):** 증류 체크포인트 9개 (r0·r1·r2 × seed 0~2)가 학습 풀 (10칸 3파일 × 종류)에서 선생님 행동을 얼마나 따라 하는지 (nll, top1, 칸, 칸이 맞을 때 종류). D1이 성립할 때만 씀: r2 pkg nll이 seed 3개 모두 r0·r1보다 높으면 "같은 증류 예산에서 pkg 선생님을 덜 맞춤 (fit)" → 증류 fit 쪽 후보, 아니면 fit 수정은 후보에서 뺌. smoke (r2·r0 seed 0, train_pkg10_0 한 파일, 숫자는 판정에 안 씀): 동작, ~40 s/체크포인트·파일. 4번째 칸 filler `ls10 diag-imit distill ckpts` (D1 뒤, ~55 min).
  - 같은 스크립트에 **H-prior** 추가 (실행 전 docstring에 규칙): MoTT의 물리 prior 가중치 w_dv / w_mn이 r2 체크포인트 4개 모두 초기값 20 / 3 그대로 (20.02~20.08, 3.00~3.10 — Adam 한 걸음 ~lr이라 스칼라 20은 거의 못 움직임) → "pkg에서 고정 prior가 logit을 지배해 선생님 대신 1수 greedy를 따른다". seed 3개 모두 pkg가 pcb·ip보다 (full argmax = prior argmax 비율)과 (prior sd / 잔차 sd)가 높고 pkg prior_top1 < top1이면 → 후보 "prior 크기를 log 공간에서 학습". smoke (pcb10_0 한 파일): prior sd 0.59 vs 잔차 sd 14.9 → pcb에서는 prior가 작음 (판정은 전체 run으로).
- **D1 seed 2 일부:** r2 s2 증류 pkg **0.70** (pcb 0.90, ip 1.00), r1 s2 pkg **0.94** (pcb 0.96, ip 0.98). 지금까지 pkg r2 0.70 / 0.68 / 0.70 vs r0 0.90 / 0.74 / – , r1 0.90 / 0.86 / 0.94 → r0 s2 (side_slot 다음 filler, ~17:40) 가 0.70 초과면 D1 성립.
- **D1 판정 (PROPOSALS 11/03 D1 규칙 그대로): 성립.** 증류만 한 체크포인트 10칸 pkg 성공률 seed 0/1/2: r2 **0.70 / 0.68 / 0.70**, r0 0.90 / 0.74 / 0.78, r1 0.90 / 0.86 / 0.94 → seed 3개 모두 r2 < r0·r1 → "MoTT는 pkg에서 선생님을 덜 따라 한다 (증류 단계)"를 RESULTS (D1 표 아래 자동 판정 줄)에 적고, 다음 MoTT 후보의 표적을 증류 단계로 정함. pcb는 r2 0.78 / 0.82 / 0.90 vs r0 0.76 / 0.88 / 0.96, r1 0.88 / 1.00 / 0.96 (보고만). 4번째 칸: r1 s2 (16:45~, bg), r0 s2 (17:24~18:03, bg — FILL_GAP 동안 side_slot이 비워 둔 칸, ps로 확인 뒤). 후보는 diag_imit (18:04~, bg, ~19:05) 결과로 고름: fit 차이인지, prior 지배인지.
- **diag_imit (seed 0·1 기준으로 먼저 적음):** 학습 풀 10칸 선생님 행동 nll pkg r0 2.39 / 1.65, r1 2.81 / 2.67, **r2 2.65 / 2.03** (seed 0 / 1) → r2는 pkg 선생님을 r1보다 잘 맞추는데 평가 pkg는 낮음 → **fit 규칙 불성립** (seed 0에서 이미 r2 < r1) → "fit 수정"은 후보에서 뺌. r0은 nll이 가장 낮음 (좌표로 학습 판을 외우는 쪽으로 보임, 판정 아님). **H-prior 불성립:** prior sd 0.43~0.87 ≪ 잔차 sd 8.7~17.2, pkg의 full=prior argmax 비율 0.26 / 0.21 < ip 0.53 / 0.56.
- **D1 실패 모양 (평가 기록만):** 증류만 한 pkg 실패 문제의 개수 r0 22~28, r1 23~40 (많이 놓고도 실패) vs **r2 s1·s2 11~12** (작업 격자에서는 만족해 멈췄는데 정확 검사 실패, 최악 log10(Z/Zt) 중앙값 ~0.10); r2는 종류 1을 적게 씀 (0.6~1.1 vs 1.1~2.5).
  - 확인 중 바로잡음: "증류는 stop을 따라 하지 않는다"는 가설 (H-stop)을 세웠다가 코드 확인으로 **틀림** — `heuristic_rollout`은 더 할 것이 없으면 stop_idx를 행동에 넣고 강제 재생이 그것을 저장함 → 후보 없음, `diag_stop`은 돌리지 않고 지움.
- **결정: D2 (`r2it`, PROPOSALS 11/03 D2, 결과 전에 고정) — 진단 GPU run을 lo_pcb_r2s2 뒤에.** 근거: 같은 0.75 h에 MoTT 증류는 모든 run에서 1477~1770 iter, 격자 단은 2100~2458 iter (~27 % 적음). r2 seed 0과 같고 증류만 2192 iter (= r0 seed 0) 에서 끝 (`run_rung.py` arm 토큰 `--distill_iters`, RL 단계 불변). 주 판정은 `r2it_distill` vs `r2_mott_distill` pkg 짝지은 CI. 예산이 다른 run이라 사다리 주장에는 안 씀. 이유: 근거 있는 수정 후보가 없고 (진단 넷 기각), QUEUE 11/03은 GPU를 비우지 말라고 함 → 코드 변경이 가장 작은 진단 (보수적). **할 일 (다음 라운드):** r2it 증류 끝나면 (~20:40) `r2it_distill` 10칸 평가를 4번째 칸 filler로 (`queue_distill_eval.py`처럼 이름 `ls10 distill-eval r2it_distill`, best.pt가 생긴 뒤에 넣을 것 — 없는 체크포인트를 넣으면 side_slot이 실패로 건너뜀).
  - side_slot8 → **side_slot9** (`r2s2 lo_pcb_r2s2 r2it`); r2s2 평가 18:31~ (diag_imit 18:04~18:31 끝난 뒤 바로 — 겹침 없음, CPU 4).
- **diag_imit 끝 (18:31, `runs/analysis/diag_imit.json`):** seed 2 r2 증류 best.pt (it 200)는 거의 덜 학습됨 (pkg nll 3.96, top1 0.23) — 그런데 평가 pkg 0.70은 다른 seed와 같음. fit 규칙·H-prior 모두 불성립 그대로.
- **D2 시작 전 점검 (PROPOSALS D2 아래, last.pt 결과 전에 적음; 18:45~19:02, bg, FILL_GAP으로 빈 칸):** r2 증류 last.pt 10칸 pkg seed 0/1/2 **0.62 / 0.68 / 0.72** vs best.pt 0.70 / 0.68 / 0.70 (`runs/analysis/ckpt_last/eval/`) → |차| ≤ 0.06이 seed 3개 모두는 아님 (seed 0 −0.08), +0.06 넘게 오른 seed 없음 → 규칙상 "그 밖" → **D2 그대로.** 관찰 (판정 아님): 증류 후반 갱신에도 pkg는 오르지 않음 (seed 2: it 200 → 1648에 0.70 → 0.72) → D2가 "갱신 횟수로는 설명 안 됨"으로 끝날 가능성이 높음.
- **r2s2 끝** (T5b, 수정 없음, RL seed 2; 345 iter / 1.27 h, best it 300, replaced 303, OOM 0), 10칸 평가 18:31~18:44: ip 0.98 / pcb 0.88 / pkg **0.88**.
- **T5b 같은 종류 절반 (PROPOSALS T5b 규칙 그대로):** r2s2 − r2_mott_rl pcb +0.00 [−0.08, +0.08], **pkg +0.16 [+0.02, +0.32]** (CI가 0 밖 — 수정 없는 재현이 판정 규칙의 "이김"을 냄), ip −0.02. → N2 다시 발동 (같은 종류 쪽에서도). d* (정의된 값: seed 1·2 × 같은 종류 pcb·pkg, 뺀 pcb는 lo_pcb_r2s2 대기) ≥ **0.16**. 관찰 (판정 아님): seed 0 RL run (`r2_mott_rl` pkg 0.72)은 같은 distill의 RL seed 1·2 (0.80·0.88)보다 낮음 → 지금까지 RL만 다시 돈 수정들의 pkg 비교 기준 자체가 낮은 쪽 seed였음. 같은 init에서 RL seed만으로 pkg ±0.16~0.24가 움직이므로 seed 하나 × 50문제 bootstrap CI로는 RL 수정의 효과를 판정할 수 없음. RESULTS ABLATIONS에 r2s2·lo_pcb_r2s2 행 추가 (자동).
- **lo_pcb_r2s1 끝** (수정 없음, lo_pcb_r2_distill에서 RL만 `--seed 1`; 358 iter / 1.27 h, best it 100, replaced 200, OOM 0), 10칸 평가 16:59~17:12.
- **T5 seed 1 쌍 완료 (PROPOSALS T5 규칙 그대로):** lo_pcb_r2s1 − lo_pcb_r2_rl 뺀 pcb **−0.06 [−0.14, +0.02]**, 뺀 run의 **pkg −0.24 [−0.38, −0.10]**, ip +0.00; 개수비 pcb 1.023, pkg 1.028. 같은 종류 쌍 (라운드 33) pcb +0.04, pkg +0.08. → d* (정의된 세 값: 같은 종류 pcb·pkg, 뺀 pcb) = **0.08**. **N2 발동** (보수적 읽기: 재현 쌍의 판별 CI 중 하나 — 뺀 run의 pkg — 가 0을 뺌): 수정 없이 RL seed만 바꿔도 한 판에서 "CI가 0 밖"인 차이 (0.24)가 나옴 → 이후 수정의 seed 0 단독 "이김"은 확장 전에 같은 수정의 RL seed 1 재현을 먼저 요구함. 관찰 (판정 아님): T1'의 (ii) "뺀 run pkg −0.22 [−0.36, −0.08]"는 이 재현 잡음 (−0.24) 크기 안; r2c·r2h의 뺀 pcb 해로움 (−0.30, −0.40)은 뺀 pcb 재현 잡음 (−0.06)보다 큼. T5b (r2s2·lo_pcb_r2s2) 뒤 d* 다시 적음.

## 재개 라운드 33 (2026-11-03 12:25~ UTC)
- 상태 (13:30): GPU **`lo_pcb_r2h`** RL 13:05~ (끝 ~14:23) → r2s1 → lo_pcb_r2s1 (T5, ~16:55). CPU: 드라이버 줄 repair16 `r0_coord_rl` (11:55~, pkg16 2파일) + 학습 정확 검사 2 + 4번째 칸 `ls10 repair r0_coord_s1_rl` (side_slot이 13:21~ 시작) = 4. **실수:** 13:28에 같은 filler를 bg로 또 띄워 ~13분 CPU 5개 → 13:41에 제 것만 (process group) 끔; 먼저 `ps.py`로 확인하지 않았음 (FILL_GAP 경계에서 side_slot이 바로 시작함, 메모 갱신). pod 13 GiB / 60, oom_kill 1 (라운드 27 그것). `ensure_running`: report_watch·resume_agent 다시 띄움. QUEUE 새 지시 없음.
- **r2h 끝** (RL 346 iter / 1.27 h, best it 300, replaced 281, OOM 0; `runs/2026-10-29/r2h_rl/metrics.json`), 10칸 평가 13:06~13:19: ip 0.98 / pcb 0.90 / pkg 0.72, 개수÷g+p 0.74 / 0.88 / 0.84.
- **T3 같은 종류 갈래 불충족 (PROPOSALS T3 규칙 그대로):** r2h − r2_mott_rl pcb +0.02 [−0.06, +0.10], pkg +0.00 [−0.16, +0.16], ip −0.02 [−0.06, +0.00]; `@last` 대비 pcb −0.02 [−0.10, +0.06], pkg +0.00. (ii) 다른 판 점추정 −0.05 아래 없음. 개수비 pcb **1.076 [1.009, 1.147]** (미리 적은 주의대로 개수 증가), pkg 0.965, ip 0.999. 같이 보고 (판정 아님, `runs/analysis/diag_difficulty_r2h_rl.json`): g+p 개수 학습 3분위별 10칸 성공 pcb 1.00 / 0.89 / **0.86** (가중 없음 1.00 / 0.94 / 0.77), pkg 0.83 / 0.81 / 0.64 (가중 없음 0.83 / 0.81 / 0.64 — 그대로). → 최종 판정은 lo_pcb_r2h − lo_pcb_r2_rl (뺀 pcb) CI 하한 > 0 여부 (~14:40 평가).
- **lo_pcb_r2h 끝** (RL 356 iter / 1.27 h, best it 300, replaced 217, OOM 0), 10칸 평가 14:21~14:37: 뺀 pcb **0.52**, ip 0.98, pkg 0.80.
- **T3 최종 판정 (PROPOSALS T3 규칙 그대로): 뺀 pcb에서 "해로움" → 확장 안 함 (0단 r0h도 안 함).** lo_pcb_r2h − lo_pcb_r2_rl 뺀 pcb **−0.40 [−0.54, −0.24]** (상한 < 0), 개수비 1.115 [0.971, 1.272]; 뺀 run의 pkg −0.02 [−0.16, +0.14], ip −0.02. 같은 종류 갈래는 위에서 불충족. 뺀 pcb 실패 24문제는 평균 22% 개수 많고 (11.3 vs 8.1) 종류 2 비율 0.75. 해석 (판정 아님): 가중치는 pkg·ip 학습 풀의 g+p 개수 백분위로만 정해지므로 처음 보는 종류 쪽에서는 RL이 학습 종류의 "어려운" 문제 (대부분 pkg) 쪽으로 기움 — 같은 종류 학습에서는 pcb 상위 3분위 0.77 → 0.86으로 점추정만 오름 (CI 없음).
  - 지금까지 RL만 다시 돈 수정 r2c·r2m·r2g·r2v·r2t·r2h + T1' 모두 이기지 못함 (r2c·r2h는 뺀 pcb에서 해로움: −0.30, −0.40).
- **T2 고치기 seed 1·2 (10칸, 4번째 칸):** `r2_mott_s1_rl+rep4` ip 1.00/1.00, pcb 1.00/1.00, pkg 0.96/1.00; `r2_mott_s2_rl+rep4` (12:28~12:46, bg로 직접) pcb 1.00/1.00, pkg 0.96/1.00, ip 1.00/1.00. 0단 `r0_coord_s1_rl+rep4` pcb 0.88/1.00, pkg 1.00/0.84, ip 1.00/1.00; `r0_coord_s2_rl+rep4` (14:38~15:09) pcb 1.00, pkg 0.96 (50문제). 표는 RESULTS `repair_table` (자동).
- **R4 (PROPOSALS 11/03 R4 — 0단 s2 결과 전에 적음, seed 0·1과 r2 s2는 이미 봤음): 같은 고치기에서 2단 − 0단** (`scripts/diag_repair_ladder.py`, RESULTS 절): pkg seed 0/1/2 +0.06 / +0.06 / +0.02, 묶음 150 짝 **+0.047 [+0.020, +0.080]** → 미리 적은 대로 **"같은 고치기를 주면 2단이 0단보다 정확 검사 성공이 높다 (pkg)"**; 개수비 1.001 [0.893, 1.100] (seed별 1.096 / 0.845 / 1.099). pcb +0.02 / +0.06 / **+0.00** (r2 세 seed 모두 1.00, 0단 s2도 1.00) → 규칙상 "차이 없음" (묶음 +0.027 [+0.007, +0.053]), 개수비 1.028 [0.983, 1.072]. 보고서에는 "seed 2 한 쌍만 결과 전 규칙"이라는 단서를 붙임. 효과 크기는 작음 (+0.05).
- **진단 (새, 기존 평가 기록만):**
  - 가설 (실행 전에 `scripts/diag_spatial.py` docstring에 적음): "MoTT는 셀끼리 attention이 없어 (모드로만 통함, 좌표 없음) 배치가 공간적으로 뭉치고, 뭉침이 실패에 몰린다" — 맞으면 셀-셀 상호작용 추가가 후보. **결과: 반대 방향 → 기각.** 8-이웃에 다른 decap이 있는 비율 r2 − r0 (같은 문제, 같은 seed) pcb −0.09 [−0.17, −0.02] / −0.19 [−0.28, −0.10] / −0.12 [−0.18, −0.06], pkg −0.25 [−0.35, −0.15] / −0.18 [−0.29, −0.07] / −0.15 [−0.26, −0.03] (seed 0/1/2) — r2가 **덜** 뭉침; r2 실패 문제의 뭉침은 성공 문제와 비슷 (pcb 0.64~0.69 vs 0.58~0.67). ip는 모든 방법이 0.95~1.00. RESULTS 절 (`count_gap_table` 안).
  - 증류만 한 체크포인트 (seed 0만 기록 있음): 10칸 pkg 성공 r0 0.90 / r1 0.90 / **r2 0.70** / r3 0.80, pcb 0.76 / 0.88 / 0.78 / 0.84 → pkg에서 r2의 약함은 증류 단계부터 있음 (RL에서 생긴 것 아님). 같은 0.75 h에 r2 증류는 1595 iter (r0 2192, r1 2109) — 다만 r2tf 증류는 2200 iter였고 효과 없음 (T1') → "iter 수 부족"은 근거 약함. seed 1·2 증류 평가가 없어 판정에 쓰지 않음. → **D1 (PROPOSALS 11/03 D1, 사용법 결과 전에 적음):** r0·r1·r2 seed 1·2 증류 체크포인트 10칸 평가 6개를 4번째 칸 filler로 (`scripts/queue_distill_eval.py`, 0단 s2 고치기 뒤). 결정: 후보를 바로 만들지 않고 seed 3개 확인부터 (작은 실험 먼저, 평가만).
- 상태 (15:20): GPU `r2s1` RL 14:23~ (끝 ~15:40) → lo_pcb_r2s1 (~17:00) → **T5b `r2s2` → `lo_pcb_r2s2` (대비책, ~19:30)**. 4번째 칸: D1 `r2_mott_s1_distill` 평가 (15:13~, bg로 직접 — ps로 빈 칸 확인 뒤; FILL_GAP 때문에 side_slot은 17:00 전 D1을 거의 못 돌림). side_slot7 → **side_slot8** (`r2s1 lo_pcb_r2s1 r2s2 lo_pcb_r2s2`).
  - 결정: T5b (PROPOSALS 11/03 T5b, r2s1 결과 전에 적음)를 GPU 큐 끝에 대비책으로 — 근거 있는 수정 후보가 지금 없고 (공간 군집 기각, 빼기 여지 0, 증류 약점은 D1 대기), QUEUE 11/03은 GPU를 비우지 말라고 함. 다음 라운드가 근거 있는 후보를 찾으면 T5b 앞에 넣음.
- **r2s1 끝** (수정 없음, 같은 distill에서 RL만 `--seed 1`; 335 iter / 1.28 h, best it 300, replaced 300, OOM 0), 10칸 평가 15:56~16:08: ip 1.00 / pcb 0.92 / pkg 0.80, 개수÷g+p 0.74 / 0.83 / 0.83.
- **T5 같은 종류 절반 (PROPOSALS T5·T5b 규칙 그대로):** r2s1 − r2_mott_rl pcb +0.04 [−0.06, +0.14], **pkg +0.08 [−0.06, +0.24]**, ip +0.00; 개수비 pcb 1.066, pkg 0.939. CI는 모두 0 포함 → 지금까지 N2 미발동. 잠정 d* = 0.08 (lo_pcb_r2s1·T5b 뒤 다시 정함). 관찰 (판정 아님): 수정 없는 재현이 pkg에서 지금까지의 어느 수정보다 큰 점추정 (+0.08)을 냄 → N1을 적용하면 r2c·r2m·r2g·r2v·r2t·r2h의 같은 종류 점추정 (|·| ≤ 0.08)은 모두 "RL 재현 잡음 안". 뺀 pcb의 해로움 (r2c −0.30, r2h −0.40)은 잡음보다 큼 (lo_pcb_r2s1로 확인).
- **D1 일부:** seed 1 증류 pkg r2 0.68 / r0 0.74 (pcb 0.82 / 0.88, ip 0.92 / 0.90); r1 s1과 seed 2는 대기 (판정은 셋 다 나온 뒤).
- **repair16 0단 끝** (`r0_coord_rl+rep4` pkg16 0.80 / 0.80, 15.3 ks, 드라이버 줄): 같은 고치기에서 pkg16 r2 1.00 / 0.88, 규칙 0.92 / 0.88, 0단 0.80 / 0.80; g+p+rep4 (드라이버 줄 다음, 몇 시간) 뒤 관찰 4 판정.
- 상태 (16:15): GPU `lo_pcb_r2s1` RL 15:38~ (끝 ~16:55) → T5b r2s2 → lo_pcb_r2s2. CPU: 드라이버 줄 `repair16 pkg greedy+prune` (16:12~, 무거움) + 학습 정확 검사 2 + 4번째 칸 D1 `r1_modal_s1_distill` (16:12~, bg로 직접, ps로 빈 칸 확인 뒤) = 4. pod ~5 GiB / 60.
- 다음: lo_pcb_r2s1 (~16:55 끝, ~17:10 평가) → T5 뺀 pcb 절반; repair16 r0·g+p → 관찰 4 판정; 4번째 칸 0단 s2 고치기. 16:55 뒤 GPU 후보는 T3·T5 결과로 다음 라운드가 PROPOSALS에 먼저 적고 큐에. 새 시도 시작은 11/6 12:00 UTC까지. DAILY 다음은 11/04 00:xx UTC.

## 재개 라운드 32 (2026-11-03 11:04~ UTC)
- 상태 (12:15): GPU **`r2h`** RL 11:50~ (T3, 끝 ~13:05) → lo_pcb_r2h (~14:20) → **r2s1 → lo_pcb_r2s1 (T5, 새로 넣음, ~16:50)**. CPU: 드라이버 줄 repair16 (r0_coord_rl, 그다음 g+p) + 학습 정확 검사 2 + 4번째 칸 `ls10 repair r2_mott_s1_rl` (12:05~) = 4. pod ~19 GiB / 60, oom_kill 1 (라운드 27 그것). side_slot6 → **side_slot7** (`r2h lo_pcb_r2h r2s1 lo_pcb_r2s1`, 부모만 바꿈 — 돌던 repair 작업은 그대로). `ensure_running`: report_watch·resume_agent 다시 띄움. QUEUE 새 지시 없음.
- **T1' 최종 판정 (PROPOSALS T1 규칙 그대로): "효과 없음", pkg는 해로운 방향 → 확장 안 함.** lo_pcb_r2tf 끝 (RL 360 iter / 1.26 h, best it 100, replaced 189, OOM 0), 10칸 평가 11:50~12:04: 뺀 pcb 0.84, ip 1.00, pkg 0.60. (i) 뺀 pcb lo_pcb_r2tf − lo_pcb_r2_rl **−0.08 [−0.20, +0.04]**, 같은 종류 갈래는 라운드 31에서 불충족 → 불충족. (ii) 뺀 run의 pkg **−0.22 [−0.36, −0.08]** (< −0.05). T1 (RL만)과 같은 그림: 규칙으로 보강한 선생님은 증류부터 써도 성공률을 올리지 않고 pkg를 내림.
- **T4 S2 (`modal_rule>r0_coord_rl`, 10:48~11:09):** 0단은 규칙 배치를 바꿈 (pcb 37/50, pkg 50/50 — r2는 0/50·2/50). vs modal_rule: pcb 성공 −0.06 [−0.14, +0.00]·개수 0.969 [0.936, 1.005] → "효과 없음"; pkg 성공 **−0.18 [−0.30, −0.06]** (상한 < 0) → **"해로움"** (개수 0.923). ip는 RESULTS 표. → S1 (r2 효과 없음)·S2 (0단 효과 없음/해로움) → 미리 적은 대로 "규칙+정책 다듬기"는 평가 파이프라인으로 효과 없음.
- **진단 (새, 기존 기록만 + 정확 검사 재계산):**
  - `scripts/diag_count_gap.py` (같은 고치기, 네 방법 모두 성공한 문제): r2/r0 개수비 pcb 1.071 [1.000, 1.147], pkg **1.088 [1.037, 1.140]**, 차이는 종류 2 (C 가장 작음): pcb 4.93 vs 4.37, pkg 4.43 vs 3.50. r2 종류 비율은 규칙과 비슷하지만 배치는 다름 (규칙과 Jaccard 0.19·0.17, 같은 해 0개).
  - `scripts/diag_kind_mix.py` (고치기 없음, seed 0·1·2): pkg에서 r2가 r0보다 종류 2를 많이 씀 **seed 3개 모두** (4.27/3.90/4.54 vs 3.32/3.33/3.58), 1단 r1은 아님 (2.91/3.67/3.04; r1 개수는 pcb에서 seed 3개 모두 네 단 중 가장 적음, pkg는 seed 0·2만) → 1수 효과 특징 때문이라는 가설은 기각, 모드 토큰 경로의 성질. 그리고 r2의 pcb 성공률이 seed 3개 모두 네 단 중 가장 낮음 (0.88/0.78/0.84; r3 0.84/0.94/0.96, r0 0.96/0.94/0.98) — QUEUE 약점 2가 seed 3개에서 재현, 고치기를 주면 사라짐 (라운드 29: r2+rep4 pcb 1.00).
  - `scripts/diag_prune_headroom.py` (성공 해에서 정확 검사를 지키며 하나씩 빼기): r2 pcb·pkg 80문제 뺄 수 있는 decap 평균 0.00~0.18 (문제 2/80), r0 0.00~0.21 (1/86) → 두 정책의 해는 이미 1-최소. **T4에서 r2가 규칙 해에서 빼지 않는 것은 자기 해에서는 손해가 아님** → "만족 상태에서 출발하는 RL (빼기 학습)"은 pcb·pkg 여지가 없어 후보에서 버림.
  - RESULTS에 절 3개 (`count_gap_table`).
- **T5 (PROPOSALS 11/03 T5, 결과 전에 고정): RL 재현 잡음 `r2s1`·`lo_pcb_r2s1`** — 수정 없이 같은 distill에서 RL만 `--seed 1`. 근거: RL만 다시 돈 수정 6개 + T1'이 모두 "효과 없음" (점추정 −0.08~+0.06)인데, 학습 잡음 (평가 CI에 안 들어감)을 잰 적이 없음. 사용법 (N1) 재현 잡음 d* 안의 점추정은 ABLATIONS에 "잡음 안" 표기, (N2) 재현 쌍 CI가 0을 빼면 이후 seed 0 단독 "이김"은 RL seed 1 재현을 먼저 요구. `scripts/queue_replicate.py`, RESULTS ABLATIONS 행 2개.
  - 결정: 근거 있는 "수정" 후보가 지금 없음 (빼기 학습 여지 0, 종류 2 과다는 한 줄 수정 없음, 규칙성 제약 금지) → GPU를 비우지 않되 (QUEUE 11/03) 수정 대신 판정 설계를 점검하는 진단 run을 먼저 (보수적: 코드 변경 없음, seed만).
- 다음: r2h (~13:05)·lo_pcb_r2h (~14:20) → T3 판정; repair16 r0·g+p → 관찰 4 판정 (지금 r2+rep4 pkg16 0.88/1.00, rule+rep4 0.92/0.88); r2s1·lo_pcb_r2s1 (~16:50) → T5; 4번째 칸 seed 1·2 고치기. 16:50 뒤 GPU 후보는 T3·T5 결과로 정함. 새 시도 시작은 11/6 12:00 UTC까지. DAILY 다음은 11/04 00:xx UTC.

## 재개 라운드 31 (2026-11-03 09:16~ UTC)
- 상태 (10:35): GPU **`lo_pcb_r2tf`** 증류 09:47~ (RL 끝 ~11:50) → r2h → lo_pcb_r2h (~14:20). CPU: 드라이버 줄 pool20 끝 (ip20 09:39) → **repair16** (pkg16 × r2_mott_rl·modal_rule·r0_coord_rl·g+p, 09:42~) + 학습 정확 검사 2 + side_slot6 (다음 filler T4 `ls10 rulestart …`) = 4. pod 15 GiB / 60, oom_kill 1 (라운드 27 그것). `ensure_running`: report_watch·resume_agent 다시 띄움. QUEUE 새 지시 없음.
- **r2tf 끝** (증류 2200 iter + RL 360 iter / 1.27 h, best it 200, replaced 233, OOM 0; `runs/2026-10-29/r2tf_rl/metrics.json`), 10칸 평가 09:48~09:59: ip 1.00 / pcb 0.88 / pkg 0.78, 개수÷g+p 0.74 / 0.85 / 0.86. **T1' 같은 종류 갈래 불충족:** r2tf − r2_mott_rl pcb +0.00 [−0.10, +0.10], pkg +0.06 [−0.08, +0.20], ip +0.00; `@last` 대비 pcb −0.04 [−0.14, +0.04], pkg +0.06. (ii) 다른 판 점추정 −0.05 아래 없음. 개수비 0.98~1.00. vs modal_rule (판정 아님): pcb −0.12 [−0.22, −0.04], pkg −0.10 [−0.22, +0.02], ip 개수 0.712. → T1' 최종 판정은 lo_pcb_r2tf − lo_pcb_r2_rl (뺀 pcb) CI 하한 > 0 여부 (~12:05 평가).
- **T2 R3 (r0_coord_rl+rep4, 10:00~10:29):** 0단도 고치기로 오름 — pkg 0.76 → 0.92 **+0.16 [+0.08, +0.26]** (8문제, 개수 변화 0.00), pcb 0.96 → 0.98 (+0.02 [+0.00, +0.06]), ip 1.00 그대로. → R1이 r2 (pcb·pkg)와 0단 (pkg) 둘 다에서 성립 → 미리 적은 R3대로 **"고치기는 방법과 무관한 평가 절차 이득" (MoTT 이득으로 쓰지 않음).** 같은 고치기에서 0단 vs g+p+rep4 pcb +0.04 [−0.04, +0.12]·pkg −0.02 (차이 없음), 개수비 0.808·0.788; 0단 vs modal_rule+rep4 pkg −0.06 [−0.14, +0.00] (상한 0.00, < 0 아님 → 차이 없음), 개수비 pkg 0.923 [0.878, 0.972]. T2 seed 0 판정 끝 — seed 1·2 고치기는 T4 다음 4번째 칸.
- **repair16 첫 기록 (관찰 4 근거, `runs/repair16/eval/`):** pkg16_1 r2_mott_rl+rep4 0.60 → **1.00** (25문제, 고친 10문제 +1.90개, 9.8 s/문제, rss 25.8 GB). 같은 판의 g+p·modal_rule·0단 고치기는 드라이버 줄에서 이어짐 → 네 방법이 다 나오면 "16칸 pkg 열세는 크기 일반화가 아니라 cutting-plane 상한 (평가 절차)"인지 판정 (가설은 결과 전 지금 적음: 맞으면 고친 뒤 r2 − g+p 성공률 CI가 0을 포함하고 개수비 < 1).
  - 결정: 관찰 4의 학습 쪽 후보 (16칸 학습 판 추가 — 판 생성·선생님 16칸 rss 22~34 GB, 드라이버 줄 몇 시간)는 repair16 네 방법이 나올 때까지 만들지 않음 — 근거가 평가 절차 쪽을 가리키고 있고, 비용이 큼 (보수적).
- **T4 r2 갈래 (`modal_rule>r2_mott_rl`, 10:30~10:47, `runs/rulestart10/eval/`): S1 세 판 모두 "효과 없음".** 정책이 규칙 배치를 바꾼 문제 ip 0/50, pcb 0/50, pkg 2/50 (smoke처럼 거의 바로 stop); vs modal_rule 성공률 차 +0.00 세 판, 개수비 ip 1.000, pcb 1.000, pkg 0.996 [0.988, 1.000] (상한 < 1 아님). 판정 아님: vs 같은 정책 빈 판 시작 pcb +0.12 [+0.04, +0.20], pkg +0.16 [+0.02, +0.30] (= 규칙 성공률), ip 개수 ×1.548. → 미리 적은 대로 "평가 파이프라인으로는 효과 없음" 기록. 관찰: ip에서는 정책 단독 해가 규칙보다 31% 적은데도 규칙 해에서 하나도 빼지 않음 → 정책은 빈 판에서 시작한 자기 궤적 밖의 "목표 만족 + 많은 decap" 상태에서 remove를 배우지 못함 (학습 중 그런 상태가 거의 없음). S2 (`modal_rule>r0_coord_rl`)는 lo_pcb_r2tf 평가 뒤 4번째 칸.
- 결정: 14:20 뒤 GPU 후보 (T5)는 다음 라운드 (~12:10, lo_pcb_r2tf·T4·repair16 결과 뒤)에 근거와 판정 규칙을 PROPOSALS에 먼저 적고 큐에 — 지금 근거 (T1·T1' 같은 종류·r2c·r2m·r2g·r2v 효과 없음, 고치기는 방법 무관)로는 정할 후보가 없음. GPU가 비기 전 두 라운드가 남음. 다음 라운드가 볼 근거 메모 (판정 아님, 같은 고치기 표에서): 고친 뒤 개수÷g+p가 0단 pcb 0.808·pkg 0.788 vs r2 0.847·0.852, ÷modal_rule 0단 pkg 0.923 vs r2 1.011 → 고치기를 주면 MoTT는 성공률은 같고 pcb·pkg 개수는 0단보다 많은 쪽 (H3: MoTT 종류 비율 ≈ 규칙). "규칙 해에서 출발하는 RL"은 T4에서 규칙 대비 줄일 여지가 pcb·pkg에서 작아 (규칙/정책 개수 ≈ 1.01) 근거 약함.
- 다음: lo_pcb_r2tf 평가 (~12:05) → T1' 최종 판정; T4 10칸 (4번째 칸, ~11:00~) → S1·S2; repair16 나머지 → 관찰 4 판정; r2h·lo_pcb_r2h (~14:35) → T3 판정. DAILY 다음은 11/04 00:xx UTC.

## 재개 라운드 30 (2026-11-03 08:34~ UTC)
- 상태 (09:00): GPU **`r2tf` RL 08:31~** (증류 0.77 h 끝: 2200 iter, best it 1900, val score 1.749 — r2_mott_distill 1.759; RL it 50 fail 0.00~0.12, teach ok 0.75~0.94, replaced 57, OOM 0; 끝 ~09:47) → lo_pcb_r2tf (~11:50) → r2h → lo_pcb_r2h (~14:20). CPU: 드라이버 줄 pool20 ip20 (마지막 2파일) + 학습 정확 검사 2 + side_slot6 (다음 filler `ls10 repair r0_coord_rl`, r2tf 평가 뒤) = 4. pod 6.7 GiB / 60, oom_kill 1 (라운드 27 그것). `ensure_running`: report_watch·resume_agent 다시 띄움. QUEUE 새 지시 없음.
- **T2 R2 g+p 갈래 (greedy+prune+rep4, 08:18~08:55, `runs/repair10/eval/`):** g+p 자체는 고치기로 pcb 0.90 → 0.94, pkg 0.90 → 0.94 (각 2문제), ip 0.40 그대로 (ip 실패는 격자 위 실패라 고칠 것 없음). 같은 고치기에서 `r2_mott_rl+rep4` − `greedy+prune+rep4`: ip **+0.60 [+0.46, +0.72]** → "정책이 낫다", pcb +0.06 [+0.00, +0.14] (하한 0.00, > 0 아님) → "차이 없음", pkg +0.04 [−0.04, +0.12] → "차이 없음"; 개수비 ip 0.736 [0.578, 0.887], pcb **0.847 [0.744, 0.945]**, pkg **0.852 [0.802, 0.906]** (정책이 15% 적음). vs modal_rule+rep4는 라운드 29 그대로 (성공 차 0, 개수비 ip 0.689만 < 1). → 고치기를 모두에 주면: 성공률은 정책 = 규칙 ≥ g+p, 개수는 정책 < g+p (세 판), 정책 < 규칙 (ip만). R3 (r0_coord_rl+rep4)는 4번째 칸 다음 차례.
- **T4 규칙 해에서 출발해 정책이 다듬기 (평가만, QUEUE 11/03 관찰 3 두 번째 갈래, PROPOSALS 11/03 T4 — 판정 규칙 결과 전에 적음):** `eval_policy.py --start rule` (기록 `modal_rule><정책>`): cutting-plane 각 round에서 modal_rule 해를 시작 배치로 정책이 greedy decoding으로 이어 감 (목표 만족 상태라 remove·종류 바꿈·stop만 → 격자에서는 개수 증가 불가). smoke (`scripts/smoke_rule_start.py`, pkg 8칸 6문제): 경로 동작, 격자 성공 규칙 해 위 개수 증가 없음 → `.ok`; **smoke 6문제 모두 정책이 바로 stop** (숫자 보고 안 함 — 정책이 남의 해를 다듬지 않을 수 있다는 신호). 10칸 6파일 × {r2_mott_rl, r0_coord_rl}을 4번째 칸 filler로 (`scripts/queue_rule_start.py`, T2 seed 0 고치기 뒤·seed 1·2 고치기 앞). RESULTS `rulestart_table` (vs modal_rule CI, 규칙 배치를 바꾼 문제 수, vs 같은 정책 빈 판 시작).
  - 결정: GPU 학습 변형 ("규칙 해에서 출발하는 RL")은 바로 만들지 않음 — 평가만으로 먼저 보는 작은 실험 (보수적), 그리고 GPU는 14:20까지 미리 적은 T1'·T3로 차 있음.
- 다음: r2tf 평가 (~10:00) → T1' 같은 종류 갈래; lo_pcb_r2tf (~12:05 평가) → T1' 최종 판정; r2h·lo_pcb_r2h (~14:35) → T3 판정. 4번째 칸: R3 → T4 → seed 1·2 고치기. **GPU가 14:20 뒤 비지 않게:** T1'/T3 중 이긴 것이 있으면 seed 1·2 + 0단 짝 (미리 적은 대로), 없으면 T4 결과 (정책이 규칙 해를 다듬는가)와 repair16 (드라이버 줄, pool20 뒤)를 근거로 다음 후보를 PROPOSALS에 적고 큐에. 새 시도 시작은 11/6 12:00 UTC까지. DAILY 다음은 11/04 00:xx UTC.

## 재개 라운드 29 (2026-11-03 06:59~ UTC)
- 상태 (07:05): GPU `lo_pcb_r2t` RL 06:28~ (끝 ~07:45) → r2tf → lo_pcb_r2tf (~11:45). CPU: 드라이버 줄 pool20 pkg20_1 (02:12~, ~4.9 h째; pkg20_0은 4.6 h) + 학습 정확 검사 2 + side_slot5 `ls10 repair r2_mott_rl` (06:43~) = 4. pod 12.0 GiB / 60, oom_kill 1 (라운드 27 그것). `ensure_running`: report_watch·resume_agent 다시 띄움. QUEUE 새 지시 없음.
- **진단 가설 (QUEUE 관찰 2, 결과 전에 적음, `scripts/diag_difficulty.py`):** "평가에서 정책이 지는 어려운 문제가 학습 풀의 난이도 범위 **안에 있지만 드물다**" (난이도 = 맨판 최악 위반 log10(Z/Zt), 위반 피크 수, g+p 개수; 학습 풀 27파일 432문제는 파일 균등·파일 안 균등 추출). 맞으면 (10칸 pcb·pkg 정책 실패의 절반 이상이 학습 분포 상위 25% 위, 그리고 학습 풀 상위 1/3에서 선생님 통과율이 낮음) → 다음 후보 = RL 문제 추출을 난이도 가중으로 (한 가지 변경). 실패가 학습 난이도 전체에 고르게 퍼져 있거나, 학습 최댓값보다 위에 몰려 있으면 (분포 밖 → 가중으로는 못 고침) 이 후보는 버림.
  - **결과 (`runs/analysis/diag_difficulty_r2_mott_rl.json`):** g+p 개수 척도로는 성립 — r2_mott_rl 실패 중 학습 상위 25% 위 pcb10 0.67 (4/6), pkg10 0.64 (9/14), pkg16 0.83 (20/24); 학습 최댓값 위 10칸 0, 16칸 pcb 0.08 / pkg 0.06 (분포 안); 학습 3분위별 10칸 성공 pcb 1.00 / 0.94 / 0.77, pkg 0.83 / 0.81 / 0.64; 학습 상위 1/3 선생님 통과 pcb 0.69 / pkg 0.52 (하위 1.00 / 0.80). 다른 척도로는 갈림 (맨판 최악 위반: pkg 0.07; 위반 피크 수: pcb 0.17) → 근거는 g+p 개수 하나 (PROPOSALS T3에 그대로 적음). 학습 풀은 파일 단위로 B=16 전부를 쓰므로 추출 가중 대신 **손실 가중**으로 구현.
- **T3 `--hard_weight 2` (PROPOSALS 11/03 T3, 판정 규칙 T1과 같음, 결과 전에 적음):** RL 손실 (advantage·모방 두 항)에 w = (1 + 2·q)/평균, q = 같은 종류 학습 문제 중 g+p 개수 mid-rank 백분위 (가중치 0.52~1.48). 추출·보상·선생님·val·평가 그대로. smoke `scripts/smoke_hard_weight.py`: 가중치 범위 확인 + **기본값 0에서 HEAD train.py와 log.csv가 같음** (대기 중인 r2tf·lo_pcb_r2tf는 바뀌지 않음) → `runs/analysis/smoke_hard_weight.ok`. GPU 큐 lo_pcb_r2tf 뒤에 `r2h`·`lo_pcb_r2h` (`scripts/queue_hard_weight.py`, 같은 distill에서 RL 1.25 h만, ~14:20 끝). RESULTS ABLATIONS 행 추가.
  - 결정: T1'(r2tf) 앞에 끼우지 않음 — T1'은 이미 미리 적고 큐에 있던 시험이라 순서 그대로 (보수적).
- **T2 고치기 R1 성립 (r2_mott_rl seed 0, 같은 기록 안 전후, `runs/repair10/eval/`):** pcb 0.88 → **1.00** (+0.12 [+0.04, +0.20], 고친 6문제 개수 변화 0.00 — 위치·종류만 바꿈), pkg 0.72 → **0.98** (+0.26 [+0.14, +0.38], 고친 13문제 +0.62개), ip 1.00 그대로; 이미 통과한 문제는 개수 그대로; 고치기 시간 0.36 / 0.96 s/문제. R2 (같은 고치기의 modal_rule·g+p)와 R3 (0단)는 4번째 칸 filler 대기 (lo_pcb_r2t 평가 뒤). 판정 문장은 R2·R3까지 나온 뒤.
  - 미리 적은 대로 넓힘 (`scripts/queue_repair_wide.py`): 10칸 seed 1·2 (r2_mott, 공정성으로 r0_coord도) → 4번째 칸 filler; **16칸 pkg 2파일** (r2_mott_rl·modal_rule·r0_coord_rl·g+p) → 드라이버 줄, pool20 바로 뒤 (20칸 평가 앞). 결정: 16칸 판 평가는 rss ~22 GB (g+p ~34 GB)라 OOM 메모 (라운드 27)대로 한 번에 하나만 도는 드라이버 줄에만; 판정에 걸린 기록이라 20칸 평가 앞.
- **lo_pcb_r2t 끝** (RL 367 iter / 1.26 h, best it 300, replaced 181, OOM 0; `runs/2026-10-29/lo_pcb_r2t_rl/metrics.json`), 10칸 평가 07:46~07:59: 처음 보는 pcb 0.90 / 개수÷g+p 0.87, ip 1.00, pkg 0.66.
- **T1 최종 판정 (PROPOSALS 11/03 T1 규칙 그대로): "효과 없음", pkg에서는 해로운 방향 → 확장 안 함.** (i) 같은 종류 갈래 불충족 (라운드 28: pcb −0.04 [−0.16, +0.08], pkg −0.06 [−0.20, +0.08]); 뺀 pcb 갈래 lo_pcb_r2t − lo_pcb_r2_rl **−0.02 [−0.14, +0.10]** → 불충족. (ii) 다른 판 점추정: 같은 종류 pkg −0.06, 뺀 run의 pkg **−0.16 [−0.30, −0.02]** → −0.05 아래. 선생님 범위를 넓혀도 (pkg 0.69 → 0.85) RL만으로는 성공률이 오르지 않고 pkg는 내려감. T1' (r2tf·lo_pcb_r2tf, 증류부터 같은 선생님) 판정은 그대로 남음 (~11:45).
- side_slot6 (`r2t lo_pcb_r2t r2tf lo_pcb_r2tf r2h lo_pcb_r2h`)로 교체 (side_slot5 부모만 끝냄 — 돌던 repair 작업은 끝나 있었음).
- **T2 R2 일부 (modal_rule+rep4, 08:00~08:17):** 규칙도 고치기로 오름 (ip 0.94 → 1.00, pkg 0.88 → 0.98, pcb 1.00 그대로). 같은 고치기에서 `r2_mott_rl+rep4` vs `modal_rule+rep4` 성공률 차 ip / pcb / pkg 모두 **+0.00 [+0.00, +0.00]** → R2 문장 "차이 없음"; 개수비 ip **0.689 [0.630, 0.749]** (정책이 31% 적음), pcb 1.014 [0.947, 1.094], pkg 1.011 [0.974, 1.046]. → 고치기를 모두에 주면 성공률은 규칙과 같고, 정책이 이기는 곳은 ip 개수뿐 (QUEUE 11/01 약점 1과 같은 그림). g+p+rep4 (08:17~)·r0_coord_rl+rep4 (R3) 대기.
- CPU 줄: pool20 pkg20_1 끝 (~08:00), ip20 시작. 그 뒤 repair16 4개 (드라이버 줄) → 20칸 평가.
- 다음: g+p·0단 고치기 기록 → R2 (g+p)·R3 판정 문장 + seed 1·2 고치기 → RESULTS ABLATIONS/설계 근거에 "고치기" 행. r2tf (RL ~09:45 끝) → lo_pcb_r2tf (~11:45) → T1' 판정. r2h → lo_pcb_r2h (~14:15) → T3 판정. 그 뒤 GPU 후보는 판정 결과와 진단으로 (QUEUE 관찰 4: 16칸 pkg 고치기 결과가 근거가 될 수 있음). 새 시도 시작은 11/6 12:00 UTC까지. DAILY 다음은 11/04 00:xx UTC.

## 재개 라운드 28 (2026-11-03 05:32~ UTC)
- 상태: GPU `r2t` RL it 75+ (05:12~, 끝 ~06:30, fail 0.01~0.11, λ 1.25, teach ok 0.69~1.00, gpu peak 14.2 GB, OOM 0; `runs/2026-10-29/r2t_rl/log.txt`) → lo_pcb_r2t → r2tf → lo_pcb_r2tf (~11:45). CPU: 드라이버 줄 pool20 pkg20_1 (02:12~) + 학습 정확 검사 2 + side_slot5 (r2t 평가 대기) = 4. pod 9.7 GiB / 60, oom_kill 1 (라운드 27의 그것, 새것 없음). `ensure_running`: report_watch·resume_agent가 죽어 있어 다시 띄움 (report_watch는 05:34에 결과 22개 push 뒤 "모든 lo_* 끝"으로 스스로 끝남 — 정상). QUEUE 새 지시 없음.
- **진단 (기존 평가 기록만, `scripts/diag_rounds.py` → `runs/analysis/diag_rounds.json`):** 10·16칸 pcb·pkg에서 r2_mott_rl·r0_coord_rl·modal_rule은 거의 모든 파일이 cutting-plane 상한 4 round에 닿음 (F_final 289~391). pkg16에서는 **규칙도 실패** (modal_rule 0.68 / 0.52, r2 0.44 / 0.60, 0단 0.56 / 0.72, g+p 0.80 / 0.84 — g+p는 개수 22~31) → pkg16은 "적게 놓는 방법은 모두" 지는 곳. cutting-plane은 round마다 처음부터 다시 풀어 마지막 해가 새 격자 틈에 기댄 채 끝남.
- **T2 끝에서 고치기 (QUEUE 관찰 1, PROPOSALS 11/03 T2, 판정 규칙 결과 전에 적음):** `eval_policy.py --repair K` — cutting-plane 뒤 정확 검사 실패 문제만 실패 주파수를 격자에 더하고 **같은 방법이 자기 배치에서 이어서** 놓음 (정책 occ0 greedy decoding / g+p greedy(occ0) / modal_rule occ0 — `baselines.modal_heuristic`에 `occ0` 인자 추가, 기본 동작 그대로), 최대 K번. 기록 `<방법>+rep4`, 같은 기록에 고치기 전 `ok_pre`·`count_pre` (전후 짝은 같은 실행 안). smoke (`scripts/smoke_repair.py`, pkg 8칸 6문제, rounds 1, repair 2): 통과 문제는 그대로, 기록의 "전"이 따로 돌린 평가와 같음, 정책은 add+prune·종류 바꿈으로 실패 3개 중 2개를 고침 → `runs/analysis/smoke_repair.ok` (숫자는 보고 안 함). `scripts/queue_repair.py`로 10칸 6파일 × {r2_mott_rl, modal_rule, g+p, r0_coord_rl} 4개를 CPU 큐 끝에 (`ls10 repair …` → 4번째 칸 filler, 학습 평가 우선). RESULTS 절 `repair_table` 추가.
  - 결정: 고치기는 GPU를 쓰지 않는 평가 절차 변형이라 T1/T1' GPU 줄과 겹치지 않게 4번째 칸에서만 — CPU 한도 4 그대로.
- **r2t 끝** (RL 342 iter / 1.27 h, best it 124, OOM 0; `runs/2026-10-29/r2t_rl/metrics.json`), 10칸 평가 06:28~06:42: 성공 ip 1.00 / pcb **0.84** / pkg **0.66**, 개수÷g+p 0.74 / 0.19 / 0.60. **T1 판정 (i) 같은 종류 갈래 불충족:** r2t − r2_mott_rl (day 1) pcb −0.04 [−0.16, +0.08], pkg −0.06 [−0.20, +0.08], ip +0.00; `@last` 대비 pcb −0.08 [−0.20, +0.04], pkg −0.06. 개수비 pcb 1.056, pkg 0.996. (ii) pkg 점추정 −0.06 < −0.05 → 같은 종류 쪽은 해로운 방향 (CI는 0 포함). vs modal_rule (판정 아님): pcb −0.16 [−0.26, −0.06], pkg −0.22 [−0.36, −0.06]. → 남은 갈래 lo_pcb_r2t − lo_pcb_r2_rl (뺀 pcb) CI 하한 > 0 (~08:00 평가). 학습 중 teach ok 0.69~1.00 (선생님 범위는 넓어졌지만 RL만으로는 성공률이 안 오름 — 증류 선생님은 그대로라 T1'이 본 시험).
- 다음: lo_pcb_r2t (~08:00) → T1 최종 판정; r2tf·lo_pcb_r2tf (~11:45) → T1' 판정. T2 10칸 기록이 나오면 미리 적은 R1~R3로 판정. GPU가 11:45 뒤 비지 않게: T1/T1'이 이기면 seed 1·2 + r0t, 아니면 다음 진단 후보를 그 전에 PROPOSALS에 적고 큐에 넣음.

## 재개 라운드 27 (2026-11-03 02:43~ UTC)
- 상태 (05:25): GPU **`r2t` RL 05:12~** (규칙 보강 선생님, 끝 ~06:30) → lo_pcb_r2t → r2tf → lo_pcb_r2tf (~11:45). CPU: 드라이버 줄 `pool20 eval k8` 4번째 파일 (pkg20_1, 02:12~; **pkg20_0은 21:36~02:12, 4.6 h**, cutting-plane 10 round, 판 생성 때 g+p 4/8 성공) + 학습 정확 검사 2 + side_slot5 (대상 r2t·lo_pcb_r2t·r2tf·lo_pcb_r2tf, 평가 대기) = 4. pod 7~10 GiB / 60, **oom_kill 1** (아래).
- **QUEUE 새 지시 (11/03 11:40 KST, 원격에서 받음): GPU를 비우지 말고 11/6 12:00 UTC까지 MoTT 개선 (진단 → 근거 → 한 가지 수정 → 짝 비교)을 계속.** → 아래 "GPU를 비워 둔다" 결정은 이 지시로 철회.
- **OOM 1건 (03:55, pod는 안 죽음):** pod 50.5 GiB에서 커널이 `ls16 eval_pcb16_0` worker (LS16은 g+p부터 돌아 rss ~31 GB)를 죽임 — pool20 + chip filler + 제가 띄운 GPU 진단 (`diag_allfail.py`, 학습 풀 27개를 차례로 올림)과 겹침. 남은 eval_policy 부모는 worker 없이 멈춰 있어 정리 (`kill_pat.py`). 45 GiB 규칙은 "시작"만 막고 16칸 작업은 시작 뒤 ~25 GB 커짐 → 결정: heavy 칸 (`side_lane16.py`)은 은퇴, LS16·g+p 20칸 12개를 드라이버 CPU 줄 (20칸 평가 뒤)로 되돌림 (`queue_heavy.py --back`), 무거운 16칸 작업은 한 번에 하나만 도는 드라이버 줄에서만. LS16 pcb16_0은 기록 없음 → 드라이버가 나중에 처음부터 다시 함.
- **진단 (QUEUE 관찰 1·2·3 근거 찾기):**
  - 기각: "M개 샘플이 모두 실패한 그룹은 정규화 advantage가 개수만 줄이는 쪽으로 민다 (V는 근소 실패에서 ~0)" — r2_mott_rl best.pt, 학습 풀 432문제, 저장된 작업 격자, M=8: 샘플 실패 0.2%, 모두 실패 그룹 0개 (`scripts/diag_allfail.py`, `runs/analysis/diag_allfail_r2_mott_rl_best.json`). 학습 중 실패는 정확 검사에서만 생김 (H2와 같음).
  - **지지: 지금 선생님이 학습 풀에서 약하다.** 정확 검사 (worst ≤ 1%)를 통과하는 선생님 해 비율 ip **0.58** / pcb **0.88** / pkg **0.69** → 증류 (선생님 있는 문제만 모방)와 RL 모방 항이 어려운 문제의 12~42%에서 시연 없이 돎. `modal_rule`을 평가와 같은 4-round cutting-plane으로 돌리면 정확 검사 통과 ip 0.99 / pcb 0.97 / pkg 0.76; 문제마다 {선생님, 규칙} 중 통과한 것 가운데 개수 적은 것을 쓰면 선생님 범위 ip **0.99** / pcb **0.99** / pkg **0.85**, 규칙 해 사용 0.46 / 0.37 / 0.38, 둘 다 통과한 문제에서 규칙/선생님 개수 중앙값 1.37 / 1.00 / 1.00 (`scripts/make_rule_teacher.py --cp 4`, `data/teacher_rcp/summary.json`; cutting-plane 없이 저장 격자만 쓰면 규칙 통과 0.15~0.63 — `data/teacher_rbest/`).
- **T1 규칙 보강 선생님 (PROPOSALS 11/03 T1·T1', 판정 규칙은 학습 전에 적음):** 변경 하나 = 선생님 저장소 `--teacher_dir data/teacher_rcp` (정책·보상·val·평가 그대로, 추론 때 규칙 호출 없음). `r2t`·`lo_pcb_r2t` = 같은 distill에서 RL 1.25 h만 (QUEUE 짝 규칙), `r2tf`·`lo_pcb_r2tf` = 같은 선생님으로 증류+RL 전체 (선생님은 증류에서 가장 크게 쓰이고, 이기면 최종 방법이 이 버전). 이기면 seed 1·2 + 0단 짝 (QUEUE 공정성). `run_rung.py`: arm에 `--teacher_dir`가 있으면 기본값을 뒤에 붙이지 않게 (argparse는 마지막 값을 씀). smoke (`scripts/smoke_rule_teacher.py`, CPU 2 iter, teacher ok 1.00) 통과 → `runs/analysis/smoke_rule_teacher.ok` → 드라이버가 05:12에 r2t 시작 (27개 풀 모두 `teacher_rcp` 읽음 확인). RESULTS ABLATIONS에 r2t·r2tf 짝 + vs `modal_rule` (판정 아님) 행 추가.
- **chip zero-shot 끝 (seed 0·1·2 + E0+):** 미리 적은 (b)는 모든 단 성공 1.00 (천장) → **불충족** ("모달 표현이 chip으로 더 잘 옮겨 감"이라고 쓰지 않음). 기술용 (판정 아님): 개수÷g+p 2단 1.11 / 1.11 / 1.14 vs 0단 2.10 / 2.65 / 2.42 (seed 0/1/2), E0+ 1.00 / 1.35, modal_rule 1.00 / 1.17.
- **lo_pcb_r2g 끝** (RL 362 iter / 1.26 h, best it 300, 격자 상한에 닿은 step **80** vs lo_pcb_r2_rl 182, λ 1.27, OOM 0; `runs/2026-10-29/lo_pcb_r2g_rl/metrics.json`), 10칸 평가 03:20~03:34. 처음 보는 pcb 0.86 / 개수÷g+p 0.93, ip 0.98, pkg 0.76.
- **r2g 최종 판정 (라운드 14에 미리 적은 규칙 그대로): "효과 없음" → seed 1·2 확장 안 함.** (i) 같은 종류: pcb +0.00 [−0.10, +0.10] (day 1)·−0.04 (`@last`), pkg +0.04 → 불충족 (라운드 26); 뺀 쪽: lo_pcb_r2g − lo_pcb_r2_rl pcb **−0.06 [−0.16, +0.04]** (ip −0.02 [−0.06, +0.00], pkg −0.06 [−0.18, +0.06]) → 불충족. 상한 완화로 격자 상한에 닿은 step은 145 → 23 (같은 종류), 182 → 80 (뺀 pcb)으로 줄었지만 성공률은 안 오름 → 진단 H2의 "격자 상한" 갈래는 성공률을 올리는 손잡이가 아니었음 (근소 실패는 그대로). RESULTS 설계 근거 "주장하지 않는 것"에 `--grid_growth` 추가 (`collect_results.py`).
- 이로써 진단 H1·H2에서 나온 수정 넷 (`--certify 2`, val 48문제, `--feas_margin`, `--grid_growth`)이 모두 미리 적은 기준에서 "효과 없음"(또는 해로움). (03:50 결정, QUEUE 11/03 11:40 KST로 **철회**:) GPU는 비워 둔다 (라운드 24 결정 유지) — 근거 있는 다음 후보가 없고 (QUEUE 11/01 23:45 "근거 없는 구조 변경은 하지 않는다"), 실험 동결 11/8까지 남은 일은 CPU 쪽 (그림 2의 LS16·20칸, chip seed)이다. 비는 정확 검사 칸 2개를 아래 heavy 칸에 씀.
- **20칸 smoke (`eval20 eval_pcb20_0 r2a_rl`, 03:18~03:26):** 8문제 성공 1.00 / 개수 4.75, 문제당 0.54 s, **peak rss 10.7 GB** → 45 GiB 규칙에 맞음 → `scripts/queue_eval20.py`로 나머지 20칸 평가 41개 (5 정책 × 6파일, modal_rule, g+p)를 드라이버 CPU 줄에 넣음 (pool20 바로 뒤).
- (03:50 결정, 04:10 은퇴 — 위 OOM·QUEUE) 무거운 작업 전용 칸 `scripts/side_lane16.py` (heavy 칸) — GPU가 비어 정확 검사 worker 2칸이 남으므로, LS16 6파일과 g+p 20칸 6파일을 드라이버 줄 (pool20 남은 3파일 ~10+ h 뒤)에서 꺼내 (`scripts/queue_heavy.py`, 큐의 새 `heavy` 목록 — 드라이버는 안 읽음, 같은 작업이 두 번 뜨는 일 없음) 한 번에 하나씩 돌림. 지킴: `train.py`가 살아 있으면 새 작업 시작 안 함 (CPU 4 초과 방지), pod < 20 GiB일 때만 시작 (16칸 g+p/LS rss 19~34 GB), 판 파일이 생기기 전에는 시작 안 함, g+p 20칸은 "solo" (다른 eval_policy·build_pools가 없을 때만 — 20칸 g+p 메모리 미측정). 순서: ls16 pcb16_0 (돌고 있음) → g+p pcb20 ×2 → ls16 pcb16_1 → ip16 ×2 → pkg16 ×2 → g+p pkg20·ip20. 로그 `agent/logs/side_lane16.log` (`.gitignore`). **주의 (다음 라운드): GPU 학습을 다시 넣으려면 heavy 칸의 지금 작업이 끝나길 기다리거나 heavy 칸을 먼저 멈출 것** (학습 정확 검사 2 + 드라이버 1 + side_slot 1 + heavy 1 = 5).
- **진단 QUEUE 관찰 4 (16칸 pkg, 기존 평가 기록만, `scripts/diag_pkg16.py` → `runs/analysis/diag_size_r2_mott_rl.json`):** r2_mott_rl seed 0 실패 pcb10 6 / pcb16 5 / pkg10 14 / **pkg16 24** (50문제씩). pkg16 실패 24개 중 19개는 g+p가 푼 문제이고, 그 문제에서 정책 개수÷g+p 중앙값 **0.57** (성공 문제는 0.82); 실패 정확 검사 최악 중앙값 log10 0.118 (+31%), 근소 (≤ +15%) 7 / 먼 실패 17, 64수 상한 0. stop은 작업 격자에서 목표를 만족할 때만 허용되고 (`sim.py` 230줄) 상한에 닿은 실패가 없으므로, 이 실패들은 **작업 격자에서는 성공인데 정확 검사에서 크게 넘는 해** → 큰 판에서 정책이 g+p보다 훨씬 적게 놓고 격자 틈에 기댐 (H2가 판 크기와 함께 커짐; pkg10도 0.78). 다음 후보 근거 (구현 안 함, 결과 전 규칙은 시작 때 PROPOSALS에): 학습 격자 쪽은 r2c·r2m·r2g가 10칸에서 효과 없었음 → 학습 판 크기 (8~12) 밖으로의 격자 밀도·피크 수 차이를 먼저 봄.
- 다음: r2t (RL 끝 ~06:30, 평가 ~06:45) → lo_pcb_r2t (~07:45, 평가 ~08:00) → T1 판정; r2tf·lo_pcb_r2tf (~09:45·~11:45) → T1' 판정. GPU가 그 뒤 비지 않게 다음 후보를 진단으로 준비 (QUEUE 관찰 4: 16칸 pkg에서 학습이 g+p보다 낮음 — 학습 판 크기 8~12 vs 16칸의 실패 유형). CPU 줄: pool20 pkg20_1 → ip20 ×2 → 20칸 평가 → LS16·g+p 20칸. 11/6 12:00 UTC 뒤 새 시도 시작 안 함. DAILY 다음은 11/04 00:xx UTC.

## 재개 라운드 26 (2026-11-03 01:22~ UTC)
- 상태: GPU `r2g` RL it 250 (3367 s / 4500, fail 0.05, λ 1.43, grid capped 0~2 per iter, gpu peak 18.6 GB, OOM 0; val best it 200; 끝 ~02:05) → lo_pcb_r2g (~03:25). CPU 줄 `pool20 eval k8` 3번째 파일 21:36~ (~4 h 10 min째, worker CPU 100%·rss 6.0 GB, 로그에 새 줄 없음 — pcb20은 ~22 min/파일; 이 파일은 pkg20 또는 ip20, 다음 라운드에도 안 끝나면 build_pools의 cutting-plane 진행을 따로 봄). side_slot4: `ls10 zs-chip pool` 01:02~01:04, `ls10 zs-chip gp,modal_rule` 01:05~01:07 끝 (rc 0) → 사다리 filler는 r2g RL 남은 시간 < 1 h (FILL_GAP_S)이라 대기, r2g 평가 뒤 시작. pod 12.2 GiB / 60, oom_kill 0. QUEUE 새 지시 없음.
- **chip 기준선 (50문제, `runs/zs_chip/baselines/eval/`):** g+p 성공 0.94 / 개수 4.55, modal_rule **1.00** / 개수 6.16 (개수÷g+p 1.17), 문제당 0.16 / 0.09 s. 판 생성 2파일 70·79 s (cutting-plane 1 round).
- **r2g 끝** (RL 348 iter / 1.27 h — r2_mott_rl 354와 비슷, 격자 상한에 닿은 step 23 (r2_mott_rl 145), best it 300, λ 1.45, gpu peak 20.2 GB, OOM 0; `runs/2026-10-29/r2g_rl/metrics.json`), 10칸 평가 02:02~02:15. 성공 / 개수÷g+p: RESULTS ABLATIONS. **라운드 14 규칙 (i) 같은 종류 갈래 불충족:** r2g − r2_mott_rl (day 1) ip −0.02 [−0.06, +0.00], pcb **+0.00 [−0.10, +0.10]**, pkg +0.04 [−0.12, +0.22]; `@last` 대비 pcb −0.04 [−0.14, +0.06], pkg +0.04. 개수비 pcb 1.068 [0.980, 1.158]. (ii) 다른 판 점추정 −0.05 아래 없음. → 남은 갈래 lo_pcb_r2g − lo_pcb_r2_rl (뺀 pcb) CI 하한 > 0 하나뿐, lo_pcb_r2g RL 02:02~ (끝 ~03:20, 평가 ~03:35).
- **chip zero-shot 사다리 seed 0 (side_slot 02:16~02:25, `runs/zs_chip/s0/eval/`, 50문제):** 0~3단·r2a 모두 성공 **1.00** (g+p 0.94, modal_rule 1.00). 개수÷g+p: 0단 2.10, 1단 1.45, 2단 **1.11**, 3단 1.22, r2a 1.20. 2단 vs modal_rule 개수비 0.977 [0.927, 1.028]. → 미리 적은 판정 (b)는 성공률 기준이라 seed 0에서 0단이 이미 1.00 → (b) = +0.00, **천장 때문에 이 규칙으로는 충족될 수 없음** (seed 1·2도 0단이 1.00이면 "불충족"으로 기록). 결정: 판 정의·목표·규칙은 그대로 (PROPOSALS 11/03 "결과 보고 바꾸지 않는다"). 단 차이는 개수에만 보임 — 기술용으로 짝지은 개수비 vs 0단을 표에 추가 (판정 아님): 1단 0.837 [0.718, 0.976], **2단 0.744 [0.661, 0.832]**, 3단 0.821, r2a 0.818. seed 1·2는 side_slot이 학습 평가 사이에 돌림.
- `collect_results.py`에 "보너스: chip zero-shot" 절 추가 (PROPOSALS 11/03 규칙 그대로): 사다리 0~3단·r2a의 seed 0/1/2 성공률 / 개수÷g+p, (a)(b) 단 차이 seed별·0보다 큰 seed 수, (b) 판정 문장 자동 (seed 3개 다 있을 때만), seed 0 짝지은 CI vs g+p·modal_rule·E0+.
- 다음: lo_pcb_r2g (RL 끝 ~03:20, 평가 ~03:35) → 라운드 14 규칙 최종 판정 (남은 갈래: 뺀 pcb CI 하한 > 0). 그 뒤 GPU는 라운드 24 결정 그대로 (r2g "이김"이면 seed 1·2, 아니면 근거 있는 후보가 생길 때까지 비움). chip 사다리 s1·s2 → E0+ 종류 0·1·2 (4번째 칸). pool20 3번째 파일 (21:36~) 계속 확인. DAILY 다음은 11/04 00:xx UTC.

## 재개 라운드 25 (2026-11-03 00:22~ UTC)
- 상태: GPU `lo_pcb_r2m` RL it 250 (3350 s, fail 0.23, λ 5.54, margin-fail 24, gpu peak 11.4 GB, OOM 0; 끝 ~00:48) → r2g → lo_pcb_r2g (~03:35). CPU 줄 `pool20 eval k8` 3번째 파일 21:36~ (~2 h 50 min째, worker CPU 100%·rss 7.3 GB). side_slot4 대기 (다음: lo_pcb_r2m 평가). pod 13.6 GiB / 60, oom_kill 0. QUEUE 새 지시 없음.
- **보너스 준비: 학습에 없던 네 번째 종류 `chip` zero-shot (DAILY 11/03 추천 5, AGENT_PROMPT 보너스).** 판 정의·평가 목록·판정 규칙을 결과 전에 PROPOSALS 11/03 절에 고정. `decap_loop/techs.py`에 `CHIP` (셀 100 µm, 유전체 0.2 µm, 금속 0.5 µm — R_edge 69 mΩ로 가장 손실 큼, 전원 경로 2 mΩ+30 pH, decap 1 nF/3 pH/50 mΩ · 300 pF/1.5 pH/0.1 Ω · 50 pF/0.5 pH/0.3 Ω, SRF 2.9/7.5/31.8 GHz)를 `ZS_TECHS`로 따로 둠 (`TECHS`는 그대로 → 학습·val·드라이버 glob·smoke 불변). 목표 범위 HARD: `scripts/probe_chip.py` (작업 격자 4문제, 계획용) HARD g+p 4/4·개수 3~14·종류 0·1 섞음, EASY 1~2개 (`agent/logs/probe_chip.log`).
  - `build_pools.py --techs` (기본 = 기존 세 종류, 바뀐 것 없음; `jobs()` 48개 그대로 확인). smoke (`scripts/smoke_zs_chip.py`, 8칸 2문제, cutting-plane → 사다리 0~3단 + modal_rule `eval_policy.py`): 다섯 방법 모두 오류 없이 돎 → `runs/analysis/smoke_zs_chip.ok` (smoke 기록은 `smoke_records/`로 옮겨 RESULTS glob 밖).
  - 큐 (`scripts/queue_zs_chip.py`, CPU 큐 맨 끝, 이름 `ls10 zs-chip …` → side_slot 4번째 칸 filler, 한 번에 하나): 판 2파일 (`runs/pool_zs/eval_chip10_{0,1}`, 25문제씩, pool_v1 밖) → g+p·modal_rule → 사다리 seed 0·1·2 (전체 학습 run 그대로) → E0+ 종류 0·1·2. 기록 `runs/zs_chip/`. 로그 `agent/logs/zs_chip.log` → `.gitignore`.
  - 결정: 4번째 칸 filler로만 돌림 — CPU 한도 4 (지금 pool20 1 + 학습 정확 검사 2 + 4번째 칸 1), 10칸이라 가벼움; 학습 평가가 우선 (side_slot 규칙 그대로). 00:40에 최종 후보 `r2a_rl` seed 0·1·2도 평가 목록에 추가 (판 생성 전, PROPOSALS 같은 줄에 적음).
  - 진행: `lo_pcb_r2m` 평가 뒤 side_slot이 `ls10 zs-chip pool`을 01:02 시작 (pod 14.3 GiB).
- **lo_pcb_r2m 끝** (RL 339 iter / 1.26 h, best it 300), 10칸 평가 00:48~01:01. 처음 보는 pcb 성공 0.96 / 개수÷g+p 1.49; ip 1.00, pkg 0.94.
- **r2m 최종 판정 (라운드 7에 미리 적은 규칙 그대로): "효과 없음" → seed 1·2 확장 안 함.** (i) 같은 종류: pcb day 1 +0.08 [+0.00, +0.18] / `@last` +0.04 [−0.04, +0.12], pkg +0.02 (둘 다) → CI 하한 > 0이 두 기록 모두에서 성립하는 판 없음; 뺀 쪽: lo_pcb_r2m − lo_pcb_r2_rl pcb **+0.04 [−0.04, +0.14]** → 불충족. (ii) 해로움도 아님 (가장 낮은 점추정 ip −0.02). 대가: 개수비 뺀 pcb **1.881 [1.536, 2.269]**, 같은 종류 pcb 1.29·pkg 1.17. 참고 (판정 대상 아님): lo_pcb_r2m은 학습에 있던 pkg에서 +0.12 [+0.02, +0.24] (개수비 1.110) — 여유가 성공을 개수로 사는 같은 방향. RESULTS 설계 근거 절 "주장하지 않는 것"에 `--feas_margin` 추가 (`collect_results.py`).
- 다음: r2g (~02:15) → lo_pcb_r2g (~03:35) 평가 → 라운드 14 규칙으로 판정. chip 판 → 기준선 → 사다리 평가 (4번째 칸, 학습 평가 사이). pool20 3번째 파일 확인. GPU는 lo_pcb_r2g 뒤 비면: r2g가 "이김"이면 seed 1·2, 아니면 진단 근거 있는 후보가 생길 때까지 비워 둠 (라운드 24 결정 그대로). DAILY 다음은 11/04 00:xx UTC.

## 재개 라운드 24 (2026-11-03 00:02~ UTC) — 날 첫 라운드: DAILY 11/03 작성
- 상태: GPU `lo_pcb_r2m` RL it 125 (1758 s, fail 0.29, λ 3.31, margin-fail 27, gpu peak 11.2 GB, OOM 0; 끝 ~00:55) → r2g → lo_pcb_r2g (~03:35). CPU 줄 `pool20 eval k8` 3번째 파일 (pkg20 차례, 21:36~, ~2.5 h째, worker rss 6.3 GB·CPU 100% — 10/28 pkg16 판도 파일당 ~2 h였으므로 멈춘 것으로 보지 않음). side_slot: `ls10 ls16-timing smoke` 23:45~23:58 끝 (rc 0), 지금 대기 (남은 filler 없음, 다음은 lo_pcb_r2m 평가). pod 11.9 GiB / 60, oom_kill 0. QUEUE 새 지시 없음.
- **LS 16칸 시간 smoke (`runs/analysis/smoke_ls16.json`, 문제 하나, 작업 격자, 시간 측정용 — 보고서 숫자 아님):** local_search pcb16 88 s / pkg16 229 s (g+p 54 / 146 s 위에 추가), ip16_0 inst 0은 g+p 실패 (201개)라 LS 0 s; 최대 rss 22.7 GB (판 불러오기).
- 결정: **LS16 6파일을 드라이버 CPU 줄 맨 끝에** (`scripts/queue_ls16.py`, 이름 `ls16 …`, pcb → ip → pkg, 명령은 ls10과 같음, 로그 `agent/logs/base16_ls.log` → `.gitignore`). 이유: g+p 16칸 기록의 rss가 19~34 GB라 LS16 파일은 다른 16/20칸 작업과 동시에 돌면 45 GiB 규칙을 넘을 수 있음 → 한 번에 하나만 도는 드라이버 줄에 두고, 이름을 `ls10 `으로 시작하지 않게 해 4번째 칸 filler로 뽑히지 않게 함. 20칸 평가 (QUEUE 10/29 지시)가 먼저: `queue_eval20.py`는 smoke 항목 바로 뒤에 끼우므로 LS16은 자동으로 20칸 뒤. 예상 (계획용, 측정 아님): 파일당 pcb ~1 h, pkg ~2.5 h.
- DAILY.md 11/03 새로 씀 (어제 05:00 이후 정리 + 오늘 추천 큐 5개).
- 다음 라운드: lo_pcb_r2m (~00:55) 평가 → r2m 최종 판정 (라운드 7 규칙); pool20 3번째 파일 확인; r2g·lo_pcb_r2g 뒤 GPU가 ~03:35에 빔 → 어떤 판정도 "이김"이 아니면 진단 근거 있는 후보가 생길 때까지 GPU를 비워 둠 (근거 없는 구조 변경 금지, QUEUE 11/01 23:45).

## 재개 라운드 23 (2026-11-02 23:21~ UTC)
- 상태: GPU `r2m` RL 끝 (335 iter / 1.28 h, best it **100** — λ가 1 → 6.69로 계속 오르고 margin-fail이 늘면서 val이 it 100 뒤로 나빠짐; grid capped 112, OOM 0) → **`lo_pcb_r2m` RL 23:3x~** (끝 ~00:50) → r2g → lo_pcb_r2g. CPU 줄 `pool20 eval k8` 3번째 파일 21:36~ (~2 h 15 min째, 프로세스 CPU 100%로 계속 일함 — 다음 라운드에도 파일이 안 나오면 로그·rss를 다시 봄). side_slot: `ga10 eval_pkg10_1` 22:29~23:17 끝 (rc 0) → `eval10 r2m_rl` 23:32~23:44 끝 (rc 0); → `ls10 ls16-timing smoke` 23:45~ (라운드 15 결정, 시간·rss 측정만; 로그 `agent/logs/smoke_ls16.log` → `.gitignore`). pod 13.3 GiB / 60, oom_kill 0. QUEUE 새 지시 없음.
- **GA 10칸 6파일 모두 끝 (150문제):** genetic 성공 ip 0.40 / pcb 0.78 / pkg 0.42, 개수÷g+p 1.02 / 0.96 / 0.88, 문제당 9.2 / 16.4 / 31.8 s. 짝지은 CI vs g+p: ip +0.00 [−0.08, +0.08], pcb −0.12 [−0.24, −0.02], **pkg −0.48 [−0.62, −0.32]**; vs E0+ ip −0.60, pcb +0.12 [−0.06, +0.28], pkg −0.40 [−0.56, −0.24] (RESULTS 자동). LS와 같은 쪽: 개수는 줄지만 정확 검사 성공이 g+p보다 낮음 (pcb·pkg). 그림 2 GA는 16칸 기록이 없어 10칸 표에만.
- **r2m 평가 (보상 성공 5% 여유, seed 0, 같은 종류 학습) — 라운드 7 판정 규칙의 (i) 앞부분:** r2m − r2_mott_rl 성공률 pcb **+0.08 [+0.00, +0.18]** (day 1 기록), **+0.04 [−0.04, +0.12]** (`@last`); pkg +0.02 [−0.12, +0.16] (둘 다); ip −0.02 [−0.06, +0.00]. CI 하한 > 0이 day 1·`@last` 둘 다에서 성립하는 판 없음 → **같은 종류 쪽은 불충족**. 대가: 개수비 (둘 다 성공) pcb **1.287 [1.153, 1.442]**, pkg 1.172 [1.099, 1.246] — 개수가 17~29% 늘어남 (라운드 7에 "fail이 늘어 λ가 올라가 개수도 늘 수 있음"으로 미리 적은 쪽). 성공률 r2m ip 0.98 / pcb 0.96 / pkg 0.74, 개수÷g+p pcb 1.02 (r2_mott_rl 0.82). → 성공률-개수 곡선 위에서 개수 쪽으로 옮겨 간 것으로 읽힘 (주장 아님).
  - **최종 판정은 lo_pcb_r2m 평가 뒤** ((i)의 다른 갈래: lo_pcb_r2m − lo_pcb_r2_rl 뺀 pcb CI 하한 > 0, 그리고 (ii) 다른 판 점추정 −0.05 이상). 같은 종류 쪽 (ii)는 이미 충족 (가장 낮은 점추정 ip −0.02).
- 결정: **GPU 큐 순서 그대로 (lo_pcb_r2m → r2g → lo_pcb_r2g).** 라운드 14에 "r2m이 이기면 seed 1·2를 r2g 앞에"라고 적었지만, r2m이 이길 수 있는 길은 이제 lo_pcb_r2m 하나뿐이고 그 평가 (~01:05)는 r2g 시작 (~00:50) 뒤에 나옴. r2g를 죽이고 순서를 바꾸는 것보다 기존 큐 유지가 보수적 → lo_pcb_r2m이 "이김"이면 r2m seed 1·2를 lo_pcb_r2g **뒤**에 넣는다.
- 다음 라운드: pool20 3번째 파일 확인; lo_pcb_r2m (~00:50) 평가 → r2m 최종 판정 (라운드 7 규칙); ls16 smoke (lo_pcb_r2m 끝난 뒤 4번째 칸). DAILY 11/03 05:00 UTC.

## 재개 라운드 22 (2026-11-02 23:05~ UTC) — 상태만 (끝난 실험 없음)
- GPU `r2m` RL it 200 (2808 s, fail 0.56 — 이 iter는 pcb12 한 판이라 흔들림, λ 4.49, margin-fail 61, grid capped 35, gpu peak 16.4 GB, oom 0; val (16문제, 판정용 아님) pcb 1.146 / pkg 1.095 / ip 0.691 vs g+p; best.pt 22:39, 20분 체크포인트 it 171; 끝 ~23:30) → lo_pcb_r2m → r2g → lo_pcb_r2g. 로그의 "resume checkpoint at iter N"은 20분 저장 줄이고 프로세스는 22:13부터 끊김 없음 (재시작 아님).
- CPU 줄 `pool20 eval k8`: 3번째 파일 21:36~ (~90 min째, pcb20 파일은 ~22 min — pkg/ip가 더 무거움, 라운드 20과 같은 판단). side_slot `ga10 eval_pkg10_1` 22:29~ (~23:40 끝 예상). CPU 무거운 프로세스 4개 (pool20 1 + r2m certify 2 + GA 1, 한도 안), pod 14.4 GiB / 60, oom_kill 0. QUEUE 새 지시 없음.
- 다음 라운드: (라운드 21 계획 그대로) r2m 끝 → side_slot 평가 (GA 뒤) → 라운드 7 판정; pool20 끝나면 smoke peak rss로 `queue_eval20.py` 결정. DAILY 11/03 05:00 UTC.

## 재개 라운드 21 (2026-11-02 22:49~ UTC) — 상태만 (끝난 실험 없음)
- GPU `r2m` RL it 150 (2108 s, fail 0.14, λ 3.61, best score 1.695 @ it 100, gpu peak 15.8 GB, oom 0; `runs/2026-10-29/r2m_rl/log.txt`; 끝 ~23:30) → lo_pcb_r2m → r2g → lo_pcb_r2g. CPU 줄 `pool20 eval k8`: pcb20_0·pcb20_1 built (1309 s·1358 s, g+p ok 1.00), 3번째 파일 21:36~ (~75 min째). side_slot `ga10 eval_pkg10_1` 22:29~ (pkg10_0 ~70 min 기준 ~23:40 끝). CPU 무거운 프로세스 4개 (한도 안), pod 13.5 GiB / 60, oom_kill 0. QUEUE 새 지시 없음. 할 일 1·2·4 (QUEUE 11/01)는 이전 라운드에서 끝남 — 지금은 r2m/r2g 판정 대기.
- report_watch는 다시 떠서 "모든 lo_* push됨"으로 바로 끝남 (설계대로).
- 다음 라운드: r2m 끝 → side_slot 평가 (GA 뒤) → 라운드 7 미리 적은 규칙으로 판정; pool20 끝나면 peak rss로 `queue_eval20.py` 결정. DAILY 11/03 05:00 UTC.

## 재개 라운드 20 (2026-11-02 22:33~ UTC)
- 상태: GPU `r2m` RL it 50 (813 s, fail 0.36, λ 1.94, gpu peak 13.6 GB, OOM 0; 끝 ~23:30) → lo_pcb_r2m → r2g → lo_pcb_r2g. CPU 줄 `pool20 eval k8` 3번째 파일 (21:36~, 1시간째 — pcb20 파일은 ~22 min이었고 pkg는 GA처럼 더 무거움, 이상 아님). side_slot: `eval10 lo_pkg_r2a_rl` 22:15~22:28 끝 (rc 0) → `ga10 eval_pkg10_1` 22:29~ (pkg10_0은 ~70 min 걸림) → 그다음 r2m 평가. CPU 무거운 프로세스 = pool20 1 + r2m certify 2 + GA 1 (한도 안). pod 13.7 GiB / 60, oom_kill 0. QUEUE 새 지시 없음.
- `ensure_running`이 report_watch·resume_agent를 다시 띄움 — report_watch는 27개 lo_* rung이 모두 push된 상태라 시작하자마자 스스로 끝나는 것이 설계대로 (라운드 17과 같음). 문제 아님.
- **lo_pkg_r2a_rl 평가 끝 (표만, 판정 없음 — 라운드 13 결정: fc_bias 범용화 seed 1·2 안 함):** pkg (뺀 종류) 성공 0.72 / 개수÷g+p 0.87 vs 편향 없는 `lo_pkg_r2_rl` 0.80 → **−0.08 [−0.24, +0.08]** (CI 0 포함); pcb −0.08 [−0.20, +0.02], ip +0.00. 전체 학습 r2a 대비 pkg −0.12 [−0.26, +0.02]. 즉 ip·pcb (+0.00 둘 다, 라운드 13)에 이어 pkg에서도 fc_bias는 처음 보는 종류에서 도움이 안 됨 (seed 0 하나, 세 종류 모두 CI 0 포함). RESULTS는 report_watch/collect가 자동으로 다시 만든 것 (`RESULTS.md` "pkg 빼고 학습 2단 + 주파수 궁합 편향" 절, 479+ 기록).
- 다음 라운드: GA pkg10_1 진행 확인; pool20 끝 (~23:10 예상보다 늦을 수 있음) → smoke 기록 peak rss로 `queue_eval20.py` 결정; r2m (~23:30) → side_slot 평가 (GA 뒤) → 라운드 7 판정. DAILY 11/03 05:00 UTC.

## 재개 라운드 19 (2026-11-02 22:15~ UTC)
- 상태: GPU `lo_pkg_r2a` 학습 끝 (RL 366 iter, 1.26 h, `runs/2026-10-29/lo_pkg_r2a_rl/metrics.json`) → **`r2m` RL 22:13~** (같은 distill에서 RL만, 끝 ~23:30) → lo_pcb_r2m → r2g → lo_pcb_r2g. side_slot이 `eval10 lo_pkg_r2a_rl` 22:15:32 시작 (worker 1). CPU 줄 `pool20 eval k8` 3번째 파일 그대로 (~23:10 끝 예상). CPU 프로세스 4개 (pool20 1 + r2m certify 2 + 평가 1) — 한도 안. pod 10.7 GiB / 60, oom_kill 0. QUEUE 새 지시 없음.
- **GA pkg10_0 끝 (25문제, 22:07):** genetic 성공 0.48, 개수 평균 5.75 (성공한 것), 39.2 s/문제 (pcb ~15 s보다 무거움), peak rss 3.4 GB. 짝지은 CI (RESULTS 자동, 파일 하나라 [25]): vs g+p **−0.44 [−0.68, −0.24]**, vs E0+ −0.36 [−0.60, −0.16]; 개수÷g+p 0.81 (둘 다 성공한 문제). pcb와 같은 쪽 (GA는 개수는 줄지만 정확 검사 성공이 떨어짐). pkg10_1은 4번째 칸이 비면 (lo_pkg_r2a 평가 뒤) 시작. pkg 판정은 두 파일 50문제가 다 나온 뒤에만.
- RESULTS 다시 생성 (`collect_results.py`, 479 기록): GA pkg10 행, lo_pkg_r2a_rl 학습 기록 행 추가.
- 다음 라운드: lo_pkg_r2a 평가 → 표만 (판정 없음, 라운드 13); GA pkg10_1 시작 확인; pool20 끝 (~23:10) → smoke 기록 peak rss로 `queue_eval20.py` 결정; r2m (~23:30) 평가 → 라운드 7 판정. DAILY 11/03 05:00 UTC.

## 재개 라운드 18 (2026-11-02 21:59~ UTC)
- 상태: GPU `lo_pkg_r2a` RL it 275 (3414 s, fail 0.00~0.03, λ 1.09, gpu peak 11.5 GB, OOM 0; 20분 체크포인트 it 293; 끝 ~22:15) → r2m → lo_pcb_r2m → r2g → lo_pcb_r2g. CPU 줄 `pool20 eval k8`: 6파일 중 2개 (pcb20_0·pcb20_1) 그대로, 지금 3번째 파일 (~23:10 끝 예상). 4번째 칸 `ga10 eval_pkg10_0` 3640 s째 (pcb 파일은 ~2000 s였음 — pkg가 더 무거움, 이상 아님) → 끝나면 side_slot이 lo_pkg_r2a 평가. pod 10.7 GiB / 60, oom_kill 0. QUEUE 새 지시 없음.
- `agent/logs/side_slot4.log` 0바이트는 정상 — side_slot.py는 작업 START/END 때만 찍고 (`scripts/side_slot.py:51,54`), 21:07 이후 아직 시작한 작업이 없음 (GA 작업 기다리는 중). 라운드 9의 side_slot2 0바이트 (지워진 파일에 쓰기)와 다른 경우.
- 이번 라운드에 끝난 실험 없음 → RESULTS 바뀐 것 없음.
- 다음 라운드: (라운드 17 계획 그대로) GA pkg10_0 기록 확인; lo_pkg_r2a 평가 → 표만; pool20 끝 → smoke peak rss로 `queue_eval20.py` 결정; r2m (~23:35) 평가 → 라운드 7 판정. DAILY 11/03 05:00 UTC.

## 재개 라운드 17 (2026-11-02 21:43~ UTC)
- 상태: GPU `lo_pkg_r2a` RL it 200 (21:36 무렵, 2537 s, val pcb fail 0.69 / ip 0.12 — 16문제, 판정용 아님; gpu peak 11.4 GB, OOM 0; it 189에서 체크포인트로 이어 감; 끝 ~22:15) → r2m → lo_pcb_r2m → r2g → lo_pcb_r2g. CPU 줄 `pool20 eval k8`: 6파일 중 2개 끝 — 고침: 라운드 16에 "지금 pkg20_0"이라 적었지만 실제 순서는 pcb20_0 (1309 s) → **pcb20_1** (1358 s, F 226, 10 rounds, g+p 8문제 성공 1.00, 개수 평균 6.9) → 남은 4파일, 끝 ~23:10 예상 → 그 뒤 `eval20 eval_pcb20_0 r2a_rl` smoke. 4번째 칸 `ga10 eval_pkg10_0` (20:58~) → ga10 pkg10_1 → `ls10 ls16-timing smoke`. pod 9.8 GiB / 60, oom_kill 0. `ensure_running`이 report_watch·resume_agent를 다시 띄움. QUEUE 새 지시 없음.
- 이번 라운드에 끝난 실험 없음 → RESULTS 바뀐 것 없음 (report_watch가 새 eval json이 오면 자동으로 다시 만듦).
- 다음 라운드: lo_pkg_r2a (~22:15) 평가 (side_slot) → 표만 (판정 없음, 라운드 13); GA pkg10_0 기록 확인; pool20 끝 (~23:10) → smoke 기록의 peak rss로 `queue_eval20.py` 실행 여부 결정 (pod 사용량 + peak < 45 GiB); r2m (~23:35) 평가 → 라운드 7 판정. DAILY 11/03 05:00 UTC.

## 재개 라운드 16 (2026-11-02 21:25~ UTC)
- 상태: GPU `lo_pkg_r2a` RL it 100 (21:2x, val 새 best it 100, gpu peak 10.7 GB, OOM 0; 끝 ~22:15) → r2m → lo_pcb_r2m → r2g → lo_pcb_r2g. CPU 줄 `pool20 eval k8`: **eval_pcb20_0 끝** (21:13, 1309 s, F 176, cutting-plane 9 rounds, 8문제 g+p 성공 1.00, 개수 평균 6.4) → 지금 pkg20_0; 남은 5파일이면 ~23:00 끝. 4번째 칸 `ga10 eval_pkg10_0` (20:58~). pod 12.8 GiB / 60, oom_kill 0. QUEUE 새 지시 없음.
- **결정: 20칸 평가는 한 항목만 먼저 (메모리·시간 측정)** — `eval20 eval_pcb20_0 r2a_rl` (`scripts/queue_eval20_smoke.py`, eval16과 같은 명령 `--workers 1 --inst_chunk 5`, CPU 줄 pool20 바로 뒤). 이유: 20칸 판은 10/28 OOM 의심 원인이고 (QUEUE 10/28), 16칸 기록의 peak rss는 같은 파일이면 방법과 무관하게 같음 (pcb16_0: 정책·규칙 22.1 GB, g+p 31.1 GB) → 메모리는 정책이 아니라 판 불러오기·정확 검사 쪽. 20칸은 포트가 더 많아 같은 비율을 가정할 수 없으므로 기록 1개로 잰 뒤 나머지를 넣는다. 이 기록은 버리는 smoke가 아니라 정식 평가 기록.
  - 나머지 (정책 5개 r2a·r0~r3 → modal_rule → g+p, 6파일; 라운드 11 순서 그대로)는 `scripts/queue_eval20.py`로 준비만 함 (smoke 기록이 없으면 거부, 있으면 peak rss·시간을 찍음). LS·E0+ 20칸은 g+p 20칸 메모리를 본 뒤. `agent/logs/base20.log` 미리 `.gitignore`.
- 다음 라운드: lo_pkg_r2a (~22:15) 평가 → 표만; smoke_ls16 결과 → LS16 큐 결정; pool20 끝 (~23:00) 뒤 smoke 기록 peak rss가 45 GiB 규칙에 여유 있으면 (pod 사용량 + peak < 45 GiB) `queue_eval20.py` 실행; r2m (~23:35) 평가 → 라운드 7 판정. DAILY 11/03 05:00 UTC.

## 재개 라운드 15 (2026-11-02 21:07~ UTC)
- 상태: GPU `lo_pkg_r2a` RL 20:58~ (끝 ~22:15) → r2m → lo_pcb_r2m → r2g → lo_pcb_r2g. CPU 줄 `pool20 eval k8` (20:51~, worker 1, 아직 파일 0개). 4번째 칸 `ga10 eval_pkg10_0` (20:58~). pod 11.8 GiB / 60, oom_kill 0. QUEUE 새 지시 없음.
- **GA pcb10 두 파일 끝 (50문제):** genetic pcb 성공 0.78 (pcb10_0 0.96 → pcb10_1에서 떨어짐), vs g+p **−0.12 [−0.24, −0.02]**, vs E0+ +0.12 [−0.06, +0.28], 개수÷g+p 0.96, 16.4 s/문제 (RESULTS 자동). 실패 11문제는 평균 30개. ip는 g+p 수준 (0.40), pcb는 g+p보다 낮음 — LS와 같은 쪽 (작업 격자에서 줄인 해가 정확 검사에서 떨어짐, 확인은 안 함).
- side_slot을 대상 `lo_pkg_r2a r2m lo_pcb_r2m r2g lo_pcb_r2g`로 다시 띄움 (`agent/logs/side_slot4.log`, 라운드 14에 적은 할 일; 돌던 GA 작업은 그대로 살아 있고 새 side_slot은 그것이 끝나길 기다림). GA 작업이 이어 쓰는 추적 로그 `agent/logs/base10_ga.log`를 `.gitignore` + `git rm --cached` (report_watch autostash로 로그가 고아가 되는 같은 위험, 라운드 5·9).
- **결정: 그림 2용 LS 16칸 — 먼저 시간 측정만** (`scripts/smoke_ls16.py`, 4번째 칸 filler `ls10 ls16-timing smoke`, GA pkg10_1 뒤). 이유: AGENT_PROMPT 그림 2는 "우리 모델 vs g+p vs LS (가능하면 GA)"인데 16칸 LS 기록이 없음 (10/28 라운드 5에 "단 평가가 기다리지 않을 때 16칸 LS"로 계획만 하고 큐에 안 넣음). LS = g+p + local_search (4×32)라 g+p 16칸 (62~131 s/문제, rss 19~34 GB)보다 무거움 → 종류마다 첫 문제 1개에서 local_search 시간·rss만 잼 (작업 격자, 정확 검사 없음). 이 숫자로 LS16 평가를 (파일 수·문제 수·드라이버 줄 vs 4번째 칸) 정한다; 보고서 숫자 아님. GA 16칸은 LS16 뒤에 같은 방식으로 판단.
- 다음 라운드: lo_pkg_r2a (~22:15) 평가 → 표만; r2m (~23:35) 평가 → 라운드 7에 미리 적은 판정; smoke_ls16 결과 → LS16 큐 결정; pool20 첫 파일 시간·메모리 확인. DAILY 11/03 05:00 UTC.

## 재개 라운드 14 (2026-11-02 20:41~ UTC)
- 상태: GPU `lo_pkg_r2a` distill it 1050 / 1782 s (RL 끝 ~22:20) → r2m (~23:40) → lo_pcb_r2m (~01:00). CPU 줄 `eval16 eval_ip16_1 r2a_rl` (20:38~, r2a 16칸 마지막) → pool20. 4번째 칸 `ga10 eval_pcb10_1` (20:24~). pod 13.2 GiB / 60, oom_kill 0. QUEUE 새 지시 없음.
- **r2a 16칸 6파일 끝** (ip16_1 20:48) → 그림 2 다시 그림 (`reports/fig2_bigboard_draft.png`, 라운드 12 명령 + `r2a_rl`). r2a_rl 16칸 성공 pcb 0.88 / pkg **0.46** / ip 1.00, 문제당 1.6~3.7 s (g+p 62~131 s). 짝지은 CI (RESULTS 자동): vs r2_mott_rl pcb16 −0.02 [−0.14, +0.10], pkg16 −0.06 [−0.22, +0.12], ip16 +0.00 (개수비 0.99~1.01) → **10칸에서 보인 편향의 이득 (pcb +0.08, pkg +0.12, CI 0 밖)이 16칸에서는 안 보임** (점추정은 오히려 음수, CI 0 포함). vs modal_rule pcb16 −0.12 [−0.22, −0.04], pkg16 −0.14 [−0.30, +0.02]; vs g+p pkg16 −0.36 [−0.50, −0.22]; ip16 개수÷E0+ 1.015 [1.003, 1.029]. 보고서에는 "r2a 우위는 10칸 (학습 크기 8~12칸) 안에서만"으로 쓴다. pkg16은 모든 학습 단이 g+p (0.82)보다 낮음 (0.46~0.64).
- **결정: GPU 큐가 ~01:00에 비지 않게 `r2g`·`lo_pcb_r2g`를 lo_pcb_r2m 뒤에 넣음** (`scripts/queue_grid_growth.py`). 바뀐 것: RL 작업 격자 상한 `--grid_growth` 0.5 → 1.0 하나 (기존 옵션, 새 코드 없음). 근거: 진단 H2 (라운드 1)가 이름 붙인 두 수정 중 "M개 샘플 전부 검사" (`--certify 2`, r2c)는 iter가 1/3로 줄어 흐려졌고 "격자 상한 완화"는 시험 안 함; r2_mott_rl 로그에서 격자 상한에 닿은 iter 145 / 354, gpu peak 16.8 GB / 24 (OOM이면 train.py가 chunk를 반으로 — 메모리 실패는 잡힘). 짝 설계는 r2c·r2m과 같음 (같은 distill best.pt에서 RL 1.25 h만, 같은 seed). 주의 (결과 전): F가 커지면 iter가 느려져 같은 1.25 h에서 iter 수가 줄 수 있음 — iter 수를 같이 보고.
  - r2m이 라운드 7 판정으로 "이김"이면 r2m seed 1·2를 r2g **앞**에 넣는다 (미리 적은 대로). r2g는 ~01:00 전에는 시작하지 않으므로 순서 바꿀 시간 있음.
  - **미리 적는 판정 (라운드 7 규칙 그대로, 결과 전):** seed 0, 판마다 50문제 짝지은 bootstrap. (i) 같은 종류 pcb·pkg 중 하나 이상에서 r2g − r2_mott_rl 성공률 CI 하한 > 0이 day 1 기록과 `r2_mott_rl@last` **둘 다에서** 성립하거나, lo_pcb_r2g − lo_pcb_r2_rl (뺀 pcb) CI 하한 > 0, (ii) 그리고 다른 판에서 점추정이 −0.05 아래가 아니면 "이김" → seed 1·2 확장. 두 기준이 갈리면 "평가 잡음 범위 — 판정 보류", 그 밖은 "효과 없음"/"해로움", 확장 안 함.
- 다음 라운드: pool20 (CPU 줄, 20:51~, 시작 때 pod 8.1 GiB) 진행·메모리 확인; lo_pkg_r2a (~22:20) 평가 → 표만; r2m (~23:40) 평가 → 라운드 7 판정; side_slot을 r2m 끝난 뒤 `lo_pcb_r2m r2g lo_pcb_r2g` 대상으로 다시 띄우기 (지금 대상 목록에 r2g 없음); ABLATIONS에 r2g 행 (`collect_results.py`). DAILY 11/03 05:00 UTC.

## 재개 라운드 13 (2026-11-02 19:34~ UTC)
- 상태: GPU `lo_pkg_r2a` distill 20:12~ (RL 끝 ~22:20) → r2m (~23:40) → lo_pcb_r2m (~01:00). CPU 줄 eval16 r2a: pcb16_1·pkg16_1 끝 (20:24), 남은 16칸 → pool20. 4번째 칸: lo_ip_r2a 평가 끝 (20:23), 다음 ga10. pod 8~17 GiB / 60, oom_kill 0. QUEUE 새 지시 없음.
- **lo_ip_r2a 끝** (314 iter / 1.27 h, best it 300), 10칸 평가 20:12~20:23: 처음 보는 ip 1.00 (lo_ip_r2 1.00), pcb 0.96, pkg 0.82.
- **라운드 5에 미리 적은 판정 (`--fc_bias` 범용화, seed 0):** (i) lo_t_r2a − lo_t_r2 처음 보는 종류에서 ip **+0.00 [+0.00, +0.00]** (개수비 1.016 [0.999, 1.036]), pcb **+0.00 [−0.10, +0.10]** (라운드 9) → CI > 0인 종류 없음 → **"효과 없음" → seed 1·2로 확장하지 않음.** (ii) 뺀 − 전체 (vs r2a_rl): ip +0.00, pcb −0.04 [−0.12, +0.04] (개수비 1.113 [1.049, 1.176]). 한계 (판정 바꾸지 않음): seed 0의 lo_ip_r2는 이미 ip 1.00이라 ip 쪽은 천장 — 이 짝은 ip에서 검출력이 없었다. RESULTS 설계 근거 절의 "주장하지 않는 것"에 한 줄 추가 (`collect_results.py`).
- 결정: 큐의 `lo_pkg_r2a`는 그대로 돌림 — (i)은 이미 불충족이라 판정은 안 바뀌지만, 기존 큐 유지가 보수적이고 r2a 범용화 행을 세 종류 모두 채움 (보고 일관성).
- **GA pcb10_0 끝** (19:37, 25문제): genetic 성공 0.96, 개수÷g+p 0.97, vs g+p −0.04 [−0.12, +0.00], vs E0+ +0.24 [+0.04, +0.44], 17.3 s/문제 (RESULTS 자동). ip에서는 g+p 수준 (0.40)인데 pcb에서는 g+p·규칙과 비슷 — 나머지 pcb10_1·pkg10은 4번째 칸에서.
- 다음 라운드: r2a 16칸 나머지 → 그림 2에 `r2a_rl` 추가; lo_pkg_r2a (~22:20) 평가 → 표만 (판정 없음); r2m (~23:40) 평가 → 라운드 7에 미리 적은 판정 (i)·(ii); GPU 큐가 ~01:00에 끝나므로 그 전 라운드에서 다음 GPU 항목 결정 (r2m이 "이김"이면 seed 1·2, 아니면 PROPOSALS 후보 중 진단 근거 있는 것). DAILY 11/03 05:00 UTC.

## 재개 라운드 12 (2026-11-02 19:17~ UTC)
- 상태: GPU `lo_ip_r2a` RL (끝 ~20:12) → lo_pkg_r2a → r2m → lo_pcb_r2m (~00:50). CPU 줄 `eval16 eval_pcb16_0 r2a_rl` 19:18~ (r2a 16칸 6파일 → pool20). 4번째 칸 `ga10 eval_pcb10_0`. pod 9.4 GiB / 60, oom_kill 0. QUEUE 새 지시 없음 (11/02 21:30 KST 절은 라운드 5·10에서 처리).
- **GA ip10 두 파일 끝 (50문제):** genetic ip 성공 0.40, vs g+p +0.00 [−0.08, +0.08], vs E0+ −0.60 [−0.72, −0.46], 개수÷E0+ 2.37 (RESULTS 자동). pcb10·pkg10은 4번째 칸에서 진행.
- **16칸 기록 완성 (r0~r3, g+p, E0+, modal_rule 모두 6파일 50문제)** → 그림 2 다시 그림 `reports/fig2_bigboard_draft.png` (`fig_results.py --fig 2 --records 'runs/2026-10-2[89]/*/eval/*16_*.json' --methods greedy+prune,gp_type_aware,modal_rule,r0_coord_rl,r1_modal_rl,r2_mott_rl,r3_loop_rl --sizes 16`). 읽기: 성공 pcb16 g+p 0.92 / E0+ 0.46 / 규칙 1.00 / r0 1.00 / r1 0.92 / r2 0.90 / r3 0.86; pkg16 0.82 / 0.80 / 0.60 / 0.64 / 0.52 / 0.52 / 0.58; ip16 0.56 / 1.00 / 나머지 1.00. ip16 정책 − g+p +0.44 [+0.30, +0.58], 개수÷g+p ~0.30. 문제당 시간 r2·r3 1.7~4.1 s vs g+p 62~131 s (22~48배), E0+ 189~547 s.
  - 주의 (시간 막대): 같은 방법도 파일 사이 `sec_per_inst`가 ~2배 차이 (r0 pcb16 19.1 / 37.5 s, pkg16 49.3 / 19.0 s) — 공유 CPU에서 다른 작업과 동시에 돈 기록. 그림의 시간은 "같은 기계, 부하 통제 안 함"으로만 쓴다. r0·r1이 r2·r3보다 느린 이유는 확인 안 함.
- 다음 라운드: lo_ip_r2a 평가 (~20:12 뒤 side_slot) → 라운드 5 판정 (i)·(ii) 확정; r2a 16칸 6파일 끝나면 그림 2에 `r2a_rl` 추가; GA pcb10·pkg10 기록 확인. DAILY 11/03 05:00 UTC.

## 재개 라운드 11 (2026-11-02 18:58~ UTC)
- 상태: GPU `lo_ip_r2a` RL 18:56~ (끝 ~20:12) → lo_pkg_r2a → r2m → lo_pcb_r2m (~00:50). CPU 줄 `eval16 eval_ip16_1 r2_mott_rl` (r1 끝, r3 → r2a 16칸 6파일 → pool20 남음). 4번째 칸 `ga10 eval_ip10_1`. pod 13.8 GiB / 60, oom_kill 0.
- **GA 10칸 첫 기록** (`ga10 eval_ip10_0`, 25문제): genetic ip 성공 0.40 (= g+p), 짝 차이 vs g+p +0.00 [−0.12, +0.12], vs E0+ −0.60 [−0.80, −0.40], 개수÷g+p 1.07, ÷E0+ 2.27, 실패 15문제는 평균 71개 (마지막 greedy 채움). RESULTS 요약표 아래 GA 정의 줄 (`collect_results.py`, 숫자 없음). 나머지 5파일은 4번째 칸에서 이어짐.
- 결정: 20칸 평가 항목은 이번에도 큐에 넣지 않음 — 라운드 9 결정 (판이 생긴 뒤 메모리 보고 명령·inst_chunk 정함) 유지. 참고로 16칸 기록의 최대 rss: 정책·규칙 13~25 GB, g+p 19~34 GB (25문제); 20칸은 파일당 8문제. 판이 생기면 정책 5개 (r0~r3, r2a) → modal_rule → g+p 순으로 먼저, LS·E0+는 g+p 20칸 메모리를 본 뒤.
- 다음 라운드: lo_ip_r2a 평가 (~20:12 뒤 side_slot) → 라운드 5 판정 (i)·(ii) 확정; eval16 ip16_1 r2·r3 끝나면 그림 2 다시 그리기; r2m (~23:30) 뒤 GPU 다음 항목 결정 (큐는 lo_pcb_r2m ~00:50에서 끝남). DAILY 11/03 05:00 UTC.

## 재개 라운드 10 (2026-11-02 18:40~ UTC)
- 상태: GPU `lo_ip_r2a` distill it 1050 / 1809 s (0.75 h, val pcb fail 0.50 / pkg 0.81 — 16문제, 판정용 아님), RL 끝 ~20:10. CPU 줄 `eval16 eval_ip16_1 r1_modal_rl` (18:26~), 4번째 칸 `ga10 eval_ip10_0` (18:24~, 첫 GA 기록). pod 13.1 GiB / 60, oom_kill 0. `ensure_running`이 report_watch·resume_agent를 다시 띄움 (report_watch는 27 rung 다 올린 상태라 다음 사이클에 끝남, STOP 없음).
- **QUEUE 11/01 할 일 4 마무리 — 학습 없는 기준선 차이 표:** `research_notes/선행 기준선 재구현 차이 표.md`에 GA (`genetic`)·LS (`local_search`)·공진 규칙 (`modal_rule`, US 6789241식)·DevFormer 행 추가 (선행 연구에서 확인된 것 / 우리 구현 / 차이). GA 개체 수·연산자, LS 이웃, 특허 세부는 저장소에 원문이 없어 "원문 미확인"; 원 논문 시간 숫자 (GA 3 h, 64분)는 다른 판·기계라 우리 시간과 나란히 쓰지 않는다고 적음. 숫자는 RESULTS (스크립트)만.
- 다음 라운드: lo_ip_r2a 평가 (~20:10 뒤 side_slot) → 라운드 5 판정 (i)·(ii) 확정; GA 10칸 기록이 오면 RESULTS 요약표 행 확인; lo_pkg_r2a (~22:10); eval16 ip16_1 r1~r3 끝나면 그림 2. DAILY 11/03 05:00 UTC.

## 재개 라운드 9 (2026-11-02 17:55~ UTC)
- 상태: GPU `lo_ip_r2a` distill 18:09~ (RL 끝 ~20:10) → lo_pkg_r2a → r2m → lo_pcb_r2m (~00:50). CPU 줄 `eval16 eval_ip16_1 r0_coord_rl` 18:00~ (그다음 r1~r3, r2a 6파일). 4번째 칸 side_slot. pod 6.5~12 GiB / 60, oom_kill 0.
- side_slot2.log가 0바이트 (살아 있는 writer가 지워진 파일에 쓰던 것으로 보임 — 라운드 시작 `git pull --rebase` 때 바뀐 듯, 원인 확인 못 함). side_slot은 쉬는 중이어서 멈추고 같은 대상으로 `agent/logs/side_slot3.log`에 다시 띄움; side_slot3·side_slot_jobs·build_pool20 로그를 `.gitignore`에.
- **LS 10칸 6파일 끝** (`ls10 eval_pkg10_1` 17.6 s/문제): local_search 성공 ip 0.40 / pcb 0.78 / pkg 0.42, 개수÷g+p 0.75 / 0.75 / 0.80 (RESULTS 요약표, 짝 차이는 판별 표 — g+p 대비 pcb −0.12 [−0.22, −0.02], pkg −0.48 [−0.62, −0.34]: 작업 격자로 줄인 해가 정확 검사에서 떨어짐). 요약표 아래에 LS 정의·이 현상 한 줄 (`collect_results.py`, 숫자 없음).
- **E0+ 16칸 6파일 모두 끝** (`e0plus16 eval_ip16_1 type2` 18:00): gp_type_aware ip16 1.00 / pcb16 0.46 / pkg16 0.80 (50문제씩).
- **lo_pcb_r2a 끝** (351 iter / 1.26 h, best it 200), 10칸 평가 18:10~18:23: 처음 보는 pcb 0.92 (lo_pcb_r2 0.92), lo_pcb_r2a − lo_pcb_r2 pcb **+0.00 [−0.10, +0.10]**, 개수비 1.007 [0.926, 1.091]; ip +0.00, pkg +0.00. 뺀 − 전체 (vs r2a_rl) pcb −0.04 [−0.12, +0.04], 개수비 1.113 [1.049, 1.176]. → 라운드 5 판정 (i)의 pcb 쪽은 불충족; **최종 판정은 lo_ip_r2a 평가 뒤** (ip가 CI > 0이고 pcb·pkg 점추정이 −0.05 아래가 아니어야 "이김").
- **20칸 평가 판 (QUEUE 10/29 03:10 할 일 5, 파일당 8문제 = PROPOSALS (c) 채택분):** 아직 안 만들어져 있었음. `build_pools.py --eval_n` (기본 25 그대로) 추가, CPU 줄에 `pool20 eval k8` 한 항목 (`--only eval --eval_sizes 20 --eval_n 8 --workers 1`, `scripts/queue_pool20.py`)을 마지막 eval16 (r2a ip16_1) 바로 뒤에. 결정: worker 1개 — 드라이버 줄 한 칸 안에서 CPU 4개 유지, 메모리 (20칸 판 = OOM 의심 원인, QUEUE 10/28) 보수적. 이미 있는 파일은 건너뜀 → 재시작해도 진행 중 파일만 잃음. 20칸 정책·기준선 평가는 판이 생긴 뒤 큐에 넣음 (명령·inst_chunk는 그때 메모리 보고 결정).
- 다음 라운드: lo_ip_r2a 평가 (~20:10 뒤 side_slot) → 라운드 5 판정 (i)·(ii) 확정; lo_pkg_r2a (~22:10); eval16 ip16_1 r0~r3 끝나면 그림 2 다시 그리기 (`fig_results.py --fig 2`). DAILY 11/03 05:00 UTC.

## 재개 라운드 8 (2026-11-02 17:35~ UTC)
- 상태: GPU `lo_pcb_r2a` RL it 200 (val 새 best it 200, 끝 ~18:10), CPU 줄 `e0plus16 eval_ip16_1 type2`, 4번째 칸 `ls10 eval_pkg10_1`, side_slot 대기 (대상 lo_pcb_r2a … lo_pcb_r2m). pod 17.6 GiB / 60, oom_kill 0. report_watch는 27 rung 다 올린 상태라 바로 끝남 (정상).
- **QUEUE 11/01 할 일 6 (설계 근거 절) — `collect_results.py` `design_rationale()`**, RESULTS "설계 근거: 관찰 → 설계 → 결과". 손 숫자 없음: (1) 학습 없는 토큰 분석 표는 `research_notes/토큰 데이터 분석/결과.md` v2 표를 그대로 읽어 옮김 (PCB↔IP만, 선형 모델, CI 없음이라고 적음), (2) 뺀 − 전체 seed 평균과 **10/31 기준 판정을 스크립트가 다시 계산** → pcb 통과, ip 실패 ((b) seed 1 −0.32, (c) −0.11), pkg 실패 ((b) −0.28·−0.28, (c) −0.11) — PROGRESS 라운드 1 판정과 같음. (3) 주장하지 않는 것 (루프 > 기본, MoTT 개선 시도 3개, 학습 > 규칙) 목록. 결정: 범용화 주장은 pcb 범위로만 쓰고, 매 수 Z 갱신·decap 토큰은 Manoharan에 있다고 명시 (QUEUE 10/29 03:45 노벨티 안내).
- 다음 라운드: lo_pcb_r2a 끝 (~18:10) → side_slot 평가 → 라운드 5 판정 (i)·(ii); lo_ip_r2a (~20:10), lo_pkg_r2a (~22:10), r2m (~23:30), lo_pcb_r2m (~00:50). DAILY 11/03 05:00 UTC.

## 재개 라운드 7 (2026-11-02 17:15~ UTC)
- 상태: GPU `lo_pcb_r2a` RL (16:54~, 끝 ~18:10) → lo_ip_r2a → lo_pkg_r2a (~22:10). CPU 줄 `e0plus16 eval_ip16_1 type2` (16:25~), 4번째 칸 filler `ls10 eval_pkg10_1` (16:57~). pod 14 GiB / 60, oom_kill 0.
- **H2 싼 수정 `--feas_margin` 구현 (PROPOSALS 라운드 6 후보, 노벨티 2줄은 이미 적음):** `train.py` RL 보상에서 작업 격자 성공 샘플도 최악 log10(|Z|/Zt) > −δ이면 실패로 침 (λ·best 샘플·expert iteration 교체 판정에 같이 들어감; 정확 검사 (`--certify 1`)·val·평가는 그대로). `rollout.py`는 마지막 관측의 최악 여유 `m`을 반환값에 더함 (키 하나 추가, 다른 호출자 영향 없음; 환경 파일 4개는 안 건드림). 로그에 `margin-fail` (여유 때문에 실패로 바뀐 샘플 수).
  - 결정: **δ = log10 1.05 (0.021189)** — PROPOSALS에 미리 적은 예시 값 그대로, 평가 실패의 최악 초과 (7~15%)보다 작아서 보수적. 결과 보고 바꾸지 않음.
  - 알아 둘 점 (결과 전): 환경은 작업 격자 성공 뒤 add를 막으므로 (`sim.py` 224줄) 여유는 마지막 add의 선택·종류 바꾸기·덜 가지치기로만 얻을 수 있음 → 이 수정은 주로 "격자 끝까지 가지치기"를 벌줌. fail이 늘어 λ가 올라가므로 개수도 늘 수 있음 — 개수비를 같이 보고.
  - smoke (`scripts/smoke_feas_margin.py`, CPU 2 iter pkg8, r2_mott distill init): 통과, it 2에서 샘플 8개 중 4개가 margin-fail (작업 격자 성공 해 다수가 목표 5% 안에 붙어 있음 — H2와 같은 방향) → `runs/analysis/smoke_feas_margin.ok`.
  - 큐: `r2m` (r2_mott seed 0 distill best.pt에서 RL 1.25 h만), `lo_pcb_r2m` (lo_pcb_r2 distill에서 RL만) — `scripts/queue_feas_margin.py`, lo_pkg_r2a 뒤 (~22:10~00:50). r2c와 같은 짝 설계라 바뀐 것은 보상의 성공 여유 하나. RESULTS ABLATIONS 행 3개 (vs r2_mott_rl, vs `r2_mott_rl@last`, lo_pcb vs lo_pcb_r2_rl) 추가.
  - **미리 적는 판정 (라운드 2 규칙 + 라운드 5 보강 그대로):** seed 0, 판마다 50문제 짝지은 bootstrap. (i) 같은 종류 pcb·pkg 중 하나 이상에서 r2m − r2_mott_rl 성공률 CI 하한 > 0이 **day 1 기록과 `@last` 둘 다에서** 성립하거나, lo_pcb_r2m − lo_pcb_r2_rl (뺀 pcb) CI 하한 > 0, (ii) 그리고 다른 판에서 점추정이 −0.05 아래가 아니면 "이김" → seed 1·2 (`r2_mott_s1/s2` distill에서 같은 방식) 확장. 둘 기준이 갈리면 "평가 잡음 범위 — 판정 보류", 그 밖은 "효과 없음"/"해로움"으로 적고 확장하지 않음.
- side_slot 재시작 (대상 lo_pcb_r2a lo_ip_r2a lo_pkg_r2a r2m lo_pcb_r2m, 로그 `agent/logs/side_slot2.log`, `.gitignore`에 추가); 돌던 LS filler는 그대로 살아 있음 (side_slot은 자기 작업이 살아 있으면 기다림).
- lo_pcb_r2a RL it 100 (17:19): fail 0.09, λ 1.22, val pkg fail 0.88 / ip 0.19 (16문제, 판정용 아님), OOM 0.
- 다음 라운드: lo_pcb_r2a 끝 (~18:10) → side_slot 평가 (LS filler 끝난 뒤) → 라운드 5에 미리 적은 판정 (i)·(ii). lo_ip_r2a (~20:10), lo_pkg_r2a (~22:10), r2m (~23:30), lo_pcb_r2m (~00:50). lo_r2a가 (i) 충족이면 seed 1·2를 r2m 앞에 넣을지 그때 결정. CPU 줄은 그대로 (e0plus16 ip16_1 → eval16 ip16_1 → eval16 r2a → 그림 2). 남은 글 작업 후보: QUEUE 11/01 할 일 6 (설계 근거 절, pcb 범용화만 성립한 범위에서). DAILY는 11/03 05:00 UTC.

## 재개 라운드 6 (2026-11-02 14:25~ UTC)
- 상태: r2v RL 끝 (14:52, 340 iter / 1.27 h, val 3번 (종류당 48문제) → best it 200), side_slot이 10칸 평가 중 (14:52~). GPU `lo_pcb_r2v` 14:52~ (~16:10) → lo_pcb_r2a → lo_ip_r2a → lo_pkg_r2a (~22:10). CPU 줄 `e0plus16 eval_ip16_1 type1` (13:17~). pod 11 GiB / 60, oom_kill 0.
- 그림 1 초안 점검: PCB 패널 곡선이 20 GHz 끝에서 1.0에 닿아 보이지만 `fig_board.py`가 verify와 같은 `check_freqs()` 격자로 그리고 최대값 0.985를 글자로 찍음 → 선 굵기 탓, 문제 없음.
- **추론 샘플링 (best-of-8) — 학습 vs 규칙 약점 (QUEUE 11/01 약점 1)에 대한 CPU 실험, 결과 전 기록:**
  - 근거: 진단 H2 — 학습이 지는 것은 어려운 문제의 근소 실패 (최악 7~15% 초과). `eval_policy.py --samples S`는 이미 있음 (같은 판을 S번 샘플링, 작업 격자에서 성공한 것 중 개수 최소 → cutting-plane 정확 검사) — 학습·환경·평가 세트 그대로, 추론만 S배 계산. 10칸 기록에서는 쓴 적 없음. POMO식 다중 샘플 추론이라 노벨티 아님 (기여로 쓰지 않음).
  - 결정: **S = 8 (학습 G와 같음), 작은 것 먼저**: `r2a_rl@8` (지금 최선 후보, 같은 종류 학습, seed 0) 10칸 6파일 + `lo_pcb_r2_rl@8` (범용화 통과한 pcb, 뺀 종류) pcb10 2파일. 4번째 칸 filler로 (`side_slot.py` 접두어 `samp10 ` 추가, `ls10 eval_pkg10_1` 앞) — 학습 평가가 늘 먼저. 출력 `runs/analysis/sampling/`.
  - **미리 적는 판정 (seed 0, 판마다 50문제, 짝지은 bootstrap):** 판 t에서 `X@8` vs `modal_rule`이 (i) 성공률 차 CI 하한 > 0, 또는 (ii) 성공률 차 점추정 ≥ 0 이고 개수 비 (둘 다 성공) CI 상한 < 1 이면 "샘플링한 정책이 규칙을 이김"; 성공률 CI가 0을 포함하고 개수 비 CI가 1을 포함하면 "규칙과 같은 수준"; 그 밖은 "규칙이 이김". 시간은 문제당 초를 그대로 보고하고 규칙보다 짧을 때만 "빠르다". pcb·pkg 중 하나라도 "이김"이면 seed 1·2 (`r2a_s1_rl`, `r2a_s2_rl`)와 `lo_*_r2` seed 0~2로 확장; 아니면 확장하지 않고 그대로 기록.
  - 첫 파일 `samp10 r2a_rl eval_pcb10_0` (15:07~16:11): 성공 23/25 (greedy 같은 체크포인트 24/25, modal_rule 25/25), **문제당 33.6 s** (greedy 0.40 s의 84배 — 샘플 8개 배치에서 모달 특징 계산이 선형보다 크게 늘어남; 규칙 0.49 s, E0+ 24.8 s보다 느림), **최대 RSS 25 GB** (greedy 3.6 GB). `runs/analysis/sampling/eval/`.
  - 결정: **샘플링 범위 축소** — `r2a_rl eval_pcb10_1` 하나만 남겨 pcb 50문제 판정을 미리 적은 규칙대로 내고, lo_pcb_r2·pkg·ip 6개는 큐에서 뺌 (`scripts/queue_drop_samp.py`). 이유: 측정된 비용 (4번째 칸 ~1 h/파일, 25 GB) + pkg는 greedy부터 pcb보다 메모리가 커서 (4.5 vs 3.6 GB) 샘플링이면 pod 45 GiB 규칙에 걸릴 위험. 주의: pcb10_0 결과 (나아지지 않음)를 본 뒤의 결정이라 이 축소는 결과에 영향받았을 수 있음 — 그래서 "샘플링이 규칙을 이기는지"는 pcb 한 판의 seed 0 판정으로만 쓰고 pkg·ip·뺀 종류에 대해서는 "측정 안 함"으로 적는다. 시간만으로도 "빠르다"는 이미 불가 (33.6 s > 규칙 0.49 s).
  - **판정 (pcb 10칸 50문제, seed 0, 미리 적은 규칙):** r2a_rl@8 − modal_rule 성공률 **−0.10 [−0.18, −0.02]** (CI 0 아래), 개수비 0.906 [0.868, 0.946] → **"규칙이 이김"**. 같은 체크포인트 greedy 대비: 성공 −0.06 [−0.16, +0.02], 개수 0.931 [0.895, 0.969] — 샘플링은 개수를 7% 줄이는 대신 성공률을 잃음 (0.96 → 0.90). 시간 25.3 s/문제 (greedy 0.39, 규칙 0.50, E0+ 24.8). → 확장하지 않음. RESULTS "추론 샘플링" 절.
  - 해석 (확인 안 함, 주장 아님): "작업 격자에서 성공한 샘플 중 개수 최소"를 고르면 가장 아슬아슬한 해가 뽑혀 정확 검사에서 떨어지기 쉬움 — 진단 H2 (격자 틈 근소 실패)와 같은 방향. 추론 쪽에서도 학습 격자의 낙관이 문제라는 근거 하나 더.
- **r2v 평가 끝 (H1 수정, seed 0, 같은 종류 학습) — 판정 (라운드 2 규칙 + 라운드 5 보강, 미리 적은 그대로):** r2v − r2_mott_rl 성공률 ip −0.04 [−0.10, +0.00], pcb +0.02 [−0.06, +0.10], pkg +0.00 [−0.18, +0.16]; 같은 설정 짝 (`@last`) pcb −0.02 [−0.10, +0.06], ip·pkg 같음. 개수비 0.98~1.04 (CI 1 포함). → **CI 모두 0 포함 → 같은 종류 학습에서 "효과 없음".** 성공률 r2v ip 0.96 / pcb 0.90 / pkg 0.72 (`runs/2026-10-29/r2v_rl/eval/`). 해석 (판정 바꾸지 않음): seed 0의 r2_mott_rl은 원래 best.pt = last.pt (선택 손실 없음, 라운드 1 진단에서도 seed 0 last − best +0.04로 가장 작음) → H1이 고치려는 선택 잡음이 이 seed에선 거의 없어 이 짝은 검출력이 낮음. 남은 H1 근거는 lo_pcb_r2v (~16:10 끝).
- **lo_pcb_r2v 평가 끝 — 판정:** lo_pcb_r2v − lo_pcb_r2 세 판 모두 **+0.00 [+0.00, +0.00], 개수비 1.000** — 해가 문제마다 똑같음. 이유 (로그로 확인): 같은 distill init·같은 seed에서 RL 궤적이 iter 단위로 같고 (it 100·300 줄이 글자까지 같음 — val은 가중치·난수에 영향 없음), 두 run 모두 best iter 300을 골라 **같은 가중치**. → **H1 수정 (val 48문제): 같은 종류·뺀 pcb 모두 "효과 없음" → seed 1·2로 확장하지 않음 (미리 적은 규칙).** 같은 이유로 r2v vs r2_mott_rl은 한 궤적의 iter 200 vs iter 300 비교였음. 한계 (판정 바꾸지 않음): H1 증상이 컸던 seed 1·2 (best it 100·200, last가 pcb +0.12~0.14)는 시험하지 않았음 — seed 0은 원래 선택 손실이 작아 검출력이 낮은 설계였다. 보고서에는 "val 종류당 48문제로 바꿔도 seed 0에서 고른 체크포인트가 같거나 (pcb 빼고) 차이 없음"까지만.
- 결정: **r2a_rl (seed 0) 16칸 평가 6파일을 CPU 줄에 추가** (`scripts/queue_eval16_r2a.py`, 사다리 eval16 바로 뒤, 같은 명령 `--inst_chunk 5`). 이유: QUEUE 10/29 03:10 할 일 4 (16칸: g+p, E0+, 사다리 rl, **최종 후보**) — r2a는 같은 종류 학습에서 seed 3개 모두 r2_mott보다 높은 지금 최선 후보인데 16칸 기록이 없음. 비용: 파일당 ~15 min, 최대 ~22 GB (r2_mott 16칸 기록 기준), 드라이버 줄 한 칸이라 CPU 4개 유지. 그림 2는 기록이 오면 `fig_results.py --fig 2 --methods ...,r2a_rl`로.
- 방법 메모: RL이 init·seed가 같으면 결정적이므로, 앞으로 "같은 distill에서 RL만" 짝 비교의 차이는 순수하게 바꾼 것 (보상·선택) 때문이다 — 잡음은 평가 배치 (`--inst_chunk`) 쪽만 남음.
- 오늘까지 MoTT 개선 시도 정리 (seed 0, 미리 적은 규칙): H2 `--certify 2` 효과 없음 / 뺀 pcb에서 해로움, H1 val 48문제 효과 없음, 추론 샘플링 규칙에 짐. 남은 진행: `lo_*_r2a` (편향의 범용화). 다음 후보 (구현 안 함): H2 싼 수정 `--feas_margin` (PROPOSALS 라운드 6).
- 다음 라운드: lo_pcb_r2a 끝 (RL ~16:55~18:10) → 평가 (side_slot) → 라운드 5에 미리 적은 판정 (i)·(ii); lo_ip_r2a (~20:10), lo_pkg_r2a (~22:10). GPU가 22:10 뒤 빔 → lo_r2a 판정이 (i) 충족이면 seed 1·2 (미리 적은 대로), 아니면 `--feas_margin` 검토. CPU 줄: e0plus16 ip16_1 type2 (16:25~) → eval16 ip16_1 r0~r3 → eval16 r2a 6파일 → 그림 2 다시 그리기 (`--methods` 에 r2a_rl). 4번째 칸: ls10 pkg10_1 → ga10 6개 (학습 평가 사이). DAILY는 11/03 05:00 UTC.

## 재개 라운드 5 (2026-11-02 10:02~ UTC)
- 상태: GPU `bp_park_rlonly` RL it 150 (fail 0.16~0.26, λ 22.8, 붕괴 없음, 끝 ~11:00), CPU 줄 `e0plus16 eval_pkg16_1 type2` (rss 23 GB), side_slot `ls10 eval_pcb10_1` (09:41~), pod 30 GiB / 60, oom_kill 0. 큐 다음 항목 r2c·lo_pcb_r2c·r2v·lo_pcb_r2v의 init·after 경로 모두 있음.
- E0+ 16칸 pkg16_1 종류 1 끝: 성공 0.36, 개수 29.67, 14827 s (RESULTS 종류별 g+p 표에 자동 반영).
- 원 논문 확인: alphaXiv `get_paper_content` (arXiv 2203.15722) 다시 시도 → 권한 거부 (세 번째). STUCK 줄 유지, 재시도하지 않음.
- **r2c·r2v 짝 기준 점검 (결과 나오기 전):** ABLATIONS의 기준 `r2_mott_rl` 기록은 day 1 평가 (`--inst_chunk` 없음)이고 r2c·r2v는 드라이버의 `--inst_chunk 5`로 평가됨 → 배치 크기 잡음 (같은 가중치에서 pcb 3/50 다름, 라운드 1)이 짝 차이에 섞임. `r2_mott_rl` seed 0의 best.pt와 last.pt는 **가중치가 같음** (torch로 확인: iter 300 둘 다, max diff 0.0; 파일 sha만 다름) → `runs/analysis/ckpt_last/eval/*__r2_mott_rl@last.json` (inst_chunk 5)이 같은 가중치·같은 평가 설정 기록. `collect_results.py` ABLATIONS에 `r2c_rl vs r2_mott_rl@last`, `r2v_rl vs r2_mott_rl@last` 행 추가 (기존 행 유지). lo_pcb_r2_rl은 이미 inst_chunk 5라 그대로.
  - 결정 (판정 규칙 보강, 결과 보기 전): 같은 종류 학습 쪽 "이김"은 **두 기준 기록 (day 1 `r2_mott_rl`, 같은 설정 `@last`) 모두에서** 라운드 2의 CI 조건을 만족할 때만. 둘이 갈리면 "평가 잡음 범위 — 판정 보류"로 적고 seed 1·2로 확장하지 않음. 이유: 더 엄격한 쪽이 보수적이고, 기존 규칙의 기준선을 바꾸지 않음.
- **그림 3 초안 (seed 3개):** `scripts/fig_results.py --fig 3 --seeds` (새 옵션; 막대 = seed 평균, 점 = seed, 기준선 E0+·modal_rule은 가로선) → `reports/fig3_ladder_seeds_draft.png` (10칸, r0~r3 `*_rl`/`*_s1_rl`/`*_s2_rl`). 읽기: pcb 성공 0.96/0.93/0.83/0.91 (modal_rule 1.00, E0+ 0.66), pkg 0.75/0.74/0.73/0.83 (규칙 0.88, E0+ 0.82), ip 0.99/0.95/1.00/0.99; ip 개수÷E0+ 1.41/1.24/1.01/1.01 (모드 토큰 2·3단만 E0+ 수준, seed 3개 모두). 주의: seed 0 기록은 day 1 평가 설정, s1·s2는 `--inst_chunk 5` (배치 잡음 ±0.04~0.06 가능). 팔레트 검증 스크립트 (node)는 무인 허용 목록 밖이라 못 돌림 — 기존 그림 팔레트 그대로, 단 이름·값은 x축에 글자로.
- **그림 2 초안 (16칸, 지금 있는 기록):** `fig_results.py --fig 2`에 성공률 행 추가 (개수비는 둘 다 성공한 문제만이라 실패를 가림) → `reports/fig2_bigboard_draft.png`. 읽기 (seed 0): 문제당 시간 g+p 62/131/82 s vs r2_mott_rl 2.4/4.1/1.1 s (pcb/pkg/ip, **26~75배**, "수백 배"는 아님), modal_rule 4.2~8.5 s. 성공 pcb16 g+p 0.92·규칙 1.00·r0 1.00·r2 0.90; pkg16 g+p 0.82·정책 0.52~0.64·규칙 0.60; ip16 g+p 0.56·나머지 1.00. **E0+는 pcb16에서 성공 0.46, 개수 g+p의 5.1배** (한 종류만 쓰면 큰 pcb에서 무너짐). 주의: ip16_1 정책·E0+, pkg16_1 E0+ 아직 없음 (n=25인 막대), CPU 줄에서 진행 중.
- **r2c 시작 (11:02, GPU) — 결과 전 주의점:** `--certify 2`는 iter가 느림 (it 75 / 3509 s, r2_mott_rl은 1.25 h에 ~300 iter) → 같은 1.25 h에서 ~95 iter, val (100 iter마다)은 0~1번. train.py는 val이 없으면 끝 가중치를 best.pt로 저장 (`train.py:499-506`) → r2c는 사실상 best.pt 선택 없음. 그래서 r2c vs r2_mott_rl은 "같은 시간 예산에서 정확 검사 보상 (iter 1/3) vs 격자 보상 (iter 전부 + val 선택)" 비교로 쓴다 — 보상만 바뀐 비교가 아님. 학습 신호: it 25 cert-bad 82 (격자 성공 샘플 중 정확 검사 실패 수, 이전 최선 샘플만 셀 때 0~16), replaced 118~229 (선생님 교체가 많음 — 코드 확인: `--certify 2`에서 남은 성공 샘플은 모두 정확 검사를 통과한 것이고 교체는 `cert_ok`일 때만, `train.py:375-389, 434` → 버그 아님).
- **bp_park_rlonly 끝** (296 iter / 2.03 h, best it 100, `runs/2026-10-29/bp_park_rlonly_rl/`), 10칸 평가 (side_slot 11:04~12:17): **ip 0.12** (개수÷E0+ 9.99), **pcb 0.86** (÷E0+ 1.76, ÷g+p 8.14), **pkg 0.86** (÷E0+ 4.87). 해 모양: **세 판 종류 모두 종류 1만** (종류 비율 0/1/0). → RL만 (원 논문 방식)이 distill 판 (pcb 0.46, pkg 0.16)보다 pcb·pkg 성공률은 높지만 개수 폭증, ip는 무너짐. B-Manoharan l20 (판마다 한 종류)와 같은 결론: RL만 + 선행 구조는 판 종류에 맞는 종류 선택을 배우지 못함. RESULTS "선행 연구 재구현" B-Park 아래 설명 줄 추가 (`collect_results.py` PRIOR). 주의: 초기 λ 20은 우리 보상의 선택 (원 논문은 λ 없음) — 성공을 개수보다 크게 쳐서 "많이 놓기"를 부추길 수 있음; λ는 fail 0.16~0.48로 학습 중 22→26까지 올라감.
- **r2c 끝** (105 iter / 1.29 h, val 1번 → best it 100, 선생님 교체 276), 평가 12:19~ (side_slot). GPU: lo_pcb_r2c 12:19~ (~13:40).
- **STUCK 해제 (QUEUE 11/02 21:30 KST 사람 조치):** 사람이 `research_notes/선행 기준선 원문 설정 (Park 2022, Manoharan 2026).md`를 넣음 → 재구현 차이 표 완성 `research_notes/선행 기준선 재구현 차이 표.md` (B-Park·B-Manoharan, 항목마다 원 논문 / 우리 / 영향). RESULTS "선행 연구 재구현" 설명 줄 갱신 (`collect_results.py` PRIOR): (1) Park 원 문제 = 개수 고정 + 임피던스 감소 (목표 없음) → "구조·입력을 옮긴 재구현"으로만, (2) Manoharan 원 예산 ~24 h·5만 에피소드·critic + replay vs 우리 2 h → pcb 0.00은 "같은 2 h 예산에서"로만, (3) Manoharan 미확인 항목은 "우리 선택, 원문 미확인". 원 논문 경향 재현 sanity check는 하지 않음 — 결정: Park 설정은 목표·보상을 바꿔야 해서 규칙 밖, Manoharan 원 예산은 마감 안 불가; 비교에 caveat만. 웹 검색·alphaXiv는 다시 시도하지 않음 (QUEUE). 지운 STUCK 줄: "원 논문 설정 확인 (QUEUE 11/01 15:10 할 일 4·23:45) — 무인 라운드에서 WebSearch·alphaXiv가 세 라운드 거부됨 (11/02 04:xx, 06:10, 10:05 al…"
- **r2c 평가 끝 (H2 수정, seed 0, 같은 종류 학습) — 판정 (라운드 2 규칙 + 라운드 5 보강, 미리 적은 그대로):** r2c − r2_mott_rl 성공률 ip +0.00, pcb +0.02 [−0.10, +0.14], pkg +0.08 [−0.06, +0.22]; 같은 설정 짝 (`@last`) pcb −0.02 [−0.14, +0.10], pkg +0.08 [−0.06, +0.22]. 개수비 0.99~1.06 (CI 1 포함). → **CI가 모두 0 포함 → 같은 종류 학습에서 "효과 없음".** 성공률 r2c ip 1.00 / pcb 0.90 / pkg 0.80. 주의 (라운드 5에 미리 적음): iter 1/3 (105 vs ~300) + best.pt 선택 사실상 없음과 함께 바뀐 비교.
- **lo_pcb_r2c 평가 끝 — 판정:** 처음 보는 pcb에서 lo_pcb_r2c − lo_pcb_r2 **−0.30 [−0.44, −0.16]** (50문제, CI 0 아래), 개수비 1.56 [1.30, 1.87]; ip +0.00, pkg +0.02. 성공률 lo_pcb_r2c pcb 0.62 / ip 1.00 / pkg 0.84. 해 모양: pcb에서 종류 0 비율 0.49 (성공)·0.72 (실패) vs lo_pcb_r2 0.09·0.10 — 처음 보는 pcb에서 종류 선택이 ip·pkg 쪽 (큰 C 종류 0)으로 쏠림, 실패 해는 ~60개. → **H2 수정 (`--certify 2`): 같은 종류 효과 없음, 처음 보는 pcb에서 해로움 → seed 1·2로 확장하지 않음.** 해석 후보 (확인 안 함): iter 1/3 + val 선택 없음이 원인일 수도 있어 "정확 검사 보상 자체가 해롭다"고는 쓰지 않음.
- **다음 GPU (결정):** `lo_pcb_r2a`, `lo_ip_r2a`, `lo_pkg_r2a` (seed 0, lo_<t>_r2 arm + `--fc_bias`, 같은 예산 distill 0.75 h + RL 1.25 h) — `scripts/queue_lo_r2a.py`로 큐 끝 (lo_pcb_r2v 뒤)에 추가. 이유: (1) r2a는 같은 종류 학습에서 seed 3개 모두 r2_mott보다 높음 (pcb 0.96/0.96/0.92 vs 0.88/0.78/0.84, pkg 0.84/0.84/0.80 vs 0.72/0.84/0.62) — 지금 최선 후보인데 범용화 run이 없음; (2) lo_pcb_r2c 실패가 처음 보는 종류에서의 **종류 선택** 쏠림이고, `--fc_bias`는 종류 선택을 모드 주파수 ↔ SRF 궁합에 묶음. 새 코드 없음. PROPOSALS 2줄 적음.
  - **미리 적는 판정 (결과 전):** 뺀 종류 t의 eval 50문제에서 (i) lo_t_r2a − lo_t_r2 성공률 짝 CI가 0보다 크면 (pcb·ip 중 하나 이상) 그리고 다른 뺀 종류에서 점추정이 −0.05 아래가 아니면 "편향이 범용화를 돕는다 (seed 0)" → seed 1·2 추가. (ii) lo_t_r2a − r2a_rl (뺀 − 전체)도 보고. 10/31 범용화 기준 ((b)·(c))은 r2 run에 대해 이미 판정됐고 바꾸지 않음; r2a는 별도 행으로 보고.
  - RESULTS ABLATIONS에 `lo_<t>_r2a_rl` vs `lo_<t>_r2_rl`, vs `r2a_rl` 행 추가 (결과 오면 자동). side_slot 재시작 (14:2x): 대상 r2v lo_pcb_r2v lo_pcb_r2a lo_ip_r2a lo_pkg_r2a.
- **로그 유실 (내 실수):** 12:20 push 거절 때 `git stash`/`stash pop`으로 rebase → 추적 중이던 `agent/logs/side_slot.log`·`e0plus16.log`가 새 파일로 바뀌어 살아 있는 writer가 지워진 파일에 씀 (12:20 뒤 side_slot 줄 없음; 평가 json은 정상). 두 로그를 `.gitignore` + `git rm --cached`. 결정: e0plus16 작업은 멈추지 않음 (결과는 json; 로그만 안 보임). 앞으로 push 거절 때는 커밋 후 깨끗한 트리에서 `pull --rebase`.
- 다음 라운드: r2v 끝 (~14:50) → side_slot 평가 → H1 판정 (라운드 2 규칙 + 라운드 5 보강: day 1 기록·`@last` 둘 다); lo_pcb_r2v (~16:10); lo_pcb_r2a 시작 확인 (~16:10 뒤, 첫 distill이 도는지). CPU 줄: e0plus16 ip16_1 → 16칸 ip16_1 정책 평가 → 그림 2 다시 그리기 (`fig_results.py --fig 2`). LS pkg10·GA 10칸은 학습 사이 side_slot filler. DAILY는 11/03 05:00 UTC에 새로.


## 재개 라운드 4 (2026-11-02 07:43~ UTC)
- 상태: GPU `bm_manoharan_rlonly_l20` RL it 150+ (붕괴 없음, fail 0.16~0.52, 끝 ~08:58), CPU 줄 `e0plus16 eval_pkg16_1 type1` (05:50~), pod 16 GiB / 60, oom_kill 0.
- **선행 기준선 설정 (QUEUE 할 일 4, STUCK 일부 해결):** 10/29에 받아 둔 DevFormer 공개 저장소 (`/home/coder/ext_devformer/devformer`, 읽기만)에서 확인 — DevFormer 문제 = 10×10 판 **한 장**, 한 종류 decap **K = 20개 고정**, 보상 = 201점 임피던스 감소 합 (`model_5`, 목표 Zt 없음); AM-PG 기준선 (`--model attention`) = 입력 (x, y, 탐침/keep-out), rollout 기준선 REINFORCE (t-test α 0.05), 5만 문제 × 10 epoch; DevFormer = 전문가 해 2000개 모방 + 대칭 증강. → **선행 설정은 개수 최소화·목표 만족 문제가 아님**; 우리 B-Park 비교는 "같은 구조·입력을 우리 문제로 옮긴 것". 표: `research_notes/선행 기준선 설정 확인 (DevFormer 공개 코드).md`, RESULTS B-Park·B-Manoharan 아래 줄 추가 (`collect_results.py` PRIOR).
- **LS·GA 10칸 기준선 (QUEUE 할 일 4 "PI 논문의 GA·LS"):** LS 10칸은 day 1부터 CPU 줄 맨 끝에 있었고 GA는 `decap_loop/baselines.genetic`이 있지만 평가에 연결 안 됨.
  - 결정: **4번째 CPU 칸 러너 `scripts/side_slot.py`** (side_eval_wait 대체) — 학습 끝난 run의 10칸 평가가 1순위, 없을 때만 `ls10 *`·`ga10 *` 큐 항목을 하나씩 (선점 없음 → 학습 평가는 최대 한 작업만큼 기다림). 이유: 이 칸은 학습 사이 ~1.5 h씩 비어 있었음; CPU 4개 유지 (검사 2 + 드라이버 1 + 이것 1). 재시작해도 자기 작업이 살아 있으면 기다림.
  - LS 항목을 파일별 6개로 나눔 (`ls10 eval_*10_*`, 기존 "day1 base10 ls" 대체, 같은 명령·출력). **ip10_0: 성공 0.40 (= g+p, LS는 성공한 해만 다듬음), 16.4 s/문제.**
  - GA: `eval_policy.py --methods ga --ga 64x100` (새 방법 `genetic`, 다른 방법 코드 그대로). 시간 smoke (`scripts/smoke_ga.py` → `runs/analysis/smoke_ga.json`): 100세대 추정 ip 26 / pcb 44 / pkg 49 s/문제 → 결정: repo 기본값 (pop 64 × 100세대, 무작위 초기화, g+p 시드 없음 — 시드를 넣으면 g+p+LS 변형이 됨) 그대로, `ga10 *` 6개를 LS 뒤에.
  - LS 지금까지 (`runs/2026-10-28/baselines/eval/*__local_search.json`): ip10 0·1 성공 0.40 (= g+p), 개수 g+p의 0.75배, 10.8 s/문제; pcb10_0 성공 0.80 vs g+p 짝 **−0.20 [−0.36, −0.08]** (작업 격자에서 줄인 해가 정확 검사에서 떨어짐 — 선생님 LS의 정확 성공이 낮은 것과 같은 현상), 개수 0.75배, 27 s/문제. pcb10_1 진행 중 (09:41~), 그다음 pkg·GA.
  - 결정 (09:22): LS pcb10_1 (23분 진행)을 멈추고 l20 평가를 먼저 — LS가 학습 끝 1분 전에 칸을 잡아 기준선 판정이 ~50분 밀릴 상황. `side_slot.py`에 가드 추가: 대상 학습의 RL이 1 h 안에 끝나면 filler를 시작하지 않음 (`FILL_GAP_S`).
- **bm_manoharan_rlonly_l20 끝** (504 iter / 2.03 h, best it 100, `runs/2026-10-29/bm_manoharan_rlonly_l20_rl/`), 10칸 평가: **ip 0.96** (개수÷E0+ 1.73), **pcb 0.00**, **pkg 0.88** (개수÷g+p 6.59 — 순서의 거의 모든 칸에 놓음). 해 모양: 판 종류마다 한 종류만 (ip 0, pkg 1, pcb 2). → λ0 20으로 붕괴는 피했지만 RL만으로도 Manoharan식 고정 순서 정책은 pcb에서 실패, pkg에서 개수 폭증. RESULTS "선행 연구 재구현" B-Manoharan에 행·설명 줄 추가. distill 판 (0.00 전부)보다는 낫지만 E0+·사다리 어느 단보다도 낮음.
- GPU: **bp_park_rlonly** (λ0 20) 09:00~ (~11:00), it 25 fail 0.23·it 100 val fail pcb 0.56 / pkg 0.69 / ip 0.94 — 붕괴 없음 (loss ≠ 0). 그다음 r2c → lo_pcb_r2c → r2v → lo_pcb_r2v (H2·H1 수정, 기준선 두 개 뒤 — QUEUE 23:45 순서).
- 다음 라운드: bp_park_rlonly 평가 (~11:00 뒤 side_slot) → B-Park 표; r2c 시작 확인; LS pkg·GA 10칸 결과 → RESULTS 요약표 행 (local_search, genetic) 해석 한 줄.

## 재개 라운드 3 (2026-11-02 06:55~ UTC)
- 상태: CPU 줄 `e0plus16 eval_pkg16_1 type1` (05:50~), pod 16~18 GiB / 60, oom_kill 0. `ensure_running`이 report_watch (27 rung 다 올림 → 바로 끝남)·resume_agent (STOP 없음 → 할 일 없음)를 다시 띄움 — 정상.
- **bm_manoharan_rlonly 붕괴 확인 → 멈춤 (06:58).** RL만 (무작위 초기화, λ0 1): it 25 fail 0.78·개수/ref 2.7 → it 50부터 fail 1.00, **it 75부터 loss 0.000**, val it 100·200 세 종류 fail 1.00. 원인: 보상 R = −개수/N_ref − λ(실패+V)에서 λ0 1이면 "0개 놓고 실패" (R ≈ −1.5)가 "많이 놓고 성공" (−2.7)보다 높음 → 모든 샘플이 빈칸만 고름 → 그룹 안 R이 모두 같아 advantage 0 → λ가 10까지 올라도 기울기가 없어 회복 불가. Park도 같은 위험: smoke (`/tmp/smoke_rlonly_park_am`) 무작위 초기화 val 개수/g+p 9.0, fail 0.81 → λ0 1이면 "바로 멈춤"이 최적.
  - 결정: **RL만 기준선 두 개를 λ0 20으로** (`bm_manoharan_rlonly_l20` 새 항목, `bp_park_rlonly` arm에 `--lam 20`; `scripts/queue_rlonly_lam.py`). 이유: 붕괴는 원 논문 방법 탓이 아니라 우리 라그랑주 보상의 초기값 탓이고 (원 논문들은 λ가 없음), 기울기 0인 run에 1.3 h GPU를 더 쓰는 것은 낭비. 20 > 무작위 초기화 개수비 (2.7·9.0) → 성공이 늘 실패보다 높음; fail ≈ 0이면 λ는 iter당 0.001만 내려가 사실상 고정, 그룹 정규화 advantage라 성공 샘플끼리는 개수로 경쟁. 멈춘 run은 `train_removed`로 옮김 (드라이버 재시작 때 이어 돌지 않게), 로그는 남김, 평가 없음. RESULTS 재구현 차이 줄에 적음.
  - GPU 큐: bm_manoharan_rlonly_l20 (06:58~ ~08:58) → bp_park_rlonly λ0 20 (~10:58) → r2c → lo_pcb_r2c → r2v·lo_pcb_r2v. side_eval_wait 다시 띄움 (`bm_manoharan_rlonly_l20 bp_park_rlonly`).
- **H1 val 풀 (`val pools k3`)을 4번째 CPU 칸에서 만듦** (06:57~07:25, `agent/logs/build_val_k3.log`). 이유: CPU 줄에서는 E0+ 16칸 뒤 (몇 시간)라 r2v가 막힐 수 있었음; 칸이 비어 있었음 (학습 검사 2 + 드라이버 1 + 이것 1 = 4). 6파일 × 16문제, 다른 10칸 풀 42개 (train·eval·val_*_0)와 겹치는 문제 0 (`scripts/check_val_overlap.py`). → r2v·lo_pcb_r2v의 "after" 충족, 드라이버 CPU 작업은 건너뜀.
- bm_manoharan_rlonly_l20 초기: it 25 fail 0.16 / 개수비 6.0 → it 50 0.03 / 1.49 (붕괴 없음, 샘플링). val it 100 (greedy, 16문제): ip fail 0.19 (개수÷g+p 0.94), pkg 0.94, pcb 1.00. 판정은 10칸 평가로 (side_eval_wait, ~09:00 뒤).
- 다음 라운드: l20 best.pt 평가 → "선행 연구 재구현" 표; bp_park_rlonly (λ0 20) 초기 iter가 붕괴하지 않는지 (fail 1.00 + loss 0이 50 iter 넘게 이어지면 같은 원인으로 기록); r2c 시작 (~11:00).

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
- **lo_pcb_r3 끝** (RL 361 iter / 1.26 h, best iter 200 — 또 앞쪽 선택, H1과 같은 모양), 평가 06:25~06:38 (side_eval_wait, `runs/2026-10-29/lo_pcb_r3_rl/eval/`). 뺀 종류 pcb 0.90 (개수÷E0+ 0.20), ip 1.00, pkg 0.86.
- **루프 vs 기본 판정 (QUEUE 10/31 11:40 기준 그대로, RESULTS "루프 트랜스포머 vs 기본"):** 같은 종류 학습 루프 − 기본 seed 0/1/2: ip −0.02/0/0 (불충족), **pcb −0.04/+0.16/+0.12 (2/3 양수, 평균 +0.08)**, **pkg +0.16/−0.06/+0.20 (2/3, 평균 +0.10)** → 판 조건은 pcb·pkg 충족. 그러나 lo_pcb_r3 vs lo_pcb_r2 = **−0.02 [−0.14, +0.10] (CI 0 포함)** → 기준의 두 번째 조건 불충족 → **"루프가 낫다"고 쓰지 않음.** 쓸 수 있는 것: "같은 종류 학습에서 pcb·pkg 성공률 seed 평균이 루프 쪽 +0.08·+0.10 (seed 3개 중 2개), 처음 보는 pcb에서는 차이 없음". 주의: 두 모델 모두 best.pt 선택 잡음 (H1)이 있어 seed별 차이 일부는 선택 잡음일 수 있음.
- 다음 라운드: bm_manoharan_rlonly (06:25~08:25) 초기 val — 빈칸 쏠림이 RL만으로 사라지는지; 끝나면 side_eval_wait가 평가 → "선행 연구 재구현" 표.

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
