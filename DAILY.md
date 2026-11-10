# DAILY — 2026-11-10 (루프 재개 9일차, 01:20 UTC 작성; 대상 기간 11/09 01:06 ~ 11/10 01:06 UTC, 라운드 209~277)

## 요약 (10줄 이내)
- **N 하네스 끝:** 서버 1 cpu 몫 `bm2_rlonly_rl+rep4` pkg16_1 (07:47 UTC, rc=0, 4.4 h) → pkg16 [50] 0.92. 서버 2 nrest 10 개 push, 마지막 `bm2_rlonly_s1_rl` pkg16_0 은 서버 2 hang 뒤 사람이 서버 1 에서 다시 돌려 19:07 UTC 끝 (2.5 h, 0.92) → pkg16 [50] 완성. `bm2_rlonly_s2` 는 OOM STUCK, 사람 결정대로 seed 0·1 로 보고.
- **N2 (뺀 pkg 모델 세 종류 하네스) 48/48 다 참** (서버 2). 최저 종류 하네스: MoTT+P1 0.99 (pkg) / 좌표 0.92 (pkg, −0.07 [−0.11, −0.04]) / B-Park 0.04 (pcb) / B-Manoharan v2 0.03 (pcb) / B-DevFormer 0.45 (pkg). 판정은 사람.
- **S (보충 seed, 사람 `seeds_s.sh`):** 서버 1 은 17:07 UTC "all done" (lo_pkg_bdf_s3/s4, bdf_s4, r2pk_s4/s5 학습 + 정책·하네스 평가). 예: r2pk_s4_rl+rep4 pcb10 1.00 / pkg10 0.94 / ip10 1.00 / pkg16 0.92; r2pk_s5_rl+rep4 1.00 / 0.98 / 1.00 / 0.82; bdf_s4_distill+rep4 pcb10 0.88 / pkg10 0.64 / ip10 0.14. 서버 2 S (bdf_s3, r0_coord_s4, bm2_rlonly_s3) 도 들어옴. 판정 없음.
- **T (속도, 서버 2):** 1차 21 기록 + PCB형 재측정 6 기록 → RESULTS "T 속도 측정" 절에만 (`speed_table.py` 그대로); 같은 앞 10문제라 본 16칸 표에는 넣지 않음 (`SPEED_EXCLUDE`).
- **U (속도 재측정, 새 문제 50개/종류, 사람 PROPOSALS 11/09 U):** 서버 2 몫, 아직 기록 push 없음 (00:52 UTC 사람 `speed_fresh.py` 폴더 생성 고침).
- **사고 하나:** 사람 커밋 31663a9 가 추적 파일 1443 개를 지움 → 16:41 UTC 복원 (682ddb1) + 사람 revert 9449762, 내용 차이 0. 그 뒤 추적 파일 수 늘기만 함 (지금 2615).
- RESULTS 1548 → **1610 기록**. 서버 1 GPU 는 S 끝 (15:08 UTC) 뒤로, cpu 는 19:07 UTC 뒤로 비어 있음 (사람 항목 없음).

## 결과표
[RESULTS.md](RESULTS.md) — 바뀐 절: "뺀 pkg 모델의 세 종류 하네스 (N2)" (48/48 완성, 최저 종류 표), "T 속도 측정" (새 절, PCB형 재측정 포함), S 보충 seed 자동 행 (r2pk_s4/s5, bdf_s3/s4, lo_pkg_bdf_s3/s4, r0_coord_s4, bm2_rlonly_s3), B-Manoharan v2 RL만 seed 1 pkg16.

## 발견한 문제
- 31663a9 대량 삭제 (위). `scripts/restore_mass_delete.py` 로 복원; 드라이버가 끝난 run 을 다시 시작하려다 코드 없이 rc=1 (GPU·체크포인트 영향 없음).
- 16칸 pkg 하네스는 B-Manoharan 류에서 2.5–4.6 h (pkg16_1 16655 s, rss 28 GB). 라운드 256 에 worker 를 `eval_policy` 패턴으로 찾다 "사라짐" 으로 잘못 적음 → 257 에 정정 (worker 는 `forkserver` 로 보임). U 의 4 h 제한과 부딪힐 수 있음 (사람에게 적음).
- `bm2_rlonly_s1_rl` 과 `bm2_rlonly_rl` 이 pkg16_1 에서 성공률·개수가 똑같음 (0.92, 146.09) — "seed 가 달라도 같은 값" 꼴, 사람 판단용 (라운드 233).
- collect_results 가 가끔 120 s 를 넘음 → `bg.py` 로 돌림. `agent/loop.sh` autostash 로 드라이버 로그 inode 문제 그대로 (드라이버는 쓸 것 없어 손실 없음).

## 오늘 추천 큐 (5개)
1. 서버 2 U 기록 push 받기 → collect_results, `runs/pool_speed`·`speed_*` 가 본 표에 섞이지 않는지 기록 수 점프 확인. 이유: PROPOSALS 11/09 U 가 보고서 주 결과 (연구 질문 1).
2. T 재측정 greedy+prune PCB (멈춘 작업) 를 U 뒤 한 번 더 (서버 2, 사람). 이유: PROPOSALS U 멈춤 처리.
3. 보고서 표·그림을 1610 기록으로 다시 만들기 (사람). 이유: N·N2·S·F 다 참, 11/16 제출.
4. N2·F·S 읽기 규칙 적용 (사람; 에이전트는 숫자와 CI 만). 이유: 판정은 사람 몫.
5. 서버 1 GPU·cpu 를 쓸 일이 있으면 사람이 QUEUE 에 적기 — 지금 비어 있음. 이유: 에이전트는 스스로 항목을 넣지 않음.

## 하지 말아야 할 것
- N2·nrest·U·T 를 서버 1 큐에 다시 넣기; `bm2_rlonly_s2` 를 같은 플래그로 다시 돌리기.
- 에이전트 스스로 새 학습·arm·seed·세트 추가 (QUEUE 11/08 11:30 6번).
- 서버 2 시간과 서버 1 시간 섞기; T 기록을 본 16칸 표에 합치기.
- `git stash`/rebase 로 추적 로그 건드리기.
