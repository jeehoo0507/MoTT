# DAILY — 2026-11-07 (루프 재개 6일차, 01:30 UTC 작성; 대상 기간 11/06 00:05 ~ 11/07 01:30 UTC, 라운드 58~64)

## 요약 (10줄 이내)
- **B1 선행 연구 재구현 seed 0·1·2 완료** (4 재구현 × 3 seed, 같은 2 h 예산). RESULTS "선행 연구 재구현 seed 0·1·2": r2pk가 3 seed 묶음 CI로 12행 중 10행에서 높음; 구분 안 됨 2행 = `bp_park_rlonly` pkg (−0.020 [−0.107, +0.067]), `bm_manoharan_rlonly_l20` ip (+0.027 [−0.007, +0.067]). 재구현은 seed 사이 폭이 매우 큼 (예: `bp_park_rlonly` ip 0.12 / 0.98 / 0.10) → 보고서는 묶음 CI만.
- **r2pk seed 2 규모 평가 끝** (16칸 6파일 + 20칸 6파일) → 규모 곡선 r2pk seed 3개 조건.
- **LS16 (`--ls_chunk 8`)**: ip16_0 0.64, pcb16_1 0.64 (문제당 114.6 s) 완주; ip16_1 진행 중 (15 h째); pcb16_0은 다른 작업과 겹쳐 두 번 중단 → 맨 끝에 혼자 다시.
- 11/06 02:06 UTC ~ 11/07 01:00 UTC: Claude 사용량 한도 + Codex 관리 기간. 사람이 Codex 코드 되돌림 (f687b21). Codex 기간 진단 (peak mask 등)은 보고서 근거에서 제외. 이 기간 학습은 없었고 B1 학습은 모두 Codex 전 코드.
- 새 수정 시도: 없음 (11/6 12:00 UTC 마감, 최종 구조 r2pk 그대로). GPU 비어 있음.

## 결과표
[RESULTS.md](RESULTS.md) — 바뀐 행: "선행 연구 재구현 seed 0·1·2" (rlonly·manoharan seed 1·2 칸), 규모 곡선 `r2pk_s2_rl` 16·20칸, LS16 pcb16_1.

## 발견한 문제
- 루프가 하루 동안 사용량 한도로 매 라운드 바로 끝났고 push도 안 됨 → 사람이 `loop.sh`에 한도 대기·pull 수리를 넣음.
- B1 평가 기록 24개 + r2pk_s2 규모 6개가 `runs/` (gitignore)라 커밋 안 돼 있었음 (report_watch는 끝난 상태) → 이번 라운드 `git add -f`.
- LS16 pcb16_0: pod 전체 기준 mem_guard라 다른 작업과 겹치면 LS가 대신 죽음 (3번째 확인).

## 오늘 추천 큐 (5개)
1. LS16 ip16_1 → pkg16_0 → pkg16_1 → pcb16_0 (드라이버, 혼자) → 그림 2 LS 열. 이유: AGENT_PROMPT 그림 2; 11/09까지 못 끝나는 파일은 캡션에 표시.
2. 11/8 동결 전 마지막 점검: 최종 구조 r2pk seed 3개가 모든 표 (10칸·뺀 학습·규모 곡선·chip)에 있는지 `collect_results` 표로 확인. 이유: 그림 1·3.
3. 그림 1·2·3 초안을 최신 RESULTS로 다시 그림 (`reports/fig{1,2,3}_*_r2pk*.png`). 이유: 11/9~10은 그림·표만.
4. 캡션 메모·재구현 차이 표 확정 (B1 seed 폭 규칙 추가함). 이유: 보고서 기준선 절.
5. GPU 새 학습 없음 유지. 이유: QUEUE 11/03·11/07, 동결 11/8.

## 하지 말아야 할 것
- Codex 기간 결과 (`reports/peak_mask_diag_20261106/`, delta/pkg_guard/curriculum/post-training)를 RESULTS·보고서에 넣기.
- LS16이 도는 동안 4번째 칸에서 16칸 정책 평가 시작; CPU 4개 초과; 새 학습·새 수정.
- 평가 세트·목표·검사·판정 규칙을 결과 보고 바꾸기; 재구현 seed 하나 숫자 쓰기; r2pk 범용화를 ip·pkg로 쓰기.
