# DAILY — 2026-11-09 (루프 재개 8일차, 01:06 UTC 작성; 대상 기간 11/08 01:15 ~ 11/09 01:06 UTC, 라운드 123~208)

## 요약 (10줄 이내)
- **F 끝:** F3 22/22 (11/08 02:01 UTC). 최종 방법 `MoTT+P1+V` 표의 10·16·20칸 칸이 다 찼다 (ip20 4개 모두 8/8, 되먹임 쓰지 않음).
- **밤 묶음 N (사람, QUEUE 11/08 11:30 KST) GPU 9/10 끝:** `lo_pkg_bp_park{,_s1,_s2}`, `lo_pkg_bm2_rlonly{,_s1,_s2}`, `bm2_rlonly{,_s1}`, `lo_pkg_r0_s3`. **`bm2_rlonly_s2` 는 STUCK** (it 300 에서 `probe_impedance` 의 CUDA OOM 이 처음 run 과 resume 에서 똑같이 남; best.pt 는 it 100 뿐, metrics.json 없음 → 평가 없음, 사람 결정 필요).
- 뺀 pkg 10칸 + `--repair 4` 3 seed 묶음 (RESULTS "N2" 절, 150문제): MoTT+P1 0.71 → **0.99**; 좌표 0.80 → 0.92 (차 −0.07 [−0.12, −0.03]); B-Park 0.79 → 0.91 (−0.09 [−0.14, −0.04]); B-Manoharan v2 0.83 → 0.89 (−0.10 [−0.15, −0.05]); B-DevFormer (서버 2) 0.20 → 0.45 (−0.54 [−0.62, −0.46]).
- 같은 종류 학습 B-Manoharan v2 RL만 (`bm2_rlonly_rl` + rep4): pcb10 **0.04 [50]** (단독과 같음, 수리 이득 없음), pkg10 0.90 [50] (단독 0.84), ip10 1.00 [50] (단독 0.96). 판정 없음.
- `lo_pkg_r0_s3` (좌표 seed 3) 정책 단독 10칸: pcb 0.98 / ip 0.98 / 뺀 pkg 0.64 [50]. 좌표 seed 1·2·3 하네스 pkg16·pkg20 기록도 들어옴 (표 "좌표+V").
- **N2 늦음 (보고만):** 사람이 48개를 서버 2로 옮김 (QUEUE 11/09 01:40 KST). 마감 11/09 00:00 UTC 지났고 01:05 UTC 에 다시 세도 서버 2 push **0 / 48** → N2 표의 MoTT+P1·좌표·B-Park·B-Manoharan v2 pcb/ip 칸과 "최저 종류" 표가 비어 있음. 서버 1 은 다시 넣지 않음.
- GPU: 11/08 21:47 UTC 부터 빔 (사람 항목 없음). cpu: N 하네스 `bm2_rlonly_rl+rep4` pkg16_0 진행 중, 그 뒤 pkg16_1, `bm2_rlonly_s1_rl+rep4` 8개 (`_s2` 8개는 metrics.json 없음 → `after` 게이트로 건너뜀).

## 결과표
[RESULTS.md](RESULTS.md) — 바뀐 절: "뺀 pkg 모델의 세 종류 하네스 (N2)" (새 표, 최저 종류 표 포함), "최종 방법 MoTT+P1+V" (F3 칸 완성), "좌표+V" pkg16/20 (seed 1·2·3), 선행 연구 재구현 (B-Park·B-Manoharan v2 뺀 pkg, `bm2_rlonly` 같은 종류) 자동 행.

## 발견한 문제
- `bm2_rlonly_s2` GPU OOM 결정적 재현 (24.5 GB 카드, it 300 val 직후 `manoharan_reward` → `probe_impedance` `torch.linalg.solve`). 같은 플래그로는 다시 돌리지 않음 (PROGRESS 첫 줄 STUCK).
- N2 가 서버 2 로 옮겨진 뒤 push 가 없음 (원인 알 수 없음, 서버 2 는 사람 관리).
- `agent/loop.sh` autostash 가 드라이버 로그 inode 를 갈아끼우는 문제 그대로 → 라운드마다 `scripts/restore_proc_log.py 1792539 agent/logs/pipeline_day2.log`.
- pcb10 하네스 평가는 B-Manoharan 류에서 한 파일 47~55 분 (pcb10_0 3286 s, pcb10_1 23:17–00:04; 성공 0.04 라 실패 문제가 되먹임을 계속 씀) → cpu 한 줄이 느림.

## 오늘 추천 큐 (5개)
1. cpu N 하네스 남은 것 (`bm2_rlonly_rl` pkg16_1, `bm2_rlonly_s1_rl` 8개) 그대로 끝내기. 이유: QUEUE 11/08 11:30 KST N 4번.
2. 서버 2 N2 48개 push 받기 → N2 표·최저 종류 표 채우기 (collect_results 자동). 이유: QUEUE 11/09 01:40 KST; 보고서 범용화 문장의 근거.
3. `bm2_rlonly_s2` 처리 결정 (사람): 예) 다른 GPU·작은 val 배치로 재실행, 또는 2 seed 로 보고. 이유: STUCK, 에이전트는 플래그를 못 바꿈.
4. 속도 측정 프로토콜 (사람, QUEUE N 7번) 이 오면 그대로 실행 — 그 전엔 속도 숫자를 보고서 표에 쓰지 않음.
5. 그림 1·2·3 최신 기록으로 다시 그리기 (사람이 서버 밖에서 그림 데이터 사용 중 — `report: export board-figure data`). 이유: 실험·그림 완료 11/10.

## 하지 말아야 할 것
- N2 48개를 서버 1 큐에 다시 넣기 (QUEUE 11/09 01:40 KST).
- `bm2_rlonly_s2` 를 같은 플래그로 다시 돌리기; 에이전트 스스로 새 학습·arm·seed·세트 추가.
- 속도 숫자를 보고서 표에 쓰기 (프로토콜 전); 서버 2 시간과 서버 1 시간 섞기.
- N2·F 읽기 판정을 RESULTS 에 적용하기 (사람 몫, 숫자와 CI 만).
- `git stash`/rebase 로 추적 로그 건드리기.
