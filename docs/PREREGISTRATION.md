# 사전 등록 타임라인 (Pre-registration timeline)

- 각 실험의 판정 규칙을 [`docs/PROPOSALS.md`](PROPOSALS.md)에 적은 시각(등록)과, 그 실험의 첫 결과 기록이 생긴 시각을 git 기록에서 뽑았다.
- **등록 시각:** 그 절의 제목이 처음 들어간 커밋의 시각이다.
- **첫 결과 시각:** 그 실험이 만든 결과 파일(`runs/…`의 평가 JSON·metrics.json) 중 가장 이른 것이다. 결과 파일을 git에 추가한 커밋 시각과 기록 안의 `time` 값 중 이른 쪽을 썼다.
- **등록 전 기록:** 첫 결과 시각이 등록보다 이른 파일 수이며, 모든 실험에서 0이다.
- 저장소: github.com/jeehoo0507/R2E. 표는 2026-11-09 13:58 KST에 git 기록에서 생성하였다.

| 실험 | 내용 | 성격 | 등록 (KST) | 등록 커밋 | 첫 결과 (KST) | 결과 커밋 | 간격 | 결과 파일 수 | 등록 전 기록 |
|---|---|---|---|---|---|---|---|---|---|
| T2 | 끝에서 고치기(수리), 모든 방법에 같은 규칙 | 판정 | 11/03 14:40 | [`8ecb65a`](https://github.com/jeehoo0507/R2E/commit/8ecb65a) | 11/03 15:50 | [`abf1ed7`](https://github.com/jeehoo0507/R2E/commit/abf1ed7) | 1시간 9분 | 233 | 0 |
| T5 | RL 단계 재현 잡음 | 판정 | 11/03 21:09 | [`8abb7fd`](https://github.com/jeehoo0507/R2E/commit/8abb7fd) | 11/04 00:59 | [`a61b481`](https://github.com/jeehoo0507/R2E/commit/a61b481) | 3시간 49분 | 13 | 0 |
| P1 | 모드 토큰 피크 보정 입력 | 판정 | 11/04 08:05 | [`4b496a5`](https://github.com/jeehoo0507/R2E/commit/4b496a5) | 11/04 08:51 | [`d0d42f3`](https://github.com/jeehoo0507/R2E/commit/d0d42f3) | 0시간 45분 | 94 | 0 |
| P2 | 중복 없는 모드 토큰 | 판정 | 11/04 10:38 | [`c4d2713`](https://github.com/jeehoo0507/R2E/commit/c4d2713) | 11/04 17:08 | [`6530206`](https://github.com/jeehoo0507/R2E/commit/6530206) | 6시간 29분 | 36 | 0 |
| P3 | P1 + P2 | 판정 | 11/04 17:59 | [`13176da`](https://github.com/jeehoo0507/R2E/commit/13176da) | 11/05 09:14 | [`664f7d2`](https://github.com/jeehoo0507/R2E/commit/664f7d2) | 15시간 14분 | 46 | 0 |
| F1 | 한 종류 빼고 학습 seed 1·2 | 판정 | 11/05 08:43 | [`ed4ced5`](https://github.com/jeehoo0507/R2E/commit/ed4ced5) | 11/05 18:38 | [`4e9d69a`](https://github.com/jeehoo0507/R2E/commit/4e9d69a) | 9시간 54분 | 38 | 0 |
| B1 | 선행 연구 재구현 seed 1·2 | 보고용 | 11/05 14:57 | [`e70de68`](https://github.com/jeehoo0507/R2E/commit/e70de68) | 11/06 03:06 | [`1b6d811`](https://github.com/jeehoo0507/R2E/commit/1b6d811) | 12시간 8분 | 36 | 0 |
| G1 | pkg 범용화 구조 후보 2개 | 판정 | 11/07 10:45 | [`a208879`](https://github.com/jeehoo0507/R2E/commit/a208879) | 11/07 13:01 | [`5b39fa7`](https://github.com/jeehoo0507/R2E/commit/5b39fa7) | 2시간 15분 | 71 | 0 |
| M | 정책 단독 pkg 방법 탐색 | 판정 | 11/07 10:46 | [`17db6a6`](https://github.com/jeehoo0507/R2E/commit/17db6a6) | 11/07 15:35 | [`a6d549e`](https://github.com/jeehoo0507/R2E/commit/a6d549e) | 4시간 48분 | 7 | 0 |
| A1·A2 | RL 설계 ablation | 보고용 | 11/07 11:00 | [`7415f04`](https://github.com/jeehoo0507/R2E/commit/7415f04) | 11/07 13:11 | [`38584c4`](https://github.com/jeehoo0507/R2E/commit/38584c4) | 2시간 10분 | 9 | 0 |
| C1 | 검증된 해 따라 하기 RL | 판정 | 11/07 14:16 | [`fb1e316`](https://github.com/jeehoo0507/R2E/commit/fb1e316) | 11/07 16:50 | [`a6d549e`](https://github.com/jeehoo0507/R2E/commit/a6d549e) | 2시간 33분 | 7 | 0 |
| C1' | 보정 피크를 만족 판정에 사용 | 판정 | 11/07 14:28 | [`dab9694`](https://github.com/jeehoo0507/R2E/commit/dab9694) | 11/07 15:33 | [`97e0745`](https://github.com/jeehoo0507/R2E/commit/97e0745) | 1시간 4분 | 6 | 0 |
| M1 | 모드 간 주파수 거리 편향 | 판정 | 11/07 19:28 | [`4b9fdbc`](https://github.com/jeehoo0507/R2E/commit/4b9fdbc) | 11/07 21:35 | [`522bd26`](https://github.com/jeehoo0507/R2E/commit/522bd26) | 2시간 6분 | 8 | 0 |
| R | 물리 상수 무작위 학습 | 판정 | 11/07 22:32 | [`ece1a02`](https://github.com/jeehoo0507/R2E/commit/ece1a02) | 11/08 01:10 | [`1637c71`](https://github.com/jeehoo0507/R2E/commit/1637c71) | 2시간 37분 | 35 | 0 |
| V | 뺀 패키지형 수리 포함 평가 | 보고용 | 11/07 22:42 | [`451ec4d`](https://github.com/jeehoo0507/R2E/commit/451ec4d) | 11/07 22:47 | [`3f3e7a1`](https://github.com/jeehoo0507/R2E/commit/3f3e7a1) | 0시간 4분 | 6 | 0 |
| F | 최종 방법 = MoTT+P1 + 검증 하네스 | 판정 | 11/08 01:31 | [`0242e56`](https://github.com/jeehoo0507/R2E/commit/0242e56) | 11/08 01:48 | [`57330c5`](https://github.com/jeehoo0507/R2E/commit/57330c5) | 0시간 16분 | 34 | 0 |
| N | 선행 기준선 원문판·보충 seed·하네스 | 판정 | 11/08 10:31 | [`ce4422f`](https://github.com/jeehoo0507/R2E/commit/ce4422f) | 11/08 12:50 | [`13f8143`](https://github.com/jeehoo0507/R2E/commit/13f8143) | 2시간 18분 | 207 | 0 |
| N2 | 뺀 pkg 모델의 PCB·인터포저 하네스 | 보고용 (결과 본 뒤 추가) | 11/09 00:14 | [`10062dd`](https://github.com/jeehoo0507/R2E/commit/10062dd) | 11/09 00:18 | [`001caca`](https://github.com/jeehoo0507/R2E/commit/001caca) | 0시간 3분 | 38 | 0 |
| S | 편차 큰·실패한 설정의 보충 seed | 보고용 (결과 본 뒤 추가) | 11/09 13:53 | [`04533ff`](https://github.com/jeehoo0507/R2E/commit/04533ff) | 진행 중 | — | — | 0 | 0 |

**읽는 법과 한계**
- 간격이 몇 분인 실험(V, F, N2)은 평가만 하는 실험이다. 평가 하나가 2~11분이라 등록 직후 첫 기록이 나왔다. 기록에 적힌 평가 소요 시간을 빼도 시작 시각이 등록 뒤이다.
- N2와 S는 앞선 결과를 **본 뒤** 추가한 실험이다. 그래서 등록 문서에 '보고용, 판정 없음'으로 적었고, 본문에서도 판정에 쓰지 않는다.
- 시각은 각 서버의 시계로 찍힌 git 커밋 시각과 기록 안의 시각이다. git 기록은 원칙적으로 다시 쓸 수 있다. 다만 이 저장소는 실험 기간 내내 공개 저장소(github.com/jeehoo0507/R2E)에 push되었으므로, 표의 커밋은 누구나 열어 확인할 수 있다.

**직접 확인하는 법**
- 표의 커밋 링크를 누르면 GitHub에 그 커밋이 올라간 내용과 시각이 나온다. 등록 커밋에는 판정 규칙이, 결과 커밋에는 평가 기록(`runs/…`)이 들어 있다.
- 정리 전 저장소 전체(에이전트 기록, 로그, 메모 포함)는 [`research-log`](https://github.com/jeehoo0507/R2E/tree/research-log) 브랜치와 [`experiments-final`](https://github.com/jeehoo0507/R2E/tree/experiments-final) 태그에 그대로 남아 있다. main의 정리 커밋은 파일을 지우기만 했고, 이전 커밋은 하나도 바꾸지 않았다.
