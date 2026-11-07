# Codex 작업 지시: R&E 보고서 그림 다시 그리기 (브랜치 `report-figs-v1`)

작성: 2026-11-07 KST, 사람(cjh). 이 브랜치는 보고서 그림 전용이다. 서버 루프는 `main`만 pull/push 하므로 여기서의 작업은 실험에 영향이 없다.

## 0. 규칙 (어기면 작업 무효)
1. **`main`에 push·merge·rebase 금지.** 이 브랜치(`report-figs-v1`)에만 커밋·push.
2. 수정 가능한 파일: `report_figs/` 아래 전부, 그리고 `scripts/fig_board.py`, `scripts/fig_rule_vs_mott.py`의 **그리기(스타일) 부분만**. 그 밖의 파일(`decap_loop/`, `train.py`, `agent/`, `PROPOSALS.md`, `RESULTS.md`, `QUEUE.md` 등)은 읽기만.
3. **숫자를 바꾸지 않는다.** 모든 값은 `report_figs/src/*.py` 안에 손으로 옮겨 둔 값이 정답이다(출처: RESULTS.md, 평가 기록). 새 값을 만들거나, 반올림 자릿수를 바꿔 의미가 달라지게 하거나, 점·막대를 빼거나 더하지 않는다. 값이 이상해 보이면 고치지 말고 `report_figs/NOTES.md`에 적는다.
4. 학습·평가·서버 명령을 실행하지 않는다. 그림은 CPU에서 matplotlib(또는 동등한 라이브러리)로만 만든다.
5. 구조 그림(Fig. 3, 4)은 **코드에서 확인한 것만** 그린다: `decap_loop/model.py`(MoTTBlock, MoTT.forward), `decap_loop/peak_env.py`(P1), `decap_loop/sim.py`. 코드에 없는 화살표·블록을 넣지 않는다.

## 1. 대상 그림 (결과가 확정된 것만)

| 보고서 번호 | 파일 이름 | 원본 함수 | 하나의 주장 |
|---|---|---|---|
| Fig. 2 | `fig_auc_flip` | `src/make_report_extra.py::auc_flip` | 좌표·한 주파수 loop L은 판 유형이 바뀌면 AUC가 0.5 아래로 뒤집히고, 정확한 1수 효과는 유지된다 |
| Fig. 3 | `fig_tokens` | `src/make_r2e_figs.py::fig_tokens` | MoTT 입력은 토큰 4종(모드·셀·decap·셀–모드 간선)이고 좌표가 없다 |
| Fig. 4 | `fig_arch_v2` | `src/make_method_figs.py::arch` | MoTT 블록(×4): 모드 self-attn → 모드가 셀을 읽음(결합도 편향) → 셀·decap이 모드를 읽음, 셀끼리 attention 없음, 출력 = 학습 점수 + 물리 사전 점수 |
| Fig. 7 | `main3_unseen` | `src/make_main_figs.py::main3` | PCB형을 빼고 학습 → 처음 보는 PCB형: 좌표 0.64 → 모달 특징 0.84 → 모드 토큰 0.90 (seed 3개) |
| Fig. 8 | `main4_p1` | `src/make_main_figs.py::main4` | P1 피크 보정: 같은 유형 pcb·pkg에서 3 seed 묶음 CI > 0, ip·처음 보는 pcb는 0 근처 |
| Fig. 1 | `fig1_board_draft_r2pk_10` | `scripts/fig_board.py` | 한 정책이 세 유형 판에 배치한 결과 + 배치 전후 \|Z\|/Zt |
| Fig. 11 | `fig_rule_vs_mott_ip16` | `scripts/fig_rule_vs_mott.py` | 같은 판에서 공진 규칙 10개 vs MoTT+P1 6개 |
| (선택) 2절·포스터 | `fig_r2e_explorations` | `src/make_r2e_explorations.py` | 2주 탐구 결과 6패널 |

지금 모습은 `report_figs/current/*.png`에 있다.

### 이번에 그리지 않는 것 (실험 결과에 따라 바뀜)
- `fig_train`, `fig_infer`: C1/C1′ 결과에 따라 학습·판정 단계가 추가될 수 있음.
- `main1_speed_quality`, `main2_size`, Table 7: 선행 재구현 16·20칸(S2)과 최종 모델이 아직 대기.
- `fig_r2e_journey`, `fig_r2e_timeline`: 마지막 단계가 진행 중.
- Fig. 4에 주파수 궁합 편향(fc_bias)이나 루프 구조를 넣지 않는다(채택되지 않았거나 판정 대기).

## 2. 스타일 사양 (모든 그림 공통)
- 인쇄: A4 한 단, 본문 폭 16.8 cm. 전체 폭 그림은 폭 16.8 cm, 반 폭은 8.0 cm로 **실제 크기**(`figsize` 인치)로 만든다. 인쇄 크기에서 글자는 최소 7 pt, 축 제목 8 pt.
- 글꼴: 한글 `Noto Sans KR`(없으면 `Apple SD Gothic Neo`), 수식은 mathtext. 한 그림 안에서 글꼴 하나.
- 색(모든 그림에서 같은 뜻으로):
  - `#C2410C` MoTT+P1 (본 연구 최종) · `#2563A8` MoTT(P1 없음) · `#93C5FD` 좌표 입력/사다리 하단
  - `#4B5563` 공진 규칙 · `#9CA3AF` 탐색 기준선(greedy+prune 등)
  - 글자 `#1F2328`, 보조 글자 `#6A737D`, 격자·테두리 `#D0D7DE`
  - 빨강·초록 대비에만 의존하지 않는다(색맹). 강조색은 그림당 하나.
- 제목은 그림 안에 넣지 않는다(캡션이 따로 있음). 패널 표시는 (a)(b)….
- 위·오른쪽 축선 없음, 격자는 옅게. 범례보다 선 끝 직접 표기를 선호.
- 오차 막대·CI·seed 점은 원본에 있는 것 그대로 유지(보고서의 통계 주장이 여기에 걸려 있음).
- 출력: `report_figs/out/<파일 이름>.pdf`(벡터, 글꼴 포함) + `.png`(300 dpi). 파일 이름은 위 표와 같게.

## 3. 그림별 메모
- **Fig. 2**: 그림 아래 주석 유지 — "판 10개씩, 신뢰구간 없음; 인터포저형 라벨은 greedy 해에 가까워 1수 효과 값이 부풀려져 있음". 방향은 PCB형에서 정하고 인터포저형에 그대로 적용했다는 점이 보여야 한다.
- **Fig. 3·4**: 지금 그림은 상자·화살표가 많아 읽기 어렵다. 정보 흐름(모드 ↔ 셀, 모드 → decap, 출력 합)이 한눈에 보이게 줄인다. 반드시 들어갈 것: K=8, 모드 토큰 채널(log f/fc, 피크 높이, 위반 여부, 유효 표시 + P1 두 채널), 셀 토큰의 정확한 1수 효과 ΔV_it·margin, decap 토큰(log SRF/fc, ESL, ESR, 1/ωC), 결합도 g_ik의 attention 편향·메시지, d=128 · 8 heads · 블록 4개, "셀끼리 attention 없음", 출력 s_it = MLP + w_Δ·ΔV − w_m·margin. Fig. 3과 4는 같은 색·같은 기호를 쓴다.
- **Fig. 7**: 막대 = seed 평균, 점 = seed 3개, 점선 = 세 유형 모두 학습했을 때. "사전 등록 기준 통과" 표기 유지.
- **Fig. 8**: 가로 CI 그래프. 0 기준선 강조, 숫자 라벨 유지.
- **Fig. 1·11** (보드 그림): pool 파일(`runs/pool_v1/*.pkl`)이 git에 없어 여기서 실제 렌더는 못 한다. `scripts/fig_board.py`, `scripts/fig_rule_vs_mott.py`의 **그리기 코드만** 고친다(판 고르기 규칙 `--pick median`, 물리 계산, CLI 인자, 입력 형식은 그대로). 미리보기는 가짜 판으로 `report_figs/out/MOCK_*.png`에만 만들고 파일 안에 "MOCK" 글자를 크게 넣는다. 실제 렌더는 사람이 서버에서 한다.
  - 판: 셀 격자 옅게, 금지 영역 회색, probe·전원 입구 기호 구분, decap 종류 3개는 모양+색 둘 다로 구분.
  - 곡선: 빈 판 점선, 배치 후 실선, 목표 = 1 검정선, 최악 \|Z\|/Zt 값 글자 유지.
- **(선택) explorations**: 6패널 구성·숫자 유지, 정리만.

## 4. 끝났을 때
1. `python report_figs/src/make_all.py report_figs/out` 한 번으로 Fig. 2·3·4·7·8(·선택)이 다시 만들어지게 한다.
2. `report_figs/NOTES.md`: 그림별로 바꾼 점 한두 줄, 확인이 필요한 점, 코드와 대조한 위치(파일:줄).
3. 이 브랜치에 커밋·push (`git push origin report-figs-v1`). `main`은 건드리지 않는다.
