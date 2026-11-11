# MoTT: Modal-Token Transformer for Generalizable Decoupling Capacitor Placement

**Decap 배치 범용화를 위한 모달 토큰 트랜스포머** · 2026 과학영재 창의연구(R&E)

<p align="center">
  <img src="assets/mott_architecture.png" width="560" alt="MoTT architecture">
</p>

전원 분배망(PDN)의 공진 피크를 목표 임피던스 아래로 낮추면서 디커플링 커패시터(decap)를 적게 쓰는 배치를 학습한다. MoTT는 판을 좌표가 아니라 **공진 모드, 칸, decap 종류의 토큰**으로 나타내므로, 학습에서 보지 못한 종류와 크기의 판에도 재학습 없이 적용된다.

## 핵심 결과

**처음 보는 종류의 판** (한 종류를 학습에서 제외, 10×10, seed 3개 × 50문제, 성공률)

| 제외한 종류 | MoTT | 좌표 입력 모델 |
|---|---|---|
| PCB형 | **0.89** | 0.64 |
| 인터포저형 | **0.99** | 0.65 |
| 패키지형 | 0.71 | 0.80 |
| 패키지형 + 검증 하네스 | **0.99** | 0.92 |

같은 하네스를 적용한 선행 연구 재구현은 처음 보는 패키지형에서 0.45–0.91이었다.

**처음 보는 크기의 판** (8–12칸으로 학습, + 검증 하네스): 16×16과 20×20에서 세 종류 모두 **0.94 이상**.

**decap 수와 속도:** 인터포저형에서 공진 규칙보다 decap을 **약 30% 적게** 사용하였고, 하네스를 포함한 전체 시간은 greedy+prune보다 인터포저형 약 4배, 패키지형 4–7배, PCB형 약 1.5배 짧았다(문제 세트를 바꿔 3회 측정).

<p align="center">
  <img src="assets/all_kinds.png" width="820" alt="Success rate when trained on all three kinds">
</p>
<p align="center">
  <img src="assets/decap_speed.png" width="820" alt="Decap count and time per problem">
</p>

## 방법

1. **입력 선택:** 한 종류의 판에서 좋은 위치를 가리키던 입력이 다른 종류에서도 같은 부호로 작동하는지 학습 전에 분석하였다. 좌표 같은 입력은 판 종류가 바뀌면 부호가 뒤집혀, decap 하나를 놓았을 때의 정확한 효과와 공진 정보만 남겼다.
2. **MoTT:** 공진 모드 토큰 8개, 칸 토큰, decap 종류 토큰 3개. 칸끼리는 직접 attention하지 않고 모드 토큰을 거쳐서만 정보를 주고받는다(계산량 O(n²) → O(nK)). 출력은 정확히 계산한 1수 효과에 학습한 보정을 더한다.
3. **학습:** 교사 해(greedy+prune + local search)를 0.75시간 모방한 뒤 1.25시간 강화학습(그룹 정규화 이점, 판마다 8개 샘플).
4. **검증 하네스:** 32개 주파수 사이에 숨은 피크를 정밀 검사로 찾고, 실패 주파수를 더해 다시 풀거나(재풀이) 이어서 고친다(수리). 모든 비교 방법에 같은 규칙으로 적용하였다.

<p align="center">
  <img src="assets/training.png" width="640" alt="Training: imitation then reinforcement learning">
</p>
<p align="center">
  <img src="assets/harness.png" width="820" alt="Verification harness">
</p>

## 연구 기록과 사전 등록

각 실험의 판정 규칙은 **결과가 나오기 전에** [`docs/PROPOSALS.md`](docs/PROPOSALS.md)에 적고 공개 저장소에 push하였다. 실험별 등록 커밋과 첫 결과 커밋을 [`docs/PREREGISTRATION.md`](docs/PREREGISTRATION.md)에 링크로 정리하였으며, 등록 전에 나온 결과 기록은 모든 실험에서 0개이다. 결과를 본 뒤 추가한 분석은 '보고용'으로 따로 표시하였다.

- 모든 수치의 원본: [`runs/`](runs) (평가마다 JSON 기록), 집계: [`docs/RESULTS.md`](docs/RESULTS.md)
- 실험은 사람이 정한 대기열을 AI 에이전트가 실행하는 방식으로 운영하였다. 에이전트는 시뮬레이터와 판정 코드를 수정하지 않도록 제한하였다: [`docs/agent/`](docs/agent)
- 정리 전 저장소 전체(에이전트 기록, 로그, 연구 메모)는 [`research-log`](../../tree/research-log) 브랜치와 [`experiments-final`](../../tree/experiments-final) 태그에 그대로 있다.

## 저장소 구성

```
decap_loop/     PDN 시뮬레이터, 정밀 검사, MoTT 모델, 탐색 기준 방법
baselines_ml/   선행 연구 재구현 (B-Park, B-DevFormer, B-Manoharan)
train.py        모방 학습 → 강화학습
scripts/        문제 생성, 교사 해, 학습 실행, 평가, 결과 집계, 속도 측정
data/           교사 해
runs/           평가 기록 (보고서의 모든 수치)
docs/           판정 규칙 기록, 사전 등록 타임라인, 결과 집계, 에이전트 운영 문서
assets/         그림
```

## 실행

```bash
pip install -r requirements.txt
python sanity_check.py                                   # 시뮬레이터·정밀 검사 확인

python scripts/build_pools.py --out runs/pool_v1         # 학습·검증·평가 문제 생성 (고정 seed)
python scripts/make_teacher.py --pool runs/pool_v1 --out data/teacher

# MoTT 학습 (모방 0.75 h → 강화학습 1.25 h) 후 평가
python scripts/run_rung.py --name r2pk --arm "--model mott --peak_refine" \
    --distill_hours 0.75 --rl_hours 1.25 --seed 0

# 검증 하네스를 포함한 평가 (수리 최대 4회)
python scripts/eval_policy.py --pool runs/pool_v1 --sets 'eval_*' \
    --policy mott=runs/<day>/r2pk_rl/best.pt --repair 4 --out runs/<day>/r2pk_rl
```

각 스크립트의 첫 주석에 전체 옵션이 있다.

## 한계

합성 판에서만 검증하였고, 최대 20×20 판과 decap 3종 규모이다. 선행 연구 재구현은 원 논문과 문제 설정·학습 예산이 달라, 원 논문의 성능이 아니라 같은 조건에서의 비교로 해석해야 한다.
