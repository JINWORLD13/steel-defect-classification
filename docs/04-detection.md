# 탐지 — 위치와 개수까지, 그리고 검사 라인의 언어로

> **Summary in English.** Faster R-CNN with the same ResNet18 backbone as v1. Design choices were justified by measurement before training: anchor ratios extended to (0.1…10) because default ratios catch only 16.3% of the elongated scratches boxes (92.8% after; overall 84.6→97.4), CPU over MPS (MPS is ~300x slower once you synchronize before timing), AdamW over SGD (0.833 vs 0.465 at equal steps in a memorization probe). The 50-image sanity gate (mAP@50≥0.9) came in at 0.667 — recorded as a miss, attributed to step budget, with the 0.04→0.67 trajectory as pipeline evidence. Final: test mAP@50 **0.734** (hand-rolled AP cross-checks torchmetrics within 0.003), per-class 0.41–0.91, and an operating point chosen on val (t=0.45) spent once on test: **19.4% miss, 1.87 false alarms/image**. The 5% miss target was unreachable at this budget — that, too, is a result.

## 왜 탐지인가

1막의 발견 — test 23장은 결함이 2종 이상 — 은 분류의 채점 문제가 아니라 문제 정의의 문제였다. 멀티라벨은 "무엇들이 있나"까지 답하지만, **어디에 몇 개**는 탐지만 답한다. 검사 라인이 실제로 묻는 것도 후자다.

## 학습 전에 만든 설계 근거

- **앵커 비율** (`anchor_recall.py`, 학습 없이 실측): 기본 (0.5,1,2)는 scratches 리콜@0.5 **16.3%** — 가늘고 긴 결함은 후보조차 없음. (0.1,0.5,1,2,10)으로 **92.8%**, 전체 84.6→97.4%. 크기는 기본 유지(512 제거안은 pitted_surface 리콜 하락으로 기각).
- **장치 = CPU**: MPS는 `torch.mps.synchronize()` 후 실측 시 수백 배 느림. 동기화 없이 재면 커널 큐 제출 시간만 재는 측정 버그가 된다.
- **AdamW**: 8장 암기 실험에서 같은 150스텝에 SGD 0.465 vs AdamW 0.833.
- **백본 = v1과 같은 ResNet18**: 분류·멀티라벨·탐지를 같은 출발점에서 비교하기 위함.

## sanity 게이트 — 미달을 기록함

50장 암기 게이트(mAP@50≥0.9)는 40에폭에 **0.667로 미달**. 학습 궤적(0.04→0.67)과 손실 하강으로 파이프라인 자체는 정상임을 확인했고, 미달 원인은 스텝 예산(암기에는 이미지당 훨씬 많은 스텝이 필요)으로 판단해 본 학습으로 진행했다. 게이트를 통과한 척하지 않는다 — `runs/detect_sanity.csv`가 기록이다.

## 본 학습과 결과

1,260장, 10에폭, CPU 약 80분 (`runs/detect_full.csv`). val mAP@50은 0.33 → **0.741**로 상승.

test 최종 (1회):

| 지표 | 값 |
|---|---|
| mAP@50 | **0.734** (손 구현 클래스 평균 0.737 — torchmetrics와 차이 0.003, 교차검증 통과) |
| mAP@50-95 | 0.350 |
| 클래스별 AP@50 | scratches 0.909 · patches 0.870 · inclusion 0.774 · pitted_surface 0.734 · rolled-in_scale 0.729 · **crazing 0.407** |

crazing이 가장 약한 이유는 실패 갤러리(`figures/10_det_failures.png`)가 보여준다 — 퍼진 그물 균열은 "박스 하나"의 경계 자체가 모호해서, 정답과 예측이 서로 다른 방식으로 영역을 쪼갠다. 분류에서는 조용히 넘어갔던 문제 정의의 어려움이 탐지에서 숫자로 드러난 것.

## 운영점 — 검사 라인의 언어

문턱값은 **val에서 훑고**(`detect_eval.py --split val`) test에는 한 번만 적용했다.

- 목표였던 **미검출 5% 이하는 이 예산의 모델로 도달 불가** (문턱 최저에서도 11.3%) — 이것도 결과다. 현장 투입 기준이 미검출 5%라면 이 모델은 아직 자격 미달이며, 그 판정을 내릴 수 있게 됐다는 것이 운영점 분석의 값어치다.
- 차선 규칙(미검출 ≤20% 중 가장 높은 문턱) → **t=0.45**. test에서 **미검출 19.4%, 헛경보 1.87건/장**.

## 재현

```bash
python anchor_recall.py                         # 설계 근거 (학습 없음)
python detect_train.py --sanity                 # 게이트 (미달 기록 포함)
python detect_train.py --epochs 10              # 본 학습 (CPU)
python detect_eval.py --split val               # 문턱 훑기
python detect_eval.py --split test --threshold 0.45
python detect_visualize.py --threshold 0.45     # 그림 08·09·10
```
