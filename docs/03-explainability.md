# 설명가능성 — "선명한 텍스처에 의존한다"는 추정을 숫자로

> **Summary in English.** We implemented Grad-CAM from scratch and verified it against the CAM identity for GAP+fc networks (max diff 1.2e-07). Because the dataset ships bounding boxes, saliency is *scored*, not eyeballed: pointing game, energy inside boxes (EBPG), and a density ratio (ADR) reported with its per-class theoretical cap. A 6-rung baseline battery shows the trained model (ADR 1.47, PG 0.81) beats a center prior (1.19) and a pure edge detector (1.36) — but the thin margin over edges is itself evidence of edge reliance. A deletion test confirms faithfulness. Under corruption the surviving heatmaps stay focused (ADR ≈ 1.5); what collapses is heatmap *existence* (dead-CAM rate 0% → 20–29%).

## 구현과 자기검증

Grad-CAM을 라이브러리 없이 직접 구현했다(`gradcam.py`). ResNet18은 끝이 GAP→fc라서 Grad-CAM이 수학적으로 fc 가중치 CAM과 일치해야 한다 — 실측 최대차 **1.2e-07**로 항등식 통과. 훅을 건 상태에서 `no_grad`를 쓰면 기울기 재료가 안 잡힌다는 것도 직접 확인했다.

## 채점 방식 — 눈이 아니라 숫자

박스 라벨이 있으므로 히트맵을 채점할 수 있다(`cam_metrics.py`).

- **PG**: 히트맵 최고점이 정답 박스 안인가
- **EBPG**: 히트맵 에너지 중 박스 안 비율
- **ADR**: 박스 안 밀도 / 전체 밀도. **함정 주의** — 상한이 1/(박스 면적비)라 클래스마다 1.42~8.10으로 다르다. 상한을 병기하고, 클래스 간 비교엔 정규화판 ADR\* = (ADR−1)/(상한−1)을 쓴다.

## 베이스라인 배터리 — "1.47이 대단한가?"에 대한 답

무작위 기준선을 1.0(균일맵)에 긋는 것은 자기기만이다. 모델도 이미지도 안 보는 "가운데 가우시안 원"이 이미 1.19를 낸다(결함이 중앙에 있는 촬영 경향 때문).

| 단계 | 쓰는 정보 | ADR | PG |
|---|---|---|---|
| uniform | 없음 | 1.000 | 0.000 |
| random-init CAM | 구조만 | 1.006 | 0.311 |
| **center prior** | 위치 사전확률 | **1.192** | 0.515 |
| **edge energy** | 이미지의 엣지만 | **1.360** | 0.522 |
| ImageNet CAM | 일반 지식 | 1.157 | 0.567 |
| **trained v1** | 결함 학습 | **1.466** | **0.811** |

읽는 법: v1은 전 베이스라인을 이긴다 — 하지만 **순수 엣지 검출기와의 마진(1.36→1.47)이 얇다.** 이것이 "모델이 엣지/텍스처 단서에 크게 기댄다"는 v1 추정의 정량 증거다. 반면 PG(0.52→0.81)는 마진이 커서, "어느 위치인가"는 확실히 학습으로 얻어진 능력이다.

## 신실성 — deletion 검사

히트맵이 가리키는 곳을 지웠을 때 정확도가 무작위 삭제보다 빨리 무너지면, 히트맵은 실제로 판정에 쓰인 곳을 가리킨 것이다. 49칸 중 10칸 삭제에서 CAM순 0.633 vs 무작위 0.759 — 모든 지점에서 CAM순이 빠르게 무너짐. 통과.

## 열화 아래에서 — 무너지는 것은 '집중'이 아니라 '존재'

| 조건 | 전멸률(dead CAM) | 살아남은 히트맵의 ADR |
|---|---|---|
| 깨끗함 | 0% | 1.47 |
| gaussian_noise s3 | **28.5%** | 1.52 |
| low_contrast s3 | **25.6%** | 1.28 |
| gaussian_blur s3 | **20.4%** | 1.50 |

살아남은 히트맵은 여전히 박스를 향해 있다. 무너지는 것은 히트맵이 **아예 생기지 않는 비율**이다 — 열화가 판정 근거 자체를 지워버리는 방식으로 작동한다는 뜻. (전멸 CAM을 0으로 평균에 섞으면 두 현상이 뭉개진다 — 그래서 분리해 보고한다.)

## 재현

```bash
python gradcam.py         # 항등식 자기검증
python cam_metrics.py     # 지표 채점 (클린 + 열화 3종)
python cam_baselines.py   # 배터리 + deletion → figures/06,07
```
