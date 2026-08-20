# 강철 표면 결함 검사 — 100%를 심문하고, 고치고, 문제를 다시 세운 기록

🇰🇷 한국어 | [🇯🇵 日本語](README.ja.md) | [🇺🇸 English](README.en.md)

전이학습 분류기가 깨끗한 test에서 **정확도 1.000**을 냈다. 이 저장소는 그 숫자를 자랑하는 대신 **심문하고(1막), 왜 무너지는지 규명하고(2막), 실제로 고치고(3막), 문제 자체를 다시 세운(4막)** 기록임.

> **요약 4줄:**
> ① 100%의 용의자 넷(누수/쉬움/작은 평가셋/헐거운 정의)을 전부 실측 — 누수 기각, "쉬움"과 "헐거운 정의"는 확정
> ② 열화 증강 재학습으로 **학습에 안 쓴 손상 6종이 전부 개선** (최대 +36.5%p) — 대가로 clean 1.000→0.967 (맥니마 p=0.004, 하락은 실재)
> ③ Grad-CAM을 직접 구현·검증해 "엣지 의존" 추정을 정량화 — 무너지는 건 히트맵의 집중이 아니라 **존재**(전멸률 0→29%)
> ④ 같은 백본으로 멀티라벨·탐지까지 — 단일 라벨로는 **원리상 불가능**했던 것(다중 결함·위치·개수)을 회복

---

## 1막 · 발견 — "1.000은 무엇이었나"

깨끗한 test 100%의 원인은 원리적으로 넷뿐임. 넷 다 실측했음. (상세: [docs/01-data-audit.md](docs/01-data-audit.md))

| 용의자 | 방법 | 판정 |
|---|---|---|
| 데이터 누수 | md5 전수 + 32×32 코사인 근접중복 감사 | **기각** — 동일 파일 0건. r>0.90인 54장(목시 결과 닮은 *다른* 판)을 다 빼도 1.000 [0.983, 1.000] |
| 평가셋이 작음 | 윌슨 신뢰구간 | **조건부** — 1.000의 정직한 표현은 "**0.986 이상**" (n=270) |
| 문제가 쉬움 | 수제 텍스처 통계 **7개** + 로지스틱 회귀 | **확정** — 딥러닝 없이 **0.933** [0.897, 0.957] |
| 채점이 헐거움 | 원본의 바운딩박스 라벨과 대조 | **확정** — 아래 |

**채점이 헐거웠다는 증거:** 이 데이터는 `NEU-CLS`라는 이름으로 배포되지만 1,800장 전부에 박스 라벨(4,189개)이 딸린 **탐지용(NEU-DET 계열)**임. 대조 결과 **123장(6.8%)**이 파일명과 다른 종류의 결함 박스를 함께 가짐 — test 기준 **23장(8.5%)은 정답이 하나가 아닌데 단일 라벨로 채점**됐음. "대표 결함"의 기준조차 갈림(개수 다수결 어긋남 22장, 면적 기준 8장, 교집합 1장). 데이터가 틀린 게 아니라 **탐지용 데이터를 단일 라벨 분류로 눌러 담은 내 문제 설정**이 틀렸던 것 — 이 발견이 4막의 이유가 됨. (`audit_data.py`, `audit_neardup.py`, `baseline_handcrafted.py`로 전부 재현)

![Baseline ladder](figures/03_baseline_ladder.png)

## 2막 · 규명 — "왜, 어떻게 무너지는가"

v1의 열화 5종 막대를 **손상 11종 × 강도 5단계** 사다리로 재설계 (`corruptions.py`). 학습에 쓸 수 있는 계열과 절대 못 쓰는 held-out 6종(near/mid/far)을 **import 시점의 assert로 물리 분리**했고, 사다리 강도는 val에서 보정 후 동결 — test는 모델당 한 번만 봄. (상세: [docs/02-robustness.md](docs/02-robustness.md))

v1 기준선 (test, 강도 평균): **clean 1.000 / seen 0.614 / near 0.643 / mid 0.565 / far 0.863**

"선명한 텍스처에 의존한다"는 v1의 추정은 Grad-CAM을 **직접 구현**해 검증했음 (`gradcam.py` — GAP∘fc 항등식으로 자기검증, 최대차 1.2e-07). 박스 라벨이 있으므로 히트맵을 **채점**할 수 있음: (상세: [docs/03-explainability.md](docs/03-explainability.md))

- 6단 베이스라인 배터리: 균일 1.00 → 중앙 사전확률 1.19 → **순수 엣지 1.36** → 학습모델 **1.47** (ADR). 전부 이기지만 **엣지와의 마진이 얇다는 것 자체가 엣지 의존의 정량 증거**. 위치 지표(PG)는 0.52→**0.81**로 마진 큼
- deletion 검사 통과 — 히트맵 상위 칸 삭제가 무작위 삭제보다 전 구간에서 빨리 정확도를 무너뜨림 (신실성)
- 열화 시 무너지는 건 살아남은 히트맵의 집중(ADR ~1.5 유지)이 아니라 **히트맵의 존재** — 전멸률 0% → 20~29%

![CAM baselines](figures/06_cam_baselines.png)

## 3막 · 해결 — "고쳤고, 고쳐졌다는 주장을 검증했다"

학습 증강에 4계열(노이즈·블러·밝기·대비)을 **연속 범위에서 추첨**해 넣고 재학습 (`train_robust.py`, 시드 5개). 평가 프리셋 고정값은 쓰지 않으며 held-out 계열은 코드가 차단함. 대표 시드는 **val 선택점수로만** 결정 — clean test가 가장 높았던 시드는 **선택하지 않았음**.

| 구간 (test, 강도 평균) | v1 | v2 | 시드 범위 |
|---|---|---|---|
| clean | 1.000 | 0.967 | 0.967~0.993 |
| seen — 학습 계열, 오르는 게 당연 | 0.614 | **0.894** | 0.886~0.895 |
| **held-out near** | 0.643 | **0.899** | 0.835~0.905 |
| **held-out mid** | 0.565 | **0.738** | 0.653~0.741 |
| **held-out far** — jpeg·픽셀레이트, 사촌조차 학습 안 함 | 0.863 | **0.906** | 0.778~0.906 |

held-out **6종 전부 개선**: speckle +0.365 · salt_pepper +0.264 · motion_blur +0.147 · gamma +0.081 · jpeg +0.047 · pixelate +0.040

**정직한 대가:** clean 1.000 → 0.967. 같은 270장의 짝지은 비교라 **맥니마 정확검정**으로 판정 — v1만 맞힌 문제 9, v2만 맞힌 문제 0, **p=0.004**. 이 하락은 노이즈가 아니라 강건성의 실제 비용임.

![Before/after](figures/05_robustness_before_after.png)

## 4막 · 재정의 — "단일 라벨로는 애초에 못 하는 것"

1막의 발견(다중 결함 23장)을 문제 정의 수준에서 풀었음. 세 모델 전부 **같은 ResNet18 백본, 같은 test 270장**.

| 모델 | 채점 기준 (박스 멀티핫) | 수치 |
|---|---|---|
| 단일 라벨 v1 (argmax) | subset accuracy | 0.915 — **자기 구조적 상한값에 정확히 도달** (다중 결함 23장은 원리상 만점 불가) |
| 멀티라벨 (BCE, 20줄 수정) | subset accuracy | **0.926** — 상한 돌파, 놓친 결함 쌍 23→18 |
| 탐지 (Faster R-CNN) | mAP@50 / 운영점 | **0.734** (mAP@50-95 0.350, 클래스별 0.41~0.91) / 미검출 19.4% · 헛경보 1.87건/장 (문턱 0.45, **val에서 결정**) |

탐지 설계 근거는 학습 없이 만들었음 (`anchor_recall.py`): 기본 앵커 비율은 가늘고 긴 scratches 박스를 **16.3%**밖에 못 잡음 → 비율 0.1/10 추가로 **92.8%** (전체 84.6→97.4%). 장치는 CPU — MPS는 `synchronize()` 후 실측 시 수백 배 느렸음(동기화 없이 재면 큐 제출 시간만 재는 함정). 50장 암기 게이트(mAP@50≥0.9)는 0.667로 미달했으나 궤적(0.04→0.67)으로 파이프라인 정상을 확인하고 본 학습으로 진행 — 미달 사실도 기록임. (상세: [docs/04-detection.md](docs/04-detection.md))

![Detection](figures/08_det_metrics.png)

## 배운 점 (v1 + v2)

- **정확도 100%는 결론이 아니라 심문의 시작임.** 원인 후보 넷을 하나씩 재는 것이 검증임.
- **조용한 버그 2건을 잡고 회귀 테스트로 못박았음.** ① 조건 키 어긋남으로 무변형 원본이 평가되던 버그 ② 재분할 시 이전 분할 위에 쌓여 train/test가 누수되던 버그 — 둘 다 "첫 실행은 멀쩡해 보임"이 공통점. `pytest` 4개가 재발을 막음.
- **경계는 문서가 아니라 코드가 지킴.** held-out 침범은 assert가, 분할의 증거는 매니페스트가, 수치의 출처는 `results/*.json`이 담당.
- **재현성의 함정들:** 파이썬 `hash()`는 프로세스마다 달라짐(crc32로 교체), 정렬 없는 셔플은 OS 의존, GPU 학습은 시드 고정으로도 완전 재현 불가(가중치 동결 백업이 정답지).
- **통계는 설계에 맞게:** 짝지은 비교엔 맥니마, 비율엔 윌슨 구간. n 없는 정확도는 쓰지 않음.
- **지표는 자기 기준선과 함께:** ADR의 무작위선은 1.0이 아니라 "가운데 원" 1.19였음. 기준선 없는 숫자는 주장이 아님.
- **음성 결과도 결과임:** TTA는 실측에서 해로웠고(clean 1.000→0.830), far 구간 개선은 +0.04로 작음 — 부풀리지 않고 그대로 적음.

## 실행 방법

```bash
# 0) 데이터 (~26MB) 와 환경
curl -sL -o NEU-CLS.zip "https://ndownloader.figshare.com/files/54094775"
unzip -q NEU-CLS.zip -d data
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# 1막 — 기반과 감사
python prepare_data.py            # 재분할 + splits/ 매니페스트
python train.py                   # v1 학습 → best_model.pth (백업: best_model_v1.pth)
python analyze.py                 # 혼동행렬
python audit_data.py              # 누수 + 문제 정의 감사
python audit_neardup.py           # 근접중복 + 부분집합 재평가
python baseline_handcrafted.py    # 수제 특징 베이스라인
python -m pytest tests/ -q        # 회귀 테스트

# 2막 — 벤치와 설명
python bench_robust.py --model best_model_v1.pth --split test --name v1
python gradcam.py && python cam_metrics.py && python cam_baselines.py

# 3막 — 재학습과 판정
for s in 42 43 44 45 46; do python train_robust.py --seed $s; done
python compare_robust.py

# 4막 — 멀티라벨과 탐지
python train_multilabel.py && python eval_multilabel.py
python anchor_recall.py
python detect_train.py --sanity   # 게이트
python detect_train.py --epochs 10          # CPU 수십 분
python detect_eval.py --split val           # 운영점 결정
python detect_eval.py --split test --threshold 0.45
python detect_visualize.py --threshold 0.45
```

> 데이터 출처: NEU 표면결함 DB (Northeastern University, Song & Yan). Figshare 배포명은 `NEU-CLS`지만 내용물은 탐지 주석 포함(NEU-DET 계열). CC BY 4.0 (재배포본 표기 기준).
> GPU 연산 특성상 시드를 고정해도 재학습 수치는 본문과 조금 다를 수 있음. v1 수치의 정답지는 로컬의 `best_model_v1.pth`임.

## 파일 구조

```
common.py / labels.py        # 공용 부품 (상수·모델·결과저장·통계 / 박스 라벨)
prepare_data.py              # 재분할 + splits/ 매니페스트
train.py → analyze.py        # v1 학습·채점 (v1 재현의 정답지, 수정 금지)
robustness.py                # v1의 열화 5종 (보존용)
audit_data.py / audit_neardup.py / baseline_handcrafted.py     # 1막
corruptions.py / bench_robust.py                                # 2막 벤치
gradcam.py / cam_metrics.py / cam_baselines.py                  # 2막 설명
train_robust.py / compare_robust.py                             # 3막
train_multilabel.py / eval_multilabel.py                        # 4막 멀티라벨
detect_dataset.py / detect_train.py / detect_eval.py
detect_visualize.py / anchor_recall.py                          # 4막 탐지
tests/                       # 회귀 테스트 (누수·무변형·파싱)
splits/  results/  figures/  runs/  docs/                       # 증거들
best_model_v1.pth            # (로컬 전용) v1 기준선 — 절대 덮어쓰지 말 것
```

## 한계

- 단일 출처의 실험실 촬영본 — 실제 현장 분포는 미지수이며, 열화는 시뮬레이션이고 복합 조건(어두우면서 흐림)은 안 걸었음
- '정상(무결함)' 클래스가 없어 어떤 이미지든 결함 6종 중 하나로 분류함
- 같은 강판의 *유사* 프레임 문제는 측정으로 완화했을 뿐 촬영 메타데이터 없인 완전 배제 불가 — 유사도 그래프 기반 group split은 v3 후보
- 탐지는 CPU 예산에 맞춘 최소 구성(10에폭)임 — 수치는 그 예산의 결과로 읽어야 함

선택하지 않은 것들과 그 이유(YOLO 라이선스, 서빙 제외, MLOps 도구 제외, TTA 기각 등): [docs/05-decisions.md](docs/05-decisions.md)
