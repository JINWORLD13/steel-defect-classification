# 강철 표면 결함 분류 (NEU-CLS) + 강건성 분석

🇰🇷 한국어 | [🇯🇵 日本語](README.ja.md) | [🇺🇸 English](README.en.md)

제조 공정에서 나오는 **강철 표면 결함 6종**을 전이학습(ResNet18)으로 분류하고,
**"실험실 정확도가 현장에서도 유지되는가"**를 강건성(robustness) 테스트로 검증한 프로젝트임.

> **요약:** 깨끗한 데이터에선 정확도 100%였지만, 현장 악조건(흐림·노이즈·저대비)을 주입하자
> 38~53%까지 떨어졌음. 즉 이 모델은 실험실 조건에 과의존적이며, 현장 투입 전 열화 증강이 필요함을 정량적으로 확인했음.

---

## 1. 문제 정의

- **목표:** 육안 검사를 대체할 표면 결함 자동 분류기
- **분류 대상 6종:** crazing(잔균열), inclusion(개재물), patches(반점), pitted_surface(점부식), rolled-in_scale(압연 스케일), scratches(긁힘)
- **접근:** 사전학습 ResNet18을 전이학습으로 미세조정

## 2. 데이터

- **출처:** NEU-CLS (Northeastern University, CC BY 4.0)
- **구성:** 1,800장, 6클래스 균형(클래스당 300장), 200×200
- **분할:** train 70% / val 15% / test 15% (클래스 비율 유지 stratified)
  - ⚠️ 원본은 검증셋이 30장뿐이라 신뢰할 수 없었음. **직접 재분할**해 평가 신뢰도를 확보함 (`prepare_data.py`)

## 3. 방법

| 항목 | 내용 |
|---|---|
| 모델 | ResNet18 (ImageNet 사전학습) + 마지막 fc층을 6종용으로 교체 |
| 증강 | RandomHorizontalFlip, RandomRotation(±15°) |
| 손실/최적화 | CrossEntropyLoss / Adam (lr=1e-3) |
| 학습 | 12 epochs, **val 정확도 최고 시점 모델 저장**(과적합 방지) |
| 장치 | CUDA·MPS(Apple GPU)·CPU 자동 선택 — 본 실험은 Apple Silicon MPS |

## 4. 결과 — 깨끗한 test 데이터

**test 정확도 100%** (270장, 클래스별 precision/recall 모두 1.000)

![Confusion Matrix](confusion_matrix.png)

혼동행렬이 완벽한 대각선 → 클래스 간 혼동 없음.
**다만 이 "너무 완벽한" 결과를 신뢰하지 않고**, 아래 강건성 테스트로 실제 실력을 검증했음.

## 5. 강건성 분석 (핵심)

현장에서 실제로 발생하는 이미지 열화 5종을 test 데이터에 주입하고 정확도 변화를 측정했음. (`robustness.py`)

| 조건 | 정확도 | 원본 대비 |
|---|---|---|
| 원본 (기준) | **1.000** | — |
| 어두움 (조명 약함) | 0.874 | −12.6%p |
| 밝음 (조명 과함/반사) | 0.767 | −23.3%p |
| 흐림 (초점·먼지) | 0.530 | −47.0%p |
| **노이즈 (저가 센서)** | **0.385** | **−61.5%p** |
| 대비 낮음 (뿌옇게) | 0.396 | −60.4%p |

![Robustness](robustness.png)

**해석:**
- 조명 변화엔 비교적 버티지만, **노이즈·저대비·흐림엔 취약함** (최대 −61%p)
- 원인: 학습 데이터가 깨끗한 실험실 촬영본이라, 모델이 **선명한 텍스처에 의존**하도록 학습된 것으로 보임
- 결론: 현장 투입 전 **노이즈·블러·대비 증강을 추가한 재학습**이 필수임

## 6. 배운 점 / 회고

- **높은 정확도 ≠ 좋은 모델.** 100%는 데이터가 쉬웠던 결과였고, 강건성 테스트로 실제 약점을 드러냈음.
- **"조용한 버그"를 직접 겪음.** 강건성 코드에서 조건 키 오타(`low_contrast`)로 대비 테스트가 무변형으로 통과해 **가짜 100%**가 나왔던 것을 발견·수정했음. 에러 없이 결과만 틀리는 버그가 가장 위험함을 체감함.
- **평가 설계가 중요함.** 원본의 부실한 val 분할을 그대로 쓰지 않고 재분할한 것이 신뢰할 수 있는 결과의 전제였음.

## 7. 다음 단계

- 열화 조건을 증강에 포함한 재학습 → 강건성 회복 실험
- 결함 위치까지 찾는 **객체 탐지(YOLO)**로 확장
- Streamlit/Gradio 데모 배포

---

## 실행 방법

```bash
# 1) 데이터 내려받기 — NEU-CLS (~26MB, CC BY 4.0, Figshare)
curl -sL -o NEU-CLS.zip "https://ndownloader.figshare.com/files/54094775"
unzip -q NEU-CLS.zip -d data

# 2) 환경 설정
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3) 파이프라인 실행
python prepare_data.py    # data/ → dataset/ 로 재분할 (train/val/test)
python train.py           # 학습 → best_model.pth 생성
python analyze.py         # 혼동행렬 + 클래스별 성적표
python robustness.py      # 강건성 테스트 → robustness.png
```

> 데이터 원본: NEU-CLS (Northeastern University). 위 curl 링크가 막히면 Figshare에서 "NEU-CLS"로 검색해 받을 수 있음.
> 학습에는 무작위성이 있어(시드를 고정해도 GPU 연산 특성상) 재학습 시 수치가 본문과 조금 다를 수 있음.

## 파일 구조

```
prepare_data.py   # NEU-CLS를 분류용 폴더 구조 + stratified 3분할
train.py          # ResNet18 전이학습 (MPS)
analyze.py        # 혼동행렬 · 클래스별 precision/recall
robustness.py     # 이미지 열화 5종 하에서의 강건성 측정
confusion_matrix.png / robustness.png   # 결과 이미지
```

## 한계

- 데이터가 단일 출처(NEU-CLS)의 깨끗한 촬영본이라 실제 산업 현장 분포와 차이가 있음
- 강건성 테스트는 인위적 열화 시뮬레이션이므로, 실제 현장 데이터 검증은 별도로 필요함
- 같은 강판을 연속 촬영한 유사 이미지가 랜덤 분할 탓에 train/test 양쪽에 들어갔을 가능성이 있음 (test 100%를 부풀린 또 하나의 요인일 수 있음)
- '정상(무결함)' 클래스가 없어 어떤 이미지든 결함 6종 중 하나로 분류함. 실제 검사 라인 투입에는 정상 클래스 추가 또는 이상 탐지 단계가 선행되어야 함
