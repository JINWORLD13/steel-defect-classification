# 데이터 감사 — "정확도 1.000"의 용의자 심문 기록

> **Summary in English.** Clean test accuracy was 1.000. That number has exactly four possible causes: leakage, an easy task, a tiny eval set, or a loosely posed problem. We measured all four. md5 found zero identical files across train/test; a cosine near-duplicate audit flagged 54 similar-looking test images, but excluding all of them still gives 1.000. A 7-feature handcrafted baseline reaches 0.933, so the task is largely easy. And the dataset ships detection boxes: 23 of 270 test images contain 2+ defect kinds yet were graded with a single label — the problem framing, not the data, was loose. All figures reproduce via `audit_data.py`, `audit_neardup.py`, `baseline_handcrafted.py`.

## 왜 이 문서가 있는가

깨끗한 test에서 1.000이 나왔다. 이 숫자의 원인은 원리적으로 넷뿐이다.

1. 데이터가 샌다 (누수)
2. 문제가 원래 쉽다
3. 평가셋이 작아 우연히 다 맞았다
4. 채점 기준 자체가 헐겁다

넷 다 실측했고, 결론은 **2번과 4번이 참, 1번과 3번은 기각**이다.

## 용의자 1 — 누수: 기각

- md5 전수 검사(`audit_data.py`): train↔test **동일 파일 0건**. train↔val 1쌍은 원본 데이터셋 자체의 중복(patches_101↔105).
- md5는 1픽셀만 달라도 무력하므로 근접중복도 쟀다(`audit_neardup.py`): 32×32 흑백 코사인 유사도로 test 각 장의 최근접 train을 찾으면 r>0.90이 **54장**. 그러나 갤러리 목시 결과 동일 강판의 재촬영이 아니라 **질감·조명이 닮은 서로 다른 판**이었고(대조군: 같은 클래스 평균 유사도 0.046 vs 최근접 중앙값 0.736), 그 54장을 가장 보수적으로 전부 빼도 나머지 216장 정확도 **1.000** (윌슨 95% [0.983, 1.000]).

## 용의자 3 — 작은 평가셋: 조건부 기각

270장에서의 1.000은 윌슨 95% 신뢰구간으로 **[0.986, 1.000]**이다. "정확히 1.000"이 아니라 "0.986 이상"이 정직한 주장이다. 원본 배포의 valid 30장이었다면 하한이 0.884까지 내려간다 — 재분할(270장)이 왜 전제였는지의 근거.

## 용의자 2 — 쉬운 문제: 확정

딥러닝 없이 **수제 텍스처 통계 7개 + 로지스틱 회귀**만으로 **0.933** [0.897, 0.957] (`baseline_handcrafted.py`). 특징은 전부 사람이 설명 가능한 것들이다(평균 밝기, 대비, 가로/세로 엣지, 엣지 산포, 강한 엣지 비율, 방향성). 즉 이 데이터의 클래스들은 손으로 만든 숫자 7개로도 대부분 갈라진다.

| 방법 | 깨끗한 test 정확도 |
|---|---|
| 무작위 | 0.167 |
| 수제 특징 7 + 로지스틱 | 0.933 [0.897, 0.957] |
| ResNet18 전이학습 (v1) | 1.000 [0.986, 1.000] |

## 용의자 4 — 헐거운 문제 정의: 확정 (이 프로젝트 최대의 발견)

이 데이터는 `NEU-CLS`라는 이름으로 배포되지만 **1,800장 전부에 바운딩박스(4,189개)가 딸린 탐지용(NEU-DET 계열)**이다. 원저자는 "한 장에 결함 여러 개·여러 종류"를 전제했다.

- **123장(6.8%)**이 파일명 클래스와 다른 종류의 결함 박스를 함께 가짐 (남의 클래스 박스 188개 = 4.5%)
- 클래스별 편차: scratches 16.7% · pitted_surface 13.0% · patches 9.7% vs inclusion·rolled-in_scale 0%
- test 기준 **23장(8.5%)**이 결함 2종 이상을 품은 채 단일 라벨로 채점됨
- "대표 결함"의 기준마저 갈린다: 파일명이 **개수 다수결**과 어긋나는 이미지 22장, **면적 다수결**과 어긋나는 이미지 8장, 교집합 1장(scratches_130). 기준에 따라 답이 바뀐다는 것 자체가 단일 정답 부재의 증거다.

데이터가 틀린 게 아니다. **탐지용 데이터를 단일 라벨 분류로 눌러 담은 내 문제 설정이 데이터 구조와 안 맞았던 것**이고, 여기서 멀티라벨(→ `train_multilabel.py`)과 탐지(→ `detect_*.py`)로의 확장이 필연이 된다.

## 재현

```bash
python audit_data.py            # 누수 + 문제 정의 감사
python audit_neardup.py         # 근접중복 + 부분집합 재평가
python baseline_handcrafted.py  # 수제 특징 베이스라인
```

수치는 전부 `results/audit_labels.json`, `results/audit_neardup.json`, `results/baselines.json`에 저장된다.
