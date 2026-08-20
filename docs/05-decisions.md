# 결정 기록 (ADR) — 선택하지 않은 것과 그 이유

> **Summary in English.** Lightweight decision records. Each entry states the decision, the measured evidence behind it, and what we deliberately did not do: ultralytics YOLO (AGPL vs this repo's MIT), FastAPI/Docker serving (no differentiation, measured OOM risk on free tiers), MLflow/W&B/DVC (90+ hours for zero model improvement at this scale), TTA (measured: destroys clean accuracy), MPS for detection (300x slower than CPU once you synchronize before timing), Python 3.12 migration (v1 reproduction is the ground truth), and re-splitting to purge near-duplicates (would invalidate every v1 number).

각 항목: **결정 / 근거 / 대안이었던 것**.

## 1. 탐지 프레임워크 — torchvision Faster R-CNN (ultralytics YOLO 아님)

- 근거: ultralytics는 **AGPL-3.0**, 이 레포는 MIT. 공개 채용용 레포에서 라이선스 충돌은 순손해. 설치 자체는 Python 3.9에서 가능함을 확인했으므로 "못 써서"가 아니라 "판단해서" 안 쓴 것.
- 백본을 v1과 같은 ResNet18로 두어 분류·멀티라벨·탐지를 같은 출발점에서 비교.

## 2. 탐지 장치 — CPU 고정 (MPS 아님)

- 근거: MPS는 비동기라 `torch.mps.synchronize()` 없이 재면 **커널을 큐에 넣는 시간만** 재게 됨. 동기화 후 실측하니 탐지 추론이 CPU보다 수백 배 느렸다. "빨라 보였던" 최초 측정은 측정 버그였고, 그 자체가 이 문서에 남길 가치가 있는 교훈.

## 3. 옵티마이저 — 탐지에 AdamW (SGD 아님)

- 근거: 8장 암기 실험에서 같은 150스텝에 SGD 0.465 vs **AdamW 0.833**. 데이터가 작고 CPU 예산이 빠듯하면 빨리 수렴하는 쪽이 정답.

## 4. 앵커 — 비율만 확장, 크기는 기본값 유지

- 근거(`anchor_recall.py`, 학습 없이 실측): 기본 비율(0.5,1,2)은 **scratches 리콜@0.5가 16.3%** — 가늘고 긴 결함을 후보조차 못 만듦. 비율에 0.1과 10을 더하면 92.8%, 전체 84.6%→97.4%. 크기에서 512를 빼는 안은 pitted_surface(면적 중앙값 0.55) 리콜을 떨어뜨려 기각.

## 5. TTA(테스트타임 증강) — 안 씀

- 근거: 실측에서 이득이 없거나 해로움 (블러 섞은 4뷰는 clean 1.000→0.830으로 파괴, 기하 전용도 노이즈 범위 내). 추론 비용만 4배. "해봤고, 쟀고, 안 됐다"로 기록.

## 6. 서빙(FastAPI/Docker/배포) — v2 범위에서 제외

- 근거: (a) ML 포트폴리오에서 가장 흔한 구성이라 변별력이 없음. (b) 무료 티어(512MB)에서 기본 스레드풀 구성이 실측 OOM. (c) 그 시간이면 held-out 평가와 데이터 감사를 다 함 — 실제로 그렇게 썼음.

## 7. MLflow / W&B / DVC / CI 배지 — 안 씀

- 근거: 이 규모(26MB 불변 데이터, 분 단위 학습)에서 모델 품질을 1%p도 안 올리는데 도입·유지비만 큼. 대신 진짜 필요한 세 가지만: `splits/*.txt` 매니페스트, `results/*.json` 단일 출처, 회귀 테스트. "왜 DVC를 안 썼나"에 숫자로 답하는 쪽이 도구 나열보다 값어치 있음.

## 8. Python 3.9 유지 — venv 갈아엎지 않음

- 근거: v1 재현이 모든 비교의 정답지인데 환경을 바꾸면 정답지가 흔들림. 필요한 것(torchmetrics 1.8.2, pycocotools 2.0.11, pytest 8.4.2)은 전부 3.9 휠이 있음을 확인.

## 9. 근접중복 제거 재분할 — 안 함

- 근거: 분할을 바꾸면 v1의 모든 실측치가 무효가 되고 before/after 자체가 성립 안 함. 대신 "진단 + 부분집합 재평가"로 같은 정보를 분할 훼손 없이 얻음 (r>0.90 전부 빼도 1.000). 유사도 그래프 연결요소 단위의 group split은 v3 후보로 남김.

## 10. 손상 사다리 — val에서 보정 후 동결

- 근거: 초기 사다리는 jpeg s1·s2 만점, salt_pepper 상단 우연 수준 등 정보 없는 칸이 많았음. **val에서 한 번 보정하고 동결**, test는 최종 보고 때 처음 봄. 시드도 crc32로 고정해 같은 명령이 같은 숫자를 냄 (파이썬 `hash()`는 프로세스마다 달라져 재현이 깨지는 함정 확인).

## 11. held-out 경계 — 코드로 강제

- 근거: pixelate를 기본 리사이즈(BILINEAR+antialias)로 만들면 사실상 블러가 되어 held-out 경계가 안에서 뚫림 — NEAREST로 강제. 학습 증강은 `TRAIN_RANGES`의 연속 구간에서만 뽑고, `corruptions.py`의 assert가 held-out 침범을 import 시점에 막음.
