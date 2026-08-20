# 강건성 — 문제를 실제로 고치고, 고쳐졌다는 주장을 검증한 기록

> **Summary in English.** We rebuilt the robustness evaluation as an 11-corruption x 5-severity ladder, calibrated on val and frozen before touching test. Retraining with augmentation drawn from 4 mechanism families (continuous ranges, never the eval presets, held-out families excluded by an import-time assert) improves every one of the 6 held-out corruptions — including the far zone the model never saw any cousin of (jpeg +0.047, pixelate +0.040; speckle +0.365, salt_pepper +0.264). The cost is real: clean accuracy drops 1.000 → 0.967 and McNemar's exact test says that drop is not noise (p = 0.004). Model selection used val only; the seed with the best clean test score was *not* selected, and we report the 5-seed range instead of hiding it.

## v1이 남긴 문제

v1의 강건성 테스트(5종 × 단일 강도)는 문제를 발견했지만 해결하지 않았다. 노이즈 0.385, 저대비 0.396, 흐림 0.530.

## 벤치마크 재설계 (M3)

- **11종 × 강도 5단계** (`corruptions.py`): seen 5종(학습 계열과 같은 메커니즘) + held-out 6종(절대 학습에 안 넣음, near/mid/far로 거리 표기)
- **사다리는 val에서 보정 후 동결.** 초기값은 jpeg s1·s2 만점, salt_pepper 상단 우연 수준 등 정보 없는 칸이 많았다. test는 최종 보고 때 처음 봤다.
- **결정성**: 손상 난수를 crc32 시드의 개인 생성기로 고정 — 같은 명령이 같은 숫자를 낸다. (`hash()`는 프로세스마다 달라 재현이 깨진다는 함정을 실측으로 확인)
- **경계는 코드가 지킨다**: `assert not (TRAIN_FAMILIES & set(BENCH_HELDOUT))`가 import 시점에 실행된다. pixelate는 NEAREST로 강제 — 기본 리사이즈로 만들면 사실상 블러가 되어 held-out이 안에서 뚫린다.

v1 기준선 (test, 강도 평균): clean 1.000 / seen 0.614 / near 0.643 / mid 0.565 / far 0.863.

## 재학습 (M4)

학습 증강은 4계열(노이즈·블러·밝기·대비)을 **연속 범위에서** 뽑아 절반 확률로 적용 (`train_robust.py`). 평가 사다리의 고정값을 그대로 쓰지 않으므로 "시험지를 외운" 것이 아니라 "그 계열에 익숙해진" 것이다. held-out 6종은 코드가 막는다.

- 시드 5개(42~46), 모델 선택은 **val의 (clean + seen s3) 평균**으로만. 선택점수: 42:0.932, 43:0.938, 44:0.930, 45:0.933, **46:0.939 (대표)**

## 결과 (test — 모델당 한 번)

| 구간 | v1 | v2(시드46) | 시드 범위 |
|---|---|---|---|
| clean | 1.000 | 0.967 | 0.967~0.993 |
| seen (당연히 오르는 구간) | 0.614 | 0.894 | 0.886~0.895 |
| held-out near | 0.643 | 0.899 | 0.835~0.905 |
| held-out mid | 0.565 | 0.738 | 0.653~0.741 |
| held-out far | 0.863 | 0.906 | 0.778~0.906 |

held-out 종별 (6종 중 **6종 개선** — 평균 하나로 뭉개지 않고 셈):
speckle +0.365 · salt_pepper +0.264 · motion_blur +0.147 · gamma +0.081 · jpeg +0.047 · pixelate +0.040

## 정직한 대가 — clean이 깎였고, 그 하락은 우연이 아니다

같은 270장을 두 모델이 풀었으므로 짝지은 비교다. **맥니마 정확검정**: v1만 맞힌 문제 9, v2만 맞힌 문제 0 → **p = 0.004**. 깨끗한 데이터 3.3%p 하락은 노이즈가 아니라 실제 비용이다. (처음엔 윌슨 구간 겹침으로 판정하려 했으나, 짝지은 설계엔 맥니마가 맞는 검정이라 바꿨다.)

또 하나: clean이 가장 높은 시드는 43(0.993)이었지만 **대표는 val이 고른 46(0.967)**이다. test를 보고 시드를 고르면 이 표의 모든 숫자가 오염된다 — 규율이 아깝게 느껴지는 순간이 규율이 일하는 순간이다.

## 재현

```bash
python bench_robust.py --model best_model_v1.pth --split test --name v1
for s in 42 43 44 45 46; do python train_robust.py --seed $s; done
python compare_robust.py     # → figures/05, results/robust_v2.json
```
