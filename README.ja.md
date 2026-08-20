# 鋼材表面欠陥検査 — 100%を尋問し、直し、問題を立て直した記録

[🇰🇷 한국어](README.md) | 🇯🇵 日本語 | [🇺🇸 English](README.en.md)

転移学習の分類器がクリーンな test で**精度 1.000** を出した。本リポジトリはその数値を誇る代わりに、**尋問し(第1幕)、なぜ崩れるかを解明し(第2幕)、実際に直し(第3幕)、問題そのものを立て直した(第4幕)**記録である。

> **要約4行:**
> ① 100%の容疑者4つ(リーク/易しさ/小さな評価セット/緩い採点)をすべて実測 — リークは棄却、「易しさ」と「緩い定義」は確定
> ② 劣化拡張の再学習で**学習に使っていない held-out 損傷6種がすべて改善**(最大 +36.5%p) — 代償として clean 1.000→0.967(McNemar p=0.004、低下は実在)
> ③ Grad-CAM を自作・検証し「エッジ依存」の推定を定量化 — 崩れるのはヒートマップの集中ではなく**存在**(全滅率 0→29%)
> ④ 同じバックボーンでマルチラベル・検出まで — 単一ラベルでは**原理的に不可能**だったもの(複数欠陥・位置・個数)を回復

---

## 第1幕 · 発見 — 「1.000 とは何だったのか」

クリーン test の 100% の原因は原理的に4つしかない。4つとも実測した。(詳細: [docs/01-data-audit.md](docs/01-data-audit.md))

| 容疑者 | 方法 | 判定 |
|---|---|---|
| データリーク | md5 全数 + 32×32 コサイン近接重複監査 | **棄却** — 同一ファイル0件。r>0.90 の54枚(目視では似た*別の*鋼板)を全部除いても 1.000 [0.983, 1.000] |
| 評価セットが小さい | Wilson 信頼区間 | **条件付き** — 1.000 の正直な読みは「**0.986 以上**」(n=270) |
| 問題が易しい | 手作りテクスチャ統計 **7個** + ロジスティック回帰 | **確定** — 深層学習なしで **0.933** [0.897, 0.957] |
| 採点が緩い | 付属のバウンディングボックスと照合 | **確定** — 下記 |

**採点が緩かった証拠:** このデータは `NEU-CLS` の名で配布されるが、1,800枚すべてに検出用ボックス(4,189個)が付く実質 **NEU-DET** である。照合の結果 **123枚(6.8%)**がファイル名と別種の欠陥ボックスを併せ持ち、test では **270枚中23枚(8.5%)が、正解が一つでないのに単一ラベルで採点**されていた。「代表欠陥」の基準すら割れる(個数多数決と22枚、面積多数決と8枚が不一致、交差1枚)。データが誤っていたのではなく、**検出用データを単一ラベル分類に押し込めた自分の問題設定**が誤っていた — この発見が第4幕の理由になる。(`audit_data.py`, `audit_neardup.py`, `baseline_handcrafted.py` で全再現)

![Baseline ladder](figures/03_baseline_ladder.png)

## 第2幕 · 解明 — 「なぜ、どのように崩れるのか」

v1 の劣化5種の棒グラフを**損傷11種 × 強度5段階**の梯子に再設計 (`corruptions.py`)。学習に使える系列と絶対に使わない held-out 6種(near/mid/far)を **import 時の assert で物理的に分離**し、梯子の強度は val で校正して凍結 — test はモデルごとに一度だけ。(詳細: [docs/02-robustness.md](docs/02-robustness.md))

v1 基準線 (test, 強度平均): **clean 1.000 / seen 0.614 / near 0.643 / mid 0.565 / far 0.863**

「鮮明なテクスチャに依存する」という v1 の推定は **Grad-CAM の自作実装**で検証した (`gradcam.py` — GAP∘fc 恒等式で自己検証、最大差 1.2e-07)。ボックスがあるのでヒートマップは目視でなく**採点**できる: (詳細: [docs/03-explainability.md](docs/03-explainability.md))

- 6段のベースライン電池: 均一 1.00 → 中央事前分布 1.19 → **純エッジ 1.36** → 学習モデル **1.47** (ADR)。すべてに勝つが、**エッジとの差が薄いこと自体がエッジ依存の定量的証拠**。位置指標(PG)は 0.52→**0.81** と差が大きい
- deletion 検査通過 — CAM 上位セルの削除が無作為削除より全域で速く精度を崩す(忠実性)
- 劣化下で崩れるのは生き残ったヒートマップの集中(ADR ~1.5 維持)ではなく**存在** — 全滅率 0% → 20~29%

![CAM baselines](figures/06_cam_baselines.png)

## 第3幕 · 解決 — 「直した。そして直ったという主張を検証した」

学習拡張に4系列(ノイズ・ぼかし・明るさ・コントラスト)を**連続範囲から抽選**して再学習 (`train_robust.py`, シード5個)。評価プリセットの固定値は使わず、held-out 系列はコードが遮断。代表シードは **val スコアのみで決定** — clean test が最も高かったシードは*選ばなかった*。

| 区間 (test, 強度平均) | v1 | v2 | シード範囲 |
|---|---|---|---|
| clean | 1.000 | 0.967 | 0.967~0.993 |
| seen — 学習系列、上がって当然 | 0.614 | **0.894** | 0.886~0.895 |
| **held-out near** | 0.643 | **0.899** | 0.835~0.905 |
| **held-out mid** | 0.565 | **0.738** | 0.653~0.741 |
| **held-out far** — jpeg・ピクセレート、従兄弟すら未学習 | 0.863 | **0.906** | 0.778~0.906 |

held-out **6種すべて改善**: speckle +0.365 · salt_pepper +0.264 · motion_blur +0.147 · gamma +0.081 · jpeg +0.047 · pixelate +0.040

**正直な代償:** clean 1.000 → 0.967。同じ270枚の対比較なので **McNemar 正確検定**で判定 — v1 だけが正解した問題9、v2 だけ0、**p = 0.004**。この低下はノイズではなく頑健性の実費である。

![Before/after](figures/05_robustness_before_after.png)

## 第4幕 · 再定義 — 「単一ラベルには原理的にできないこと」

第1幕の発見(多重欠陥 test 23枚)を問題定義のレベルで解いた。3モデルすべて**同じ ResNet18 バックボーン、同じ test 270枚**。

| モデル | 採点 (ボックスのマルチホット) | 数値 |
|---|---|---|
| 単一ラベル v1 (argmax) | subset accuracy | 0.915 — **自らの構造的上限にちょうど到達**(多重欠陥23枚は原理的に満点不可) |
| マルチラベル (BCE, 20行の変更) | subset accuracy | **0.926** — 上限突破、見逃した欠陥ペア 23→18 |
| 検出 (Faster R-CNN) | mAP@50 / 運用点 | **0.734** (mAP@50-95 0.350, クラス別 0.41~0.91) / 見逃し 19.4% · 誤警報 1.87件/枚 (閾値 0.45, **val で決定**) |

検出設計の根拠は学習なしで作った (`anchor_recall.py`): 既定のアンカー比率は細長い scratches ボックスを **16.3%** しか捕まえられない → 比率 0.1/10 の追加で **92.8%**(全体 84.6→97.4)。デバイスは CPU — MPS は `synchronize()` 後の実測で数百倍遅かった(同期なしではキュー投入時間だけを測る罠)。50枚暗記ゲート(mAP@50≥0.9)は 0.667 で未達だが、軌跡(0.04→0.67)でパイプライン正常を確認して本学習に進んだ — 未達の事実も記録である。(詳細: [docs/04-detection.md](docs/04-detection.md))

![Detection](figures/08_det_metrics.png)

## 学んだこと (v1 + v2)

- **精度100%は結論ではなく尋問の始まり。** 原因候補4つを一つずつ測るのが検証である。
- **静かなバグ2件を捕まえ回帰テストで釘付け。** ① 条件キーの不一致で無変形の原本が評価されていた ② 再分割時に旧分割の上に積み重なり train/test がリークした — 共通点は「初回実行は無事に見える」。pytest 4件が再発を防ぐ。
- **境界は文書ではなくコードが守る。** held-out 侵犯は assert が、分割の証拠はマニフェストが、数値の出所は `results/*.json` が担う。
- **再現性の罠:** Python の `hash()` はプロセスごとに変わる(crc32 に交換)、ソートなしのシャッフルは OS 依存、GPU 学習はシード固定でも完全再現不可(重みの凍結バックアップが答案)。
- **統計は設計に合わせる:** 対比較には McNemar、比率には Wilson。n のない精度は書かない。
- **指標には自前のベースラインを:** ADR の無作為線は 1.0 ではなく「中央の円」の 1.19 だった。基準線のない数字は主張ではない。
- **陰性結果も結果:** TTA は実測で有害(clean 1.000→0.830)、far 区間の改善は +0.04 と小さい — 膨らませずそのまま書く。

## 実行方法

```bash
# 0) データ (~26MB) と環境
curl -sL -o NEU-CLS.zip "https://ndownloader.figshare.com/files/54094775"
unzip -q NEU-CLS.zip -d data
python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt

# 第1幕 — 基盤と監査
python prepare_data.py
python train.py
python analyze.py
python audit_data.py
python audit_neardup.py
python baseline_handcrafted.py
python -m pytest tests/ -q

# 第2幕 — ベンチと説明
python bench_robust.py --model best_model_v1.pth --split test --name v1
python gradcam.py && python cam_metrics.py && python cam_baselines.py

# 第3幕 — 再学習と判定
for s in 42 43 44 45 46; do python train_robust.py --seed $s; done
python compare_robust.py

# 第4幕 — マルチラベルと検出
python train_multilabel.py && python eval_multilabel.py
python anchor_recall.py
python detect_train.py --sanity
python detect_train.py --epochs 10
python detect_eval.py --split val
python detect_eval.py --split test --threshold 0.45
python detect_visualize.py --threshold 0.45
```

> データ出典: NEU 表面欠陥データベース (Northeastern University, Song & Yan)。Figshare の配布名は `NEU-CLS` だが中身は検出アノテーション付き(NEU-DET 系)。CC BY 4.0(再配布物の表記に基づく)。
> GPU 演算はシード固定でも非決定的で、再学習の数値は本文と多少異なりうる。v1 数値の答案はローカルの `best_model_v1.pth`。

## ファイル構成

```
common.py / labels.py        # 共用部品 (定数・モデル・結果・統計 / ボックスラベル)
prepare_data.py              # 再分割 + splits/ マニフェスト
train.py → analyze.py        # v1 学習・採点 (v1 再現の答案 — 変更禁止)
robustness.py                # v1 の劣化5種 (保存用)
audit_data.py / audit_neardup.py / baseline_handcrafted.py     # 第1幕
corruptions.py / bench_robust.py                                # 第2幕ベンチ
gradcam.py / cam_metrics.py / cam_baselines.py                  # 第2幕説明
train_robust.py / compare_robust.py                             # 第3幕
train_multilabel.py / eval_multilabel.py                        # 第4幕マルチラベル
detect_dataset.py / detect_train.py / detect_eval.py
detect_visualize.py / anchor_recall.py                          # 第4幕検出
tests/                       # 回帰テスト (リーク・無変形・パース)
splits/  results/  figures/  runs/  docs/                       # 証拠
best_model_v1.pth            # (ローカル専用) v1 基準線 — 絶対に上書きしない
```

## 限界

- 単一出所の実験室撮影 — 実際の現場分布は未知であり、劣化はシミュレーションで複合条件(暗くかつぼやけ)は掛けていない
- 「正常(無欠陥)」クラスがなく、どの画像も欠陥6種のいずれかに分類される
- 同一鋼板の*類似*フレーム問題は測定で緩和しただけで、撮影メタデータなしには完全排除できない — 類似度グラフの group split は v3 候補
- 検出は CPU 予算に合わせた最小構成(10エポック) — 数値はその予算の結果として読むべき

選ばなかったものとその理由(YOLO ライセンス、サービング除外、MLOps ツール除外、TTA 棄却など): [docs/05-decisions.md](docs/05-decisions.md)
