# 鋼材表面欠陥の分類 (NEU-CLS) + ロバスト性分析

[🇰🇷 한국어](README.md) | 🇯🇵 日本語

製造工程で発生する**鋼材表面の欠陥6種**を転移学習(ResNet18)で分類し、
**「実験室での精度は現場でも保たれるのか」**をロバスト性(robustness)テストで検証したプロジェクト。

> **要約:** クリーンなデータでは精度100%だったが、現場の悪条件(ぼけ・ノイズ・低コントラスト)を注入すると
> 38〜53%まで低下。つまり本モデルは実験室条件に過度に依存しており、現場投入前に劣化オーグメンテーションが
> 必要であることを定量的に確認した。

---

## 1. 課題設定

- **目的:** 目視検査を代替する表面欠陥の自動分類器
- **対象6クラス:** crazing(微細亀裂)、inclusion(介在物)、patches(斑点)、pitted_surface(点食)、rolled-in_scale(圧延スケール)、scratches(擦り傷)
- **アプローチ:** 事前学習済み ResNet18 を転移学習でファインチューニング

## 2. データ

- **出典:** NEU-CLS (Northeastern University, CC BY 4.0)
- **構成:** 1,800枚、6クラス均衡(各300枚)、200×200
- **分割:** train 70% / val 15% / test 15%(クラス比率を維持した stratified 分割)
  - ⚠️ 原データは検証セットが30枚しかなく信頼できなかったため、**自前で再分割**して評価の信頼性を確保 (`prepare_data.py`)

## 3. 手法

| 項目 | 内容 |
|---|---|
| モデル | ResNet18 (ImageNet 事前学習) + 最終 fc 層を6クラス用に置換 |
| オーグメンテーション | RandomHorizontalFlip, RandomRotation(±15°) |
| 損失/最適化 | CrossEntropyLoss / Adam (lr=1e-3) |
| 学習 | 10 epochs、**val 精度が最良の時点のモデルを保存**(過学習対策) |
| デバイス | Apple Silicon GPU (PyTorch MPS) |

## 4. 結果 — クリーンな test データ

**test 精度 100%**(270枚、クラス別 precision/recall いずれも 1.000)

![Confusion Matrix](confusion_matrix.png)

混同行列は完全な対角線 → クラス間の取り違えなし。
**ただしこの「出来すぎた」結果を鵜呑みにせず**、以下のロバスト性テストで実力を検証した。

## 5. ロバスト性分析(本題)

現場で実際に起こりうる画像劣化5種を test データに注入し、精度の変化を測定した。(`robustness.py`)

| 条件 | 精度 | 原本との差 |
|---|---|---|
| 原本(基準) | **1.000** | — |
| 暗い(照明不足) | 0.874 | −12.6%p |
| 明るい(照明過多・反射) | 0.767 | −23.3%p |
| ぼけ(ピント・粉塵) | 0.530 | −47.0%p |
| **ノイズ(安価なセンサー)** | **0.385** | **−61.5%p** |
| 低コントラスト(白っぽい) | 0.396 | −60.4%p |

![Robustness](robustness.png)

**考察:**
- 照明変化には比較的耐えるが、**ノイズ・低コントラスト・ぼけには脆弱**(最大 −61%p)
- 原因: 学習データがクリーンな実験室撮影のため、モデルが**鮮明なテクスチャに依存**して学習したと考えられる
- 結論: 現場投入前に**ノイズ・ブラー・コントラスト劣化を加えた再学習**が必須

## 6. 学び / 振り返り

- **高い精度 ≠ 良いモデル。** 100%はデータが易しかった結果にすぎず、ロバスト性テストが実際の弱点を露わにした。
- **「静かなバグ」を身をもって経験。** ロバスト性コードで条件キーの綴り誤り(`low_contrast`)により、コントラストのテストが無変換のまま通過し**偽の100%**が出ていたのを発見・修正。エラーを出さず結果だけが誤るバグが最も危険だと実感した。
- **評価設計の重要性。** 原データの不十分な val 分割をそのまま使わず再分割したことが、信頼できる結果の前提だった。

## 7. 次のステップ

- 劣化条件をオーグメンテーションに組み込んだ再学習 → ロバスト性の回復を検証
- 欠陥の位置まで特定する**物体検出(YOLO)**への拡張
- Streamlit/Gradio によるデモ公開

---

## 実行方法

```bash
# 1) データ取得 — NEU-CLS (約26MB, CC BY 4.0, Figshare)
curl -sL -o NEU-CLS.zip "https://ndownloader.figshare.com/files/54094775"
unzip -q NEU-CLS.zip -d data

# 2) 環境構築
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3) パイプライン実行
python prepare_data.py    # data/ → dataset/ へ再分割 (train/val/test)
python train.py           # 学習 → best_model.pth を生成
python analyze.py         # 混同行列 + クラス別スコア
python robustness.py      # ロバスト性テスト → robustness.png
```

> データ出典: NEU-CLS (Northeastern University)。上記 curl リンクが使えない場合は Figshare で「NEU-CLS」を検索して取得可能。

## ファイル構成

```
prepare_data.py   # NEU-CLS を分類用フォルダ構造へ整形 + stratified 3分割
train.py          # ResNet18 の転移学習 (MPS)
analyze.py        # 混同行列・クラス別 precision/recall
robustness.py     # 画像劣化5種におけるロバスト性の測定
confusion_matrix.png / robustness.png   # 結果画像
```

## 限界

- データが単一出典(NEU-CLS)のクリーンな撮影画像であり、実際の産業現場の分布とは差がある
- ロバスト性テストは人為的な劣化シミュレーションであり、実データによる検証は別途必要
