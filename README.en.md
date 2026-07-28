# Steel Surface Defect Classification (NEU-CLS) + Robustness Analysis

[🇰🇷 한국어](README.md) | [🇯🇵 日本語](README.ja.md) | 🇺🇸 English

Classifies **6 types of steel surface defects** from manufacturing using transfer learning (ResNet18),
then asks the question that actually matters — **does lab accuracy survive the factory floor?** — and answers it with a robustness test.

> **TL;DR:** Accuracy was 100% on clean data, but injecting real-world degradations (blur, noise, low contrast)
> dropped it to 38–53%. In other words, the model is over-fitted to lab conditions, and this project
> quantifies exactly how much degradation augmentation is needed before field deployment.

---

## 1. Problem

- **Goal:** an automatic surface-defect classifier to replace manual visual inspection
- **6 target classes:** crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches
- **Approach:** fine-tune a pretrained ResNet18 via transfer learning

## 2. Data

- **Source:** NEU-CLS (Northeastern University, CC BY 4.0)
- **Composition:** 1,800 images, 6 balanced classes (300 each), 200×200
- **Split:** train 70% / val 15% / test 15% (stratified, class ratios preserved)
  - ⚠️ The original validation set held only 30 images — far too small to trust. I **re-split the data myself**
    to make evaluation meaningful (`prepare_data.py`)

## 3. Method

| Item | Detail |
|---|---|
| Model | ResNet18 (ImageNet pretrained), final fc layer replaced for 6 classes |
| Augmentation | RandomHorizontalFlip, RandomRotation(±15°) |
| Loss / Optimizer | CrossEntropyLoss / Adam (lr=1e-3) |
| Training | 12 epochs, **checkpoint saved at best val accuracy** (guards against overfitting) |
| Device | Auto-selects CUDA / MPS (Apple GPU) / CPU — this run used Apple Silicon MPS |

## 4. Results — clean test set

**100% test accuracy** (270 images, precision and recall both 1.000 for every class)

![Confusion Matrix](confusion_matrix.png)

A flawless diagonal → no confusion between classes.
**Rather than taking this "too perfect" result at face value**, I measured the model's real capability with the robustness test below.

## 5. Robustness analysis (the core of this project)

Five image degradations that actually occur on a production line were injected into the test set, and accuracy was re-measured. (`robustness.py`)

| Condition | Accuracy | vs. baseline |
|---|---|---|
| Original (baseline) | **1.000** | — |
| Dark (weak lighting) | 0.874 | −12.6pp |
| Bright (overexposure / reflection) | 0.767 | −23.3pp |
| Blur (defocus / dust on lens) | 0.530 | −47.0pp |
| **Noise (cheap sensor)** | **0.385** | **−61.5pp** |
| Low contrast (hazy capture) | 0.396 | −60.4pp |

![Robustness](robustness.png)

**Interpretation:**
- The model holds up reasonably well against lighting changes, but is **fragile against noise, low contrast, and blur** (up to −61.5pp)
- Why: the training data is clean lab photography, so the model appears to have learned to **rely on sharp texture**
- Conclusion: **retraining with noise, blur, and contrast augmentation is mandatory** before field deployment

## 6. What I learned

- **High accuracy ≠ a good model.** The 100% reflected an easy dataset; the robustness test exposed the real weakness.
- **I hit a "silent bug" firsthand.** A typo in a condition key (`low_contrast`) made the contrast test run on unmodified images and pass at a **fake 100%**. I found and fixed it. Bugs that raise no error and only corrupt the result are the most dangerous kind.
- **Evaluation design comes first.** Refusing to reuse the original's inadequate validation split was the precondition for any trustworthy number.

## 7. Next steps

- Retrain with degradations included in augmentation → measure how much robustness recovers
- Extend to **object detection (YOLO)** to localize defects, not just classify them
- Ship a Streamlit/Gradio demo

---

## How to run

```bash
# 1) Download the data — NEU-CLS (~26MB, CC BY 4.0, Figshare)
curl -sL -o NEU-CLS.zip "https://ndownloader.figshare.com/files/54094775"
unzip -q NEU-CLS.zip -d data

# 2) Set up the environment
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 3) Run the pipeline
python prepare_data.py    # data/ → dataset/, stratified re-split (train/val/test)
python train.py           # training → produces best_model.pth
python analyze.py         # confusion matrix + per-class report
python robustness.py      # robustness test → robustness.png
```

> Data source: NEU-CLS (Northeastern University). If the curl link above is blocked, search Figshare for "NEU-CLS".
> Training involves randomness (GPU arithmetic is non-deterministic even with a fixed seed), so numbers may differ slightly on retraining.

## Project structure

```
prepare_data.py   # NEU-CLS → classification folder layout + stratified 3-way split
train.py          # ResNet18 transfer learning
analyze.py        # confusion matrix · per-class precision/recall
robustness.py     # robustness measurement under 5 image degradations
confusion_matrix.png / robustness.png   # result figures
```

## Limitations

- The data comes from a single source (NEU-CLS) of clean lab captures, which differs from a real industrial distribution
- The robustness test uses artificial degradation simulation; validation on real field data is still required
- Near-duplicate frames of the same steel plate may have landed in both train and test due to random splitting (possibly another factor inflating the 100% test accuracy)
- There is no "normal / defect-free" class, so any image is forced into one of the 6 defect types. Real inspection-line deployment requires adding a normal class or an anomaly-detection stage first
