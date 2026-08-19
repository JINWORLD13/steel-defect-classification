# Steel Surface Defect Classification (NEU-CLS) + Robustness Analysis

[🇰🇷 한국어](README.md) | [🇯🇵 日本語](README.ja.md) | 🇺🇸 English

Classifies **6 types of steel surface defects** from manufacturing using transfer learning (ResNet18),
then asks the question that actually matters — **does lab accuracy survive the factory floor?** — and answers it with a robustness test.

> **TL;DR:** Accuracy was 100% on clean data, but injecting real-world degradations (blur, noise, low contrast)
> dropped it to 38–53%. In other words, the model is over-fitted to lab conditions, and this project quantifies
> exactly how much that costs — establishing the baseline for the degradation-augmented retraining that comes next.

---

## 1. Problem

- **Goal:** an automatic surface-defect classifier to replace manual visual inspection
- **6 target classes:** crazing, inclusion, patches, pitted_surface, rolled-in_scale, scratches
- **Approach:** fine-tune a pretrained ResNet18 via transfer learning

## 2. Data

- **Source:** NEU-CLS (Northeastern University, CC BY 4.0)
- **Composition:** 1,800 images, 6 balanced classes (300 each), 200×200
- **Split:** train **1,260** (70%) / val **270** (15%) / test **270** (15%) — stratified, class ratios preserved, fixed `seed 42`
  - ⚠️ The original ships as train 1,770 / valid 30, so **every single misclassified image swings validation accuracy by 3.3pp**.
    You cannot pick a best checkpoint off a set like that, so I **re-split the data myself** and grew the validation set
    **9× (to 270 images)** (`prepare_data.py`)
  - A `sorted()` call before shuffling removes any dependence on filesystem ordering — fixing the seed without sorting first still breaks reproducibility on another machine
  - The shuffle happens **within each class**. Shuffling everything at once lets one class pile up in the test split, which makes the per-class report untrustworthy

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
Each intensity is dialed to the point where **a human can still recognize the defect**. If a person can tell the classes apart and the model cannot, that gap is evidence of which cue the model was leaning on.

| Condition | Injection | Accuracy | vs. baseline |
|---|---|---|---|
| Original (baseline) | — | **1.000** | — |
| Dark | brightness **×0.4** — weakly lit line | 0.874 | −12.6pp |
| Bright | brightness **×1.7** — overexposure / metal reflection | 0.767 | −23.3pp |
| Blur | Gaussian blur **k=9, σ=3.0** — defocus / dust on lens | 0.530 | −47.0pp |
| **Noise** | Gaussian noise **σ=0.12** — cheap sensor | **0.385** | **−61.5pp** |
| Low contrast | contrast **×0.3** — hazy capture | 0.396 | −60.4pp |

![Robustness](robustness.png)

**Interpretation:**
- The model holds up reasonably well against lighting changes, but is **fragile against noise, low contrast, and blur** (up to −61.5pp)
- Why: the training data is clean lab photography, so the model appears to have learned to **rely on sharp texture**
- Conclusion: **retraining with noise, blur, and contrast augmentation is mandatory** before field deployment

## 6. What I learned

- **High accuracy ≠ a good model.** The 100% reflected an easy dataset; the robustness test exposed the real weakness.
- **I hit a "silent bug" firsthand.** The keys in `CONDITIONS` and the branch names in `make_transform()` had drifted apart, so **untouched originals were being evaluated with no transform applied at all**. Python raised nothing — the result wasn't wrong, it just *looked good*. Bugs that raise no error and only corrupt the result are the most dangerous kind, and what caught this one was not a debugger but the suspicion "why is this number like that?"
- **You have to make it visible to catch it.** I changed the script to print each condition's result and its drop against the baseline on its own line, so a condition that sailed through untransformed stands out immediately.
- **A second silent bug — this time in the reproduction script.** `prepare_data.py` overwrote its output folder without ever clearing it, so re-running with a different seed or split ratio left **the previous split in place and stacked the new one on top of it.** Reproduced on a 20-image stand-in, six images ended up in train and test at the same time. What it shared with the first bug is that a single run looks perfectly fine — a reproduction script is not something that has to work once, it is something that **has to give the same answer however many times you run it**.
- **Order of transforms decides what you are measuring.** Brightness, contrast, and blur belong in the PIL stage; noise has to be added after `ToTensor()` (on the 0–1 range) for the σ value to mean anything fixed. Drop the `clamp(0,1)` and you are measuring **normalization distortion**, not noise. `Normalize` always goes last.
- **Peeking at test during training defeats the point.** Even if the code never trains on test, **I end up choosing hyperparameters by looking at the test score** — and at that moment test is no longer unseen data.
- **A fixed seed alone is not reproducibility.** Sorting, splitting, noise, and training seeds all have to be pinned together before the same command yields the same result.
- **Evaluation design comes first.** Refusing to reuse the original's inadequate validation split was the precondition for any trustworthy number.

## 7. Next steps

- Retrain with degradations included in augmentation → measure how much robustness recovers
- Extend to **object detection (YOLO)** to localize defects, not just classify them — which is also the principled fix for the multi-label problem above (the labels already exist)
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
python audit_data.py      # data audit (split leakage · label definition)
```

> Data source: NEU-CLS (Northeastern University). If the curl link above is blocked, search Figshare for "NEU-CLS".
> Training involves randomness (GPU arithmetic is non-deterministic even with a fixed seed), so numbers may differ slightly on retraining.

## Project structure

```
prepare_data.py   # NEU-CLS → classification folder layout + stratified 3-way split
train.py          # ResNet18 transfer learning (MPS)
analyze.py        # confusion matrix · per-class precision/recall
robustness.py     # robustness measurement under 5 image degradations
audit_data.py     # split-leakage & label-definition audit (reproduces the Limitations figures)
confusion_matrix.png / robustness.png   # result figures
```

## Limitations

- The data comes from a single source (NEU-CLS) of clean lab captures, which differs from a real industrial distribution
- The robustness test uses **artificial degradation simulation** and applies **one condition at a time**; a real line is a **compound condition** — dim *and* out of focus at once — so validation on real field data is still required
- **File-level leakage was ruled out by an md5 check over every image** — **zero** identical files between train and test (the single train↔val pair is a duplicate present in the source dataset itself). Near-duplicate frames of the same steel plate may still have landed in both train and test through random splitting
- **The label definition itself is loose.** Cross-checked against the detection bounding boxes shipped with the source data (present for all 1,800 images, 4,189 boxes), **123 images (6.8%) carry a defect box of a class other than the one in their filename** — 188 foreign boxes, 4.5% of all boxes. The spread across classes is wide: `scratches` **16.7%**, `pitted_surface` 13.0%, `patches` 9.7%, against **0%** for `inclusion` and `rolled-in_scale`. Counted directly, **23 of the 270 test images (8.5%) have more than one right answer yet were graded against a single label.** So the 100% reflects not just an easy dataset but **a loosely posed problem** — the task needs restating as multi-label classification, or as detection (the figures above are reproducible with `audit_data.py`)
- There is no "normal / defect-free" class, so any image is forced into one of the 6 defect types. Real inspection-line deployment requires adding a normal class or an anomaly-detection stage first
