"""
열화 증강 재학습 — v1의 취약성을 실제로 고치는 단계 (v2 모델)

v1과 다른 점은 학습 증강 하나뿐임:
  RandomCorrupt — 배치마다 절반 확률로, TRAIN_FAMILIES(노이즈·블러·밝기·대비) 중
  하나를 골라 '연속 범위에서 뽑은 무작위 강도'로 적용함.

정직성 규칙 (corruptions.py의 분리를 학습 쪽에서 지키는 방법):
  1) held-out 계열(모션블러·스펙클·소금후추·감마·jpeg·픽셀레이트)은 절대 안 씀
     -> 파일 안 assert가 지킴
  2) 평가 사다리의 고정값을 그대로 쓰지 않고 연속 범위에서 뽑음
     -> "평가 문제를 외운" 게 아니라 "그 계열에 익숙해진" 것이 되게 함
  3) 모델 선택(best)은 val에서만 — clean과 seen s3의 평균으로 고름

실행: ./venv/bin/python train_robust.py --seed 42   (시드 5개: 42~46)
"""
import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from bench_robust import accuracy_under
from common import DEVICE, MEAN, STD, build_model, save_result
from corruptions import BENCH_SEEN, TRAIN_FAMILIES, clean_transform, make_corrupt_transform

BATCH_SIZE = 32
EPOCHS = 12
LR = 1e-3

# 학습용 강도 '범위' — 평가 사다리의 고정값이 아니라 연속 구간에서 매번 뽑음
TRAIN_RANGES = {
    "gaussian_noise": (0.02, 0.18),    # σ
    "gaussian_blur": (0.5, 4.0),       # σ (커널은 σ에서 유도)
    "brightness": (0.30, 1.90),        # 배율 (어두움~밝음을 한 축으로)
    "contrast": (0.25, 1.00),          # 배율 (낮추는 쪽만 — 현장 열화의 방향)
}
assert set(TRAIN_RANGES) == TRAIN_FAMILIES     # 학습은 선언된 계열만 쓸 수 있음


class RandomCorrupt:
    """텐서(0~1) 단계에서 동작하는 학습용 열화. 절반은 그대로 통과시킴 —
    깨끗한 이미지도 계속 봐야 clean 성능이 무너지지 않음."""

    def __init__(self, p=0.5, rng_seed=0):
        self.p = p
        self.rng = random.Random(rng_seed)     # 증강 전용 난수기 (전역과 분리)

    def __call__(self, t):
        if self.rng.random() > self.p:
            return t
        kind = self.rng.choice(sorted(TRAIN_RANGES))
        lo, hi = TRAIN_RANGES[kind]
        s = self.rng.uniform(lo, hi)           # 연속 범위에서 강도 추첨
        if kind == "gaussian_noise":
            return torch.clamp(t + torch.randn_like(t) * s, 0, 1)
        if kind == "gaussian_blur":
            k = int(2 * round(3 * s) + 1)
            return transforms.functional.gaussian_blur(t, kernel_size=k, sigma=s)
        if kind == "brightness":
            return torch.clamp(t * s, 0, 1)    # 텐서 곱 = 밝기 배율
        if kind == "contrast":
            mean = t.mean()                    # 평균 쪽으로 끌어당기면 대비가 줄어듦
            return torch.clamp((t - mean) * s + mean, 0, 1)
        return t


def make_train_tf(seed):
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),      # v1의 증강은 그대로 유지하고
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        RandomCorrupt(p=0.5, rng_seed=seed),    # 열화 증강만 추가 (0~1 텐서 단계)
        transforms.Normalize(MEAN, STD),
    ])


def selection_score(model):
    """모델 선택 점수 = val의 (clean + seen 5종 s3) 평균. test는 안 봄."""
    accs = [accuracy_under(model, "val", clean_transform())]
    for kind in BENCH_SEEN:
        accs.append(accuracy_under(model, "val", make_corrupt_transform(kind, 3)))
    return sum(accs) / len(accs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    train_ds = datasets.ImageFolder("dataset/train", transform=make_train_tf(args.seed))
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = build_model(pretrained=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best = 0.0
    out = f"best_model_robust_s{args.seed}.pth"
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

        score = selection_score(model)
        print(f"[seed {args.seed}][{epoch:2d}/{EPOCHS}] val 선택점수 {score:.3f}", flush=True)
        if score > best:
            best = score
            torch.save(model.state_dict(), out)
            print(f" best 갱신 -> {out} ({score:.3f})", flush=True)

    save_result(f"train_robust_s{args.seed}", {"seed": args.seed, "val_selection": round(best, 4)})
    print(f"\n끝. {out}, val 선택점수 {best:.3f}")


if __name__ == "__main__":
    main()
