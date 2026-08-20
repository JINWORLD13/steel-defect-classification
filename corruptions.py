"""
손상(corruption) 은행 — 강도 5단계 사다리 + 학습용/평가용의 물리적 분리

이 파일의 존재 이유는 분리임:
  TRAIN_FAMILIES : 재학습(M4) 증강에 쓸 수 있는 손상 '계열'
  BENCH_SEEN     : 평가 손상 중 학습 계열과 같은 메커니즘 (오르는 게 당연한 구간)
  BENCH_HELDOUT  : 학습에 절대 넣지 않는 손상 (진짜 일반화를 재는 구간)
                   near/mid/far = 학습 계열과 메커니즘이 가까운/중간/먼 정도

평가에 쓸 손상을 학습에 넣는 순간 그 개선폭은 '시험지를 미리 본' 점수가 됨.
파일 맨 아래 assert가 이 경계를 코드로 지킴 — 문서 약속이 아니라 실행되는 약속임.

사다리 강도는 val에서 보정했고 test는 최종 보고 때 한 번만 봄 (bench_robust.py).

결정성: 무작위가 필요한 손상(노이즈류)은 자기만의 난수 생성기를 (종류, 강도)로
시드해서 씀 — 같은 명령을 두 번 돌리면 완전히 같은 숫자가 나와야 재현임.
"""
import io as _io
import zlib

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from common import MEAN, STD

# ── 분리 선언 ──────────────────────────────────────────────────────────
TRAIN_FAMILIES = {"gaussian_noise", "gaussian_blur", "brightness", "contrast"}

BENCH_SEEN = ["gaussian_noise", "gaussian_blur", "dark", "bright", "low_contrast"]
SEEN_MECHANISM = {                    # 평가 이름 -> 학습 계열 (같은 메커니즘 표)
    "gaussian_noise": "gaussian_noise", "gaussian_blur": "gaussian_blur",
    "dark": "brightness", "bright": "brightness", "low_contrast": "contrast",
}

BENCH_HELDOUT = ["motion_blur", "speckle", "salt_pepper", "gamma", "jpeg", "pixelate"]
HELDOUT_DISTANCE = {                  # 학습 계열과의 메커니즘 거리
    "motion_blur": "near",            # 블러의 사촌 (방향성만 다름)
    "speckle": "near",                # 노이즈의 사촌 (곱셈식)
    "salt_pepper": "mid",             # 충격 노이즈 — 가우시안과 분포가 다름
    "gamma": "mid",                   # 비선형 밝기 — 선형 밝기와 다름
    "jpeg": "far",                    # 압축 아티팩트 — 학습에 비슷한 것도 없음
    "pixelate": "far",                # 블록화 — NEAREST로 만들어야 진짜 픽셀레이트
}

# ── 강도 사다리 (1~5단계, val에서 보정) ────────────────────────────────
# 사다리는 val에서 한 번 보정한 뒤 동결했음 (test는 최종 보고 때 처음 봄).
# 보정 이유: 초기값에서 jpeg s1·s2와 gamma s1이 만점(정보 없음), salt_pepper
# s4·s5와 노이즈류 상단이 우연 수준(0.167)으로 포화(정보 없음)였음.
LADDER = {
    "gaussian_noise": [0.03, 0.06, 0.10, 0.14, 0.20],   # σ (0~1 텐서 기준)
    "gaussian_blur": [0.8, 1.5, 2.4, 3.5, 5.0],          # σ (커널은 σ에서 유도)
    "dark": [0.65, 0.50, 0.40, 0.30, 0.22],              # 밝기 배율
    "bright": [1.3, 1.5, 1.7, 2.0, 2.4],
    "low_contrast": [0.60, 0.45, 0.35, 0.25, 0.18],      # 대비 배율
    "motion_blur": [3, 5, 9, 13, 19],                    # 가로 블러 커널 길이(px)
    "speckle": [0.06, 0.12, 0.18, 0.28, 0.40],           # 곱셈 노이즈 σ
    "salt_pepper": [0.005, 0.01, 0.02, 0.04, 0.08],      # 뒤집을 픽셀 비율
    "gamma": [1.6, 2.0, 2.4, 2.9, 3.5],                  # t**γ (γ>1 = 어두운 쪽 뭉갬)
    "jpeg": [40, 25, 15, 10, 6],                         # 저장 품질 (낮을수록 손상 큼)
    "pixelate": [75, 56, 37, 28, 20],                    # 축소 변 길이 (224 기준)
}

ALL_KINDS = BENCH_SEEN + BENCH_HELDOUT


def _gen(kind, severity):
    """(종류, 강도)마다 고정 시드의 개인 난수기 — 전역 시드에 안 기댐.

    hash()가 아니라 crc32를 쓰는 이유: 파이썬 문자열 hash는 보안상 프로세스마다
    달라짐(PYTHONHASHSEED) — 그걸 시드로 쓰면 "같은 명령 = 같은 숫자"가 깨짐.
    crc32는 언제 어디서 돌려도 같은 값을 냄.
    """
    g = torch.Generator()
    g.manual_seed(zlib.crc32(f"{kind}-{severity}".encode()))
    return g


def make_corrupt_transform(kind, severity):
    """(종류, 강도 1~5) -> 전처리 파이프라인.

    순서 규칙은 robustness.py와 동일함:
    PIL 단계(밝기·대비·블러·jpeg·픽셀레이트) → ToTensor → 텐서 단계(노이즈류) → Normalize
    """
    p = LADDER[kind][severity - 1]
    pil_steps, tensor_steps = [], []

    if kind == "dark" or kind == "bright":
        pil_steps.append(transforms.ColorJitter(brightness=(p, p)))
    elif kind == "low_contrast":
        pil_steps.append(transforms.ColorJitter(contrast=(p, p)))
    elif kind == "gaussian_blur":
        k = int(2 * round(3 * p) + 1)                    # σ의 ±3σ를 덮는 홀수 커널
        pil_steps.append(transforms.GaussianBlur(kernel_size=k, sigma=(p, p)))
    elif kind == "jpeg":
        def jpeg_pil(img, q=p):
            buf = _io.BytesIO()
            img.save(buf, format="JPEG", quality=int(q))  # 품질 q로 압축했다가
            buf.seek(0)
            return Image.open(buf).convert("RGB")          # 다시 열면 아티팩트만 남음
        pil_steps.append(transforms.Lambda(jpeg_pil))
    elif kind == "pixelate":
        def pixelate_pil(img, s=int(p)):
            # NEAREST가 핵심 — 기본값(BILINEAR+antialias)으로 하면 블록이 아니라
            # 블러가 돼서 held-out이 아니라 학습 계열의 사촌이 돼 버림
            small = img.resize((s, s), Image.NEAREST)
            return small.resize((224, 224), Image.NEAREST)
        pil_steps.append(transforms.Lambda(pixelate_pil))

    if kind == "gaussian_noise":
        g = _gen(kind, severity)
        tensor_steps.append(transforms.Lambda(
            lambda t: torch.clamp(t + torch.randn(t.shape, generator=g) * p, 0, 1)))
    elif kind == "speckle":
        g = _gen(kind, severity)
        tensor_steps.append(transforms.Lambda(                 # 곱셈 노이즈: 밝은 곳이 더 흔들림
            lambda t: torch.clamp(t * (1 + torch.randn(t.shape, generator=g) * p), 0, 1)))
    elif kind == "salt_pepper":
        g = _gen(kind, severity)
        def sp(t, f=p, g=g):
            r = torch.rand(t.shape[1:], generator=g)           # 픽셀 위치별 주사위 (채널 공통)
            t = t.clone()
            t[:, r < f / 2] = 0.0                              # 절반은 검정(pepper)
            t[:, r > 1 - f / 2] = 1.0                          # 절반은 하양(salt)
            return t
        tensor_steps.append(transforms.Lambda(sp))
    elif kind == "gamma":
        tensor_steps.append(transforms.Lambda(lambda t: t.pow(p)))
    elif kind == "motion_blur":
        k = int(p)
        kernel = torch.zeros(1, 1, 1, k)
        kernel[..., :] = 1.0 / k                               # 가로 일자 평균 = 수평 흔들림
        def mb(t, kn=kernel, k=k):
            t = t.unsqueeze(0)                                 # (3,H,W) -> (1,3,H,W)
            out = F.conv2d(t, kn.expand(3, 1, 1, k), padding=(0, k // 2), groups=3)
            return out.squeeze(0)
        tensor_steps.append(transforms.Lambda(mb))

    return transforms.Compose(
        [transforms.Resize((224, 224))] + pil_steps
        + [transforms.ToTensor()] + tensor_steps
        + [transforms.Normalize(MEAN, STD)]
    )


def clean_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


# ── 경계 검문 — import 되는 순간 실행됨 ────────────────────────────────
# held-out은 학습 계열과 이름이 겹치면 안 되고, seen은 전부 학습 계열에 대응해야 함
assert not (TRAIN_FAMILIES & set(BENCH_HELDOUT)), "held-out이 학습 계열에 뚫림!"
assert set(SEEN_MECHANISM) == set(BENCH_SEEN)
assert set(HELDOUT_DISTANCE) == set(BENCH_HELDOUT)
assert all(len(v) == 5 for v in LADDER.values())


if __name__ == "__main__":
    # 손상 격자 그림: 행=종류 11, 열=강도 5. 눈으로 강도가 말이 되는지 검수하는 용도.
    import os
    import matplotlib.pyplot as plt
    from labels import read_manifest

    sample = read_manifest("test")[0]                          # 아무 test 이미지 한 장
    img = Image.open(sample).convert("RGB")

    fig, axes = plt.subplots(len(ALL_KINDS), 5, figsize=(10, 22))
    for r, kind in enumerate(ALL_KINDS):
        for c in range(5):
            t = make_corrupt_transform(kind, c + 1)(img)       # 정규화된 텐서
            # 보기용으로 정규화를 되돌림 (t*std+mean)
            show = t * torch.tensor(STD).view(3, 1, 1) + torch.tensor(MEAN).view(3, 1, 1)
            axes[r][c].imshow(show.clamp(0, 1).permute(1, 2, 0))
            axes[r][c].axis("off")
            if c == 0:
                axes[r][c].set_title(f"{kind}", loc="left", fontsize=9)
    fig.suptitle("Corruption bank: 11 kinds x 5 severities (row title = kind)")
    plt.tight_layout()
    os.makedirs("figures", exist_ok=True)
    plt.savefig("figures/04_corruption_grid.png", dpi=110)
    print("그림 저장 -> figures/04_corruption_grid.png")
