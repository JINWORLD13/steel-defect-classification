"""
히트맵 지표의 베이스라인 배터리 — "이 숫자가 공짜 점수는 아닌가?"를 검문

ADR의 무작위 기준선은 1.0이지만, 그건 '완전 균일 맵'의 얘기일 뿐임.
모델도 이미지도 안 보는 '가운데 가우시안 원'이 이미 1.0을 훌쩍 넘길 수 있음 —
결함이 화면 중앙에 있는 경향(촬영 습관) 때문임. 그래서 학습된 모델의 점수는
균일맵이 아니라 이 '중앙 사전확률'과 '순수 엣지 검출기'를 이겨야 의미가 있음.

배터리 6단 (뒤로 갈수록 정보를 더 씀):
  1 uniform    : 아무 정보 없음 (수학적으로 ADR=1)
  2 rand-init  : 무작위 가중치 모델의 Grad-CAM (구조만 있음)
  3 center     : 가운데 가우시안 원 (모델·이미지 무시, 위치 사전확률만)
  4 edge       : 이미지의 엣지 세기 (이미지만 보고 모델은 없음)
  5 imagenet   : ImageNet 그대로의 ResNet18 (일반 지식, 결함 학습 없음)
  6 trained    : 우리 v1 (결함을 학습함)

+ deletion 검사: 히트맵 상위 칸부터 지웠을 때 정확도가 무작위 지우기보다
  빨리 무너지면, 히트맵이 '실제로 판정에 쓰인 곳'을 가리킨 것임 (신실성 검사).

실행: ./venv/bin/python cam_baselines.py
"""
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import models

from common import CLASSES, DEVICE, MEAN, STD, build_model, save_result
from corruptions import clean_transform
from cam_metrics import mask_of, score_cam
from gradcam import GradCAM, upsample
from labels import read_manifest, stem_of


def battery_scores(saliency_fn, name):
    """saliency_fn(경로) -> (224,224) 히트맵. test 전체를 ADR/EBPG/PG로 채점."""
    scores = []
    for path in read_manifest("test"):
        stem = stem_of(path)
        cls_idx = CLASSES.index(stem.rsplit("_", 1)[0])
        s = score_cam(saliency_fn(path, cls_idx), mask_of(stem, cls_idx))
        if s is not None:
            scores.append(s)
    adr = sum(s["adr"] for s in scores) / len(scores)
    ebpg = sum(s["ebpg"] for s in scores) / len(scores)
    pg = sum(s["pg"] for s in scores) / len(scores)
    print(f"{name:12s}: ADR {adr:.3f} | EBPG {ebpg:.3f} | PG {pg:.3f}")
    return {"adr": round(adr, 3), "ebpg": round(ebpg, 3), "pg": round(pg, 3)}


def main():
    tf = clean_transform()

    # 6단 배터리의 saliency 함수들 ---------------------------------------
    def uniform(path, c):
        return torch.ones(224, 224)

    torch.manual_seed(0)
    rand_model = build_model(pretrained=False)          # 무작위 가중치 (학습 없음)
    rand_cam = GradCAM(rand_model)
    def randinit(path, c):
        img = tf(Image.open(path).convert("RGB")).to(DEVICE)
        cam, _ = rand_cam(img, class_idx=c)
        return upsample(cam).cpu()

    yy, xx = torch.meshgrid(torch.arange(224.), torch.arange(224.), indexing="ij")
    center_map = torch.exp(-(((yy - 112) ** 2 + (xx - 112) ** 2) / (2 * 67.0 ** 2)))
    def center(path, c):
        return center_map                                # 모델도 이미지도 안 봄

    def edge(path, c):
        g = np.asarray(Image.open(path).convert("L").resize((224, 224)), dtype=np.float32)
        gy, gx = np.gradient(g)
        return torch.from_numpy(np.sqrt(gx * gx + gy * gy))

    imnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(DEVICE)
    imnet_cam = GradCAM(imnet)
    def imagenet(path, c):
        img = tf(Image.open(path).convert("RGB")).to(DEVICE)
        cam, _ = imnet_cam(img, class_idx=None)          # 자기가 고른 ImageNet 클래스 기준
        return upsample(cam).cpu()

    v1 = build_model()
    v1.load_state_dict(torch.load("best_model_v1.pth", map_location=DEVICE))
    v1_cam = GradCAM(v1)
    def trained(path, c):
        img = tf(Image.open(path).convert("RGB")).to(DEVICE)
        cam, _ = v1_cam(img, class_idx=c)
        return upsample(cam).cpu()

    print("===== 베이스라인 배터리 (깨끗한 test 270장) =====")
    battery = {}
    for name, fn in [("uniform", uniform), ("rand-init", randinit), ("center", center),
                     ("edge", edge), ("imagenet", imagenet), ("trained-v1", trained)]:
        battery[name] = battery_scores(fn, name)

    # 배터리 그림 — 무작위선을 1.0이 아니라 center 수준에 긋는 게 정직한 표시임
    names = list(battery)
    adrs = [battery[n]["adr"] for n in names]
    plt.figure(figsize=(7.5, 4.5))
    colors = ["gray"] * (len(names) - 1) + ["seagreen"]
    bars = plt.bar(names, adrs, color=colors)
    plt.axhline(battery["center"]["adr"], linestyle="--", color="crimson", linewidth=1,
                label=f"center prior ({battery['center']['adr']:.2f}) — the bar to beat")
    plt.axhline(1.0, linestyle=":", color="black", linewidth=1, label="uniform (1.0)")
    for b, v in zip(bars, adrs):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
    plt.ylabel("ADR (density ratio, live CAMs)")
    plt.title("Is the trained model's saliency earning its score?")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("figures/06_cam_baselines.png", dpi=150)
    print("그림 저장 -> figures/06_cam_baselines.png")

    # ---------------- deletion 검사 (신실성) ----------------------------
    # 히트맵 7x7 칸을 값이 큰 순서로 지워가며 v1 정확도를 측정.
    # 비교군: 같은 개수의 칸을 무작위로 지움 (고정 시드).
    paths = read_manifest("test")
    labels = torch.tensor([CLASSES.index(stem_of(p).rsplit("_", 1)[0]) for p in paths])
    imgs = torch.stack([tf(Image.open(p).convert("RGB")) for p in paths])   # (270,3,224,224)

    cams = []
    for i, p in enumerate(paths):
        cam, _ = v1_cam(imgs[i].to(DEVICE), class_idx=int(labels[i]))
        cams.append(cam.cpu())
    cams = torch.stack(cams)                             # (270,7,7)

    g = torch.Generator().manual_seed(0)
    rand_order = torch.stack([torch.randperm(49, generator=g) for _ in paths])

    fractions = [0, 5, 10, 20, 30, 40]                   # 지울 칸 수 (49칸 중)
    curves = {"cam": [], "random": []}
    with torch.no_grad():
        for k in fractions:
            for mode in ("cam", "random"):
                x = imgs.clone()
                for i in range(len(paths)):
                    if k:
                        if mode == "cam":
                            order = cams[i].flatten().argsort(descending=True)[:k]
                        else:
                            order = rand_order[i][:k]
                        for cell in order:
                            r, c = divmod(int(cell), 7)
                            x[i, :, r * 32:(r + 1) * 32, c * 32:(c + 1) * 32] = 0
                correct = 0
                for b in range(0, len(paths), 64):
                    out = v1(x[b:b + 64].to(DEVICE))
                    correct += (out.argmax(1).cpu() == labels[b:b + 64]).sum().item()
                curves[mode].append(correct / len(paths))
            print(f"칸 {k:2d}개 삭제: CAM순 {curves['cam'][-1]:.3f} vs 무작위 {curves['random'][-1]:.3f}")

    plt.figure(figsize=(6.5, 4.5))
    plt.plot(fractions, curves["cam"], "o-", color="seagreen", label="delete top-CAM cells")
    plt.plot(fractions, curves["random"], "s--", color="gray", label="delete random cells")
    plt.xlabel("deleted cells (of 49)")
    plt.ylabel("accuracy (clean test)")
    plt.title("Deletion check: does the heatmap point at what the model uses?")
    plt.legend()
    plt.tight_layout()
    plt.savefig("figures/07_deletion_curve.png", dpi=150)
    print("그림 저장 -> figures/07_deletion_curve.png")

    save_result("cam_baselines", {"battery": battery,
                                  "deletion": {"cells": fractions, **curves}})


if __name__ == "__main__":
    main()
