"""
탐지 학습 — Faster R-CNN + ResNet18 백본

분류 학습과 구조가 다른 점 두 가지:
  1) criterion이 없음. torchvision 탐지 모델은 학습 모드에서 (이미지, 정답)을 주면
     손실 딕셔너리를 스스로 계산해 돌려줌. 우리는 그 합을 backward하면 됨.
  2) 장치가 CPU 고정임. MPS에서 탐지 추론을 실측하니 CPU보다 수백 배 느렸음
     (동기화 없이 재면 빨라 보이는 함정까지 확인함) — 그래서 CPU가 정답임.

백본을 v1과 같은 ResNet18로 두는 이유: 분류·멀티라벨·탐지 세 문제를
같은 출발점(같은 사전학습 가중치)에서 비교하기 위함임.

실행:
  ./venv/bin/python detect_train.py --sanity     # 50장으로 파이프라인 검증 (게이트)
  ./venv/bin/python detect_train.py              # 본 학습 (CPU 수 시간)
"""
import argparse
import csv
import os
import time

import torch
from torch.utils.data import DataLoader
from torchvision.models import ResNet18_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from detect_dataset import DetectionSet, collate_fn, sanity_check

DEVICE = torch.device("cpu")               # 탐지는 CPU 고정 (위 주석 참고)
NUM_CLASSES = 7                            # 결함 6 + 배경 1


def build_detector():
    """ResNet18-FPN 백본의 Faster R-CNN.

    앵커 비율에 0.1과 10.0을 넣은 이유: scratches는 가늘고 긴 세로줄,
    inclusion은 가로로 긴 띠라서 기본 비율(0.5~2.0)로는 못 감쌈.
    크기(sizes)는 기본값 유지 — 512를 빼면 pitted_surface(면적 중앙값 0.55)의
    리콜이 실측으로 떨어짐. anchor_recall.py가 이 선택의 근거를 학습 없이 보여줌.
    """
    backbone = resnet_fpn_backbone("resnet18", weights=ResNet18_Weights.DEFAULT)
    anchors = AnchorGenerator(
        sizes=((32,), (64,), (128,), (256,), (512,)),
        aspect_ratios=((0.1, 0.5, 1.0, 2.0, 10.0),) * 5,
    )
    return FasterRCNN(backbone, num_classes=NUM_CLASSES,
                      rpn_anchor_generator=anchors,
                      min_size=320, max_size=320)   # 200px 원본을 320으로 키워 처리


@torch.no_grad()
def quick_map50(model, loader):
    """mAP@50 한 숫자만 — 에폭마다 val로 best를 고르기 위한 채점."""
    from torchmetrics.detection import MeanAveragePrecision
    model.eval()
    metric = MeanAveragePrecision(iou_thresholds=[0.5])
    for images, targets in loader:
        preds = model([img.to(DEVICE) for img in images])
        metric.update(preds, list(targets))
    return float(metric.compute()["map_50"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sanity", action="store_true", help="50장 축소 실행 (게이트)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--init", default=None,
                    help="이어달리기: 저장된 가중치(pth)에서 시작 (예: last_det.pth)")
    args = ap.parse_args()

    torch.manual_seed(42)
    sanity_check()

    if args.sanity:
        # 게이트: 50장을 외울 수 있어야 파이프라인이 맞는 것 (mAP@50 >= 0.9)
        # 탐지는 분류보다 훨씬 많은 스텝이 필요함 — 에폭을 넉넉히 잡음
        train_ds = DetectionSet("train", limit=50)
        val_ds = train_ds                          # 같은 50장으로 암기 확인
        epochs = args.epochs or 40
        tag = "sanity"
    else:
        train_ds = DetectionSet("train")
        val_ds = DetectionSet("val")
        epochs = args.epochs or 8
        tag = "full"

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True,
                              collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False,
                            collate_fn=collate_fn)

    model = build_detector().to(DEVICE)
    if args.init:
        model.load_state_dict(torch.load(args.init, map_location=DEVICE))
        print(f"{args.init} 에서 이어서 학습")
    params = [p for p in model.parameters() if p.requires_grad]
    # AdamW를 쓰는 이유: 8장 암기 실험에서 같은 150스텝에 SGD 0.465 vs AdamW 0.833 —
    # 데이터가 작고 CPU 예산이 빠듯할 때는 빨리 수렴하는 쪽이 정답임 (실측 근거)
    optimizer = torch.optim.AdamW(params, lr=1e-4)

    os.makedirs("runs", exist_ok=True)
    log_path = f"runs/detect_{tag}.csv"
    best = 0.0
    with open(log_path, "w", newline="") as logf:
        writer = csv.writer(logf)
        writer.writerow(["epoch", "train_loss", "val_map50", "seconds"])
        for epoch in range(1, epochs + 1):
            model.train()
            t0 = time.time()
            running, n = 0.0, 0
            for images, targets in train_loader:
                images = [img.to(DEVICE) for img in images]
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)     # 모델이 손실을 스스로 계산
                loss = sum(loss_dict.values())         # RPN 2종 + head 2종의 합
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                running += loss.item() * len(images)
                n += len(images)

            m50 = quick_map50(model, val_loader)
            sec = time.time() - t0
            writer.writerow([epoch, round(running / n, 4), round(m50, 4), round(sec)])
            logf.flush()
            print(f"[{epoch:2d}/{epochs}] loss {running / n:.4f} | "
                  f"val mAP@50 {m50:.3f} | {sec:.0f}s", flush=True)

            torch.save(model.state_dict(), "last_det.pth")
            if m50 > best:
                best = m50
                torch.save(model.state_dict(), "best_det.pth")
                print(f" best 갱신 -> best_det.pth (mAP@50 {m50:.3f})", flush=True)

    print(f"\n끝. 로그: {log_path}, best mAP@50 {best:.3f}")
    if args.sanity:
        print("게이트:", "통과 (>=0.9)" if best >= 0.9 else "실패 — 파이프라인 점검 필요")


if __name__ == "__main__":
    main()
