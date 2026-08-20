"""
앵커 리콜 분석 — 앵커 설계의 근거를 '학습 없이' 만든다

앵커(anchor)는 탐지 모델이 후보로 깔아놓는 고정 박스들임. 정답 박스가
어떤 앵커와도 안 겹치면(IoU가 낮으면) 모델은 그 결함을 배울 기회조차 없음.
그래서 학습을 돌리기 전에 "우리 정답 박스들이 앵커에 얼마나 잡히는가"를
직접 재면, 앵커 구성 선택의 근거가 실측 수치로 남음.

비교: 기본 비율 (0.5, 1.0, 2.0)  vs  우리 구성 (0.1, 0.5, 1.0, 2.0, 10.0)
      — scratches(세로로 김)와 inclusion(가로로 김) 때문에 극단 비율을 추가함

실행: ./venv/bin/python anchor_recall.py
"""
import numpy as np
import torch
from torchvision.models import ResNet18_Weights
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone

from common import CLASSES, save_result
from detect_dataset import DetectionSet
from detect_eval import iou_matrix


def anchors_of(ratios):
    """해당 비율 구성의 모델이 실제로 까는 앵커 전부 (320x320 기준 좌표)."""
    backbone = resnet_fpn_backbone("resnet18", weights=ResNet18_Weights.DEFAULT)
    gen = AnchorGenerator(sizes=((32,), (64,), (128,), (256,), (512,)),
                          aspect_ratios=(tuple(ratios),) * 5)
    model = FasterRCNN(backbone, num_classes=7, rpn_anchor_generator=gen,
                       min_size=320, max_size=320).eval()
    with torch.no_grad():
        images, _ = model.transform([torch.zeros(3, 200, 200)], None)  # 200 -> 320 확대 포함
        feats = model.backbone(images.tensors)
        anchors = model.rpn.anchor_generator(images, list(feats.values()))[0]
    return anchors.numpy(), 320.0 / 200.0          # (전체 앵커, 좌표 배율)


def main():
    ds = DetectionSet("train")                      # 근거는 train 라벨로만 만듦
    gts = {c: [] for c in range(1, 7)}
    for i in range(len(ds)):
        _, t = ds[i]
        for box, lab in zip(t["boxes"].numpy(), t["labels"].numpy()):
            gts[int(lab)].append(box)

    result = {}
    for name, ratios in [("default(0.5,1,2)", (0.5, 1.0, 2.0)),
                         ("ours(+0.1,+10)", (0.1, 0.5, 1.0, 2.0, 10.0))]:
        anchors, scale = anchors_of(ratios)
        print(f"\n[{name}] 앵커 {len(anchors):,}개")
        print(f"{'class':18s} {'GT수':>5s} {'리콜@0.5':>8s} {'리콜@0.7':>8s}")
        rows = {}
        all50, all70, n_all = 0, 0, 0
        for c in range(1, 7):
            boxes = np.array(gts[c]) * scale        # 정답도 320 좌표계로
            best = np.array([iou_matrix([b], anchors)[0].max() for b in boxes])
            r50, r70 = float((best >= 0.5).mean()), float((best >= 0.7).mean())
            all50 += (best >= 0.5).sum(); all70 += (best >= 0.7).sum(); n_all += len(best)
            print(f"{CLASSES[c - 1]:18s} {len(boxes):5d} {r50:8.1%} {r70:8.1%}")
            rows[CLASSES[c - 1]] = {"n": len(boxes), "r50": round(r50, 4), "r70": round(r70, 4)}
        print(f"{'전체':16s} {n_all:5d} {all50 / n_all:8.1%} {all70 / n_all:8.1%}")
        result[name] = {"per_class": rows,
                        "overall": {"r50": round(all50 / n_all, 4),
                                    "r70": round(all70 / n_all, 4)}}
    save_result("anchor_recall", result)


if __name__ == "__main__":
    main()
