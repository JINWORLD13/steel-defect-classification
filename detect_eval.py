"""
탐지 최종 평가 — mAP와 '검사 라인의 언어(미검출률·헛경보)'로 채점

세 부분:
  1) mAP@50 / mAP@50-95 / 클래스별 AP@50 (torchmetrics)
     + AP@50을 손으로도 계산해 라이브러리와 대조 — 틀린 mAP는 신뢰를 통째로 무너뜨림
  2) 부트스트랩 신뢰구간 — test 270장을 복원추출로 다시 뽑아 AP가 얼마나 흔들리는지
  3) 운영점(operating point): "결함을 몇 % 놓칠 것인가"의 문턱값을
     ★val에서 골라★ test에 한 번만 적용 (test로 문턱을 고르면 그 순간 자랑이 오염됨)

실행:
  ./venv/bin/python detect_eval.py --split val    # 문턱 고르기 (몇 번을 봐도 됨)
  ./venv/bin/python detect_eval.py --split test --threshold 0.XX   # 최종 1회
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from common import CLASSES, save_result
from detect_dataset import DetectionSet, collate_fn
from detect_train import DEVICE, build_detector


def iou_matrix(a, b):
    """박스집합 a(n,4) x b(m,4) -> IoU (n,m). 좌표는 [x1,y1,x2,y2]."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


@torch.no_grad()
def collect(model, split):
    """예측과 정답을 전부 모아둠 — 이후 채점은 numpy로만 (모델 재실행 없이)."""
    model.eval()
    loader = DataLoader(DetectionSet(split), batch_size=4, shuffle=False,
                        collate_fn=collate_fn)
    records = []                                    # 이미지 하나당 dict 하나
    for images, targets in loader:
        preds = model([img.to(DEVICE) for img in images])
        for p, t in zip(preds, targets):
            records.append({
                "p_boxes": p["boxes"].cpu().numpy(), "p_scores": p["scores"].cpu().numpy(),
                "p_labels": p["labels"].cpu().numpy(),
                "t_boxes": t["boxes"].numpy(), "t_labels": t["labels"].numpy(),
            })
    return records


def ap50_per_class(records, cls):
    """AP@50 직접 구현 — 101점 보간 없이 정밀 곡선의 면적(모든 점 방식).

    절차(탐욕 매칭): 점수 높은 예측부터, 아직 안 짝지어진 정답과 IoU>=0.5면 TP,
    아니면 FP. 그 순서로 정밀도-재현율 곡선을 만들고 면적을 적분함.
    """
    entries, n_gt = [], 0
    for i, r in enumerate(records):
        gt = r["t_boxes"][r["t_labels"] == cls]
        n_gt += len(gt)
        sel = r["p_labels"] == cls
        for box, score in zip(r["p_boxes"][sel], r["p_scores"][sel]):
            entries.append((float(score), i, box))
    if n_gt == 0:
        return None
    entries.sort(key=lambda e: -e[0])                       # 점수 내림차순

    matched = {}                                            # (이미지, 정답번호) 사용 여부
    tp, fp = [], []
    for score, i, box in entries:
        gt = records[i]["t_boxes"][records[i]["t_labels"] == cls]
        ious = iou_matrix([box], gt)[0] if len(gt) else np.array([])
        j = int(ious.argmax()) if len(ious) else -1
        if j >= 0 and ious[j] >= 0.5 and (i, j) not in matched:
            matched[(i, j)] = True
            tp.append(1); fp.append(0)
        else:
            tp.append(0); fp.append(1)
    tp, fp = np.cumsum(tp), np.cumsum(fp)
    recall = tp / n_gt
    precision = tp / (tp + fp)
    # 정밀도를 오른쪽에서 왼쪽으로 단조화한 뒤 재현율 축으로 적분 (all-point AP)
    prec = np.concatenate([[0.0], precision, [0.0]])
    rec = np.concatenate([[0.0], recall, [1.0]])
    for k in range(len(prec) - 2, -1, -1):
        prec[k] = max(prec[k], prec[k + 1])
    idx = np.where(rec[1:] != rec[:-1])[0]
    return float(np.sum((rec[idx + 1] - rec[idx]) * prec[idx + 1]))


def line_metrics(records, threshold):
    """검사 라인의 언어: 미검출률(놓친 정답 박스 비율)과 이미지당 헛경보 수."""
    miss, n_gt, fa = 0, 0, 0
    for r in records:
        keep = r["p_scores"] >= threshold
        pb, pl = r["p_boxes"][keep], r["p_labels"][keep]
        used = set()
        for gi, (gb, gl) in enumerate(zip(r["t_boxes"], r["t_labels"])):
            n_gt += 1
            ious = iou_matrix([gb], pb)[0] if len(pb) else np.array([])
            ok = [(j, ious[j]) for j in range(len(pb))
                  if pl[j] == gl and ious[j] >= 0.5 and j not in used]
            if ok:
                used.add(max(ok, key=lambda x: x[1])[0])
            else:
                miss += 1
        fa += len(pb) - len(used)                            # 정답과 못 짝지은 예측
    return miss / n_gt, fa / len(records)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--threshold", type=float, default=None,
                    help="test 때 val에서 고른 문턱을 넘겨줌")
    ap.add_argument("--model", default="best_det.pth")
    args = ap.parse_args()

    model = build_detector().to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    records = collect(model, args.split)
    print(f"{args.split} {len(records)}장 예측 수집 완료")

    # ---- mAP: 라이브러리 + 손 구현 대조 --------------------------------
    from torchmetrics.detection import MeanAveragePrecision
    metric = MeanAveragePrecision(class_metrics=True)
    metric.update(
        [{"boxes": torch.tensor(r["p_boxes"]), "scores": torch.tensor(r["p_scores"]),
          "labels": torch.tensor(r["p_labels"])} for r in records],
        [{"boxes": torch.tensor(r["t_boxes"]), "labels": torch.tensor(r["t_labels"])}
         for r in records])
    m = metric.compute()
    print(f"\nmAP@50 {float(m['map_50']):.4f} | mAP@50-95 {float(m['map']):.4f}")

    manual, lib = {}, {}
    classes_idx = m["classes"].tolist() if m["classes"].ndim else [int(m["classes"])]
    for k, cls in enumerate(classes_idx):
        lib[CLASSES[cls - 1]] = float(m["map_50_per_class"][k]) if m["map_50_per_class"].ndim else float(m["map_50_per_class"])
    print(f"{'class':18s} {'AP@50(lib)':>10s} {'AP@50(손)':>10s}")
    for cls in range(1, 7):
        manual[CLASSES[cls - 1]] = ap50_per_class(records, cls)
        print(f"{CLASSES[cls - 1]:18s} {lib.get(CLASSES[cls - 1], float('nan')):10.4f} "
              f"{manual[CLASSES[cls - 1]]:10.4f}")

    # ---- 부트스트랩 CI (이미지 단위 복원추출 500회) --------------------
    rng = np.random.default_rng(0)
    boot = {c: [] for c in range(1, 7)}
    for _ in range(500):
        sample = [records[i] for i in rng.integers(0, len(records), len(records))]
        for c in range(1, 7):
            a = ap50_per_class(sample, c)
            if a is not None:
                boot[c].append(a)
    ci = {CLASSES[c - 1]: [round(float(np.percentile(boot[c], 2.5)), 4),
                           round(float(np.percentile(boot[c], 97.5)), 4)]
          for c in range(1, 7) if boot[c]}

    # ---- 운영점 ---------------------------------------------------------
    if args.split == "val":
        print("\n문턱값 훑기 (val — 여기서 고르고 test엔 한 번만 적용):")
        sweep = []
        for t in np.arange(0.05, 0.96, 0.05):
            miss, fa = line_metrics(records, t)
            sweep.append({"threshold": round(float(t), 2), "miss": round(miss, 4),
                          "false_alarms_per_image": round(fa, 3)})
            print(f"  t={t:.2f}: 미검출 {miss:5.1%} | 헛경보 {fa:4.2f}건/장")
        pick = max((s for s in sweep if s["miss"] <= 0.05),
                   key=lambda s: s["threshold"], default=None)
        print("\n미검출 5% 이하를 만족하는 가장 높은 문턱:",
              pick if pick else "없음 — 문턱을 낮춰도 5%를 못 맞춤")
        save_result("detection_val", {"sweep": sweep, "picked": pick,
                                      "map50": round(float(m["map_50"]), 4)})
    else:
        assert args.threshold is not None, "test에는 val에서 고른 --threshold가 필수임"
        miss, fa = line_metrics(records, args.threshold)
        print(f"\n운영점 t={args.threshold} (val에서 결정): "
              f"미검출 {miss:.1%} | 헛경보 {fa:.2f}건/장")
        save_result("detection", {
            "map50": round(float(m["map_50"]), 4), "map5095": round(float(m["map"]), 4),
            "ap50_lib": {k: round(v, 4) for k, v in lib.items()},
            "ap50_manual": {k: round(v, 4) for k, v in manual.items()},
            "bootstrap_ci95": ci,
            "operating_point": {"threshold": args.threshold,
                                "miss_rate": round(miss, 4),
                                "false_alarms_per_image": round(fa, 3)},
        })


if __name__ == "__main__":
    main()
