"""
탐지 결과 시각화 — 성공만 골라 보여주는 갤러리는 광고지, 실패를 같이 보여야 보고서

그림 두 장:
  figures/08_det_metrics.png    클래스별 AP@50 + 부트스트랩 CI 오차막대
  figures/10_det_failures.png   운영점 기준 성공 3장 + 실패 3장 (놓침/헛경보)
선정 기준을 코드로 남김: 실패는 '놓친 정답 박스가 가장 많은 순', 성공은
'정답과 예측이 전부 짝지어진 이미지 중 박스가 많은 순' — 사람이 고르지 않음.

실행: ./venv/bin/python detect_visualize.py --threshold 0.XX  (val에서 고른 문턱)
"""
import argparse

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from common import CLASSES, load_result
from detect_dataset import DetectionSet
from detect_eval import collect, iou_matrix
from detect_train import DEVICE, build_detector

PALETTE = ["red", "deepskyblue", "lime", "orange", "magenta", "yellow"]


def match_stats(r, threshold):
    """이미지 한 장의 (놓친 정답 수, 헛경보 수, 짝 목록)을 계산."""
    keep = r["p_scores"] >= threshold
    pb, pl = r["p_boxes"][keep], r["p_labels"][keep]
    used, missed = set(), 0
    for gb, gl in zip(r["t_boxes"], r["t_labels"]):
        ious = iou_matrix([gb], pb)[0] if len(pb) else np.array([])
        ok = [(j, ious[j]) for j in range(len(pb))
              if pl[j] == gl and ious[j] >= 0.5 and j not in used]
        if ok:
            used.add(max(ok, key=lambda x: x[1])[0])
        else:
            missed += 1
    return missed, len(pb) - len(used), keep


def draw(ax, ds, i, r, threshold):
    img, _ = ds[i]
    ax.imshow(img.permute(1, 2, 0))
    for gb, gl in zip(r["t_boxes"], r["t_labels"]):          # 정답 = 실선
        ax.add_patch(plt.Rectangle((gb[0], gb[1]), gb[2] - gb[0], gb[3] - gb[1],
                                   fill=False, edgecolor=PALETTE[gl - 1], linewidth=2))
    keep = r["p_scores"] >= threshold
    for pb, pl, s in zip(r["p_boxes"][keep], r["p_labels"][keep], r["p_scores"][keep]):
        ax.add_patch(plt.Rectangle((pb[0], pb[1]), pb[2] - pb[0], pb[3] - pb[1],
                                   fill=False, edgecolor=PALETTE[pl - 1],
                                   linewidth=1.5, linestyle="--"))
        ax.text(pb[0], pb[1] - 2, f"{s:.2f}", fontsize=7, color=PALETTE[pl - 1])
    ax.axis("off")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, required=True)
    args = ap.parse_args()

    model = build_detector().to(DEVICE)
    model.load_state_dict(torch.load("best_det.pth", map_location=DEVICE))
    ds = DetectionSet("test")
    records = collect(model, "test")

    # ---- 그림 08: 클래스별 AP@50 + CI ----------------------------------
    det = load_result("detection")
    names = list(det["ap50_manual"])
    vals = [det["ap50_manual"][n] for n in names]
    ci = det["bootstrap_ci95"]
    yerr = [[vals[i] - ci[n][0] for i, n in enumerate(names)],
            [ci[n][1] - vals[i] for i, n in enumerate(names)]]
    plt.figure(figsize=(8, 4.5))
    bars = plt.bar(names, vals, color="steelblue", yerr=yerr, capsize=4)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.2f}", ha="center", fontsize=9)
    plt.ylabel("AP@50 (test, bootstrap 95% CI)")
    plt.title(f"Per-class detection quality — mAP@50 {det['map50']:.3f}")
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1.15)
    plt.tight_layout()
    plt.savefig("figures/08_det_metrics.png", dpi=150)
    print("그림 저장 -> figures/08_det_metrics.png")

    # ---- 그림 09: val 문턱 훑기 곡선 + test 최종점 ----------------------
    # "문턱은 val에서 골랐고 test는 그 값 하나로 한 번만 봤다"를 그림 한 장으로
    val = load_result("detection_val")
    det = load_result("detection")
    ts = [s["threshold"] for s in val["sweep"]]
    plt.figure(figsize=(7, 4.5))
    plt.plot(ts, [s["miss"] for s in val["sweep"]], "o-", color="crimson",
             label="miss rate (val sweep)")
    plt.plot(ts, [s["false_alarms_per_image"] for s in val["sweep"]], "s--",
             color="steelblue", label="false alarms / image (val sweep)")
    op = det["operating_point"]
    plt.axvline(op["threshold"], color="gray", linestyle=":")
    plt.scatter([op["threshold"]], [op["miss_rate"]], marker="*", s=180,
                color="crimson", zorder=5,
                label=f"TEST (once): miss {op['miss_rate']:.1%}")
    plt.scatter([op["threshold"]], [op["false_alarms_per_image"]], marker="*",
                s=180, color="steelblue",
                label=f"TEST (once): {op['false_alarms_per_image']:.2f} FA/img")
    plt.axhline(0.05, color="black", linewidth=0.8, linestyle="--")
    plt.text(ts[0], 0.055, "miss 5% target", fontsize=8)
    plt.xlabel("score threshold")
    plt.title("Operating point chosen on val, spent once on test")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("figures/09_det_threshold_val_test.png", dpi=150)
    print("그림 저장 -> figures/09_det_threshold_val_test.png")

    # ---- 그림 10: 성공 3 + 실패 3 --------------------------------------
    stats = [match_stats(r, args.threshold) for r in records]
    fail_order = sorted(range(len(records)),
                        key=lambda i: -(stats[i][0] * 10 + stats[i][1]))[:3]
    succ = [i for i in range(len(records))
            if stats[i][0] == 0 and stats[i][1] == 0]
    succ_order = sorted(succ, key=lambda i: -len(records[i]["t_boxes"]))[:3]

    fig, axes = plt.subplots(2, 3, figsize=(12, 8.5))
    for ax, i in zip(axes[0], succ_order):
        draw(ax, ds, i, records[i], args.threshold)
        ax.set_title("success (all matched)", fontsize=9)
    for ax, i in zip(axes[1], fail_order):
        m, f, _ = stats[i]
        draw(ax, ds, i, records[i], args.threshold)
        ax.set_title(f"failure — missed {m}, false alarm {f}", fontsize=9)
    handles = [mpatches.Patch(color=PALETTE[k], label=CLASSES[k]) for k in range(6)]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9)
    fig.suptitle(f"Detections at operating point t={args.threshold} "
                 "(solid = ground truth, dashed = prediction)")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig("figures/10_det_failures.png", dpi=150)
    print("그림 저장 -> figures/10_det_failures.png")


if __name__ == "__main__":
    main()
