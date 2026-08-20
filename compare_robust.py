"""
v1 vs v2 최종 비교 — 개선이 '본 계열'을 넘어 전이됐는지 판정

보고 규율:
  - 대표 v2는 val 선택점수가 가장 높은 시드임 (test를 보고 고르면 그 순간 오염됨)
  - 헤드라인은 seen(오르는 게 당연한 구간)이 아니라 held-out, 특히 far(jpeg·픽셀레이트)
  - 깨끗한 test의 변화는 짝지은 비교이므로 맥니마 정확검정으로 판정
  - 시드 5개의 범위를 오차막대로 같이 보고 (평균 하나로 뭉개지 않음)

실행: ./venv/bin/python compare_robust.py
"""
import glob

import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt

from bench_robust import accuracy_under
from common import CLASSES, DEVICE, build_model, load_result, mcnemar_exact, save_result
from corruptions import (ALL_KINDS, BENCH_HELDOUT, BENCH_SEEN, HELDOUT_DISTANCE,
                         clean_transform, make_corrupt_transform)
from labels import read_manifest, stem_of


@torch.no_grad()
def per_image_correct(model, transform):
    """test 각 장의 맞음/틀림 — 맥니마의 재료 (경로 순서 = 매니페스트 순서)."""
    model.eval()
    out = []
    for path in read_manifest("test"):
        img = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(DEVICE)
        pred = int(model(img).argmax(1))
        out.append(pred == CLASSES.index(stem_of(path).rsplit("_", 1)[0]))
    return out


def bench(model):
    """모델 하나를 (clean + 11종x5강도) 전부 채점해 표로."""
    table = {"clean": accuracy_under(model, "test", clean_transform())}
    for kind in ALL_KINDS:
        table[kind] = [accuracy_under(model, "test", make_corrupt_transform(kind, s))
                       for s in range(1, 6)]
    return table


def main():
    # 1) 대표 시드 선정 — val 선택점수 기준 (train_robust가 저장해 둔 값)
    seeds = {}
    for f in sorted(glob.glob("results/train_robust_s*.json")):
        r = load_result(f.split("/")[-1][:-5])
        seeds[r["seed"]] = r["val_selection"]
    primary = max(seeds, key=seeds.get)
    print(f"시드별 val 선택점수: {seeds} → 대표 시드 {primary}")

    v1 = build_model()
    v1.load_state_dict(torch.load("best_model_v1.pth", map_location=DEVICE))

    # 2) 시드 전부 test 벤치 (범위 보고용) — 선택은 이미 val에서 끝났음
    per_seed = {}
    for s in seeds:
        m = build_model()
        m.load_state_dict(torch.load(f"best_model_robust_s{s}.pth", map_location=DEVICE))
        per_seed[s] = bench(m)
        print(f"seed {s}: clean {per_seed[s]['clean']:.3f}")

    v1_table = load_result("robust_v1")["per_kind"]
    v1_clean = load_result("robust_v1")["summary"]["clean"]

    # 3) 구간 요약 + 헤드라인
    def zone(table, kinds):
        return float(np.mean([a for k in kinds for a in table[k]]))
    zones = {}
    for name, kinds in [("seen", BENCH_SEEN),
                        ("near", [k for k in BENCH_HELDOUT if HELDOUT_DISTANCE[k] == "near"]),
                        ("mid", [k for k in BENCH_HELDOUT if HELDOUT_DISTANCE[k] == "mid"]),
                        ("far", [k for k in BENCH_HELDOUT if HELDOUT_DISTANCE[k] == "far"])]:
        v1z = zone(v1_table, kinds)
        v2z = [zone(per_seed[s], kinds) for s in seeds]
        zones[name] = {"v1": round(v1z, 4), "v2_primary": round(zone(per_seed[primary], kinds), 4),
                       "v2_min": round(min(v2z), 4), "v2_max": round(max(v2z), 4)}
        print(f"{name:5s}: v1 {v1z:.3f} → v2 {zone(per_seed[primary], kinds):.3f} "
              f"(시드 범위 {min(v2z):.3f}~{max(v2z):.3f})")

    # held-out 종별로 개선 여부를 개수로 셈 (평균 하나로 뭉개지 않기)
    improved = []
    for k in BENCH_HELDOUT:
        d = float(np.mean(per_seed[primary][k]) - np.mean(v1_table[k]))
        improved.append((k, round(d, 4)))
        print(f"  held-out {k:12s}: {'+' if d >= 0 else ''}{d:.3f}")

    # 4) 깨끗한 test — 맥니마
    c1 = per_image_correct(v1, clean_transform())
    m2 = build_model()
    m2.load_state_dict(torch.load(f"best_model_robust_s{primary}.pth", map_location=DEVICE))
    c2 = per_image_correct(m2, clean_transform())
    b = sum(1 for a, bb in zip(c1, c2) if a and not bb)   # v1만 맞힌 문제
    c = sum(1 for a, bb in zip(c1, c2) if bb and not a)   # v2만 맞힌 문제
    p = mcnemar_exact(b, c)
    print(f"\nclean test: v1 {sum(c1)}/270 vs v2 {sum(c2)}/270 "
          f"(v1만 맞힘 {b}, v2만 맞힘 {c}, 맥니마 p={p:.3f})")

    # 5) 그림 — 종별 강도평균 막대, seen/held-out 구역을 배경색으로 분리
    kinds = ALL_KINDS
    v1_means = [float(np.mean(v1_table[k])) for k in kinds]
    v2_means = [float(np.mean(per_seed[primary][k])) for k in kinds]
    v2_lo = [min(float(np.mean(per_seed[s][k])) for s in seeds) for k in kinds]
    v2_hi = [max(float(np.mean(per_seed[s][k])) for s in seeds) for k in kinds]
    x = np.arange(len(kinds))
    plt.figure(figsize=(12, 5))
    plt.axvspan(-0.5, len(BENCH_SEEN) - 0.5, color="lightyellow", alpha=0.6)
    plt.axvspan(len(BENCH_SEEN) - 0.5, len(kinds) - 0.5, color="lavender", alpha=0.5)
    plt.bar(x - 0.2, v1_means, width=0.4, label="v1 (clean-trained)", color="lightgray",
            edgecolor="gray")
    yerr = [np.array(v2_means) - np.array(v2_lo), np.array(v2_hi) - np.array(v2_means)]
    plt.bar(x + 0.2, v2_means, width=0.4, label=f"v2 (aug-trained, seed {primary})",
            color="seagreen", yerr=yerr, capsize=3)
    plt.axhline(1 / 6, linestyle=":", color="black", linewidth=1, label="chance (0.167)")
    plt.xticks(x, kinds, rotation=30, ha="right")
    plt.text(len(BENCH_SEEN) / 2 - 0.5, 1.06, "SEEN (trained mechanisms)", ha="center", fontsize=9)
    plt.text(len(BENCH_SEEN) + len(BENCH_HELDOUT) / 2 - 0.5, 1.06, "HELD-OUT (never trained)",
             ha="center", fontsize=9)
    plt.ylim(0, 1.12)
    plt.ylabel("accuracy (mean over severities 1-5, test)")
    plt.legend(loc="lower left", fontsize=9)
    plt.tight_layout()
    plt.savefig("figures/05_robustness_before_after.png", dpi=150)
    print("그림 저장 -> figures/05_robustness_before_after.png")

    save_result("robust_v2", {
        "primary_seed": primary, "val_selection": seeds,
        "clean": {"v1": v1_clean, "v2": round(sum(c2) / 270, 4),
                  "mcnemar": {"v1_only": b, "v2_only": c, "p": round(p, 4)}},
        "zones": zones,
        "heldout_delta": dict(improved),
        "per_seed_zone_clean": {s: round(per_seed[s]["clean"], 4) for s in seeds},
        "per_kind_v2_primary": {k: [round(a, 4) for a in per_seed[primary][k]] for k in kinds},
    })


if __name__ == "__main__":
    main()
