"""
손상 벤치마크 — 모델 하나를 (11종 × 5강도 + 깨끗한) 56개 조건에서 채점

규율 두 가지:
  1) --split val  : 사다리 보정·구성 판단용. 몇 번을 봐도 됨.
     --split test : 최종 보고용. 모델당 한 번만 봄.
     (test를 보며 사다리를 고치면 이후 모든 개선폭이 그 test에 맞춰 부풀게 됨)
  2) 같은 명령을 두 번 돌리면 완전히 같은 숫자 — 손상의 난수가 고정돼 있어서임.

실행:
  ./venv/bin/python bench_robust.py --model best_model_v1.pth --split val --name v1_val
  ./venv/bin/python bench_robust.py --model best_model_v1.pth --split test --name v1
"""
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from common import DEVICE, build_model, save_result
from corruptions import (ALL_KINDS, BENCH_HELDOUT, BENCH_SEEN, HELDOUT_DISTANCE,
                         clean_transform, make_corrupt_transform)


@torch.no_grad()
def accuracy_under(model, split, transform):
    ds = datasets.ImageFolder(f"dataset/{split}", transform=transform)
    loader = DataLoader(ds, batch_size=64, shuffle=False)
    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        outputs = model(images.to(DEVICE))
        correct += (outputs.argmax(1).cpu() == labels).sum().item()
        total += labels.size(0)
    return correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--name", required=True, help="results/robust_<name>.json 으로 저장")
    args = ap.parse_args()

    model = build_model()
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))

    clean = accuracy_under(model, args.split, clean_transform())
    print(f"clean ({args.split}) : {clean:.3f}")

    table = {}                                   # kind -> [s1..s5 정확도]
    for kind in ALL_KINDS:
        accs = []
        for s in range(1, 6):
            acc = accuracy_under(model, args.split, make_corrupt_transform(kind, s))
            accs.append(round(acc, 4))
        tag = "seen" if kind in BENCH_SEEN else f"heldout-{HELDOUT_DISTANCE[kind]}"
        print(f"{kind:14s} [{tag:12s}]: " + " ".join(f"{a:.3f}" for a in accs))
        table[kind] = accs

    # 구간 요약 — 강도 전체 평균을 seen / near / mid / far로 묶음
    def zone_mean(kinds):
        vals = [a for k in kinds for a in table[k]]
        return round(sum(vals) / len(vals), 4)
    summary = {
        "clean": round(clean, 4),
        "seen": zone_mean(BENCH_SEEN),
        "near": zone_mean([k for k in BENCH_HELDOUT if HELDOUT_DISTANCE[k] == "near"]),
        "mid": zone_mean([k for k in BENCH_HELDOUT if HELDOUT_DISTANCE[k] == "mid"]),
        "far": zone_mean([k for k in BENCH_HELDOUT if HELDOUT_DISTANCE[k] == "far"]),
    }
    print("\n구간 평균:", summary)

    save_result(f"robust_{args.name}", {
        "model": args.model, "split": args.split,
        "summary": summary, "per_kind": table,
    })


if __name__ == "__main__":
    main()
