"""
멀티라벨 최종 평가 — 단일 라벨(v1) vs 멀티라벨을 같은 test 270장에서 비교

채점 기준이 서로 다르므로 각자의 기준과 '교차 기준'을 모두 잰다:
  - subset accuracy : 6칸 전부 일치해야 정답 (멀티라벨의 엄격한 채점)
  - Jaccard         : (맞힌 결함 수) / (정답∪예측 결함 수) — 부분 점수 인정
  - v1을 멀티라벨 기준으로 채점 : argmax 하나만 답하는 모델이 박스 전체를 몇 %나 커버하나
    -> 단일 라벨의 '구조적 상한'이 숫자로 드러남 (다중 결함 이미지는 원리상 만점 불가)

실행: ./venv/bin/python eval_multilabel.py
"""
import torch
from torch.utils.data import DataLoader

from common import CLASSES, DEVICE, build_model, save_result, wilson
from labels import multi_hot, read_manifest, stem_of
from train_multilabel import MultiLabelSet, eval_tf


@torch.no_grad()
def collect(model, loader, mode):
    """mode='multi' : 시그모이드>0.5 벡터 / mode='single' : argmax 원핫 벡터"""
    model.eval()
    preds = []
    for images, _ in loader:
        logits = model(images.to(DEVICE))
        if mode == "multi":
            p = (torch.sigmoid(logits) > 0.5).float()
        else:                                      # 단일 라벨 모델은 1등 하나만 켬
            p = torch.zeros_like(logits)
            p[torch.arange(len(logits)), logits.argmax(1)] = 1.0
        preds.append(p.cpu())
    return torch.cat(preds)                        # (270, 6)


def score(preds, targets, name):
    exact = (preds == targets).all(dim=1).float()  # 행 단위 완전 일치 여부
    inter = ((preds == 1) & (targets == 1)).sum(1).float()   # 교집합 크기
    union = ((preds == 1) | (targets == 1)).sum(1).float()   # 합집합 크기
    jacc = (inter / union.clamp(min=1)).mean()
    missed = int(((targets == 1) & (preds == 0)).sum())      # 놓친 (이미지,결함) 쌍
    k = int(exact.sum())
    lo, hi = wilson(k, len(exact))
    print(f"{name:24s}: subset {exact.mean():.3f} [{lo:.3f},{hi:.3f}] | "
          f"Jaccard {jacc:.3f} | 놓친 결함 {missed}쌍")
    return {"subset": round(float(exact.mean()), 4), "wilson": [round(lo, 4), round(hi, 4)],
            "jaccard": round(float(jacc), 4), "missed_pairs": missed}


def main():
    loader = DataLoader(MultiLabelSet("test", eval_tf), batch_size=32, shuffle=False)
    targets = torch.stack([torch.tensor(multi_hot(stem_of(p)))
                           for p in read_manifest("test")])

    ml = build_model()
    ml.load_state_dict(torch.load("best_model_multilabel.pth", map_location=DEVICE))
    v1 = build_model()
    v1.load_state_dict(torch.load("best_model_v1.pth", map_location=DEVICE))

    print("===== 같은 test 270장, 박스 기준(멀티핫) 채점 =====")
    r_ml = score(collect(ml, loader, "multi"), targets, "멀티라벨 (BCE)")
    r_v1 = score(collect(v1, loader, "single"), targets, "단일라벨 v1 (argmax)")

    # 단일 라벨의 구조적 상한: 결함 2종 이상 이미지는 원리상 subset 정답이 불가능
    multi_imgs = int((targets.sum(1) > 1).sum())
    ceiling = (len(targets) - multi_imgs) / len(targets)
    print(f"\n단일 라벨의 구조적 상한: {ceiling:.3f} "
          f"(다중 결함 {multi_imgs}장은 원리상 만점 불가)")
    print("멀티라벨이 상한을", "넘음" if r_ml["subset"] > ceiling else "못 넘음",
          f"({r_ml['subset']:.3f} vs {ceiling:.3f})")

    save_result("multilabel", {
        "multilabel": r_ml, "v1_as_multilabel": r_v1,
        "single_label_ceiling": round(ceiling, 4),
        "multi_defect_test_images": multi_imgs,
    })


if __name__ == "__main__":
    main()
