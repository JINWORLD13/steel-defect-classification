"""
근접중복(near-duplicate) 감사 — md5가 못 잡는 '거의 같은' 이미지를 찾음

왜 필요한가:
  md5는 1픽셀만 달라도 다른 지문을 냄. 그래서 md5 전수검사(audit_data.py)로
  "동일 파일 0건"을 확인했어도, 같은 강판을 연속 촬영한 '거의 같은' 사진이
  train과 test에 나뉘어 들어갔을 가능성은 남아 있음. 그게 남아 있으면
  test 점수는 '본 적 없는 데이터' 점수가 아니라 '거의 본 데이터' 점수가 됨.

방법:
  1) 모든 이미지를 32x32 흑백으로 줄이고 밝기·대비를 정규화해 1,024차원 벡터로 만듦
  2) test 각 장에 대해 train 전체와 코사인 유사도를 재서 가장 닮은 train 이미지를 찾음
  3) ★대조군과 같이 봄 — "원래 강판 텍스처는 다 비슷한 것 아닌가?"를 배제하려면
     '진짜 닮음'이 '그 클래스의 평균 닮음'보다 얼마나 튀는지 봐야 함
  4) 문턱(r>0.90)을 넘는 test 이미지를 뺀 부분집합에서 v1 모델을 재평가
     → 점수가 떨어지면 근접중복이 100%를 부풀린 것. 안 떨어지면 그 의혹은 해소.

실행: ./venv/bin/python audit_neardup.py
"""
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

from common import DEVICE, MEAN, STD, build_model, save_result, wilson

THRESHOLD = 0.90     # 이보다 닮으면 '근접중복 의심'으로 표시
GALLERY_TOP = 12     # 갤러리에 그릴 상위 쌍 수


def read_manifest(split):
    """splits/<split>.txt에서 경로 목록을 읽음 — 분할의 단일 출처를 그대로 씀."""
    with open(os.path.join("splits", split + ".txt")) as f:
        return [line.split("\t")[0] for line in f.read().splitlines() if line]


def to_vector(path):
    """이미지 한 장 -> 1,024차원 비교용 벡터.

    32x32 흑백으로 줄이는 이유: 픽셀 단위의 자잘한 차이(노이즈)를 뭉개고
    '전체 생김새'만 남기기 위함. 근접중복은 생김새가 같으니 이걸로 충분히 잡힘.
    """
    img = Image.open(path).convert("L").resize((32, 32))   # L = 흑백
    v = np.asarray(img, dtype=np.float32).ravel()          # 32x32 -> 1,024개 숫자 한 줄
    v = v - v.mean()                                        # 밝기 차이 제거 (평균을 0으로)
    norm = np.linalg.norm(v)                                # 벡터 길이
    return v / norm if norm > 0 else v                      # 길이를 1로 -> 내적이 곧 코사인 유사도


def class_of(path):
    """dataset/test/crazing/xxx.jpg -> 'crazing' (폴더명이 곧 클래스)"""
    return os.path.basename(os.path.dirname(path))


def main():
    # ---------------------------------------------------- 1) 벡터 만들기
    train_paths = read_manifest("train")
    test_paths = read_manifest("test")
    print(f"train {len(train_paths)}장, test {len(test_paths)}장 벡터화 중...")

    train_mat = np.stack([to_vector(p) for p in train_paths])   # (1260, 1024)
    test_mat = np.stack([to_vector(p) for p in test_paths])     # (270, 1024)
    train_cls = np.array([class_of(p) for p in train_paths])

    # ---------------------------------------------------- 2) 유사도 행렬
    # 행렬곱 한 번이면 (test 270) x (train 1260) 모든 쌍의 코사인 유사도가 나옴
    # errstate로 경고를 끄는 이유: macOS Accelerate BLAS가 float32 행렬곱에서
    # 허위 경고(divide by zero 등)를 내는 알려진 현상임. 실제 결과를 검증했음 —
    # NaN 0개, float64 재계산과 최대 차이 2e-6, 값 전부 코사인 범위(-1~1) 안.
    with np.errstate(all="ignore"):
        sims = test_mat @ train_mat.T                            # (270, 1260)
    assert not np.isnan(sims).any(), "유사도 행렬에 NaN — 허위 경고가 아니라 진짜 문제임"

    # ---------------------------------------------------- 3) 최근접 + 대조군
    pairs = []          # (유사도, test경로, train경로)
    ctrl_same_mean = [] # 대조군1: 같은 클래스 train들과의 '평균' 유사도
    ctrl_other_max = [] # 대조군2: 다른 클래스 train들과의 '최대' 유사도
    for i, tp in enumerate(test_paths):
        row = sims[i]
        j = int(row.argmax())                                    # 가장 닮은 train의 위치
        pairs.append((float(row[j]), tp, train_paths[j]))

        same = row[train_cls == class_of(tp)]                    # 같은 클래스만 골라서
        other = row[train_cls != class_of(tp)]
        ctrl_same_mean.append(float(same.mean()))
        ctrl_other_max.append(float(other.max()))

    pairs.sort(reverse=True)                                     # 닮은 순으로 정렬
    flagged = [p for p in pairs if p[0] > THRESHOLD]

    print(f"\n===== 근접중복 감사 (문턱 r>{THRESHOLD}) =====")
    print(f"최근접 유사도 분포: 중앙값 {np.median([p[0] for p in pairs]):.3f}, "
          f"최대 {pairs[0][0]:.3f}")
    print(f"대조군 — 같은 클래스 평균 유사도의 평균: {np.mean(ctrl_same_mean):.3f}")
    print(f"대조군 — 다른 클래스 최대 유사도의 평균: {np.mean(ctrl_other_max):.3f}")
    print(f"문턱을 넘은 test 이미지: {len(flagged)}장 / {len(test_paths)}장")
    for r, tp, trp in flagged[:20]:
        print(f"  r={r:.3f}  {os.path.basename(tp)} ↔ {os.path.basename(trp)}")

    # ---------------------------------------------------- 4) 갤러리 그림
    os.makedirs("figures", exist_ok=True)
    top = pairs[:GALLERY_TOP]
    fig, axes = plt.subplots(4, 6, figsize=(13, 9))              # 4행 6열 = 12쌍 (위 test, 아래 train)
    for k, (r, tp, trp) in enumerate(top):
        row, col = (0 if k < 6 else 2), k % 6                    # 앞 6쌍은 1-2행, 뒤 6쌍은 3-4행
        axes[row][col].imshow(Image.open(tp).convert("L"), cmap="gray")
        axes[row][col].set_title(f"test r={r:.3f}", fontsize=8)
        axes[row + 1][col].imshow(Image.open(trp).convert("L"), cmap="gray")
        axes[row + 1][col].set_title("nearest train", fontsize=8)
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"Top-{GALLERY_TOP} nearest test-train pairs (cosine, 32x32 gray)")
    plt.tight_layout()
    plt.savefig("figures/01_neardup_gallery.png", dpi=150)
    print("\n그림 저장 -> figures/01_neardup_gallery.png")

    # ---------------------------------------------------- 5) 부분집합 재평가
    # 의심 이미지를 뺀 나머지에서 v1 모델의 정확도를 다시 잼 (재학습 없음 — 채점만 다시)
    eval_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    ds = datasets.ImageFolder("dataset/test", transform=eval_tf)
    loader = DataLoader(ds, batch_size=32, shuffle=False)

    model = build_model()
    model.load_state_dict(torch.load("best_model_v1.pth", map_location=DEVICE))
    model.eval()

    correct_by_path = {}                                          # 경로 -> 맞혔나(True/False)
    idx = 0
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images.to(DEVICE))
            preds = outputs.argmax(1).cpu()
            for k in range(len(labels)):
                path = ds.samples[idx][0]                         # 이 배치 k번째의 원본 경로
                correct_by_path[path] = bool(preds[k] == labels[k])
                idx += 1

    flagged_set = {tp for _, tp, _ in flagged}
    all_paths = list(correct_by_path)
    kept = [p for p in all_paths if p not in flagged_set]

    acc_full = sum(correct_by_path[p] for p in all_paths) / len(all_paths)
    acc_kept = sum(correct_by_path[p] for p in kept) / len(kept) if kept else 0.0
    lo_f, hi_f = wilson(sum(correct_by_path[p] for p in all_paths), len(all_paths))
    lo_k, hi_k = wilson(sum(correct_by_path[p] for p in kept), len(kept))

    print(f"\n===== v1 모델 부분집합 재평가 =====")
    print(f"전체 test {len(all_paths)}장     : {acc_full:.3f}  (윌슨 95% [{lo_f:.3f}, {hi_f:.3f}])")
    print(f"의심 제외 {len(kept)}장 : {acc_kept:.3f}  (윌슨 95% [{lo_k:.3f}, {hi_k:.3f}])")
    if acc_kept >= acc_full:
        print("→ 근접중복 의심분을 빼도 점수가 안 떨어짐 — 이 요인으로 부풀지 않았음")
    else:
        print("→ 점수 하락 — 근접중복이 test 점수를 부풀리고 있었음")

    # ---------------------------------------------------- 6) 결과 저장
    save_result("audit_neardup", {
        "threshold": THRESHOLD,
        "nearest_median": round(float(np.median([p[0] for p in pairs])), 4),
        "nearest_max": round(pairs[0][0], 4),
        "control_same_class_mean": round(float(np.mean(ctrl_same_mean)), 4),
        "control_other_class_max": round(float(np.mean(ctrl_other_max)), 4),
        "flagged_count": len(flagged),
        "flagged_pairs": [
            {"r": round(r, 4), "test": tp, "train": trp} for r, tp, trp in flagged
        ],
        "acc_full_test": round(acc_full, 4),
        "acc_without_flagged": round(acc_kept, 4),
        "wilson_full": [round(lo_f, 4), round(hi_f, 4)],
        "wilson_without_flagged": [round(lo_k, 4), round(hi_k, 4)],
    })


if __name__ == "__main__":
    main()
