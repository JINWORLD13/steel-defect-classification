"""
수제 특징 베이스라인 — 딥러닝 없이 이 과제가 얼마나 풀리는지 잰다

왜 필요한가:
  "ResNet18로 1.000"이 대단한지 아닌지는 더 싼 방법의 점수를 알아야 판단됨.
  손으로 만든 텍스처 통계 7개 + 로지스틱 회귀(가장 단순한 분류기)가 이미 높게 나온다면,
  이 데이터는 깊은 모델이 필요 없을 만큼 클래스 간 텍스처가 갈리는 데이터라는 뜻임.
  1.000의 세 번째 용의자("문제가 원래 쉽다")를 재는 실험임.

실행: ./venv/bin/python baseline_handcrafted.py
"""
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from common import CLASSES, save_result, wilson
from labels import read_manifest, stem_of


def features_of(path):
    """이미지 한 장 -> 텍스처 통계 7개.

    깊은 모델이 '배우는' 특징 대신, 사람이 '정의한' 특징만 씀.
    각 숫자는 눈으로 설명 가능함 — 그게 이 베이스라인의 존재 이유임.
    """
    img = np.asarray(Image.open(path).convert("L"), dtype=np.float32)  # 흑백 (200,200)
    gy, gx = np.gradient(img)                    # 세로/가로 방향의 밝기 변화량
    mag = np.sqrt(gx * gx + gy * gy)             # 변화량의 크기 = 엣지 세기
    ax, ay = np.abs(gx).mean(), np.abs(gy).mean()
    return np.array([
        img.mean(),                              # 1 평균 밝기
        img.std(),                               # 2 밝기 산포 (전역 대비)
        ax,                                      # 3 가로 방향 엣지 평균
        ay,                                      # 4 세로 방향 엣지 평균
        mag.std(),                               # 5 엣지 세기의 산포
        (mag > 30).mean(),                       # 6 강한 엣지 픽셀의 비율
        (ax - ay) / (ax + ay + 1e-6),            # 7 방향성: 가로/세로 엣지의 비대칭
    ], dtype=np.float32)


def load_split(split):
    paths = read_manifest(split)
    X = np.stack([features_of(p) for p in paths])          # (n, 7)
    # 정답: 폴더명 클래스의 인덱스 (CLASSES 알파벳순)
    y = np.array([CLASSES.index(stem_of(p).rsplit("_", 1)[0]) for p in paths])
    return X, y


def main():
    print("특징 추출 중 (7개/장)...")
    X_tr, y_tr = load_split("train")
    X_te, y_te = load_split("test")

    # 스케일 표준화 — 로지스틱 회귀는 특징들의 크기 단위가 다르면 큰 특징에 끌려감
    scaler = StandardScaler().fit(X_tr)                     # 기준은 train에서만 배움
    X_tr, X_te = scaler.transform(X_tr), scaler.transform(X_te)

    clf = LogisticRegression(max_iter=2000).fit(X_tr, y_tr)
    acc = float((clf.predict(X_te) == y_te).mean())
    lo, hi = wilson(int(acc * len(y_te)), len(y_te))

    chance = 1.0 / len(CLASSES)                             # 균형 6클래스 → 무작위 0.167
    print(f"\n===== 베이스라인 사다리 (깨끗한 test {len(y_te)}장) =====")
    print(f"무작위 찍기          : {chance:.3f}")
    print(f"수제 특징 7 + 로지스틱 : {acc:.3f}  (윌슨 95% [{lo:.3f}, {hi:.3f}])")
    print(f"ResNet18 전이학습    : 1.000  (v1 실측)")

    # 사다리 그림 — 막대 3개면 충분함
    names = ["chance", "handcrafted 7\n+ logistic", "ResNet18\n(transfer)"]
    vals = [chance, acc, 1.0]
    errs = [0, acc - lo, 1.0 - wilson(270, 270)[0]]
    plt.figure(figsize=(6.5, 4.5))
    bars = plt.bar(names, vals, color=["gray", "steelblue", "seagreen"],
                   yerr=errs, capsize=4)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.3f}", ha="center")
    plt.ylim(0, 1.1)
    plt.ylabel("Accuracy (clean test, n=270)")
    plt.title("How hard is this task without deep learning?")
    plt.tight_layout()
    plt.savefig("figures/03_baseline_ladder.png", dpi=150)
    print("\n그림 저장 -> figures/03_baseline_ladder.png")

    save_result("baselines", {
        "chance": round(chance, 4),
        "handcrafted_logreg": round(acc, 4),
        "handcrafted_wilson": [round(lo, 4), round(hi, 4)],
        "resnet18_v1": 1.0,
        "features": ["mean", "std", "|gx|mean", "|gy|mean", "grad_std",
                     "edge_frac(>30)", "anisotropy"],
    })


if __name__ == "__main__":
    main()
