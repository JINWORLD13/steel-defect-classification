import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt


DATA_DIR ="dataset"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 32
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

torch.manual_seed(42)

# 테스트할 현장 상황 목록(영문키, 한글설명) - 영문키는 그래프용(한글 폰트 깨짐)
CONDITIONS = [
    ("original", "원본(기준"),
    ("dark", "어두움 - 조명이 약해진 상황"),
    ("bright", "밝음 - 조명이 과한/반사 상황"),
    ("blur", "흐림 - 초점 안 맞거나 렌즈에 먼지"),
    ("noise", "노이즈 - 저가 센서 * 저조도 촬영 "),
    ("low_contrast", "대비 낮음 - 뿌옇게 찍힌 상황")
]

# 상황별(망가뜨리기 위함) 전처리 만들기
def make_transform(kind):
    steps = [transforms.Resize((224, 224))]


    if kind == "dark":
        # 밝기 40%로 낮춤
        steps.append(transforms.ColorJitter(brightness = (0.4, 0.4))) # 대비를 낮추니 뿌옇게 됨.
    elif kind == "bright":
        # 밝기 170%로 올림
        steps.append(transforms.ColorJitter(brightness=(1.7, 1.7))) # 대비를 1.7배로
    elif kind == "blur":
        # 흐리게
        steps.append(transforms.GaussianBlur(kernel_size=9, sigma=(3.0, 3.0))) # 흐리게 함. (sigma가 클수록 더 흐림)
    elif kind == "low_contrast":
        # 대비 30%로 낮춤
        steps.append(transforms.ColorJitter(contrast=(0.3, 0.3)))
    # 숫자(텐서) 상태
    steps.append(transforms.ToTensor())

    # 0-1 숫자에 무작위 잡음을 더하고 범위를 다시 0-1로 자름.
    if kind == "noise":
        steps.append(transforms.Lambda( # 내가 만든 임의의 처리를 파이프라인에 끼워넣는 도구
            # torch.clamp : 0.0 - 1.0 범위로 잘라내는 함수 (노이즈 범위 벗어남 방지)
            # torch.randn_like : t와 같은 모양의 무작위 잡음을 만드는 함수
            lambda t: torch.clamp(t + torch.randn_like(t) * 0.12, 0.0, 1.0 # 
        )))
    # 정규화 (마지막)
    steps.append(transforms.Normalize(MEAN, STD))

    return transforms.Compose(steps)

# 모델 뼈대(train.py와 동일) - 지식 담을 몸통 만듦기.
def build_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)

# 특정 상황에서의 정확도 측정 (망가뜨린 전처리로 test 데이터를 다시 읽어 정확도만 계산)
@torch.no_grad()
def accuracy_under(model, transform):
    ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=transform)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False)

    model.eval()
    correct, total = 0, 0
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return correct / total

def main():
    # 클래스 수 파악 (모델 뼈대 만들 때 필요)
    probe_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"))
    classes = probe_ds.classes
    num_classes = len(classes)

    # 학습된 최고의 모델 불러오기
    model = build_model(num_classes)
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE))

    # 원본->어두움->밝음->흐림->노이즈->저대비 순으로 시험을 6번(set) 봄.
    print("\n===== 강건성 테스트 (test 270장) =====")
    keys, accs = [], []
    
    for key, desc in CONDITIONS:
        acc = accuracy_under(model, make_transform(key))
        keys.append(key)
        accs.append(acc)
        print(f"{desc:32s} : {acc:.3f}")

    base = accs[0]
    # 기준(원본) 대비 얼마나 떨어졌는지 요약 (약점 집어냄)
    print("\n----- 원본 대비 하락폭 -----")
    for (key, desc), acc in zip(CONDITIONS[1:], accs[1:]):
        drop = (base - acc) * 100 # %p 단위 하락
        print(f"{desc:32s} : -{drop:.1f}%p")

    # 막대그래프로 저장 (README에 넣을 두 번째 이미지)
    plt.figure(figsize=(9, 5))
    bars = plt.bar(keys, accs, color="steelblue") # 막대그래프 그리기
    bars[0].set_color("seagreen")
    plt.axhline(base, linestyle="--", linewidth=1, color="gray") # 기준선
    plt.ylim(0, 1.05)
    plt.ylabel("Accuracy")
    plt.title("Robustness under real-world image degradations")

    for i, a in enumerate(accs):
        # 막대 위 숫자 표시
        plt.text(i, a + 0.02, f"{a:.3f}", ha="center", fontsize=9)
    
    plt.tight_layout()
    plt.savefig("robustness.png", dpi=150)
    print("\n 그럼 저장 완료 -> robustness.png")

if __name__ == "__main__":
    main()