"""
멀티라벨 분류기 — 단일 라벨의 구조적 상한을 20줄 수정으로 깨 본다

무엇이 다른가 (train.py 대비):
  1) 정답이 번호 하나가 아니라 길이 6의 0/1 벡터임 (박스가 말하는 결함 '전부')
  2) 손실이 CrossEntropyLoss -> BCEWithLogitsLoss
     - CrossEntropy는 "6개 중 정답은 딱 하나"를 가정하고 확률을 서로 뺏게 만듦
     - BCE는 클래스마다 독립적인 예/아니오 문제 6개로 봄 -> 동시에 여러 개가 1일 수 있음
  3) 예측은 argmax(1등 하나)가 아니라 "확률 0.5를 넘는 전부"

왜 하는가:
  test 270장 중 23장은 결함이 2종 이상인데 단일 라벨은 구조적으로 하나만 답할 수 있음.
  그 상한을 실제로 깨는 데 몇 줄이 필요한지, 깨면 뭐가 좋아지는지 재는 실험임.

실행: ./venv/bin/python train_multilabel.py
"""
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from common import DEVICE, MEAN, STD, build_model
from labels import multi_hot, read_manifest, stem_of

BATCH_SIZE = 32
EPOCHS = 12
LR = 1e-3
torch.manual_seed(42)

train_tf = transforms.Compose([                    # v1 train.py와 동일한 증강
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


class MultiLabelSet(Dataset):
    """splits 매니페스트의 이미지 + 박스에서 만든 멀티핫 정답."""

    def __init__(self, split, tf):
        self.paths = read_manifest(split)
        self.tf = tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        img = self.tf(Image.open(path).convert("RGB"))
        target = torch.tensor(multi_hot(stem_of(path)))    # 예: [0,1,0,0,0,1]
        return img, target


@torch.no_grad()
def subset_accuracy(model, loader):
    """전부 맞아야 정답 — 6칸 벡터가 정답 벡터와 완전히 일치한 비율.

    멀티라벨에서 가장 엄격한 채점임. 5칸 맞고 1칸 틀리면 0점.
    """
    model.eval()
    correct, total = 0, 0
    for images, targets in loader:
        logits = model(images.to(DEVICE))
        preds = (torch.sigmoid(logits) > 0.5).float().cpu()  # 클래스별 예/아니오
        correct += (preds == targets).all(dim=1).sum().item()  # 6칸 전부 일치한 행 수
        total += targets.size(0)
    return correct / total


def main():
    train_loader = DataLoader(MultiLabelSet("train", train_tf), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(MultiLabelSet("val", eval_tf), batch_size=BATCH_SIZE, shuffle=False)

    model = build_model(pretrained=True)               # 시작점은 v1과 같은 ImageNet 지식
    criterion = nn.BCEWithLogitsLoss()                 # 클래스별 독립 예/아니오 손실
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_acc = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for images, targets in train_loader:
            images, targets = images.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), targets)   # 로짓 (32,6) vs 정답 (32,6)
            loss.backward()
            optimizer.step()
            running += loss.item() * images.size(0)

        val_acc = subset_accuracy(model, val_loader)   # 선택 기준: val의 subset accuracy
        print(f"[{epoch:2d}/{EPOCHS}] train loss {running / len(train_loader.dataset):.4f} "
              f"| val subset acc {val_acc:.3f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model_multilabel.pth")
            print(f" best 갱신 -> 저장 (val subset acc {val_acc:.3f})")

    print("\n학습 끝. 최종 평가는 eval_multilabel.py에서 (test는 딱 한 번).")


if __name__ == "__main__":
    main()
