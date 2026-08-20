"""
탐지용 Dataset — 분류와 무엇이 다른가

분류의 정답은 숫자 하나였음. 탐지의 정답은 '박스 여러 개 + 각 박스의 클래스'라서
정답이 딕셔너리가 됨:
    {"boxes": (박스수, 4) 픽셀좌표 [x1,y1,x2,y2], "labels": (박스수,) 클래스번호}

규칙 두 가지를 외울 것:
  1) 배경이 0번임. torchvision 탐지 모델은 "아무것도 아님"을 0번으로 예약하므로
     우리 결함 6종은 1~6번이 됨 (num_classes = 6 + 1).
  2) 이미지마다 박스 수가 달라서 기본 배치(쌓기)가 불가능함 -> collate_fn으로
     '쌓지 말고 리스트로 묶기'를 지정함.
"""
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from labels import boxes_of, read_manifest, stem_of


def sanity_check():
    """세 분할의 모든 이미지에 라벨 파일이 있는지 — 학습 전에 반드시 한 번."""
    for split in ("train", "val", "test"):
        for p in read_manifest(split):
            boxes_of(stem_of(p))          # 라벨이 없으면 여기서 KeyError로 즉사
    print("sanity_check 통과: 1,800장 전부 라벨 존재")


class DetectionSet(Dataset):
    def __init__(self, split, limit=None):
        self.paths = read_manifest(split)
        if limit:                          # sanity run용: 앞 limit장만
            self.paths = self.paths[:limit]
        self.to_tensor = transforms.ToTensor()   # 탐지 모델은 정규화를 내부에서 함

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        path = self.paths[i]
        img = Image.open(path).convert("RGB")
        W, H = img.size                    # 200, 200

        boxes, cls = [], []
        for c, cx, cy, w, h in boxes_of(stem_of(path)):
            # YOLO의 중심+크기(0~1) -> 픽셀 좌표의 좌상단·우하단
            x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
            x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
            # 좌표를 이미지 안으로 자르고, 넓이 0짜리 박스는 버림
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(W), x2), min(float(H), y2)
            if x2 > x1 and y2 > y1:
                boxes.append([x1, y1, x2, y2])
                cls.append(c + 1)          # 배경=0 예약이라 결함은 1~6번

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(cls, dtype=torch.int64),
        }
        return self.to_tensor(img), target


def collate_fn(batch):
    """[(이미지, 정답), ...] -> (이미지 리스트, 정답 리스트).

    이미지마다 박스 수가 달라 텐서로 쌓을 수 없으므로 리스트 그대로 넘김 —
    torchvision 탐지 모델은 리스트 입력을 받도록 설계돼 있음.
    """
    return tuple(zip(*batch))


if __name__ == "__main__":
    sanity_check()
    ds = DetectionSet("train")
    img, tgt = ds[0]
    print("이미지:", tuple(img.shape), "| 박스:", tuple(tgt["boxes"].shape),
          "| 라벨:", tgt["labels"].tolist())
