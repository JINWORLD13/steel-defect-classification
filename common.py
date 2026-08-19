"""
공용 부품 모음 — 여러 스크립트가 똑같이 쓰는 상수와 함수를 한 곳에 모음

왜 필요한가:
  analyze.py와 robustness.py가 MEAN/STD/build_model을 각자 복사해 갖고 있었음.
  복사본이 흩어져 있으면 한 곳만 고치고 다른 곳을 빼먹는 사고가 남.
  (v1에서 겪은 "조용한 버그" 두 개가 전부 이런 어긋남에서 나왔음)

사용법: 각 스크립트 맨 위에서  from common import CLASSES, DEVICE, build_model, ...
"""
import json
import math
import os

import torch
import torch.nn as nn
from torchvision import models

# 클래스 이름 목록 — ImageFolder가 폴더명을 알파벳순으로 정렬한 순서와 동일해야 함
# (dataset/train/ 밑 폴더명이 곧 라벨이고, 그 정렬 순서가 곧 숫자 라벨 0~5임)
CLASSES = ["crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"]

# 정규화 기준값 — ImageNet 사전학습 모델이 쓴 값 그대로 (train.py와 동일해야 함)
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# 계산 장치 자동 선택 (NVIDIA GPU → 애플 GPU → CPU 순)
DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

RESULTS_DIR = "results"  # 모든 수치의 단일 출처 — 문서의 숫자는 전부 여기서 나와야 함


def build_model(num_classes=len(CLASSES), pretrained=False):
    """ResNet18 뼈대 만들기.

    pretrained=False (기본): 뼈대만 필요할 때. 저장된 가중치로 덮어쓸 거라 다운로드 불필요.
    pretrained=True        : 전이학습 시작점이 필요할 때 (train 계열 스크립트만 씀).
    """
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)  # 마지막 출력층만 우리 클래스 수로 교체
    return model.to(DEVICE)


def save_result(name, payload):
    """실험 결과를 results/<name>.json 으로 저장함.

    README나 노션에 적는 숫자는 전부 이 파일에서 나와야 함.
    '문서에 적힌 수치인데 재현할 스크립트가 없다'는 상황을 막기 위한 장치임.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name + ".json")
    with open(path, "w", encoding="utf-8") as f:
        # ensure_ascii=False = 한글을 \uXXXX로 깨뜨리지 않고 그대로 저장
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"  결과 저장 -> {path}")


def load_result(name):
    """save_result로 저장한 결과를 다시 읽음."""
    path = os.path.join(RESULTS_DIR, name + ".json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def wilson(k, n, z=1.96):
    """윌슨 95% 신뢰구간 — 'n장 중 k장 정답'이라는 관측이 허용하는 진짜 정확도의 범위.

    왜 필요한가: test 270장에서 270장 전부 맞혔다고 진짜 실력이 딱 1.000인 건 아님.
    270번의 동전던지기로 확인할 수 있는 건 '최소 0.986 이상'까지임 — 그 하한을 계산함.
    (표본이 작을수록 구간이 넓어짐. 원본 valid 30장이 왜 못 믿을 검증셋이었는지도 이 공식이 말해줌)

    반환: (하한, 상한) — 예: wilson(270, 270) -> (0.986, 1.000)
    """
    if n == 0:
        return (0.0, 0.0)
    p = k / n                                   # 관측된 비율
    denom = 1 + z * z / n                       # 보정 분모
    center = (p + z * z / (2 * n)) / denom      # 보정된 중심 (0.5 쪽으로 살짝 끌려감)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom  # 반폭
    return (max(0.0, center - half), min(1.0, center + half))
