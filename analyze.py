import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

# 표와 성적표를 대신 만들어줌(채점과 통계)
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
# 그래프, 그림 그리고 파일로 저장해줌.
import matplotlib.pyplot as plt # from matplotlib import pyplot as plt

DATA_DIR = "dataset"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 32
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# 학습 때 평가용 전치리(증강 없음)와 동일해야 함.
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# 모델 뼈대 만들기
def build_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model.to(DEVICE)

# 예측 모으기 함수(정답지(y_true)와 내가 쓴 답(모델 예측-y_pred)을 두 줄로 쭉 받아적음 - test 데이터 전체 돎)
@torch.no_grad() # 채점만 하므로 기울기 계산 끔
def collect_predictions(model, loader):
    model.eval()
    y_true, y_pred = [], []

    for images, label in loader:
        images = images.to(DEVICE) # 이미지만 gpu로. labels은 정답 비교용이라 cpu에 둬도 됨)
        outputs = model(images) # 예측 점수 계산
        preds = outputs.argmax(1).cpu() # 6점수 중 최고 번호(예측값)와 그걸 cpu로 내려 리스트화 준비
        
        y_true.extend(label.tolist()) # tolist() : 텐서를 파이썬 리스트로.
        y_pred.extend(preds.tolist())

    return y_true, y_pred


# 전체 실행
def main():
    # train,py의 test와 동일(실전 문제집만 꺼내 컨베이어에 얹음)
    test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=eval_tf)
    classes = test_ds.classes
    num_classes = len(classes)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    # 모델 빼대 만들고 최고의 모델 지식(가중치) 불러오기
    model = build_model(num_classes)
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE)) # 해당 파일(좌)을 지금 장치(우)에 맞게 불러오기

    # test 전체 예측 모으기
    y_true, y_pred = collect_predictions(model, test_loader)
    
    # 클래스별 성적표 출력 (결함 종류별 성적표)
    print("\n==== 클래스별 성적표 ====")
    print(classification_report(y_true, y_pred, target_names=classes, digits=3))

    # 혼동행렬 계산(6x6표. 대각선상은 정답 아닌건 착각한 것)
    cm = confusion_matrix(y_true, y_pred)
    print("혼동행렬(숫자):\n", cm)

    # 혼동행렬을 그림으로 그려서 파일로 저장 (README에 넣을 이미지 - plt가 표를 색칠해 png로 저장)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes) # 혼동 행렬 표 만들기
    fig, ax = plt.subplots(figsize=(8, 7))
    disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, colorbar=True) # 표에 색칠 - 파란계열, x축 글자 45도 회전
    plt.title("Confusion Matrix (test set)") # matplotlib은 한글 폰트 깨짐.
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("\n 그림 저장 완료 -> confusion_matrix.png")


if __name__ == "__main__":
    main()