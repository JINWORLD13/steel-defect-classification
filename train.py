import os # 파일 폴더 경로를 다루는 표준 라이브러리
import torch # 텐서 계산 담당
import torch.nn as nn # 신경망 부품 모음
import torch.optim as optim # 옵티마이저(모델 고치는 도구)

from torch.utils.data import DataLoader # 데이터를 배치 단위로 꺼내줌

from torchvision import datasets, transforms, models # 이미지 데이터셋/전처리(크기, 정규화)/사전학습모델 모음

DATA_DIR = "dataset" # 데이터가 들어 있는 폴더 이름

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu") # 계산 장치 자동 선택 (NVIDIA GPU → 애플 GPU → CPU 순)

BATCH_SIZE = 32
EPOCHS = 12
LR = 1e-3 # 0.001을 뜻하는 표기(e는 10의 몇 제곱을 뜻) / 학습률(모델을 한 번에 얼마나 크게 고칠지)

torch.manual_seed(42) # 무작위 요소(가중치 초기화·셔플·증강)를 고정해 재실행해도 비슷한 결과가 나오게 함

MEAN = [0.485, 0.456, 0.406] # 정규화(숫자들의 단위 척도를 맞춤)의 평균 (사전 학습 모델이 쓴 기준)
STD = [0.229, 0.224, 0.225] # 정규화의 표준편차 (사전 학습 모델이 쓴 기준)

# 전처리1
train_tf = transforms.Compose([
    transforms.Resize((224, 224)), # 이미지 크기 통일
    transforms.RandomHorizontalFlip(), # 데이터 증강(Augmentation) : 사진 변형해 데이터 수 뻥튀기
    transforms.RandomRotation(15), # 데이터 증강
    transforms.ToTensor(), # 이미지를 0-1 범위 숫자 텐서로 바꿔주는 함수
    transforms.Normalize(MEAN, STD) # (값-평균)/표준편차 로 값을 0 근처로 재배치하는 함수
])
# 전처리2 (시험 문제는 원본 그대로. 증강 없음)
eval_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# 모델 만들기 함수
def build_model(num_classes):
    # ResNet18 구조 모델을 불러옴. 기본 추천 가중치 사용.
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # 마지막 출력층 교체
    model.fc = nn.Linear(model.fc.in_features, num_classes) # 입력수(디폴트로 설정), 출력수

    return model.to(DEVICE)

#  학습 함수 만들기
# loader : 배치를 하나씩 공급하는 DataLoader(학습, 평가, 테스트용 로더 main()에 세팅)
# optimizer : 틀린 정도를 줄이도록 모델을 고치는 도구
def train_one_epoch(model, loader, criterion, optimizer):
    model.train() # 모델을 학습모드로 전환
    
    running_loss, correct, total = 0.0, 0, 0

    # 배치를 하나씩 꺼내기
    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE) # 이번 배치를 GPU로 옮김

        optimizer.zero_grad() # (평가함수에 x) 이전 배치의 기울기를 0으로 지우는 메서드
        
        outputs = model(images) # 이미지를 모델에 넣어 예측을 계산 (문제 풀기)

        # 손실 : 틀린 정도, 낮을 수록 잘한 것
        loss = criterion(outputs, labels) # 예측과 정답 차이 계산

        loss.backward() # (평가함수에 x) 손실 줄이기 위한 기울기에 대한 역계산

        optimizer.step() # (평가함수에 x) 계산된 기울기 방향으로 파라미터를 한 걸음 갱신

        running_loss += loss.item() * images.size(0)
        # 예측(가장 큰 1개 out of 6)과 정답 비교
        correct += (outputs.argmax(1) == labels).sum().item() # True 개수 합쳐 파이썬 숫자로 꺼냄
        total += labels.size(0)

    # 평균 손실 : 손실 / 전체 개수
    # 정확도 : 맞힌 개수 / 전체 개수
    return running_loss / total, correct / total

@torch.no_grad() # 기울기 계산 끄기 (데코레이션 : 문법 바꾸기)
def evaluate(model, loader, criterion):
    # 평가모드로 전환
    model.eval()

    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        
        outputs = model(images)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


# 전체 조합
def main():
    # 폴더명을 클래스 라벨로 자동 인식해 데이터셋을 만들기
    # transform= 각 이미지에 적용할 전처리 지정.
    train_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tf)
    val_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=eval_tf)
    test_ds = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=eval_tf)

    # ImageFolder가 폴더명에서 뽑은 클래스 이름 목록
    classes = train_ds.classes
    num_classes = len(classes)
    print("클래스:", classes)

    # DataLoader 3개 만들기
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    # 모델 생성
    model = build_model(num_classes)

    # 손실 함수 생성
    criterion = nn.CrossEntropyLoss() # 다중분류용 표준 손실함수 만드는 함수 - 소프트맥스(점수들을 확률로 바꿈) 내장됨.
    
    # 옵티마이저 생성(손실과 옵티마이저는 형제)
    # 학습률 lr : 한 걸음 크기 지정
    optimizer = optim.Adam(model.parameters(), lr=LR) # 옵티마이저 Adam을 만듦 - 학습으로 조정할 모델의 모든 값을 넘겨주는 매서드를 인자로.

    # 최고 성능 추적용 변수
    best_acc = 0.0

    # 에폭 반복: 학습->검증->출력->best면 저장
    for epoch in range(1, EPOCHS + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        
        print(f"[{epoch:2d}/{EPOCHS}] " f"train loss {tr_loss:.3f} acc {tr_acc:.3f} | " f"val loss {val_loss:.3f} acc {val_acc:.3f}") # d는 정수 형식. 2d는 정수를 최소 2칸 폭으로 출력하라. f는 소수 형식 .3은 소수점 뒤 3자리
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth") # 모델의 현재 가중치(지식)을 통째로 꺼내 파일로 저장
            print(f" best 갱신 -> 저장 (val acc {val_acc:.3f})")
        
    # '저장된 최고의 모델(이게 val과 차이)'을 불러와 test로 최종 평가.
    model.load_state_dict(torch.load("best_model.pth", map_location=DEVICE)) # 저장 시 장치와 달라도 지금 장치에 맞게 불러옴
    test_loss, test_acc = evaluate(model, test_loader, criterion) # 현재 최고의 모델로 설정
    print(f"\n=== 최종 테스트 정확도: {test_acc:.3f} ===")
    
if __name__ == "__main__":
    main()


# 간단한 학습 흐름
# main에서 학습, 평가, best 가중치(지능) 갱신을 한 셋으로 epoch만큼 반복함.
# main에서 일련의 과정을 다 거치면 최고의 가중치 파일을 불러 모델에 입힌 후 테스트(최종시험)을 함.
# main에서 일련의 과정을 거치려면 model, loss함수(criterion), optimizer(틀린 정도를 줄이도록 모델을 고치는 도구), loader를 생성
# model 생성시 여기선 전이학습인데 출력만 변경(클래스 수 전달) => 데이터셋 설정 필요(train_ds)
# main에서 loader(데이터셋을 배치 단위로 깨내줌)를 train, val, test용 각각 생성
# main에서 데이터셋을 설정 => 전처리 설정(정의) 필요(main 밖, 데이터셋 꺼내면서 전처리 자동으로 함)
# main에서 최고 모델(학습 평가시) 기록용 변수 생성(이 기록으로 모델을 불러 test)

# main 밖에서 model, train, evaluate 정의 및 전처리 정의(loader 때문)해야 함.
# loss함수, optimizer는 라이브러리로 생성. 단, 수치는 내가 설정