"""
NEU-CLS 데이터 정리 스크립트
- 원본(data/): YOLO 탐지 형식 (train 1770 + valid 30), 검증셋이 너무 작음
- 목표(dataset/): 분류용 ImageFolder 구조 + 제대로 된 train/val/test 분할

실행: ./venv/bin/python prepare_data.py
"""
import glob
import os
import random
import shutil

SRC_DIRS = [
    "data/train/train/images",
    "data/valid/valid/images",
]
OUT_DIR = "dataset"
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}
SEED = 42  # 재현성을 위한 고정 시드


def class_of(path):
    """파일명에서 클래스 추출: 'crazing_10.jpg' -> 'crazing'"""
    name = os.path.basename(path)
    stem = os.path.splitext(name)[0]
    # 뒤쪽 '_숫자'만 제거 (rolled-in_scale_10 -> rolled-in_scale)
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem


def main():
    # 0) 이전 실행 결과를 통째로 비우고 시작
    #    안 지우면: 시드나 비율을 바꿔 다시 돌릴 때 예전 분할이 그대로 남아 있고,
    #    그 위에 새 분할이 덮여 쌓임 -> 같은 이미지가 train과 test에 동시에 존재하게 됨.
    #    이게 데이터 누수(leakage)이고, 시험 문제를 미리 보고 푼 꼴이라 정확도가 부풀려짐.
    if os.path.isdir(OUT_DIR):
        print(f"기존 {OUT_DIR}/ 삭제 후 재생성")
        shutil.rmtree(OUT_DIR)

    # 1) 모든 이미지 경로 수집
    paths = []
    for d in SRC_DIRS:
        paths += glob.glob(os.path.join(d, "*.jpg"))
    print(f"총 이미지: {len(paths)}장")

    # 2) 클래스별로 묶기
    by_class = {}
    for p in paths:
        by_class.setdefault(class_of(p), []).append(p)

    # 3) 클래스별로 셔플 후 비율대로 분할 (stratified)
    rng = random.Random(SEED)
    counts = {s: {} for s in SPLITS}
    for cls, files in sorted(by_class.items()):
        files = sorted(files)          # 먼저 정렬해 재현성 확보
        rng.shuffle(files)             # 고정 시드로 섞기
        n = len(files)
        n_train = int(n * SPLITS["train"])
        n_val = int(n * SPLITS["val"])
        split_files = {
            "train": files[:n_train],
            "val": files[n_train:n_train + n_val],
            "test": files[n_train + n_val:],
        }
        for split, flist in split_files.items():
            dst_dir = os.path.join(OUT_DIR, split, cls)
            os.makedirs(dst_dir, exist_ok=True)
            for src in flist:
                shutil.copy2(src, os.path.join(dst_dir, os.path.basename(src)))
            counts[split][cls] = len(flist)

    # 4) 결과 출력
    print(f"\n출력 폴더: {OUT_DIR}/")
    for split in SPLITS:
        total = sum(counts[split].values())
        print(f"\n[{split}] 총 {total}장")
        for cls in sorted(by_class):
            print(f"  {cls:18s}: {counts[split][cls]}장")


if __name__ == "__main__":
    main()
