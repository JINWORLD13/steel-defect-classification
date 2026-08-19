"""
데이터 감사 스크립트 — README의 '한계' 항목에 적은 수치를 재현함

두 가지를 확인함:
  1) 분할 간 누수: train/val/test에 '내용이 완전히 같은 파일'이 걸쳐 있는지 (md5 해시로 전수 대조)
  2) 라벨 정의 검증: 파일명에서 딴 클래스가 정말 그 이미지의 유일한 정답인지
     (원본에 딸린 탐지용 바운딩박스 라벨과 대조)

왜 필요한가:
  정확도 100%가 나왔을 때 의심할 것은 셋임 — 데이터 누수 / 문제가 쉬움 / 평가셋이 작음.
  이 스크립트는 그중 '누수'를 배제하고, 대신 '문제 정의가 느슨했다'는 네 번째 가능성을 찾아냄.

실행: ./venv/bin/python audit_data.py
      (prepare_data.py를 먼저 돌려 dataset/이 있어야 함)
"""
import collections
import glob
import hashlib
import os

DATASET_DIR = "dataset"                      # prepare_data.py가 만든 분할 폴더
LABEL_GLOB = "data/*/*/labels/*.txt"          # 원본에 딸린 YOLO 형식 바운딩박스 라벨
SPLITS = ["train", "val", "test"]


def md5_of(path):
    """파일 내용을 16진수 지문 한 줄로 요약함. 내용이 1바이트라도 다르면 지문이 달라짐."""
    with open(path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def class_of(path):
    """파일명에서 클래스 추출: 'crazing_10.jpg' -> 'crazing' (prepare_data.py와 동일 규칙)"""
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem


# ---------------------------------------------------------------- 1) 누수 검사
def check_leakage():
    """같은 내용의 파일이 서로 다른 분할에 동시에 들어있는지 찾음."""
    print("=" * 60)
    print("1) 분할 간 누수 검사 (md5 전수 대조)")
    print("=" * 60)

    # 지문 -> [(분할이름, 경로), ...] 로 모음. 한 지문에 2개 이상 달리면 중복임.
    by_hash = collections.defaultdict(list)
    for split in SPLITS:
        for path in glob.glob(os.path.join(DATASET_DIR, split, "*", "*.jpg")):
            by_hash[md5_of(path)].append((split, path))

    total = sum(len(v) for v in by_hash.values())
    if total == 0:
        print(f"  {DATASET_DIR}/ 가 비어 있음. prepare_data.py를 먼저 실행할 것.")
        return

    # 서로 다른 분할에 걸친 중복만 문제임 (같은 분할 안의 중복은 누수가 아님)
    cross = []
    for files in by_hash.values():
        if len({s for s, _ in files}) > 1:
            cross.append(files)

    print(f"  검사 대상: {total}장")
    if not cross:
        print("  분할에 걸친 동일 파일: 0건")
    for files in cross:
        pair = " ↔ ".join(sorted({s for s, _ in files}))
        print(f"  [{pair}] " + " / ".join(p for _, p in files))

    # train↔test 누수만이 test 점수를 직접 부풀림 (시험 문제를 미리 본 격)
    train_test = [f for f in cross if {"train", "test"} <= {s for s, _ in f}]
    print(f"\n  → train↔test 누수: {len(train_test)}건 "
          f"({'test 점수는 이 요인으로는 부풀지 않았음' if not train_test else '★ test 점수가 부풀려졌음'})")


# ------------------------------------------------------- 2) 라벨 정의 검증
def check_label_definition():
    """파일명 라벨이 그 이미지의 유일한 정답인지, 바운딩박스 라벨과 대조함."""
    print("\n" + "=" * 60)
    print("2) 라벨 정의 검증 (파일명 클래스 vs 바운딩박스 클래스)")
    print("=" * 60)

    files = glob.glob(LABEL_GLOB)
    if not files:
        print(f"  {LABEL_GLOB} 에 라벨이 없음. 원본 data/ 를 먼저 받을 것.")
        return

    def boxes_in(path):
        """라벨 파일 한 줄 = 박스 하나. 맨 앞 숫자가 클래스 id임."""
        with open(path) as f:
            return [int(line.split()[0]) for line in f if line.strip()]

    # 라벨 파일에는 클래스 '이름'이 없고 id(0~5)만 있음.
    # 데이터에서 직접 대응을 추론함: 각 파일명 클래스에서 가장 많이 나온 id가 그 클래스의 id.
    votes = collections.Counter()
    for path in files:
        for cid in boxes_in(path):
            votes[(class_of(path), cid)] += 1
    id2name = {}
    for name in sorted({n for n, _ in votes}):
        _, cid = max((v, i) for (n, i), v in votes.items() if n == name)
        id2name[cid] = name
    print("  추론된 id→클래스:", {k: id2name[k] for k in sorted(id2name)})

    # 파일명 클래스와 '다른' 클래스의 박스를 품은 이미지를 셈
    n_mixed, n_foreign, n_box = 0, 0, 0
    mixed_by_class = collections.Counter()
    total_by_class = collections.Counter()
    combos = collections.Counter()

    for path in files:
        own = class_of(path)
        total_by_class[own] += 1
        ids = boxes_in(path)
        n_box += len(ids)
        foreign = [i for i in ids if id2name[i] != own]
        n_foreign += len(foreign)
        if foreign:
            n_mixed += 1
            mixed_by_class[own] += 1
            for other in sorted({id2name[i] for i in foreign}):
                combos[(own, other)] += 1

    n_img = len(files)
    print(f"\n  전체 {n_img}장 · 박스 {n_box}개 (이미지당 평균 {n_box / n_img:.2f}개)")
    print(f"  파일명 클래스와 다른 결함 박스를 함께 가진 이미지: "
          f"{n_mixed}장 ({n_mixed / n_img * 100:.1f}%)")
    print(f"  그런 '남의 클래스' 박스: {n_foreign}개 / {n_box}개 ({n_foreign / n_box * 100:.1f}%)")

    print("\n  클래스별 (해당 클래스 이미지 중 몇 장이 다른 결함도 품고 있나):")
    for cls in sorted(total_by_class):
        bad, tot = mixed_by_class[cls], total_by_class[cls]
        print(f"    {cls:18s}: {bad:3d}/{tot}장 ({bad / tot * 100:4.1f}%)")

    print("\n  흔한 혼재 조합 top 5:")
    for (a, b), cnt in combos.most_common(5):
        print(f"    {a:18s} + {b:18s}: {cnt}장")

    # test 분할에서 실제로 몇 장이 해당되는지 (있으면)
    test_paths = glob.glob(os.path.join(DATASET_DIR, "test", "*", "*.jpg"))
    if test_paths:
        label_of = {os.path.splitext(os.path.basename(p))[0]: p for p in files}
        hit = 0
        for p in test_paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            lab = label_of.get(stem)
            if lab and any(id2name[i] != class_of(p) for i in boxes_in(lab)):
                hit += 1
        print(f"\n  → test {len(test_paths)}장 중 {hit}장은 정답이 하나가 아님에도 "
              f"단일 라벨로 채점됨 ({hit / len(test_paths) * 100:.1f}%)")
        print("    즉 test 100%는 '데이터가 쉬웠다'만이 아니라 '문제 정의가 느슨했다'의 결과이기도 함")


def main():
    check_leakage()
    check_label_definition()


if __name__ == "__main__":
    main()
