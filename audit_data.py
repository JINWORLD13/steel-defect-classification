"""
데이터 감사 스크립트 — README의 '한계' 항목에 적은 수치를 재현함

두 가지를 확인함:
  1) 분할 간 누수: train/val/test에 '내용이 완전히 같은 파일'이 걸쳐 있는지 (md5 해시로 전수 대조)
  2) 문제 정의 검증: 파일명에서 딴 클래스가 정말 그 이미지의 유일한 정답인지
     (원본에 딸린 탐지용 바운딩박스 라벨과 대조 — 이 데이터는 NEU-DET 계열이라 박스가 전부 있음)

왜 필요한가:
  정확도 100%가 나왔을 때 의심할 것은 셋임 — 데이터 누수 / 문제가 쉬움 / 평가셋이 작음.
  이 스크립트는 그중 '누수'를 배제하고, 대신 '내 문제 정의가 데이터 구조와 안 맞았다'는
  네 번째 가능성을 수치로 드러냄.

'다수결'을 두 기준으로 재는 이유:
  파일명 클래스가 그 이미지의 '대표 결함'이라면, 무엇으로 대표를 정했는지가 문제임.
  박스 개수 기준과 박스 면적 기준은 서로 다른 답을 낼 수 있고,
  기준에 따라 숫자가 바뀐다는 사실 자체가 "이 데이터에 단일 정답이 없다"는 증거가 됨.

실행: ./venv/bin/python audit_data.py
      (prepare_data.py를 먼저 돌려 dataset/과 splits/가 있어야 함)
"""
import collections
import glob
import hashlib
import os

from common import CLASSES, save_result

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


def boxes_in(path):
    """라벨 파일 -> [(클래스id, 면적), ...]  한 줄 = 박스 하나.

    YOLO 형식: 클래스id 중심x 중심y 너비 높이 (전부 0~1로 정규화된 값)
    면적 = 너비 x 높이 (이미지 전체 대비 비율)
    """
    out = []
    with open(path) as f:
        for line in f:
            if line.strip():
                p = line.split()
                out.append((int(p[0]), float(p[3]) * float(p[4])))
    return out


def split_of_stems():
    """splits/*.txt 매니페스트에서 파일이름(stem) -> 분할이름 대응표를 만듦."""
    table = {}
    for split in SPLITS:
        with open(os.path.join("splits", split + ".txt")) as f:
            for line in f.read().splitlines():
                if line:
                    stem = os.path.splitext(os.path.basename(line.split("\t")[0]))[0]
                    table[stem] = split
    return table


# ---------------------------------------------------------------- 1) 누수 검사
def check_leakage():
    """같은 내용의 파일이 서로 다른 분할에 동시에 들어있는지 찾음."""
    print("=" * 60)
    print("1) 분할 간 누수 검사 (md5 전수 대조)")
    print("=" * 60)

    by_hash = collections.defaultdict(list)
    for split in SPLITS:
        for path in glob.glob(os.path.join(DATASET_DIR, split, "*", "*.jpg")):
            by_hash[md5_of(path)].append((split, path))

    total = sum(len(v) for v in by_hash.values())
    if total == 0:
        print(f"  {DATASET_DIR}/ 가 비어 있음. prepare_data.py를 먼저 실행할 것.")
        return None

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

    train_test = [f for f in cross if {"train", "test"} <= {s for s, _ in f}]
    print(f"\n  → train↔test 누수: {len(train_test)}건 "
          f"({'test 점수는 이 요인으로는 부풀지 않았음' if not train_test else '★ test 점수가 부풀려졌음'})")
    return {"total": total, "cross_split_dups": len(cross), "train_test_dups": len(train_test)}


# ------------------------------------------------------- 2) 문제 정의 검증
def check_label_definition():
    """파일명 라벨이 그 이미지의 유일한 정답인지, 바운딩박스 라벨과 대조함."""
    print("\n" + "=" * 60)
    print("2) 문제 정의 검증 (파일명 클래스 vs 바운딩박스 클래스)")
    print("=" * 60)

    files = glob.glob(LABEL_GLOB)
    if not files:
        print(f"  {LABEL_GLOB} 에 라벨이 없음. 원본 data/ 를 먼저 받을 것.")
        return None

    # 라벨 파일에는 클래스 '이름'이 없고 id(0~5)만 있음.
    # 데이터에서 직접 대응을 추론함: 각 파일명 클래스에서 가장 많이 나온 id가 그 클래스의 id.
    votes = collections.Counter()
    for path in files:
        for cid, _ in boxes_in(path):
            votes[(class_of(path), cid)] += 1
    id2name = {}
    for name in sorted({n for n, _ in votes}):
        _, cid = max((v, i) for (n, i), v in votes.items() if n == name)
        id2name[cid] = name
    print("  추론된 id→클래스:", {k: id2name[k] for k in sorted(id2name)})

    split_table = split_of_stems()                  # 파일이름 -> train/val/test

    n_box = 0
    mixed = []                                       # 다른 클래스 박스를 품은 이미지들의 stem
    mixed_by_class = collections.Counter()
    total_by_class = collections.Counter()
    combos = collections.Counter()
    cooccur = collections.Counter()                  # (클래스A, 클래스B) 공존 이미지 수
    mismatch_count = []                              # 파일명이 '개수 다수결'과 어긋나는 stem
    mismatch_area = []                               # 파일명이 '면적 다수결'과 어긋나는 stem
    mixed_split = collections.Counter()              # 다중클래스 이미지의 분할별 분포

    for path in files:
        own = class_of(path)
        stem = os.path.splitext(os.path.basename(path))[0]
        total_by_class[own] += 1
        boxes = boxes_in(path)
        n_box += len(boxes)

        count_by = collections.Counter()             # 클래스별 박스 '개수'
        area_by = collections.defaultdict(float)     # 클래스별 박스 '면적 합'
        for cid, area in boxes:
            count_by[id2name[cid]] += 1
            area_by[id2name[cid]] += area

        present = set(count_by)                       # 이 이미지에 들어있는 클래스들
        if present - {own}:                           # 파일명과 다른 클래스가 하나라도 있으면
            mixed.append(stem)
            mixed_by_class[own] += 1
            mixed_split[split_table.get(stem, "?")] += 1
            for other in sorted(present - {own}):
                combos[(own, other)] += 1
        for a in present:                             # 공존 행렬 (순서 무관 쌍)
            for b in present:
                if a < b:
                    cooccur[(a, b)] += 1

        # 다수결 판정 — 동점이면 파일명 클래스의 손을 들어줌 (보수적 판정)
        top_count = max(count_by.values())
        if count_by[own] < top_count:
            mismatch_count.append(stem)
        top_area = max(area_by.values())
        if area_by[own] < top_area:
            mismatch_area.append(stem)

    n_img = len(files)
    n_foreign = sum(combos.values())                  # 근사 아님: 조합 단위가 아니라 아래서 다시 셈
    n_foreign = 0
    for path in files:
        own = class_of(path)
        n_foreign += sum(1 for cid, _ in boxes_in(path) if id2name[cid] != own)

    print(f"\n  전체 {n_img}장 · 박스 {n_box}개 (이미지당 평균 {n_box / n_img:.2f}개)")
    print(f"  파일명 클래스와 다른 결함 박스를 함께 가진 이미지: "
          f"{len(mixed)}장 ({len(mixed) / n_img * 100:.1f}%)")
    print(f"  그런 '남의 클래스' 박스: {n_foreign}개 / {n_box}개 ({n_foreign / n_box * 100:.1f}%)")

    print("\n  클래스별 (해당 클래스 이미지 중 몇 장이 다른 결함도 품고 있나):")
    for cls in sorted(total_by_class):
        bad, tot = mixed_by_class[cls], total_by_class[cls]
        print(f"    {cls:18s}: {bad:3d}/{tot}장 ({bad / tot * 100:4.1f}%)")

    print("\n  다중클래스 이미지의 분할별 분포:",
          {s: mixed_split[s] for s in SPLITS})

    print("\n  파일명 vs 다수결 (동점은 파일명 인정):")
    print(f"    개수 다수결과 어긋남: {len(mismatch_count)}장 — "
          + ", ".join(sorted(mismatch_count)[:10]) + ("..." if len(mismatch_count) > 10 else ""))
    print(f"    면적 다수결과 어긋남: {len(mismatch_area)}장 — "
          + ", ".join(sorted(mismatch_area)[:10]) + ("..." if len(mismatch_area) > 10 else ""))
    both = sorted(set(mismatch_count) & set(mismatch_area))
    print(f"    두 기준 모두 어긋남(교집합): {len(both)}장 — " + ", ".join(both))
    print("    → 기준에 따라 숫자가 달라짐 = 이 데이터에 '단일 정답'은 없다는 뜻")

    print("\n  클래스 공존 행렬 (같은 이미지에 함께 등장한 횟수):")
    header = "    " + " ".join(f"{c[:7]:>8s}" for c in CLASSES)
    print(header)
    for a in CLASSES:
        row = []
        for b in CLASSES:
            if a == b:
                row.append(f"{'—':>8s}")
            else:
                row.append(f"{cooccur[tuple(sorted((a, b)))]:>8d}")
        print(f"    {a[:7]:>7s} " + " ".join(row))

    print("\n  흔한 혼재 조합 top 5 (파일명클래스 → 같이 들어있는 클래스):")
    for (a, b), cnt in combos.most_common(5):
        print(f"    {a:18s} + {b:18s}: {cnt}장")

    # test 분할에서 실제로 몇 장이 해당되는지
    test_mixed = [s for s in mixed if split_table.get(s) == "test"]
    n_test = sum(1 for s in split_table.values() if s == "test")
    print(f"\n  → test {n_test}장 중 {len(test_mixed)}장은 결함을 2종 이상 품은 채 "
          f"단일 라벨로 채점됨 ({len(test_mixed) / n_test * 100:.1f}%)")
    print("    데이터가 틀린 게 아니라, 탐지용 데이터를 단일 라벨 분류로 눌러 담은 문제 설정의 결과임")

    return {
        "images": n_img, "boxes": n_box,
        "mixed_images": len(mixed), "foreign_boxes": n_foreign,
        "mixed_by_class": {c: mixed_by_class[c] for c in sorted(total_by_class)},
        "mixed_split_distribution": {s: mixed_split[s] for s in SPLITS},
        "mismatch_count_majority": sorted(mismatch_count),
        "mismatch_area_majority": sorted(mismatch_area),
        "mismatch_both": both,
        "cooccurrence": {f"{a}+{b}": v for (a, b), v in sorted(cooccur.items())},
        "test_mixed_images": sorted(test_mixed),
    }, id2name, split_table, mixed


# ------------------------------------------------- 3) 다중결함 실례 그림
def draw_examples(id2name, split_table, mixed):
    """test의 다중클래스 이미지 6장에 박스를 그려 저장 — 주장을 눈으로 확인하게 함."""
    from PIL import Image, ImageDraw
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    # 클래스별 색 (구분만 되면 됨)
    palette = ["red", "deepskyblue", "lime", "orange", "magenta", "yellow"]
    color_of = {name: palette[i] for i, name in enumerate(CLASSES)}

    # test에 속한 다중클래스 이미지 중 앞 6장
    targets = sorted(s for s in mixed if split_table.get(s) == "test")[:6]
    if not targets:
        print("  test에 다중클래스 이미지가 없어 그림 생략")
        return

    label_of = {os.path.splitext(os.path.basename(p))[0]: p
                for p in glob.glob(LABEL_GLOB)}

    os.makedirs("figures", exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8.5))
    for ax, stem in zip(axes.ravel(), targets):
        cls = class_of(stem + ".jpg")
        img_path = os.path.join(DATASET_DIR, "test", cls, stem + ".jpg")
        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        W, H = img.size
        present = set()
        with open(label_of[stem]) as f:
            for line in f:
                if line.strip():
                    p = line.split()
                    cid, cx, cy, w, h = int(p[0]), *map(float, p[1:5])
                    name = id2name[cid]
                    present.add(name)
                    # YOLO의 중심+크기(0~1) -> 픽셀 좌표의 좌상단·우하단
                    x1, y1 = (cx - w / 2) * W, (cy - h / 2) * H
                    x2, y2 = (cx + w / 2) * W, (cy + h / 2) * H
                    draw.rectangle([x1, y1, x2, y2], outline=color_of[name], width=2)
        ax.imshow(img)
        ax.set_title(f"{stem}  (filename: {cls})", fontsize=9)
        ax.axis("off")
    handles = [mpatches.Patch(color=color_of[c], label=c) for c in CLASSES]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=9)
    fig.suptitle("Multi-defect test images — one filename label, several defect kinds")
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig("figures/02_multilabel_examples.png", dpi=150)
    print("\n  그림 저장 -> figures/02_multilabel_examples.png")


def main():
    leak = check_leakage()
    result = check_label_definition()
    if result:
        labels, id2name, split_table, mixed = result
        draw_examples(id2name, split_table, mixed)
        save_result("audit_labels", {"leakage": leak, **labels})


if __name__ == "__main__":
    main()
