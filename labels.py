"""
바운딩박스 라벨 읽기 — 멀티라벨 분류·탐지·Grad-CAM이 공용으로 쓰는 부품

원본 라벨은 YOLO 형식임: 한 줄 = 박스 하나 = "클래스id 중심x 중심y 너비 높이" (전부 0~1).
클래스id 0~5는 CLASSES의 알파벳순 인덱스와 정확히 일치함 —
audit_data.py의 다수결 추론으로 검증된 사실임 (0=crazing ... 5=scratches).
"""
import glob
import os

from common import CLASSES

LABEL_GLOB = "data/*/*/labels/*.txt"

_cache = None


def label_files():
    """{파일이름(stem): 라벨경로} 대응표. 한 번만 만들고 재사용함."""
    global _cache
    if _cache is None:
        _cache = {os.path.splitext(os.path.basename(p))[0]: p
                  for p in glob.glob(LABEL_GLOB)}
        assert _cache, f"{LABEL_GLOB} 에 라벨이 없음 — 원본 data/ 를 먼저 받을 것"
    return _cache


def boxes_of(stem):
    """stem -> [(클래스인덱스, cx, cy, w, h), ...]  (좌표는 0~1 정규화 그대로)"""
    out = []
    with open(label_files()[stem]) as f:
        for line in f:
            if line.strip():
                p = line.split()
                out.append((int(p[0]), *map(float, p[1:5])))
    return out


def multi_hot(stem):
    """stem -> 길이 6의 0/1 리스트. 그 이미지에 든 결함 종류마다 1.

    단일 라벨(파일명)과 달리 박스가 말하는 '전부'를 정답으로 삼음 —
    멀티라벨 분류의 정답 벡터가 됨.
    """
    vec = [0.0] * len(CLASSES)
    for cls_idx, *_ in boxes_of(stem):
        vec[cls_idx] = 1.0
    return vec


def read_manifest(split):
    """splits/<split>.txt -> 경로 목록 (분할의 단일 출처)"""
    with open(os.path.join("splits", split + ".txt")) as f:
        return [line.split("\t")[0] for line in f.read().splitlines() if line]


def stem_of(path):
    """'dataset/test/crazing/crazing_102.jpg' -> 'crazing_102'"""
    return os.path.splitext(os.path.basename(path))[0]
