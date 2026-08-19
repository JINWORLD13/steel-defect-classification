"""
robustness.py 회귀 테스트

지키는 버그: v1에서 겪은 첫 번째 "조용한 버그" —
CONDITIONS 키와 make_transform() 분기 이름이 어긋나
아무 변형도 안 된 원본이 그대로 평가되던 문제.
파이썬은 여기서 에러를 안 내므로, 테스트로 잡는 수밖에 없음.
"""
import numpy as np
import torch
from PIL import Image

import robustness


def test_every_condition_actually_transforms():
    """original을 뺀 모든 조건이 원본과 '다른' 텐서를 내야 함."""
    # 고정 시드로 만든 무작위 이미지 한 장 (내용은 무엇이든 됨 — 변형 여부만 봄)
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (200, 200, 3), dtype=np.uint8))

    base = robustness.make_transform("original")(img)   # 기준: 무변형 텐서

    for key, _ in robustness.CONDITIONS:
        if key == "original":
            continue
        out = robustness.make_transform(key)(img)
        # allclose = 두 텐서가 사실상 같은가. 같으면 그 조건은 아무 일도 안 한 것
        assert not torch.allclose(out, base), (
            f"'{key}' 조건이 아무 변형도 하지 않음 — v1의 조용한 버그 재발"
        )
