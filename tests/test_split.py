"""
prepare_data.py 회귀 테스트

회귀 테스트 = 한 번 잡은 버그가 다시 못 들어오게 못박는 테스트.
여기서 지키는 버그: 커밋 d1c54d1 — 출력 폴더를 안 지우고 덮어써서
시드를 바꿔 재실행하면 이전 분할 위에 새 분할이 쌓여 train/test가 누수되던 문제.
"""
import glob
import os

import prepare_data


def test_reseed_does_not_leak(fake_neu, monkeypatch):
    """시드를 바꿔 두 번 돌려도 파일 수가 늘지 않고 train∩test가 비어 있어야 함."""
    prepare_data.main()                                  # 1차 실행 (SEED 42)
    n1 = len(glob.glob("dataset/*/*/*.jpg"))

    monkeypatch.setattr(prepare_data, "SEED", 7)         # 시드만 바꿔서
    prepare_data.main()                                  # 2차 실행 — 버그가 있으면 여기서 쌓임
    n2 = len(glob.glob("dataset/*/*/*.jpg"))

    # 파일 총수가 그대로여야 함 (쌓였다면 20 -> 28처럼 늘어남)
    assert n1 == n2 == 20

    # 같은 파일이 train과 test 양쪽에 있으면 안 됨 (= 데이터 누수)
    train = {os.path.basename(p) for p in glob.glob("dataset/train/*/*.jpg")}
    test = {os.path.basename(p) for p in glob.glob("dataset/test/*/*.jpg")}
    assert not (train & test), f"train/test 누수 발생: {sorted(train & test)}"


def test_split_is_reproducible(fake_neu):
    """같은 시드로 두 번 돌리면 파일 배치가 완전히 같아야 함 (재현성)."""
    prepare_data.main()
    first = sorted(glob.glob("dataset/*/*/*.jpg"))
    prepare_data.main()
    second = sorted(glob.glob("dataset/*/*/*.jpg"))
    assert first == second


def test_class_of_keeps_underscored_names():
    """클래스 이름에 밑줄이 있어도('rolled-in_scale') 뒤의 번호만 떼야 함."""
    assert prepare_data.class_of("rolled-in_scale_10.jpg") == "rolled-in_scale"
    assert prepare_data.class_of("crazing_7.jpg") == "crazing"
    assert prepare_data.class_of("pitted_surface_123.jpg") == "pitted_surface"
