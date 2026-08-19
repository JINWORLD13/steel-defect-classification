"""
pytest 공용 준비물(fixture) 모음

conftest.py는 pytest가 자동으로 읽는 특별한 파일임.
여기에 만든 fixture는 tests/ 안 모든 테스트가 인자 이름만 적으면 받아 쓸 수 있음.
"""
import os
import sys

import pytest

# tests/ 폴더에서 한 칸 위(프로젝트 루트)를 import 경로에 추가
# → 테스트가 prepare_data, robustness 같은 루트의 모듈을 import할 수 있게 됨
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


@pytest.fixture
def fake_neu(tmp_path, monkeypatch):
    """가짜 NEU 데이터 20장 — 진짜 데이터 없이도 분할 로직을 시험할 수 있게 함.

    tmp_path    = pytest가 만들어주는 1회용 빈 폴더 (테스트 끝나면 알아서 치워짐)
    monkeypatch = 테스트 동안만 무언가를 바꿨다가 끝나면 원상복구해주는 도구
    """
    src = tmp_path / "data" / "train" / "train" / "images"
    src.mkdir(parents=True)
    for i in range(1, 21):
        # 분할 로직은 파일 '내용'을 보지 않으므로 내용은 아무 글자면 됨
        (src / f"crazing_{i}.jpg").write_text(f"img{i}")
    # prepare_data가 찾는 두 번째 원본 폴더도 (비어 있게) 만들어 둠
    (tmp_path / "data" / "valid" / "valid" / "images").mkdir(parents=True)

    # 테스트가 tmp 폴더 안에서 돌게 함 → 진짜 dataset/ 을 절대 건드리지 않음
    monkeypatch.chdir(tmp_path)
    return tmp_path
