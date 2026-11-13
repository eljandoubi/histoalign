import pytest

from tests.slides import SAMPLE, write_slide


@pytest.fixture
def fake_slide(tmp_path):
    return write_slide(tmp_path / SAMPLE)
