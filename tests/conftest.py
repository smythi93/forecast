import shutil
from pathlib import Path

import pytest

from forecast import subject


@pytest.fixture(scope="session", autouse=True)
def subject_dir():
    subject.SUBJECT_DIR = Path(__file__).parent / "test_subjects"
    yield
    # Only tests that actually create subjects leave this directory behind.
    shutil.rmtree(subject.SUBJECT_DIR, ignore_errors=True)
