import pathlib

import pytest


@pytest.fixture
def repo_root() -> pathlib.Path:
    return pathlib.Path(__file__).resolve().parents[1]


@pytest.fixture
def example_coding_dir(repo_root: pathlib.Path) -> pathlib.Path:
    return repo_root / "examples" / "coding-systems"
