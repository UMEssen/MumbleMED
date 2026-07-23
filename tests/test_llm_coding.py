from pathlib import Path

from mumblemed.utils.llm import DEFAULT_CODING_SYSTEM_FILES, load_coding_tables


def test_load_coding_tables_example_dir(example_coding_dir: Path):
    assert example_coding_dir.is_dir()
    tables = load_coding_tables(example_coding_dir)
    assert set(tables.keys()) == set(DEFAULT_CODING_SYSTEM_FILES.keys())
    for name, df in tables.items():
        assert "display" in df.columns
        assert "code" in df.columns
        assert len(df) >= 1
