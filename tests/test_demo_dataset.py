import json
from pathlib import Path

import pandas as pd

from mumblemed.datasets.demo import DemoDatasetConfig, generate_demo_dataset


def test_generate_demo_dataset_creates_ten_samples(tmp_path: Path):
    audio_dir = tmp_path / "audio"
    csv_dir = tmp_path / "csv"

    generate_demo_dataset(
        DemoDatasetConfig(
            dataset_path=audio_dir,
            csv_path=csv_dir,
            num_samples=10,
            duration_seconds=0.1,
        )
    )

    wav_files = sorted(audio_dir.glob("*.wav"))
    assert len(wav_files) == 10

    split_counts = {}
    for split in ["train", "val", "test"]:
        split_path = csv_dir / f"{split}.csv"
        assert split_path.exists()
        split_counts[split] = len(pd.read_csv(split_path, sep=";"))

    assert split_counts == {"train": 8, "val": 1, "test": 1}

    stats = json.loads((csv_dir / "stats.json").read_text(encoding="utf-8"))
    assert stats["total_samples"] == 10
    assert stats["splits"]["train"]["samples"] == 8
