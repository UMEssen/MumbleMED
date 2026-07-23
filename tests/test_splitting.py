import json

import pandas as pd

from mumblemed.datasets.real import _split_and_write
from mumblemed.utils.splitting import split_dataframe_by_group, split_group_ids


def test_split_group_ids_uses_80_10_10_for_ten_groups():
    splits = split_group_ids(list(range(10)))

    assert {name: len(ids) for name, ids in splits.items()} == {
        "train": 8,
        "val": 1,
        "test": 1,
    }


def test_split_dataframe_by_group_prevents_patient_leakage():
    df = pd.DataFrame(
        {
            "patient_id": ["p1", "p1", "p2", "p2", "p3", "p3", "p4", "p4"],
            "document_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "label_chunk": ["text"] * 8,
        }
    )

    splits = split_dataframe_by_group(df, group_column="patient_id", seed=13)
    split_patients = {
        name: set(split_df.patient_id.unique())
        for name, split_df in splits.items()
    }

    assert split_patients["train"].isdisjoint(split_patients["val"])
    assert split_patients["train"].isdisjoint(split_patients["test"])
    assert split_patients["val"].isdisjoint(split_patients["test"])


def test_real_split_writer_keeps_patients_in_one_csv(tmp_path):
    df = pd.DataFrame(
        {
            "patient_id": [f"p{i}" for i in range(10) for _ in range(2)],
            "document_id": [f"d{i}_{j}" for i in range(10) for j in range(2)],
            "label_chunk": ["clinical text"] * 20,
            "duration_in_seconds": [1.0] * 20,
            "speaker_id": ["default"] * 20,
            "audio_path": ["sample.wav"] * 20,
        }
    )

    _split_and_write(df, tmp_path, suffix="demo", whisper_mode=False, seed=7)

    split_patients = {}
    for split in ("train", "val", "test"):
        split_df = pd.read_csv(tmp_path / f"{split}_demo.csv", sep=";")
        split_patients[split] = set(split_df.patient_id.unique())

    assert {name: len(ids) for name, ids in split_patients.items()} == {
        "train": 8,
        "val": 1,
        "test": 1,
    }
    assert split_patients["train"].isdisjoint(split_patients["val"])
    assert split_patients["train"].isdisjoint(split_patients["test"])
    assert split_patients["val"].isdisjoint(split_patients["test"])

    stats = json.loads((tmp_path / "stats_demo.json").read_text(encoding="utf-8"))
    assert stats["split_group"] == "patient_id"
    assert stats["total_patients"] == 10
