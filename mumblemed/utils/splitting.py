"""Group-aware train/validation/test splitting helpers."""

from __future__ import annotations

import random
from collections.abc import Hashable

import pandas as pd


def split_dataframe_by_group(
    df: pd.DataFrame,
    group_column: str,
    seed: int | None = None,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> dict[str, pd.DataFrame]:
    """Split rows by group so one document or patient cannot cross splits."""
    if group_column not in df.columns:
        raise KeyError(f"Split group column not found: {group_column}")

    group_ids = list(pd.unique(df[group_column]))
    if seed is not None:
        rng = random.Random(seed)
        rng.shuffle(group_ids)

    split_ids = split_group_ids(
        group_ids=group_ids,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
    )

    return {
        name: df[df[group_column].isin(ids)].copy()
        for name, ids in split_ids.items()
    }


def split_group_ids(
    group_ids: list[Hashable],
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> dict[str, set[Hashable]]:
    """Assign group ids to train/val/test with sensible behavior for small n."""
    n_groups = len(group_ids)
    if n_groups == 0:
        return {"train": set(), "val": set(), "test": set()}
    if n_groups == 1:
        return {"train": set(group_ids), "val": set(), "test": set()}
    if n_groups == 2:
        return {"train": {group_ids[0]}, "val": set(), "test": {group_ids[1]}}

    train_count = max(1, int(n_groups * train_fraction))
    val_count = max(1, int(n_groups * val_fraction))

    while train_count + val_count > n_groups - 1:
        if train_count > val_count and train_count > 1:
            train_count -= 1
        elif val_count > 1:
            val_count -= 1
        else:
            break

    train_ids = set(group_ids[:train_count])
    val_ids = set(group_ids[train_count:train_count + val_count])
    test_ids = set(group_ids[train_count + val_count:])

    return {"train": train_ids, "val": val_ids, "test": test_ids}
