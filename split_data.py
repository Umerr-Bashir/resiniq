from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

DATA_PATH = Path(__file__).resolve().parent / "data" / "rows.csv"

FEATURES = [
    "elapsed",
    "av_init",
    "temp_set",
    "catalyst",
    "glycerol_ratio",
    "purity_meas",
    "lab_av",
    "lab_av_prev",
    "lab_visc",
    "torque",
    "torque_slope",
    "temp_last",
    "temp_mean",
    "water_cum",
    "water_rate",
]

TARGET = "time_remaining"
GROUP_COL = "batch_id"
RANDOM_STATE = 42
TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    return df.sort_values([GROUP_COL, "elapsed"]).reset_index(drop=True)


def _resolve_batches(indexed_groups: pd.Series, row_indices) -> set:
    return set(indexed_groups.iloc[row_indices].unique())


def _split_by_group(df: pd.DataFrame) -> dict[str, pd.Index]:
    features = df[FEATURES]
    target = df[TARGET]
    groups = df[GROUP_COL]

    first_split = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_val_idx, test_idx = next(first_split.split(features, target, groups))

    second_split = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_SIZE / (1 - TEST_SIZE),
        random_state=RANDOM_STATE,
    )
    train_idx, val_idx = next(
        second_split.split(
            features.iloc[train_val_idx],
            target.iloc[train_val_idx],
            groups.iloc[train_val_idx],
        )
    )

    train_batches = _resolve_batches(groups.iloc[train_val_idx], train_idx)
    val_batches = _resolve_batches(groups.iloc[train_val_idx], val_idx)
    test_batches = set(groups.iloc[test_idx].unique())

    return {
        "train_batches": train_batches,
        "val_batches": val_batches,
        "test_batches": test_batches,
    }


def make_splits(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    groups = df[GROUP_COL]
    batch_split = _split_by_group(df)

    train_mask = groups.isin(batch_split["train_batches"])
    val_mask = groups.isin(batch_split["val_batches"])
    test_mask = groups.isin(batch_split["test_batches"])

    train_batches = set(df.loc[train_mask, GROUP_COL].unique())
    val_batches = set(df.loc[val_mask, GROUP_COL].unique())
    test_batches = set(df.loc[test_mask, GROUP_COL].unique())

    assert train_batches.isdisjoint(val_batches)
    assert train_batches.isdisjoint(test_batches)
    assert val_batches.isdisjoint(test_batches)

    train_df = df.loc[train_mask].copy()
    val_df = df.loc[val_mask].copy()
    test_df = df.loc[test_mask].copy()

    return {
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "train_batches": train_batches,
        "val_batches": val_batches,
        "test_batches": test_batches,
    }


df = load_dataset()
splits = make_splits(df)

train_df = splits["train_df"]
val_df = splits["val_df"]
test_df = splits["test_df"]

X_train = train_df[FEATURES]
y_train = train_df[TARGET]
X_val = val_df[FEATURES]
y_val = val_df[TARGET]
X_calib = X_val
y_calib = y_val
X_test = test_df[FEATURES]
y_test = test_df[TARGET]

train_batches = splits["train_batches"]
val_batches = splits["val_batches"]
calib_batches = val_batches
test_batches = splits["test_batches"]

SPLIT_METADATA = {
    "random_state": RANDOM_STATE,
    "test_size": TEST_SIZE,
    "validation_size": VALIDATION_SIZE,
    "rows": {
        "train": len(train_df),
        "validation": len(val_df),
        "test": len(test_df),
    },
    "batches": {
        "train": len(train_batches),
        "validation": len(val_batches),
        "test": len(test_batches),
    },
}

