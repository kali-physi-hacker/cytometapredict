import os
from pathlib import Path
from typing import Tuple

import pandas as pd


DATA_DIR = Path("data")


def load_indices() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load Train.csv (file->sample map), cytokine_profiles.csv (targets), and Train_Subjects.csv.

    Returns:
        train_df: columns [filename, SampleType, SubjectID, SampleID]
        cytok_df: cytokine targets keyed by SampleID
        subjects_df: subject covariates keyed by SubjectID
    """
    train_path = DATA_DIR / "Train.csv"
    cytok_path = DATA_DIR / "cytokine_profiles.csv"
    subjects_path = DATA_DIR / "Train_Subjects.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing {train_path}")
    if not cytok_path.exists():
        raise FileNotFoundError(f"Missing {cytok_path}")
    if not subjects_path.exists():
        raise FileNotFoundError(f"Missing {subjects_path}")

    train_df = pd.read_csv(train_path)
    cytok_df = pd.read_csv(cytok_path)
    subjects_df = pd.read_csv(subjects_path)
    return train_df, cytok_df, subjects_df


def ensure_dirs(*dirs: os.PathLike) -> None:
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)

