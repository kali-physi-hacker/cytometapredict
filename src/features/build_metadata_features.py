from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.io import load_indices


CYTOKINE_META_COLS = {
    "SampleID",
    "Plate",
    "CollectionDate",
    "CL1",
    "CL2",
    "CL3",
    "CL4",
}


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    feature_names: List[str]
    target_names: List[str]
    groups: np.ndarray  # SubjectID for GroupKFold
    sample_ids: np.ndarray
    index: pd.Index
    transformer: ColumnTransformer


def _infer_target_columns(cytok_df: pd.DataFrame) -> List[str]:
    """Infer cytokine columns by excluding known metadata columns and non-numeric columns."""
    candidates = []
    for col in cytok_df.columns:
        if col in CYTOKINE_META_COLS:
            continue
        if pd.api.types.is_numeric_dtype(cytok_df[col]):
            candidates.append(col)
    return candidates


def _deduplicate_sample_records(
    df: pd.DataFrame,
    preferred_sample_types: Optional[Sequence[str]] = None,
    allowed_sample_types: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if allowed_sample_types is not None:
        allowed_set = set(allowed_sample_types)
        df = df[df["SampleType"].isin(allowed_set)].copy()

    if df.empty:
        return df

    if preferred_sample_types is None:
        preferred_sample_types = list(df["SampleType"].value_counts().index)

    priority = {sample_type: idx for idx, sample_type in enumerate(preferred_sample_types)}
    default_priority = len(preferred_sample_types)

    df = df.copy()
    df["__sample_priority"] = df["SampleType"].map(priority).fillna(default_priority)
    df.sort_values(["SampleID", "__sample_priority", "filename"], inplace=True)
    deduped = df.drop_duplicates(subset="SampleID", keep="first").copy()
    deduped.drop(columns=["__sample_priority"], inplace=True)
    return deduped


def make_metadata_dataset(
    include_subject_covariates: bool = True,
    preferred_sample_types: Optional[Sequence[str]] = None,
    allowed_sample_types: Optional[Sequence[str]] = None,
    deduplicate_sample_ids: bool = True,
) -> Dataset:
    """Build a metadata-only dataset by joining indices and selecting basic features.

    Features:
      - SampleType (categorical)
      - Subject covariates (Gender, Ethnicity, Adj.age, BMI, FPG_Mean, SSPG, FPG, OGTT) if available

    Targets:
      - All numeric cytokines in cytokine_profiles.csv (auto-inferred)

    Args:
        preferred_sample_types: Optional ordering that picks which SampleType to keep when a SampleID
            has multiple entries. Earlier values win; unspecified types are used last.
        allowed_sample_types: Optional whitelist; records outside the list are discarded before
            deduplication.
        deduplicate_sample_ids: When True (default) enforce one row per SampleID after filters.
    """
    train_df, cytok_df, subjects_df = load_indices()

    # Join train to cytokines via SampleID
    df = train_df.merge(cytok_df, on="SampleID", how="inner", validate="m:1")
    # Join subject covariates
    df = df.merge(subjects_df, on="SubjectID", how="left", validate="m:1")

    # Identify targets
    target_cols = _infer_target_columns(cytok_df)
    if not target_cols:
        raise ValueError("No numeric cytokine targets found.")

    # Feature columns
    base_cat = ["SampleType"]
    subj_cat = ["Gender", "Ethnicity"] if include_subject_covariates else []
    cat_cols = [c for c in base_cat + subj_cat if c in df.columns]

    subj_num = [
        "Adj.age",
        "BMI",
        "FPG_Mean",
        "SSPG",
        "FPG",
        "OGTT",
    ] if include_subject_covariates else []
    num_cols = [c for c in subj_num if c in df.columns]

    # Drop rows with all-targets missing
    df_targets = df[target_cols]
    mask_any_target = df_targets.notna().any(axis=1)
    df = df.loc[mask_any_target].copy()

    if deduplicate_sample_ids:
        df = _deduplicate_sample_records(
            df,
            preferred_sample_types=preferred_sample_types,
            allowed_sample_types=allowed_sample_types,
        )
        if df.empty:
            raise ValueError("No records remain after applying sample type filters.")

    # Prepare transformers
    transformers = []
    feature_names: List[str] = []

    if num_cols:
        num_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(("num", num_pipe, num_cols))
        feature_names.extend(num_cols)

    if cat_cols:
        cat_pipe = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "ohe",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                ),
            ]
        )
        transformers.append(("cat", cat_pipe, cat_cols))
        # Placeholder names; will be expanded after fit
        feature_names.extend(cat_cols)

    if not transformers:
        raise ValueError("No features found. Enable subject covariates or add feature extraction.")

    ct = ColumnTransformer(transformers=transformers, remainder="drop")

    # Targets (y)
    y = df[target_cols].to_numpy(dtype=float)
    # Groups for GroupKFold
    groups = df["SubjectID"].astype(str).to_numpy()
    sample_ids = df["SampleID"].astype(str).to_numpy()

    # Fit transformer to get X and expanded feature names
    X = ct.fit_transform(df)

    # Compute final feature names after OneHot expansion
    out_feature_names: List[str] = []
    for name, trans, cols in ct.transformers_:
        if name == "remainder":
            continue
        if hasattr(trans, "named_steps") and "ohe" in trans.named_steps:
            ohe: OneHotEncoder = trans.named_steps["ohe"]
            # Prepend original column names to categories for clarity
            cat_feature_names = []
            for col, cats in zip(cols, ohe.categories_):
                cat_feature_names.extend([f"{col}__{c}" for c in cats])
            out_feature_names.extend(cat_feature_names)
        else:
            out_feature_names.extend(cols)

    return Dataset(
        X=X,
        y=y,
        feature_names=out_feature_names,
        target_names=target_cols,
        groups=groups,
        sample_ids=sample_ids,
        index=df.index,
        transformer=ct,
    )
