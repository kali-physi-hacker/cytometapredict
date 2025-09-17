from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GroupKFold

from src.features.build_metadata_features import make_metadata_dataset, Dataset
from src.models.baseline import build_model, BaselineConfig
from src.utils.io import ensure_dirs


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def evaluate_fold(y_true: np.ndarray, y_pred: np.ndarray, target_names: List[str]) -> pd.DataFrame:
    rows = []
    for i, name in enumerate(target_names):
        yt = y_true[:, i]
        yp = y_pred[:, i]
        rows.append({
            "cytokine": name,
            "rmse": rmse(yt, yp),
            "mae": float(mean_absolute_error(yt, yp)),
        })
    df = pd.DataFrame(rows)
    df.loc["macro_avg"] = {
        "cytokine": "macro_avg",
        "rmse": df["rmse"].mean(),
        "mae": df["mae"].mean(),
    }
    return df


def main():
    parser = argparse.ArgumentParser(description="Train baseline cytokine predictor (metadata-only)")
    parser.add_argument("--model", choices=["random_forest", "ridge"], default="random_forest")
    parser.add_argument("--n_splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="reports")
    parser.add_argument("--models_dir", type=str, default="models/baseline")
    args = parser.parse_args()

    ensure_dirs(args.outdir, args.models_dir)

    # Build dataset
    dataset: Dataset = make_metadata_dataset(include_subject_covariates=True)

    X, y, groups = dataset.X, dataset.y, dataset.groups
    target_names = dataset.target_names

    # Optionally apply log1p to targets to stabilize heavy tails
    # Keep transform simple (apply across all targets)
    apply_log = True
    if apply_log:
        y_trans = np.log1p(np.clip(y, a_min=0, a_max=None))
    else:
        y_trans = y

    # GroupKFold by SubjectID
    unique_groups = np.unique(groups)
    n_splits = min(args.n_splits, len(unique_groups)) if len(unique_groups) >= 2 else 2
    gkf = GroupKFold(n_splits=n_splits)

    fold_metrics = []
    oof_pred = np.zeros_like(y_trans)

    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X, y_trans, groups=groups), start=1):
        model = build_model(BaselineConfig(model_type=args.model, random_state=args.seed + fold))
        model.fit(X[tr_idx], y_trans[tr_idx])
        pred_log = model.predict(X[va_idx])
        # Invert log if applied
        if apply_log:
            pred = np.expm1(pred_log)
        else:
            pred = pred_log

        # Clip negative predictions (physically non-negative cytokines)
        pred = np.clip(pred, a_min=0, a_max=None)

        # Evaluate
        fold_df = evaluate_fold(y_true=y[va_idx], y_pred=pred, target_names=target_names)
        fold_df.insert(0, "fold", fold)
        fold_metrics.append(fold_df)

        # Save model per fold
        model_path = Path(args.models_dir) / f"baseline_{args.model}_fold{fold}.joblib"
        joblib.dump({
            "model": model,
            "transformer": dataset.transformer,
            "target_names": target_names,
            "feature_names": dataset.feature_names,
            "apply_log": apply_log,
        }, model_path)

    metrics_df = pd.concat(fold_metrics, ignore_index=True)
    metrics_csv = Path(args.outdir) / f"baseline_metrics_{args.model}.csv"
    metrics_df.to_csv(metrics_csv, index=False)

    # Summary
    summary = (
        metrics_df[metrics_df["cytokine"] == "macro_avg"]["rmse"].mean(),
        metrics_df[metrics_df["cytokine"] == "macro_avg"]["mae"].mean(),
    )
    print(json.dumps({"macro_rmse": summary[0], "macro_mae": summary[1], "n_splits": n_splits}))
    print(f"Saved metrics to {metrics_csv}")


if __name__ == "__main__":
    main()

