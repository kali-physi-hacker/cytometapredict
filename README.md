# CytoMetaPredict
This is basically a model to predict cytokine levels given biome or metagenomic data. This project is hackerton from [Zindi Africa](https://zindi.africa/).
All data used for training is provided by Zindi Africa and can be found on the competition page. 


## Installation and Setup
This project uses conda for package management. To set up the environment, follow these steps:
The `environment.yml` file contains the necessary dependencies for this project. You can create a conda environment using the following command:

```bash
conda env create -f environment.yml
```

**NB**: Make sure you have conda installed on your system. If you don't have it, you can download and install Anaconda or Miniconda.

## Quickstart: Metadata‑Only Baseline

This repo includes a minimal, reproducible baseline that predicts cytokine levels using sample type and subject covariates (no `.mgb` decoding yet). It performs GroupKFold validation grouped by `SubjectID` and writes per‑cytokine metrics.

1) Activate the environment

```bash
conda env create -f environment.yml
conda activate cytometapredict
```

2) Run the baseline training

```bash
python -m src.train_baseline --model random_forest --n_splits 5
```

Outputs:
- `reports/baseline_metrics_random_forest.csv` – per‑fold RMSE/MAE per cytokine and macro average
- `models/baseline/` – one model per fold with the fitted feature transformer

Notes:
- Targets are inferred from numeric columns in `data/cytokine_profiles.csv` (non‑target metadata like `Plate`, `CollectionDate`, `CL1–CL4` are ignored).
- Features include: `SampleType`, `Gender`, `Ethnicity`, and numeric subject covariates (`Adj.age`, `BMI`, `FPG_Mean`, `SSPG`, `FPG`, `OGTT`) when available, with imputation and one‑hot/standard scaling.
- Predictions are clipped to non‑negative and evaluated on the original scale.

## Next Steps (Optional)

- Implement `.mgb` decoding to derive microbiome features (taxonomic/functional), then extend `src/features` to merge them into the dataset.
- Try `ridge` baseline: `--model ridge`.
- Add hyperparameter search and feature importance reporting.

## End‑to‑End Pipeline

This section outlines the full process to build, validate, and extend a cytokine prediction model from the data in this repo.

**Data Layout**
- `data/Train.csv`: maps `.mgb` files to `SampleID`, `SampleType`, and `SubjectID`.
- `data/cytokine_profiles.csv`: targets (cytokines) keyed by `SampleID`; includes `Plate`, `CollectionDate`, `CL1–CL4`.
- `data/Train_Subjects.csv`: subject covariates keyed by `SubjectID` (demographics, metabolic markers).
- `data/TrainFiles/*.mgb`, `data/ViomeData/*.mgb`: metagenomic binaries (decoding required, see `docs/MPEGG_Track1_PRD.pdf`).

**Project Structure**
- `src/utils/io.py`: CSV loaders and directory helpers.
- `src/features/build_metadata_features.py`: joins indices and builds metadata features/targets.
- `src/models/baseline.py`: baseline model factory (RandomForest, Ridge).
- `src/train_baseline.py`: GroupKFold training, metrics, artifacts.
- `reports/`: metrics outputs; `models/`: saved models and transformers.

**1) Environment Setup**
- Create and activate env: `conda env create -f environment.yml && conda activate cytometapredict`.
- Ensure `data/` contains the CSVs and `.mgb` files as listed above.

**2) Decode `.mgb` Files**
- Use the helper CLI to convert MPEG-G archives into per-sample folders (supports parallelism and progress bars):
  ```bash
  python -m src.pipeline.decode_mgb \
      --decoder-bin genie \
      --command-template "{decoder} run -i {input} -o {output}" \
      --mgb-dir data/TrainFiles \
      --output-root data/decoded \
      --manifest reports/decode_manifest.jsonl \
      --workers 4 --skip-existing
  ```
- Outputs land under `data/decoded/<filename>/` with per-sample FASTQ files (override the filename pattern with `--output-name-template`, e.g., `{stem}.fastq.gz`).
- Install `tqdm` inside the environment for richer progress bars (optional). Add `--no-progress` to suppress progress output in CI logs.

**3) Trim FASTQ Files (Optional but Recommended)**
- Once decoding is complete, run the trimming CLI to remove adapters or low-quality tails (defaults to `fastp` but any tool can be wrapped via the template):
  ```bash
  python -m src.pipeline.trim_fastq \
      --tool-bin fastp \
      --command-template "{tool} -i {input} -o {output} --trim_front1 5 --length_required 50" \
      --fastq-root data/decoded \
      --output-root data/trimmed \
      --manifest reports/trim_manifest.jsonl \
      --workers 4 --skip-existing
  ```
- Customize `--command-template` with the flags your trimmer needs; outputs default to `data/trimmed/<stem>/<stem>_trimmed.fastq.gz`. Set `--output-name-template` to change the filename and add `--no-progress` for quiet logs.

**4) Data Audit (Recommended)**
- Check join keys: `Train.csv.SampleID` ↔ `cytokine_profiles.csv.SampleID`, `Train.csv.SubjectID` ↔ `Train_Subjects.csv.SubjectID`.
- Inspect target distributions; consider log1p transform for heavy tails (baseline already applies this internally).
- Note multiple `.mgb` per `SubjectID` and across body sites; plan aggregation policy for future feature extraction.

**5) Train Baseline (Metadata-Only)**
- Run CV training (grouped by `SubjectID`): `python -m src.train_baseline --model random_forest --n_splits 5`.
- Outputs:
  - Metrics CSV: `reports/baseline_metrics_random_forest.csv`.
  - Fold models: `models/baseline/baseline_random_forest_fold*.joblib` (includes fitted transformer and metadata).
- Alternatives: `--model ridge` (linear baseline) or adjust `--n_splits`.

**6) Evaluate Results**
- Open the metrics CSV and review per‑cytokine and macro averages.
- Compare against a null or simple baseline (e.g., per‑cytokine mean) to quantify signal from metadata.
- Validate grouping: ensure subjects don’t leak across folds (already enforced by GroupKFold).

**7) Extend With `.mgb`‑Derived Microbiome Features (Optional)**
- Decode `.mgb` to analysis‑ready data using official MPEG‑G tooling (see `docs/MPEGG_Track1_PRD.pdf`). Common paths:
  - Taxonomic abundances (species/genus) per sample.
  - Functional profiles (pathways/KO/EC) per sample.
- Write exported feature tables to `data/features/` (suggested):
  - `data/features/taxonomic_abundance.csv` with columns: `filename`, `feature`, `value` (long) or wide matrix keyed by `filename`.
  - `data/features/functional_abundance.csv` similarly keyed.
- Extend `src/features/` with a loader that:
  - Merges exported features to `Train.csv` via `filename`.
  - Aggregates across body sites per `SampleID` (e.g., mean/median) or keeps site‑specific prefixes.
  - Applies compositional transforms (log, CLR) and scaling; one‑hot encodes `SampleType`.
- Update `src/train_baseline.py` (or add a new trainer) to include these features alongside metadata and rerun CV.

**8) Modeling Upgrades (Optional)**
- Try gradient‑boosted trees (LightGBM/XGBoost) for stronger baselines.
- Use multi‑task learning (`MultiOutputRegressor` wrapper already supported) or PLS for correlated cytokines.
- Add feature importance/SHAP for interpretability; prune features via variance/importance thresholds.

**9) Reproducibility**
- Grouped CV by `SubjectID` prevents leakage.
- Seeds: use `--seed` to control model seeds; artifacts include fitted transformers for exact replay.
- Persist outputs: models in `models/`, metrics in `reports/` with timestamps or model tags as needed.

**10) Inference (Conceptual)**
- Load a saved fold model and its transformer, transform input features with the same pipeline, predict, and invert log if used.
- Example (Python snippet):

```python
import joblib, numpy as np, pandas as pd

bundle = joblib.load('models/baseline/baseline_random_forest_fold1.joblib')
model = bundle['model']
transformer = bundle['transformer']

# df_new should contain the same feature columns used by the transformer
df_new = pd.DataFrame([
    {"SampleType": "Stool", "Gender": "F", "Ethnicity": "C", "Adj.age": 55.0,
     "BMI": 28.0, "FPG_Mean": 0.95, "SSPG": 120.0, "FPG": 95.0, "OGTT": 1.1}
])
X_new = transformer.transform(df_new)
pred = model.predict(X_new)
pred = np.clip(np.expm1(pred), a_min=0, a_max=None)  # invert log1p, clip negatives
```

If you want, we can add a small CLI (`src/predict_baseline.py`) to batch this over a CSV in the same schema.

**11) Quality Checks**
- Temporal leakage: avoid using `CollectionDate` or infection phase labels (`CL1–CL4`) as predictive features unless carefully justified.
- Site bias: evaluate per `SampleType` performance; consider site‑specific models or ensembling.
- Scaling: ensure consistent transforms between train and inference via saved transformers.
