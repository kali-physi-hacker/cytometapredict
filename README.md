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
