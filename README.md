# Filtered-Classifier Phishing Detection

A Python implementation of FilteredClassifier-style phishing website detection using scikit-learn pipelines. The project evaluates three configurations on the UCI Phishing Websites Dataset (dataset ID 327) and compares a pass-through filter and top-15 mutual-information feature selection with J48-style decision trees and a random forest.

## Models

| Pipeline | Filter | Classifier |
| :--- | :--- | :--- |
| `FC_Default_J48` | Identity pass-through | Decision tree (`criterion="entropy"`) |
| `FC_InfoGain15_J48` | Mutual-information feature selection (`k=15`) | Decision tree (`criterion="entropy"`) |
| `FC_InfoGain15_RF` | Mutual-information feature selection (`k=15`) | Random forest (100 trees) |

The implementation treats the original UCI target values as binary classes: `-1` is mapped to phishing (`0`) and `1` is mapped to legitimate (`1`).

## Results

The saved benchmark in [`results/metrics_summary.csv`](results/metrics_summary.csv) reports 10-fold stratified cross-validation and an 80/20 stratified holdout split.

| Model | 10-Fold Accuracy | 10-Fold MCC | Holdout Accuracy | Holdout Precision | Holdout Recall | Holdout F1 | Holdout MCC | Test Latency |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `FC_Default_J48` | 94.25% | 0.8835 | 94.75% | 95.21% | 95.37% | 95.29% | 0.8937 | 0.0048 s |
| `FC_InfoGain15_J48` | 93.75% | 0.8734 | 94.21% | 95.02% | 94.56% | 94.79% | 0.8828 | 0.0054 s |
| `FC_InfoGain15_RF` | **96.35%** | **0.9260** | **96.79%** | **96.55%** | **97.73%** | **97.13%** | **0.9349** | 0.1133 s |

Metrics are empirical results for the fixed random seeds used in `src/train.py`; rerunning the benchmark may produce slightly different timing values.

## Dataset

The pipeline downloads the UCI Phishing Websites Dataset through [`ucimlrepo`](https://archive.ics.uci.edu/dataset/327/phishing+websites), then performs local preprocessing and evaluation.

- Dataset ID: `327`
- Instances: 11,055
- Source features: 30
- Selected features: 15
- URL/DOM feature values: `-1`, `0`, and `1`
- Target values: `-1` (phishing) and `1` (legitimate)

An internet connection is required on the first run to fetch the dataset.

## Setup

Python 3.9 or newer is recommended.

```bash
git clone https://github.com/maithilimukherjee/Filtered-Classifier-PhishingDetection.git
cd Filtered-Classifier-PhishingDetection

python -m venv .venv
source .venv/bin/activate
# On Windows: .venv\Scripts\activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the Benchmark

Run the benchmark from the repository root:

```bash
python src/train.py
```

This command:

- downloads the UCI dataset;
- runs 10-fold stratified cross-validation;
- evaluates each pipeline on an 80/20 stratified holdout split;
- writes `results/metrics_summary.csv`;
- ranks the 30 features in `results/infogain_feature_ranking.csv`; and
- saves the best pipeline as `models/fc_infogain15_rf.joblib`.

Generate plots and the detailed holdout classification report after training:

```bash
python src/evaluate.py
```

The evaluation script writes:

- `results/confusion_matrix.png`
- `results/infogain_top15.png`

## Tests

Run the test suite with:

```bash
pytest
```

The test verifies that the mutual-information filter reduces 30 dummy features to 15 and that the random-forest pipeline predicts only binary classes.

## Repository Structure

```text
.
├── models/
│   └── .gitkeep
├── results/
│   ├── infogain_feature_ranking.csv
│   ├── metrics_summary.csv
│   └── metrics.py
├── src/
│   ├── evaluate.py
│   └── train.py
├── tests/
│   └── test_pipeline.py
├── .gitignore
├── pytest.ini
├── requirements.txt
└── README.md
```

## Reproducibility

- The cross-validation split and holdout split use `random_state=42`.
- Mutual-information scoring uses `random_state=42` where supported.
- Decision-tree and random-forest estimators use `random_state=42`.
- The random forest uses 100 estimators and parallel execution (`n_jobs=-1`).

## Generated Files

The following files are generated locally by `src/train.py` or `src/evaluate.py`:

- `models/fc_infogain15_rf.joblib`
- `results/metrics_summary.csv`
- `results/infogain_feature_ranking.csv`
- `results/confusion_matrix.png`
- `results/infogain_top15.png`
