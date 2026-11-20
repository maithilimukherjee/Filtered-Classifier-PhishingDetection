from pathlib import Path
from typing import Dict, Optional, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
    confusion_matrix,
    make_scorer,
)

RESULTS_DIR = Path(__file__).resolve().parent
DEFAULT_SUMMARY_CSV = RESULTS_DIR / "metrics_summary.csv"
DEFAULT_OUTPUT_CSV = RESULTS_DIR / "literature_comparison.csv"

# Published empirical benchmarks from the Literature Review (UCI Dataset 1: ~11,055 samples, 30 features)
LITERATURE_BASELINES = [
    {
        "Model": "[Ref] Alazaidah et al. (2024) - WEKA FilteredClassifier / J48",
        "Test_Accuracy (%)": 90.76,
        "Test_Precision (%)": None,
        "Test_Recall (%)": None,
        "Test_F1 (%)": None,
        "Test_MCC": 0.8340,
        "Test_Latency (s)": None,
    },
    {
        "Model": "[Ref] Alazaidah et al. (2024) - InfoGain (15 feat) + RF",
        "Test_Accuracy (%)": 96.58,
        "Test_Precision (%)": None,
        "Test_Recall (%)": None,
        "Test_F1 (%)": None,
        "Test_MCC": None,
        "Test_Latency (s)": None,
    },
    {
        "Model": "[Ref] Shahrivari et al. (2020) - Random Forest (30 feat)",
        "Test_Accuracy (%)": 97.27,
        "Test_Precision (%)": None,
        "Test_Recall (%)": None,
        "Test_F1 (%)": None,
        "Test_MCC": None,
        "Test_Latency (s)": 0.0219,
    },
    {
        "Model": "[Ref] Shahrivari et al. (2020) - XGBoost (30 feat)",
        "Test_Accuracy (%)": 98.32,
        "Test_Precision (%)": 98.72,
        "Test_Recall (%)": 98.10,
        "Test_F1 (%)": 97.68,
        "Test_MCC": None,
        "Test_Latency (s)": 0.0062,
    },
]


def get_cv_scoring() -> Dict[str, object]:
    """Return standardized scoring dictionary for 10-fold cross-validation."""
    return {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "mcc": make_scorer(matthews_corrcoef),
        "roc_auc": "roc_auc",
    }


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    train_time: Optional[float] = None,
    test_latency: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    """Compute full binary classification, security error-rate, and latency metrics."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # Phishing misclassified as Legitimate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # Legitimate blocked as Phishing

    return {
        "Accuracy (%)": round(accuracy_score(y_true, y_pred) * 100, 2),
        "Precision (%)": round(precision_score(y_true, y_pred, zero_division=0) * 100, 2),
        "Recall (%)": round(recall_score(y_true, y_pred, zero_division=0) * 100, 2),
        "F1 (%)": round(f1_score(y_true, y_pred, zero_division=0) * 100, 2),
        "MCC": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "ROC_AUC": round(float(roc_auc_score(y_true, y_prob)), 4) if y_prob is not None else None,
        "FPR (%)": round(fpr * 100, 2),
        "FNR (%)": round(fnr * 100, 2),
        "TP": int(tp),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "Train_Time (s)": round(train_time, 4) if train_time is not None else None,
        "Test_Latency (s)": round(test_latency, 4) if test_latency is not None else None,
    }


def generate_literature_comparison(
    summary_csv: Union[str, Path] = DEFAULT_SUMMARY_CSV,
    output_csv: Union[str, Path] = DEFAULT_OUTPUT_CSV,
) -> pd.DataFrame:
    """Merge local FilteredClassifier runs with published literature baselines."""
    summary_path = Path(summary_csv)
    output_path = Path(output_csv)

    if not summary_path.exists():
        raise FileNotFoundError(
            f"'{summary_path}' not found. Run 'python src/train.py' first to generate model results."
        )

    local_df = pd.read_csv(summary_path)
    cols = [
        "Model",
        "Test_Accuracy (%)",
        "Test_Precision (%)",
        "Test_Recall (%)",
        "Test_F1 (%)",
        "Test_MCC",
        "Test_Latency (s)",
    ]

    combined_df = pd.concat([local_df[cols].copy(), pd.DataFrame(LITERATURE_BASELINES)[cols]], ignore_index=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_df.to_csv(output_path, index=False)
    return combined_df


if __name__ == "__main__":
    comparison = generate_literature_comparison()
    print("=== Local FilteredClassifier vs. Literature Benchmarks ===")
    print(comparison.fillna("-").to_string(index=False))
    print(f"\nSaved comparison table to {DEFAULT_OUTPUT_CSV}")