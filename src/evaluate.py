import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, classification_report

os.makedirs("results", exist_ok=True)


def generate_plots():
    # 1. Load model and dataset using the exact holdout split from train.py
    model_path = "models/fc_infogain15_rf.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError("Run 'python src/train.py' first to generate the saved model.")

    pipe = joblib.load(model_path)
    dataset = fetch_ucirepo(id=327)
    X = dataset.data.features.copy()
    y = (dataset.data.targets.values.ravel() == 1).astype(int)

    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    y_pred = pipe.predict(X_test)

    print("=== Detailed Holdout Classification Report ===")
    print(classification_report(y_test, y_pred, target_names=["Phishing (0)", "Legitimate (1)"]))

    # 2. Plot & Save Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,
        y_pred,
        display_labels=["Phishing", "Legitimate"],
        cmap="Blues",
        ax=ax,
        colorbar=False,
    )
    ax.set_title("FilteredClassifier (InfoGain Top-15 + RF) — Confusion Matrix")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=300)
    plt.close(fig)

    # 3. Plot & Save Top-15 InfoGain Features (50% Dimensionality Reduction)
    ranking_path = "results/infogain_feature_ranking.csv"
    if os.path.exists(ranking_path):
        df_rank = pd.read_csv(ranking_path).head(15).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(df_rank["Feature"], df_rank["InfoGain_Score"], color="#2b5c8f")
        ax.set_xlabel("Information Gain (Mutual Information Score)")
        ax.set_title("Top 15 Selected URL & DOM Attributes (50% Reduction)")
        plt.tight_layout()
        plt.savefig("results/infogain_top15.png", dpi=300)
        plt.close(fig)

    print("Saved plots to results/confusion_matrix.png and results/infogain_top15.png")


if __name__ == "__main__":
    generate_plots()