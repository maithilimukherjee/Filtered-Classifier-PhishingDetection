import os
import time
from functools import partial
import joblib
import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    make_scorer,
)

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)


def load_data():
    """Fetch UCI Phishing Websites Dataset (ID 327: 11,055 samples, 30 features)."""
    dataset = fetch_ucirepo(id=327)
    X = dataset.data.features.copy()
    # Map {-1: Phishing, 1: Legitimate} -> {0: Phishing, 1: Legitimate}
    y = (dataset.data.targets.values.ravel() == 1).astype(int)
    return X, y


def build_pipelines():
    """Construct FilteredClassifier equivalents using scikit-learn Pipelines."""
    exact_infogain = partial(mutual_info_classif, discrete_features=True, random_state=42)

    return {
        # 1. Default WEKA FilteredClassifier (Pass-through Discretize + J48 / C4.5 Tree)
        "FC_Default_J48": Pipeline([
            ("filter", FunctionTransformer(validate=False)),
            ("classifier", DecisionTreeClassifier(
                criterion="entropy",
                ccp_alpha=0.001,
                random_state=42
            )),
        ]),
        # 2. InfoGain (Top 15 / 50% Reduction) + J48 Decision Tree
        "FC_InfoGain15_J48": Pipeline([
            ("filter", SelectKBest(score_func=exact_infogain, k=15)),
            ("classifier", DecisionTreeClassifier(
                criterion="entropy",
                ccp_alpha=0.001,
                random_state=42
            )),
        ]),
        # 3. InfoGain (Top 15 / 50% Reduction) + Random Forest
        "FC_InfoGain15_RF": Pipeline([
            ("filter", SelectKBest(score_func=exact_infogain, k=15)),
            ("classifier", RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )),
        ]),
    }


def main():
    X, y = load_data()
    pipelines = build_pipelines()

    # --- Phase 1: 10-Fold Stratified Cross-Validation (WEKA Protocol) ---
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "mcc": make_scorer(matthews_corrcoef),
    }

    print("Running 10-Fold Stratified Cross-Validation...")
    cv_records = []
    for name, pipe in pipelines.items():
        scores = cross_validate(pipe, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        cv_records.append({
            "Model": name,
            "CV_Accuracy (%)": round(scores["test_accuracy"].mean() * 100, 2),
            "CV_Precision (%)": round(scores["test_precision"].mean() * 100, 2),
            "CV_Recall (%)": round(scores["test_recall"].mean() * 100, 2),
            "CV_F1 (%)": round(scores["test_f1"].mean() * 100, 2),
            "CV_MCC": round(scores["test_mcc"].mean(), 4),
        })

    cv_df = pd.DataFrame(cv_records)
    print("\n=== 10-Fold Cross-Validation Summary ===")
    print(cv_df.to_string(index=False))

    # --- Phase 2: 80/20 Holdout Split + Latency Profiling ---
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    holdout_records = []
    for name, pipe in pipelines.items():
        t0 = time.perf_counter()
        pipe.fit(X_train, y_train)
        train_time = time.perf_counter() - t0

        t1 = time.perf_counter()
        y_pred = pipe.predict(X_test)
        test_time = time.perf_counter() - t1

        holdout_records.append({
            "Model": name,
            "Test_Accuracy (%)": round(accuracy_score(y_test, y_pred) * 100, 2),
            "Test_Precision (%)": round(precision_score(y_test, y_pred) * 100, 2),
            "Test_Recall (%)": round(recall_score(y_test, y_pred) * 100, 2),
            "Test_F1 (%)": round(f1_score(y_test, y_pred) * 100, 2),
            "Test_MCC": round(matthews_corrcoef(y_test, y_pred), 4),
            "Train_Time (s)": round(train_time, 4),
            "Test_Latency (s)": round(test_time, 4),
        })

    holdout_df = pd.DataFrame(holdout_records)
    print("\n=== 80/20 Holdout & Latency Summary ===")
    print(holdout_df.to_string(index=False))

    # Save metrics and best reduced-feature model
    summary_df = pd.merge(cv_df, holdout_df, on="Model")
    summary_df.to_csv("results/metrics_summary.csv", index=False)

    best_pipe = pipelines["FC_InfoGain15_RF"]
    mask = best_pipe.named_steps["filter"].get_support()
    scores_mi = best_pipe.named_steps["filter"].scores_

    feature_ranking = (
        pd.DataFrame({"Feature": X.columns, "InfoGain_Score": scores_mi, "Selected": mask})
        .sort_values(by="InfoGain_Score", ascending=False)
    )
    feature_ranking.to_csv("results/infogain_feature_ranking.csv", index=False)
    joblib.dump(best_pipe, "models/fc_infogain15_rf.joblib")

    print("\nSaved results to results/metrics_summary.csv and models/fc_infogain15_rf.joblib")


if __name__ == "__main__":
    main()