# Phishing Website Detection — FilteredClassifier & InfoGain Benchmark

A leak-free `scikit-learn` pipeline implementation of the WEKA `FilteredClassifier` architecture evaluated on the UCI Phishing Websites Dataset (11,055 instances, 30 URL/DOM features), benchmarking baseline decision trees against 50% Information Gain dimensionality reduction.

## Evaluated Pipelines
1. **`FC_Default_J48`**: Identity pass-through filter paired with an entropy-based pruned Decision Tree (C4.5 / J48 equivalent).
2. **`FC_InfoGain15_J48`**: `SelectKBest` (discrete Mutual Information / InfoGain, $k=15$) paired with a J48 Decision Tree.
3. **`FC_InfoGain15_RF`**: `SelectKBest` (discrete Mutual Information / InfoGain, $k=15$) paired with a 100-tree Random Forest classifier.

## Setup & Execution

```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/train.py