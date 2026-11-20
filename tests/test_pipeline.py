import numpy as np
import pandas as pd
from src.train import build_pipelines


def test_infogain_50_percent_reduction():
    rng = np.random.default_rng(42)
    X_dummy = pd.DataFrame(
        rng.choice([-1, 0, 1], size=(80, 30)),
        columns=[f"feat_{i}" for i in range(30)],
    )
    y_dummy = rng.integers(0, 2, size=80)

    pipelines = build_pipelines()
    fc_rf = pipelines["FC_InfoGain15_RF"]
    fc_rf.fit(X_dummy, y_dummy)

    X_transformed = fc_rf.named_steps["filter"].transform(X_dummy)
    assert X_transformed.shape == (80, 15), "InfoGain filter must reduce 30 features to 15."

    preds = fc_rf.predict(X_dummy)
    assert preds.shape == (80,)
    assert set(np.unique(preds)).issubset({0, 1})