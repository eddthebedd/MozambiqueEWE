import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score


def build_super_learner(seed: int = 42) -> StackingClassifier:
    """
    Returns an UNFITTED Super Learner (StackingClassifier) with:
      - Base learners: LogisticRegression, RandomForest, XGBClassifier, GaussianNB
      - Meta-learner: LogisticRegression
    Passthrough=True so meta-learner sees original features + base learner outputs.

    Note: This function does NOT read data and does NOT fit the model.
    """
    base_learners = [
        ('lr', LogisticRegression(max_iter=500)),
        ('rf', RandomForestClassifier(n_estimators=200, random_state=seed)),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=seed)),
        ('nb', GaussianNB())
    ]

    meta_learner = LogisticRegression(max_iter=500)

    clf = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=seed),
        passthrough=True
    )
    return clf


# ----------------------------------------------------------------------
# Optional demo block so `python3 super_learner.py` runs without errors.
# This DOES NOT affect evaluate_model.py (which imports and fits the model).
# ----------------------------------------------------------------------
if __name__ == "__main__":
    print("Running standalone demo fit for super_learner.py ...")

    df = pd.read_csv("vaccine_weather_features.csv")

    # Filter to labeled rows to avoid y NaN errors
    if "vaccine_drop_flag" not in df.columns:
        raise RuntimeError("Expected 'vaccine_drop_flag' in vaccine_weather_features.csv")

    df = df[df["vaccine_drop_flag"].notna()].copy()
    df["vaccine_drop_flag"] = df["vaccine_drop_flag"].astype(int)

    # Default feature set (match evaluate_model.py)
    base_features = [
        'tp', 'temperature_avg', 'rh',
        'Rota1_lag2', 'Rota2_lag1', 'completion_rate',
        'rota_data_missing', 'weather_data_missing'
    ]
    # Include weather lags if present
    lag_features = [c for c in [
        'tp_lag1','tp_lag2','temperature_avg_lag1','temperature_avg_lag2','rh_lag1','rh_lag2'
    ] if c in df.columns]

    features = [c for c in base_features + lag_features if c in df.columns]

    X = df[features].fillna(0)
    y = df["vaccine_drop_flag"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    model = build_super_learner(seed=42)
    model.fit(X_train, y_train)

    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    print("Super Learner Evaluation (demo split):")
    print(f"ROC-AUC:   {roc_auc_score(y_test, y_proba):.3f}")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
    print(f"F1-score:  {f1_score(y_test, y_pred):.3f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
    print(f"Precision: {precision_score(y_test, y_pred):.3f}")
    print("Demo fit OK.")
