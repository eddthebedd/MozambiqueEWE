import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, precision_score

# SUPERLEARNERWIP
# Trying to figure out how to initiate XGBoost 

df = pd.read_csv('vaccine_weather_features.csv')

# Define predictors and target
features = [
    'tp', 'temperature_avg', 'rh', 'season_rainy',
    'Rota1_lag2', 'Rota2_lag1', 'completion_rate',
    'rota_data_missing', 'weather_data_missing'
]
target = 'vaccine_drop_flag'

X = df[features]
y = df[target]

# Impute any remaining missing values with 0
X = X.fillna(0)

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    stratify=y,
    random_state=42
)

# Scale numeric features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define base learners
base_learners = [
    ('lr', LogisticRegression(max_iter=500)),
    ('rf', RandomForestClassifier(n_estimators=200, random_state=42)),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)),
    ('nb', GaussianNB())
]

# Define meta-learner
meta_learner = LogisticRegression(max_iter=500)

# Build Super Learner (StackingClassifier)
super_learner = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=StratifiedKFold(n_splits=5),
    passthrough=True
)

# Fit model
super_learner.fit(X_train_scaled, y_train)

# Predict probabilities
y_pred_proba = super_learner.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

# Evaluate
roc_auc = roc_auc_score(y_test, y_pred_proba)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print("Super Learner Evaluation:")
print(f"ROC-AUC: {roc_auc:.3f}")
print(f"Accuracy: {accuracy:.3f}")
print(f"F1-score: {f1:.3f}")
print(f"Recall: {recall:.3f}")
print(f"Precision: {precision:.3f}")
