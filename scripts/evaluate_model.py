import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
)
import shap
from scipy.stats import chi2_contingency

from super_learner import build_super_learner

DF_PATH = "vaccine_weather_features.csv"
os.makedirs("outputs", exist_ok=True)

print(f"Loading: {DF_PATH}")
df = pd.read_csv(DF_PATH)
df = df.sort_values(by=["province", "year", "month"]).copy()

# ===== Targets =====
if "vaccine_drop_flag" not in df.columns:
    raise RuntimeError("vaccine_weather_features.csv must contain 'vaccine_drop_flag'.")

# Alternative thresholds from completion_rate MoM change, per province
for thresh, label in zip([0.05, 0.15, 0.20], ["drop_5", "drop_15", "drop_20"]):
    if "completion_rate" in df.columns:
        df[label] = (
            df.groupby("province")["completion_rate"]
              .transform(lambda x: (x.pct_change(fill_method=None) <= -thresh).astype(int))
        )
    else:
        df[label] = np.nan

# Labeled subset
df_model = df[df["vaccine_drop_flag"].notna()].copy()
df_model["vaccine_drop_flag"] = df_model["vaccine_drop_flag"].astype(int)

# Ensure cyclone flags exist
for c in ["cyclone_exposure", "cyclone_lag1", "cyclone_lag2"]:
    if c not in df_model.columns:
        df_model[c] = 0
df_model[["cyclone_exposure","cyclone_lag1","cyclone_lag2"]] = (
    df_model[["cyclone_exposure","cyclone_lag1","cyclone_lag2"]].fillna(0).astype(int)
)

# ===== Feature sets =====
# Full (vaccine + weather)
base_features = [
    'tp', 'temperature_avg', 'rh',
    'Rota1_lag2', 'Rota2_lag1', 'completion_rate',
    'rota_data_missing', 'weather_data_missing'
]
weather_lags = ['tp_lag1','tp_lag2','temperature_avg_lag1','temperature_avg_lag2','rh_lag1','rh_lag2']
features_full = [c for c in base_features + weather_lags if c in df_model.columns]

# Weather-only (strict): use ONLY lagged weather + weather missing flag
features_weather_only = [c for c in (weather_lags + ['weather_data_missing']) if c in df_model.columns]

print("\nFeature sets in use:")
print("  Full:         ", features_full)
print("  Weather-only: ", features_weather_only)

def evaluate_feature_set(name: str, feature_cols: list, target_col: str = "vaccine_drop_flag"):
    print(f"\n=== Evaluating {name} ===")
    X = df_model[feature_cols].fillna(0)
    y = df_model[target_col].astype(int)

    seeds = [0,1,2,3,4]
    k = 5
    fold_results = []
    for seed in seeds:
        kfold = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
        seed_scores = []
        for train_idx, test_idx in kfold.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = build_super_learner(seed)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            seed_scores.append({
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_proba),
                'recall': recall_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred)
            })
        avg = pd.DataFrame(seed_scores).mean()
        print(f"Seed {seed} - F1: {avg['f1']:.3f} | AUC: {avg['roc_auc']:.3f}")
        fold_results.append(avg)
    res = pd.DataFrame(fold_results).mean().rename(name)
    return res

# ===== Run both models (primary threshold) =====
full_metrics    = evaluate_feature_set("Full (vax+weather)", features_full, "vaccine_drop_flag")
weather_metrics = evaluate_feature_set("Weather-only", features_weather_only, "vaccine_drop_flag")

compare = pd.concat([full_metrics, weather_metrics], axis=1)
print("\n============================")
print("Full vs Weather-only (CV)")
print("============================")
print(compare.round(3))

# ===== Original sensitivity runs across thresholds (using FULL features) =====
targets = {
    'vaccine_drop_flag': '≥10% Drop (Primary)',
    'drop_5': '≥5% Drop',
    'drop_15': '≥15% Drop',
    'drop_20': '≥20% Drop'
}
summary = []
for target, label in targets.items():
    if target not in df_model.columns:
        print(f"\n[WARN] Target {target} not found; skipping.")
        continue
    print(f"\nEvaluating thresholds with FULL features: {label}")
    X = df_model[features_full].fillna(0)
    y = df_model[target].astype(int)

    results = []
    seeds = [0, 1, 2, 3, 4]
    for seed in seeds:
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        fold_scores = []
        for train_idx, test_idx in kfold.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            model = build_super_learner(seed)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]

            fold_scores.append({
                'accuracy': accuracy_score(y_test, y_pred),
                'f1': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_proba),
                'recall': recall_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred)
            })

        avg_metrics = pd.DataFrame(fold_scores).mean()
        print(f"Seed {seed} - F1: {avg_metrics['f1']:.3f} | AUC: {avg_metrics['roc_auc']:.3f}")
        results.append(avg_metrics)

    results_df = pd.DataFrame(results)
    print(f"\n[{label}] Avg Across Seeds (FULL):")
    print(results_df.mean())
    summary.append(results_df.mean().rename(label))

summary_df = pd.DataFrame(summary)
print("\n==========================")
print("Summary Across Thresholds (FULL features)")
print("==========================")
print(summary_df.T.round(3))

# ===== Full-data fit (FULL features) for SHAP + exports =====
print("\nFitting final model on ALL labeled rows (FULL features) for SHAP & export...")
y_full = df_model['vaccine_drop_flag'].astype(int)
X_full = df_model[features_full].fillna(0)

final_model = build_super_learner(seed=42)
final_model.fit(X_full, y_full)

y_pred_full = final_model.predict(X_full)
cm = confusion_matrix(y_full, y_pred_full)
ConfusionMatrixDisplay(cm).plot()
plt.title("Confusion Matrix: Primary Outcome")
plt.savefig("confusion_matrix.png")
plt.close()

print("\nCalculating SHAP values...")
try:
    explainer = shap.Explainer(final_model.named_estimators_["xgb"])
    shap_values = explainer(X_full)
    shap.summary_plot(shap_values, X_full, show=False)
    plt.savefig("shap_summary.png")
    plt.close()
except Exception as e:
    print(f"[WARN] SHAP skipped: {e}")

# ===== TP-based cyclone overlap export =====
print("\nWriting TP-based cyclone overlap export...")
id_cols = ["province", "year", "month"]
for c in ["cyclone_exposure", "cyclone_lag1", "cyclone_lag2"]:
    if c not in df_model.columns:
        df_model[c] = 0
df_model[["cyclone_exposure","cyclone_lag1","cyclone_lag2"]] = (
    df_model[["cyclone_exposure","cyclone_lag1","cyclone_lag2"]].fillna(0).astype(int)
)

y_proba = final_model.predict_proba(X_full)[:, 1]
pred_rows = df_model[id_cols + ["cyclone_exposure", "cyclone_lag1", "cyclone_lag2"]].copy()
pred_rows["actual"] = y_full.values
pred_rows["predicted"] = y_pred_full.astype(int)
pred_rows["proba"] = y_proba

pred_rows = pred_rows.sort_values(id_cols).copy()
pred_rows["cyclone_lead1"] = pred_rows.groupby("province")["cyclone_exposure"].shift(-1).fillna(0).astype(int)
pred_rows["cyclone_conc_or_lag1"] = ((pred_rows["cyclone_exposure"]==1) | (pred_rows["cyclone_lag1"]==1)).astype(int)
pred_rows["cyclone_pm1"] = ((pred_rows["cyclone_exposure"]==1) | (pred_rows["cyclone_lag1"]==1) | (pred_rows["cyclone_lead1"]==1)).astype(int)

pred_path = "outputs/predictions_with_cyclone.csv"
pred_rows.to_csv(pred_path, index=False)
print(f"Saved: {pred_path}")

tp = pred_rows[(pred_rows["actual"]==1) & (pred_rows["predicted"]==1)]
tp_total = len(tp)
def pct(x): return float(np.round(100.0 * x, 2))
metrics = {
    "tp_total": int(tp_total),
    "tp_with_concurrent_%": pct((tp["cyclone_exposure"]==1).mean()) if tp_total else np.nan,
    "tp_with_lag1_%": pct((tp["cyclone_lag1"]==1).mean()) if tp_total else np.nan,
    "tp_with_conc_or_lag1_%": pct((tp["cyclone_conc_or_lag1"]==1).mean()) if tp_total else np.nan,
    "tp_with_pm1_%": pct((tp["cyclone_pm1"]==1).mean()) if tp_total else np.nan,
    "exposure_concurrent_rate_%": pct((pred_rows["cyclone_exposure"]==1).mean()),
    "exposure_lag1_rate_%": pct((pred_rows["cyclone_lag1"]==1).mean()),
    "exposure_conc_or_lag1_rate_%": pct((pred_rows["cyclone_conc_or_lag1"]==1).mean()),
    "exposure_pm1_rate_%": pct((pred_rows["cyclone_pm1"]==1).mean()),
}
summary_path = "outputs/tp_cyclone_overlap_summary.txt"
with open(summary_path, "w") as f:
    f.write("TP-based cyclone co-occurrence summary\n")
    for k,v in metrics.items():
        f.write(f"{k}: {v}\n")
    f.write("\nConfusion matrix [ [TN, FP], [FN, TP] ]:\n")
    f.write(str(confusion_matrix(y_full, y_pred_full)))
print(f"Saved: {summary_path}")
print("TP-based co-occurrence (key):", {k:metrics[k] for k in [
    "tp_total","tp_with_concurrent_%","tp_with_lag1_%","tp_with_conc_or_lag1_%","tp_with_pm1_%"
]})

# ===== Missingness vs Cyclone diagnostic =====
print("\nRunning Missingness vs Cyclone diagnostic...")
def _norm_province(s):
    if pd.isna(s): return s
    import unicodedata
    s = str(s).lower().strip()
    s = ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    s = s.replace("cidade de maputo", "maputo city")
    s = s.replace("maputo cidade", "maputo city")
    s = s.replace("maputo provincia", "maputo province")
    return s

df_missing = df.copy()
if "province" in df_missing.columns:
    df_missing["province"] = df_missing["province"].apply(_norm_province)
for c in ["year","month"]:
    if c in df_missing.columns:
        df_missing[c] = df_missing[c].astype(int, errors="ignore")

df_missing["cyclone_exposure"] = df_missing.get("cyclone_exposure", 0)
df_missing = df_missing.sort_values(["province","year","month"]).copy()
df_missing["cyclone_lag1_diag"] = df_missing.groupby("province")["cyclone_exposure"].shift(1).fillna(0).astype(int)
df_missing["cyclone_lead1_diag"] = df_missing.groupby("province")["cyclone_exposure"].shift(-1).fillna(0).astype(int)
df_missing["cyclone_conc_or_lag1_diag"] = ((df_missing["cyclone_exposure"]==1) | (df_missing["cyclone_lag1_diag"]==1)).astype(int)
df_missing["cyclone_pm1_diag"] = ((df_missing["cyclone_exposure"]==1) |
                                 (df_missing["cyclone_lag1_diag"]==1) |
                                 (df_missing["cyclone_lead1_diag"]==1)).astype(int)

if "rota_data_missing" in df_missing.columns:
    df_missing["vaccine_missing_flag"] = df_missing["rota_data_missing"].fillna(0).astype(int)
elif "completion_rate" in df_missing.columns:
    df_missing["vaccine_missing_flag"] = df_missing["completion_rate"].isna().astype(int)
else:
    cand_cols = [c for c in df_missing.columns if c.lower().startswith("rota")]
    df_missing["vaccine_missing_flag"] = (
        df_missing[cand_cols].isna().any(axis=1).astype(int) if cand_cols else 0
    )

def _rates_and_test(flag_col, exposure_col):
    tab = pd.crosstab(df_missing[flag_col], df_missing[exposure_col]).reindex(index=[0,1], columns=[0,1], fill_value=0)
    rate_exp   = tab.loc[1,1] / (tab.loc[0,1] + tab.loc[1,1]) if (tab.loc[0,1]+tab.loc[1,1])>0 else np.nan
    rate_unexp = tab.loc[1,0] / (tab.loc[0,0] + tab.loc[1,0]) if (tab.loc[0,0]+tab.loc[1,0])>0 else np.nan
    chi2, p, dof, _ = chi2_contingency(tab.values) if tab.values.sum() > 0 else (np.nan, np.nan, np.nan, None)
    return {
        "n_exposed": int(tab[1].sum()),
        "n_unexposed": int(tab[0].sum()),
        "missing_rate_exposed_%": None if np.isnan(rate_exp) else round(100*rate_exp, 2),
        "missing_rate_unexposed_%": None if np.isnan(rate_unexp) else round(100*rate_unexp, 2),
        "diff_%": None if (np.isnan(rate_exp) or np.isnan(rate_unexp)) else round(100*(rate_exp-rate_unexp), 2),
        "chi2": None if np.isnan(chi2) else round(chi2, 3),
        "p_value": None if np.isnan(p) else round(p, 4)
    }, tab

out = {}
tabs = {}
for name, col in [
    ("concurrent", "cyclone_exposure"),
    ("lag1", "cyclone_lag1_diag"),
    ("pm1", "cyclone_pm1_diag"),
]:
    stats, tab = _rates_and_test("vaccine_missing_flag", col)
    out[name] = stats
    tabs[name] = tab

miss_df = pd.DataFrame(out).T
txt = "outputs/missingness_vs_cyclone.txt"
csv = "outputs/missingness_vs_cyclone.csv"
with open(txt, "w") as f:
    f.write("Vaccine missingness vs Cyclone exposure\n")
    f.write("(flagged via rota_data_missing or completion_rate isna)\n\n")
    f.write(miss_df.to_string())
    f.write("\n\n2x2 tables (rows: missing 0/1, cols: exposure 0/1):\n")
    for k, tab in tabs.items():
        f.write(f"\n[{k}]\n")
        f.write(str(tab))
        f.write("\n")
miss_df.to_csv(csv, index=True)
print(f"Saved: {txt}")
print(f"Saved: {csv}")
print("Missingness vs Cyclone (concurrent/lag1/pm1):")
print(miss_df)

print("\nAll evaluations complete. Outputs saved.")
