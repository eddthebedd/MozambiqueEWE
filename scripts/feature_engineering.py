import pandas as pd
import numpy as np

IN_PATH  = "vaccine_weather_merged.csv"
OUT_PATH = "vaccine_weather_features.csv"

# Contemporary weather columns in the merged file
WEATHER_VARS = ["tp", "temperature_avg", "rh"]

def neighbor_impute(series: pd.Series) -> pd.Series:
    """
    Neighbor-based impute within a province:
      - If value is NaN, average previous and next month (when present).
      - If only one neighbor exists, use that.
      - If neither neighbor exists, leave NaN (province/global fallback handles it).
    """
    s = series.copy()
    prev_ = s.shift(1)
    next_ = s.shift(-1)

    have_prev = (~prev_.isna()).astype(int)
    have_next = (~next_.isna()).astype(int)
    denom = (have_prev + have_next).replace(0, np.nan)

    neighbor_mean = (prev_.fillna(0) + next_.fillna(0)) / denom
    s = s.where(~s.isna(), neighbor_mean)
    return s

def main():
    df = pd.read_csv(IN_PATH)
    df = df.sort_values(["province", "year", "month"]).copy()

    # Flag weather missingness BEFORE imputation (keep this feature)
    df["weather_data_missing"] = df[WEATHER_VARS].isna().any(axis=1).astype(int)

    # Impute contemporary weather values (so lags won’t be NaN except at series head)
    for col in WEATHER_VARS:
        # 1) neighbor impute within province
        df[col] = (
            df.groupby("province", group_keys=False)[col]
              .apply(neighbor_impute)
        )
        # 2) fallback province mean
        df[col] = df[col].fillna(df.groupby("province")[col].transform("mean"))
        # 3) fallback global mean
        df[col] = df[col].fillna(df[col].mean())

    # Create LAGGED weather features (AFTER imputation)
    df = df.sort_values(["province", "year", "month"]).copy()
    for col in WEATHER_VARS:
        df[f"{col}_lag1"] = df.groupby("province")[col].shift(1)
        df[f"{col}_lag2"] = df.groupby("province")[col].shift(2)

    # Optional: lightly fill the very first months' lags so models don't drop rows
    for lag_col in [f"{v}_lag1" for v in WEATHER_VARS] + [f"{v}_lag2" for v in WEATHER_VARS]:
        df[lag_col] = df[lag_col].fillna(df.groupby("province")[lag_col].transform("mean"))
        df[lag_col] = df[lag_col].fillna(df[lag_col].mean())

    # **Remove contemporaries** so downstream code cannot accidentally use them
    df = df.drop(columns=WEATHER_VARS)

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved: {OUT_PATH}")

if __name__ == "__main__":
    main()
