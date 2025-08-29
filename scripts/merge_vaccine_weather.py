import pandas as pd
import numpy as np
import unicodedata

VAX_PATH     = "vaccine_cleaned_with_flags.csv"
WEATHER_PATH = "weather_monthly_aggregated.csv"
CYCLONE_PATH = "Cyclones.csv"
OUT_PATH     = "vaccine_weather_merged.csv"

START_Y, END_Y = 2017, 2022  

def norm_province(s):
    if pd.isna(s):
        return s
    s = str(s).lower().strip()
    s = ''.join(c for c in unicodedata.normalize('NFKD', s) if not unicodedata.combining(c))
    # common variants → canonical
    s = s.replace("cidade de maputo", "maputo city")
    s = s.replace("maputo cidade", "maputo city")
    s = s.replace("maputo provincia", "maputo province")
    return s

def load_vaccine():
    v = pd.read_csv(VAX_PATH)
    v["province"] = v["province"].apply(norm_province)
    v["year"] = v["year"].astype(int)
    v["month"] = v["month"].astype(int)
    return v

def load_weather():
    w = pd.read_csv(WEATHER_PATH)
    w["province"] = w["province"].apply(norm_province)
    w["year"] = w["year"].astype(int)
    w["month"] = w["month"].astype(int)
    return w

def load_cyclones():
    c = pd.read_csv(CYCLONE_PATH)
    # Accept either "Affected areas" or "province"
    prov_col = "Affected areas" if "Affected areas" in c.columns else ("province" if "province" in c.columns else None)
    if prov_col is None:
        raise ValueError("Cyclones.csv must have 'Affected areas' or 'province' column")
    c = c.rename(columns={prov_col: "province"})
    c["province"] = c["province"].apply(norm_province)

    # month may be strings like "Mar"
    month_map = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
    if c["month"].dtype == object:
        c["month"] = c["month"].map(month_map).fillna(c["month"]).astype(int)
    else:
        c["month"] = c["month"].astype(int)
    c["year"] = c["year"].astype(int)

    # Collapse to a unique (province, year, month) flag
    c = (c.dropna(subset=["province","year","month"])
           .drop_duplicates(subset=["province","year","month"]))
    c["cyclone_exposure"] = 1
    return c[["province","year","month","cyclone_exposure"]]

def build_calendar(vax, wthr, cycl):
    provs = sorted(set(vax["province"]).union(wthr["province"]).union(cycl["province"]))
    idx = pd.MultiIndex.from_product([provs, range(START_Y, END_Y+1), range(1,13)],
                                     names=["province","year","month"])
    cal = idx.to_frame(index=False)
    return cal

def add_cyclone_lags(df):
    df = df.sort_values(["province","year","month"]).copy()
    for i in (1,2,3,4):
        df[f"cyclone_lag{i}"] = df.groupby("province")["cyclone_exposure"].shift(i).fillna(0).astype(int)
    return df

def main():
    vax = load_vaccine()
    wth = load_weather()
    cyc = load_cyclones()

    # Build a master calendar so we don't lose provinces with no vaccine rows
    cal = build_calendar(vax, wth, cyc)

    # Left-join sources
    merged = (cal
              .merge(vax, how="left", on=["province","year","month"])
              .merge(wth, how="left", on=["province","year","month"])
              .merge(cyc, how="left", on=["province","year","month"]))

    # Fill cyclone_exposure with 0 (no record ⇒ no cyclone that month for that province)
    merged["cyclone_exposure"] = merged["cyclone_exposure"].fillna(0).astype(int)

    # Create cyclone lags
    merged = add_cyclone_lags(merged)

    # Diagnostics
    print("Provinces present (merged calendar):", sorted(merged["province"].unique().tolist()))
    print("Rows:", len(merged))
    print("Cyclone months (concurrent):", int(merged["cyclone_exposure"].sum()))
    print("Cyclone months (lag1):", int(merged["cyclone_lag1"].sum()))

    # Save
    merged.to_csv(OUT_PATH, index=False)
    print(f"Saved merged dataset: {OUT_PATH}")

if __name__ == "__main__":
    main()
