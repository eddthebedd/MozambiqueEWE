import pandas as pd
import numpy as np

# === Load and preprocess m2 weather data ===
m2 = pd.read_excel("../data/m2.xlsx")

# Normalize province names
m2['province'] = m2['province'].str.lower().str.strip()

# Parse date from year + week
m2['date'] = pd.to_datetime(
    m2['year'].astype(str) + m2['week'].astype(str) + '1',
    format='%Y%W%w', errors='coerce'
)
m2 = m2.dropna(subset=['date'])

# Extract year and month
m2['year'] = m2['date'].dt.year
m2['month'] = m2['date'].dt.month

# Compute average temperature
m2['temperature_avg'] = (m2['tmin'] + m2['tmax']) / 2

# Monthly aggregation
monthly_weather = (
    m2.groupby(['province', 'year', 'month'])
    .agg({
        'tp': 'sum',
        'temperature_avg': 'mean',
        'rh': 'mean'
    })
    .reset_index()
)

# Mark any rows with missing weather data
monthly_weather['weather_data_missing'] = monthly_weather[['tp', 'temperature_avg', 'rh']].isna().any(axis=1).astype(int)

# Save output
monthly_weather.to_csv("weather_monthly_aggregated.csv", index=False)
print("Saved: weather_monthly_aggregated.csv (no rain/dry season flags)")
