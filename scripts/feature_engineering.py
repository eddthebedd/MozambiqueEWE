import pandas as pd
import numpy as np

# This script does feature engineering on the merged dataset and imputes missing values (for now...)

df = pd.read_csv('vaccine_weather_merged.csv')

# Inspect
print("Loaded merged dataset. Columns:")
print(df.columns)

# Impute numeric weather variables
weather_cols = ['tp', 'temperature_avg', 'rh']

# Impute with province means (fallback to global mean)
for col in weather_cols:
    # Compute province means
    province_means = df.groupby('province')[col].transform('mean')
    global_mean = df[col].mean()
    
    df[col] = df[col].fillna(province_means)
    df[col] = df[col].fillna(global_mean)

# Encode season as dummy variable
df['season'] = df['season'].fillna('unknown')
df['season_rainy'] = np.where(df['season'] == 'rainy', 1,
                        np.where(df['season'] == 'dry', 0, -1))

# Confirm no remaining NAs in predictors
print("Any missing tp?", df['tp'].isnull().any())
print("Any missing temperature_avg?", df['temperature_avg'].isnull().any())
print("Any missing rh?", df['rh'].isnull().any())
print("Season_rainy unique values:", df['season_rainy'].unique())

# Save clean dataset
df.to_csv('vaccine_weather_features.csv', index=False)

print("Feature engineering has been completed! Saved as vaccine_weather_features.csv.")
