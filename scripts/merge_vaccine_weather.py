import pandas as pd

# This script merges our vaccine dataset that we saved with the weather dataset that we saved

vaccine_df = pd.read_csv('vaccine_cleaned_with_flags.csv')

weather_df = pd.read_csv('weather_monthly_aggregated.csv')

# Standardize province names to lowercase and strip spaces
vaccine_df['province'] = vaccine_df['province'].str.lower().str.strip()
weather_df['province'] = weather_df['province'].str.lower().str.strip()

# Diagnostic before renaming
print("Provinces in vaccine data before mapping:", vaccine_df['province'].unique())
print("Provinces in weather data before mapping:", weather_df['province'].unique())

# Province name mapping ONLY for spelling alignment
province_map = {
    'maputo cidade': 'maputo city',
    'maputo provincia': 'maputo province'
}
weather_df['province'] = weather_df['province'].replace(province_map)

# Diagnostic after mapping
print("Provinces in weather data after mapping:", weather_df['province'].unique())

# Merge on province, year, month
merged_df = pd.merge(
    vaccine_df,
    weather_df,
    on=['province', 'year', 'month'],
    how='left' 
)

# Create flag for rows with any missing weather variables
merged_df['weather_data_missing'] = merged_df[['tp', 'temperature_avg', 'rh', 'season']].isnull().any(axis=1).astype(int)

# Check merge result
print("Merged data preview:")
print(merged_df.head())

# Count rows with missing weather data
num_missing = merged_df['weather_data_missing'].sum()
print(f"Rows with missing weather data: {num_missing}")

# Save merged dataset
merged_df.to_csv('vaccine_weather_merged.csv', index=False)

print("Merged dataset has been saved as vaccine_weather_merged.csv!")
