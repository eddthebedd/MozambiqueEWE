import pandas as pd

# This script will transform raw vaccine data into a monthly aggregated format with lag variables and flags.

weather_df = pd.read_excel('/Users/edderoni/Desktop/project/data/m2.xlsx')

# Inspect column names
print(weather_df.columns)

# Standardize province name column
weather_df['province'] = weather_df['province'].str.lower().str.strip()

# List provinces and years present
print("Provinces in weather data:", weather_df['province'].unique())
print("Years in weather data:", weather_df['year'].unique())

# Create a date column from year and week
weather_df['date'] = pd.to_datetime(
    weather_df['year'].astype(str) + weather_df['week'].astype(str) + '1',
    format='%Y%W%w'
)

# Extract year and month
weather_df['year'] = weather_df['date'].dt.year
weather_df['month'] = weather_df['date'].dt.month

# Confirm column names to aggregate
print(weather_df.head())

# You can calculate avg temp as (tmin + tmax)/2
weather_df['temperature_avg'] = (weather_df['tmin'] + weather_df['tmax']) / 2

# Group by province/year/month and aggregate
monthly_weather = (
    weather_df
    .groupby(['province', 'year', 'month'])
    .agg({
        'tp': 'sum',                        # total precipitation summed
        'temperature_avg': 'mean',          # mean temperature
        'rh': 'mean',                       # mean humidity
        'season': lambda x: x.mode().iloc[0]  # most frequent season label
    })
    .reset_index()
)

# List provinces and years after aggregation
print("Provinces in aggregated weather data:", monthly_weather['province'].unique())
print("Years in aggregated weather data:", monthly_weather['year'].unique())

# Save output
monthly_weather.to_csv('weather_monthly_aggregated.csv', index=False)

print("Weather data is aggregated and saved!")

# Load the vaccine dataset and list provinces and years
vaccine_df = pd.read_csv('vaccine_cleaned_with_flags.csv')

vaccine_df['province'] = vaccine_df['province'].str.lower().str.strip()

print("Provinces in vaccine data:", vaccine_df['province'].unique())
print("Years in vaccine data:", vaccine_df['year'].unique())
