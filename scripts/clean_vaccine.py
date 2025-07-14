import pandas as pd

# This script cleans the dataset of vaccine data

vaccine_df = pd.read_excel('/Users/edderoni/Desktop/project/data/vaccine_cleaned.xlsx')
print(vaccine_df.columns)

# Rename columns for consistency
vaccine_df = vaccine_df.rename(columns={
    'Mês': 'month',
    'Total Rota 1ª Dose  (0-11 meses)': 'Rota1',
    'Total Rota 2ª Dose  (0-11 meses)': 'Rota2'
})

month_map = {
    'Jan': 1,
    'Feb': 2,
    'Mar': 3,
    'Apr': 4,
    'May': 5,
    'Jun': 6,
    'Jul': 7,
    'Aug': 8,
    'Sep': 9,
    'Oct': 10,
    'Nov': 11,
    'Dec': 12
}

vaccine_df['month'] = vaccine_df['month'].map(month_map)

# Ensure year and month are integers
vaccine_df['year'] = vaccine_df['year'].astype(int)
vaccine_df['month'] = vaccine_df['month'].astype(int)

# Create a date column
vaccine_df['date'] = pd.to_datetime(vaccine_df[['year', 'month']].assign(day=1))

# Sort
vaccine_df = vaccine_df.sort_values(['province', 'date'])

# Create lag variables
vaccine_df['Rota1_lag2'] = vaccine_df.groupby('province')['Rota1'].shift(2)
vaccine_df['Rota2_lag1'] = vaccine_df.groupby('province')['Rota2'].shift(1)

# Completion rate
vaccine_df['completion_rate'] = vaccine_df['Rota2'] / vaccine_df['Rota1_lag2']

# Month-over-month percent change
vaccine_df['completion_rate_pct_change'] = (
    vaccine_df.groupby('province')['completion_rate'].pct_change(fill_method=None)
)

# Create target flag
vaccine_df['vaccine_drop_flag'] = (
    vaccine_df['completion_rate_pct_change'] <= -0.10
).astype(int)

#flag columns for missing data
vaccine_df['rota_data_missing'] = vaccine_df[['Rota1', 'Rota2']].isnull().any(axis=1).astype(int)

#fill missing counts with 0
vaccine_df['Rota1'] = vaccine_df['Rota1'].fillna(0)
vaccine_df['Rota2'] = vaccine_df['Rota2'].fillna(0)

vaccine_df.to_csv('vaccine_cleaned_with_flags.csv', index=False)
