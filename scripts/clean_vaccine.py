import pandas as pd

# Load the Excel file
vaccine_df = pd.read_excel('/Users/edderoni/Desktop/project/data/vaccine_cleaned.xlsx')
print(vaccine_df.columns)

# Rename columns
vaccine_df = vaccine_df.rename(columns={
    'Mês': 'month',
    'Total Rota 1ª Dose  (0-11 meses)': 'Rota1',
    'Total Rota 2ª Dose  (0-11 meses)': 'Rota2'
})

# Month name → number mapping
month_map = {'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
             'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12}
vaccine_df['month'] = vaccine_df['month'].map(month_map)

# Ensure types and add date column
vaccine_df['year'] = vaccine_df['year'].astype(int)
vaccine_df['month'] = vaccine_df['month'].astype(int)
vaccine_df['date'] = pd.to_datetime(vaccine_df[['year', 'month']].assign(day=1))
vaccine_df = vaccine_df.sort_values(['province', 'date'])

# Lag features
vaccine_df['Rota1_lag2'] = vaccine_df.groupby('province')['Rota1'].shift(2)
vaccine_df['Rota2_lag1'] = vaccine_df.groupby('province')['Rota2'].shift(1)

# Completion rate & % change
vaccine_df['completion_rate'] = vaccine_df['Rota2'] / vaccine_df['Rota1_lag2']
vaccine_df['completion_rate_pct_change'] = (
    vaccine_df.groupby('province')['completion_rate'].pct_change(fill_method=None)
)

# Drop flag: ≥10% drop
vaccine_df['vaccine_drop_flag'] = (vaccine_df['completion_rate_pct_change'] <= -0.10).astype(int)

# Missing flag and safe imputation
vaccine_df['rota_data_missing'] = vaccine_df[['Rota1', 'Rota2']].isnull().any(axis=1).astype(int)
vaccine_df['Rota1'] = vaccine_df['Rota1'].fillna(0)
vaccine_df['Rota2'] = vaccine_df['Rota2'].fillna(0)

# Save
vaccine_df.to_csv('vaccine_cleaned_with_flags.csv', index=False)
