import pandas as pd
import numpy as np

def clean_and_prepare_data(file_path):
    # 1. Load data
    df = pd.read_csv(file_path)
    
    # 2. Convert timestamp to datetime and sort chronological per equipment
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['equipment_id', 'timestamp']).reset_index(drop=True)
    
    # 3. Handle missing values: Forward fill then backward fill within each equipment group
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df.groupby('equipment_id')[numeric_cols].transform(
        lambda group: group.ffill().bfill()
    )
    
    # 4. Feature Engineering: Efficiency & Contextual Indicators
    # Energy per load ratio (kWh / RT)
    df['Energy_per_Load'] = df['Chiller Energy Consumption (kWh)'] / (df['Building Load (RT)'] + 1e-5)
    
    # Rolling 3-hour mean of energy consumption (6 intervals of 30 mins)
    df['Energy_Rolling_3h'] = df.groupby('equipment_id')['Chiller Energy Consumption (kWh)'].transform(
        lambda x: x.rolling(window=6, min_periods=1).mean()
    )
    
    # Time context features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    
    return df

if __name__ == "__main__":
    cleaned_df = clean_and_prepare_data('development_dataset.csv')
    print("=== PREPROCESSING COMPLETE ===")
    print(f"Processed Shape: {cleaned_df.shape}")
    print(f"Remaining Missing Values: {cleaned_df.isnull().sum().sum()}")
    print("\nNew Features Created:")
    print(cleaned_df[['Energy_per_Load', 'Energy_Rolling_3h', 'hour', 'day_of_week']].head())