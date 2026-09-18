import pandas as pd

# 1. Load the dataset
df = pd.read_csv('development_dataset.csv')

# 2. Check basic shape (rows, columns)
print("=== DATASET SHAPE ===")
print(f"Total Rows: {df.shape[0]}")
print(f"Total Columns: {df.shape[1]}\n")

# 3. View column names & data types
print("=== COLUMNS & TYPES ===")
print(df.dtypes)
print("\n")

# 4. Check for missing values per column
print("=== MISSING VALUES ===")
print(df.isnull().sum())
print("\n")

# 5. Check distinct Chiller IDs
if 'equipment_id' in df.columns:
    print("=== UNIQUE CHILLERS ===")
    print(df['equipment_id'].value_counts())