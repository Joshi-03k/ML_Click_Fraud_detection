import pandas as pd

print("Program started...")

df = pd.read_csv("click_fraud_dataset_cleaned.csv")

print("Dataset loaded successfully!")

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nFirst 5 rows:")
print(df.head())

print("\nColumn names:")
print(df.columns.tolist())

print("\nMissing values:")
print(df.isnull().sum())

print("\nTarget distribution:")
print(df["is_fraudulent"].value_counts())

print("\nProgram completed!")
