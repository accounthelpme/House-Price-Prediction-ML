import pandas as pd

df = pd.read_csv("Housing.csv")

print(df.head())
print(df.info())
print(df.describe())
print(df.isnull().sum())

# Save original data (optional)
df.to_csv("processed_data.csv", index=False)

print("Program 1 Completed")