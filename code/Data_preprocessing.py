import pandas as pd

df = pd.read_csv("processed_data.csv")

binary_columns = [
    "mainroad",
    "guestroom",
    "basement",
    "hotwaterheating",
    "airconditioning",
    "prefarea"
]

for col in binary_columns:
    df[col] = df[col].map({"yes":1,"no":0})

df = pd.get_dummies(df,
                    columns=["furnishingstatus"],
                    drop_first=True)

print(df.head())

# Save processed data
df.to_csv("final_data.csv", index=False)

print("Program 2 Completed")