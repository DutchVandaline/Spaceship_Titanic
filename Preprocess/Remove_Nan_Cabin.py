import pandas as pd

file_path = "../Dataset/train.csv"
df = pd.read_csv(file_path)

df_cleaned = df.dropna(subset=["Cabin"])

df_cleaned.to_csv("../Dataset/cabin_cleaned_train.csv", index=False)

print(f"Remaining rows after dropping rows with missing Cabin: {df_cleaned.shape[0]}")
print(df_cleaned.head())
