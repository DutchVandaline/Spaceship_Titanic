import pandas as pd

# Load the dataset
file_path = "../Dataset/train.csv"  # 파일 경로를 적절히 변경하세요.
df = pd.read_csv(file_path)

# Drop rows with any missing values
df_cleaned = df.dropna()

# Save the cleaned dataset (optional)
df_cleaned.to_csv("../Dataset/fully_cleaned_train.csv", index=False)

# Display the number of remaining rows after cleaning
print(f"Remaining rows after dropping missing values: {df_cleaned.shape[0]}")

# Display the first few rows to check the transformation
print(df_cleaned.head())
