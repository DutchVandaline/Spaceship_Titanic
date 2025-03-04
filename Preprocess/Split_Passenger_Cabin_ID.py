import pandas as pd

file_path = "../Dataset/test.csv"
df = pd.read_csv(file_path)

df[['Group', 'GroupMember']] = df['PassengerId'].str.split('_', expand=True)
df['GroupMember'] = df['GroupMember'].astype(int)

df[['Deck', 'CabinNum', 'Side']] = df['Cabin'].str.split('/', expand=True)

df['CabinNum'] = pd.to_numeric(df['CabinNum'], errors='coerce')

df.to_csv("../Dataset/test_splitted.csv", index=False)

print(df.head())
