import pandas as pd

df = pd.read_csv("../Dataset/train.csv")

df["GroupID"] = df["PassengerId"].astype(str).str.split("_").str[0]
df["IndividualID"] = df["PassengerId"].astype(str).str.split("_").str[1].astype(int)

df["Deck"] = df["Cabin"].astype(str).str.split("/").str[0]
df["CabinNum"] = df["Cabin"].astype(str).str.split("/").str[1]
df["Side"] = df["Cabin"].astype(str).str.split("/").str[2]

df.to_csv("../Dataset/preprocessed_passenger_cabin.csv", index=False)
