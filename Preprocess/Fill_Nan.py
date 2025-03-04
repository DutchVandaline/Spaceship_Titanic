import pandas as pd

file_path = "../Dataset/cabin_no_nan_train_splitted.csv"
df = pd.read_csv(file_path)

spending_cols = ["VRDeck", "Spa", "RoomService", "ShoppingMall", "FoodCourt"]

df.loc[df["CryoSleep"] == True, spending_cols] = 0

df[spending_cols] = df.groupby("Cabin")[spending_cols].transform(lambda x: x.fillna(x.median()))
df[spending_cols] = df.groupby("HomePlanet")[spending_cols].transform(lambda x: x.fillna(x.median()))

for col in spending_cols:
    df[col].fillna(df[col].median(), inplace=True)

df["HomePlanet"] = df.groupby("Group")["HomePlanet"].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x))
df["HomePlanet"] = df.groupby("Cabin")["HomePlanet"].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x))
df["HomePlanet"].fillna(df["HomePlanet"].mode()[0], inplace=True)

df["CryoSleep"] = df.groupby("Group")["CryoSleep"].transform(lambda x: x if x.notna().all() else x.fillna(x.mode()[0] if not x.mode().empty else x))

df["Age"] = df.groupby("Group")["Age"].transform(lambda x: x.fillna(x.mean()))
df["Age"] = df.groupby("Cabin")["Age"].transform(lambda x: x.fillna(x.mean()))
df["Age"].fillna(df["Age"].mean(), inplace=True)

df["VIP"] = df.groupby("Group")["VIP"].transform(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else x))
df.loc[df["RoomService"] > df["RoomService"].median(), "VIP"] = True
df.loc[df["RoomService"] == 0, "VIP"] = False
df["VIP"].fillna(df["VIP"].mode()[0], inplace=True)

df.to_csv("../Dataset/fully_filled_train.csv", index=False)

print("Remaining missing values:\n", df.isnull().sum())
