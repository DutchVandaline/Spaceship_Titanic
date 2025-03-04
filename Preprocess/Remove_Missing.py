import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("../Dataset/train.csv")

# Extract GroupID and IndividualID
df["GroupID"] = df["PassengerId"].astype(str).str.split("_").str[0]
df["IndividualID"] = df["PassengerId"].astype(str).str.split("_").str[1].astype(int)

# Extract Cabin details
df["Deck"] = df["Cabin"].astype(str).str.split("/").str[0]
df["CabinNum"] = df["Cabin"].astype(str).str.split("/").str[1]
df["Side"] = df["Cabin"].astype(str).str.split("/").str[2]

# Handle missing values in Cabin-related columns
df["Deck"].fillna("Missing", inplace=True)
df["CabinNum"].fillna("Missing", inplace=True)
df["Side"].fillna("Missing", inplace=True)

# Drop unnecessary columns
df = df.drop(columns=["Name", "Cabin"])

# Handle missing values
df.fillna({"HomePlanet": df["HomePlanet"].mode()[0], "Destination": df["Destination"].mode()[0]}, inplace=True)
df.fillna({"Age": df["Age"].median(), "VRDeck": df["VRDeck"].median()}, inplace=True)
df.fillna({col: 0 for col in ["RoomService", "FoodCourt", "ShoppingMall", "Spa"]}, inplace=True)

# VIP: Convert to integer and fill missing values with mode
df["VIP"] = df["VIP"].astype(float)  # Convert True/False to 1/0
df["VIP"].fillna(df["VIP"].mode()[0], inplace=True)

# Encode categorical variables
label_encoders = {}
categorical_cols = ["HomePlanet", "CryoSleep", "Destination", "VIP", "GroupID", "Deck", "Side", "CabinNum"]
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Define features and target
y = df["Transported"].astype(int)  # Convert boolean to integer (0 or 1)
X = df.drop(columns=["Transported"])

# Save preprocessed data without normalization
preprocessed_data = pd.concat([X, pd.DataFrame(y, columns=["Transported"])], axis=1)
preprocessed_data.to_csv("../Dataset/preprocessed_data.csv", index=False)