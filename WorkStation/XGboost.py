import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the dataset
file_path = "../Dataset/cabin_no_nan_train_splitted.csv"
df = pd.read_csv(file_path)

df = df.drop(columns=["PassengerId", "Name", "Cabin"])

# Convert categorical variables to numerical using one-hot encoding
df = pd.get_dummies(df, columns=["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side"])

# Split features and target
X = df.drop(columns=["Transported"])
y = df["Transported"].astype(int)

# Split into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the XGBoost model
model = xgb.XGBClassifier(
    n_estimators=200,  # 트리 개수
    learning_rate=0.05,  # 학습률
    max_depth=6,  # 트리의 최대 깊이
    subsample=0.8,  # 샘플링 비율
    colsample_bytree=0.8,  # 컬럼 샘플링 비율
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost Model Accuracy: {accuracy:.4f}")
