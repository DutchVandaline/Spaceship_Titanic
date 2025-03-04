import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load dataset
train_file = "filled_spending_train.csv"
df = pd.read_csv(train_file)

# Drop unnecessary columns
drop_cols = ["PassengerId", "Name", "Cabin"]
df = df.drop(columns=drop_cols)

# Define categorical and numerical features
categorical_features = ["HomePlanet", "CryoSleep", "Destination", "VIP", "Deck", "Side"]
numerical_features = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Group", "GroupMember",
                      "CabinNum"]

# Preprocessing pipelines
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),  # 최빈값 채우기
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))  # 원-핫 인코딩
])

numerical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),  # 중앙값 채우기
    ("scaler", StandardScaler())  # 스케일링
])

# Column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Split dataset
X = df.drop(columns=["Transported"])
y = df["Transported"].astype(int)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess data
X_train_transformed = preprocessor.fit_transform(X_train)
X_valid_transformed = preprocessor.transform(X_valid)

# Get categorical feature dimensions
num_categorical = len(categorical_features)
categorical_dim = len(preprocessor.named_transformers_["cat"].get_feature_names_out())


# TabTransformer 모델 정의
class TabTransformer(keras.Model):
    def __init__(self, num_numerical, num_categorical, embed_dim=32, num_heads=4, num_transformer_layers=2):
        super(TabTransformer, self).__init__()

        # Transformer for categorical features
        self.cat_embedding = layers.Dense(embed_dim, activation="relu")
        self.transformer_layers = [
            layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim) for _ in range(num_transformer_layers)
        ]
        self.cat_norm_layers = [layers.LayerNormalization() for _ in range(num_transformer_layers)]

        # Fully Connected layers for numerical + categorical embeddings
        self.concat_layer = layers.Concatenate()
        self.fc1 = layers.Dense(128, activation="relu")
        self.fc2 = layers.Dense(64, activation="relu")
        self.fc3 = layers.Dense(32, activation="relu")
        self.output_layer = layers.Dense(1, activation="sigmoid")

    def call(self, inputs):
        num_inputs, cat_inputs = inputs[:, :num_numerical], inputs[:, num_numerical:]

        # Transformer Encoding for Categorical Features
        cat_embedded = self.cat_embedding(cat_inputs)
        for norm_layer, transformer_layer in zip(self.cat_norm_layers, self.transformer_layers):
            cat_embedded = transformer_layer(cat_embedded, cat_embedded)
            cat_embedded = norm_layer(cat_embedded)

        # Concatenate numerical features with categorical embeddings
        x = self.concat_layer([num_inputs, cat_embedded])
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return self.output_layer(x)


# Instantiate model
num_numerical = len(numerical_features)
model = TabTransformer(num_numerical=num_numerical, num_categorical=categorical_dim)

# Compile model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train model
history = model.fit(X_train_transformed, y_train, epochs=50, batch_size=32,
                    validation_data=(X_valid_transformed, y_valid))

# Evaluate model
val_loss, val_acc = model.evaluate(X_valid_transformed, y_valid)
print(f"TabTransformer Validation Accuracy: {val_acc:.4f}")
