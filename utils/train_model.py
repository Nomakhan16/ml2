import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

# Load dataset (make sure you have data/housing.csv)
df = pd.read_csv("data/housing.csv")

# Features and target
X = df.drop("price", axis=1)
y = df["price"]

# Categorical, numeric, and binary columns
categorical_cols = ["location", "property_type"]
numeric_cols = ["size", "bedrooms", "bathrooms", "year_built", "lot_size"]
amenity_cols = ["garage", "garden", "swimming_pool", "home_gym"]

# One-hot encode categorical features
X_encoded = pd.get_dummies(X, columns=categorical_cols)

# Force all columns to numeric
X_encoded = X_encoded.apply(pd.to_numeric, errors="coerce").fillna(0)

# Save feature columns for use during prediction
os.makedirs("models", exist_ok=True)
joblib.dump(list(X_encoded.columns), "models/features.pkl")

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train XGBoost model
model = XGBRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("✅ Model trained successfully")

# Save trained model
joblib.dump(model, "models/model.pkl")
print("✅ Model saved at models/model.pkl")


