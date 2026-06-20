"""
Data preprocessing and feature engineering for the
Car Price Prediction project.

Handles:
- Missing value imputation
- Categorical encoding
- Feature engineering (derived features)
- Train/test splitting
- Feature scaling (for Linear Regression)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


def load_raw_data(path="data/used_car_data.csv"):
    return pd.read_csv(path)


def engineer_features(df):
    """Add derived features that help the models capture pricing logic."""
    df = df.copy()

    # Mileage per year of age (driving intensity) — avoid divide by zero
    df["mileage_per_year"] = df["mileage_km"] / (df["age_years"] + 1)

    # Power-to-engine ratio
    df["hp_per_liter"] = df["horsepower"] / df["engine_size_l"].replace(0, np.nan)

    # Age bucket (captures non-linear depreciation bands)
    df["age_bucket"] = pd.cut(
        df["age_years"],
        bins=[-1, 2, 5, 10, 20],
        labels=["0-2yrs", "3-5yrs", "6-10yrs", "11+yrs"],
    )

    return df


def build_preprocessing_pipeline(numeric_features, categorical_features):
    """ColumnTransformer that imputes, scales numeric features, and
    one-hot encodes categorical ones. Used inside each model pipeline."""
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ])
    return preprocessor


def get_feature_lists():
    numeric_features = [
        "age_years", "mileage_km", "engine_size_l", "horsepower",
        "owners", "accident_history", "mileage_per_year", "hp_per_liter",
    ]
    categorical_features = [
        "brand", "fuel_type", "transmission", "condition", "age_bucket",
    ]
    return numeric_features, categorical_features


def prepare_data(path="data/used_car_data.csv", test_size=0.2, random_state=42):
    df = load_raw_data(path)
    df = engineer_features(df)

    numeric_features, categorical_features = get_feature_lists()
    feature_cols = numeric_features + categorical_features

    X = df[feature_cols]
    y = df["price_usd"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, numeric_features, categorical_features


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, num_feats, cat_feats = prepare_data(
        path="/home/claude/car_price_project/data/used_car_data.csv"
    )
    print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
    print("Numeric features:", num_feats)
    print("Categorical features:", cat_feats)
    print(X_train.head())
