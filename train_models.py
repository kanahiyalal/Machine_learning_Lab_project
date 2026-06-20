

import time
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from preprocessing import prepare_data, build_preprocessing_pipeline

DATA_PATH = "/home/claude/car_price_project/data/used_car_data.csv"
MODELS_DIR = "/home/claude/car_price_project/models"
OUTPUTS_DIR = "/home/claude/car_price_project/outputs"


def regression_accuracy(y_true, y_pred, tolerance=0.10):
    """% of predictions within `tolerance` (default 10%) of true price.
    This gives an 'accuracy'-style metric for a regression task, since
    the dashboard requirement asks for Accuracy alongside other metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    pct_error = np.abs((y_pred - y_true) / y_true)
    return float(np.mean(pct_error <= tolerance) * 100)


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start

    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    acc_10pct = regression_accuracy(y_test, y_pred, tolerance=0.10)
    acc_15pct = regression_accuracy(y_test, y_pred, tolerance=0.15)

    metrics = {
        "model": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R2_Score": round(r2, 4),
        "Accuracy_within_10pct": round(acc_10pct, 2),
        "Accuracy_within_15pct": round(acc_15pct, 2),
        "Training_Time_sec": round(train_time, 4),
    }
    return model, metrics, y_pred


def main():
    X_train, X_test, y_train, y_test, num_feats, cat_feats = prepare_data(DATA_PATH)
    preprocessor = build_preprocessing_pipeline(num_feats, cat_feats)

    results = []
    predictions = {"y_test": y_test.reset_index(drop=True)}

    # ---- Model 1: Linear Regression ----
    lr_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression()),
    ])
    lr_model, lr_metrics, lr_pred = evaluate_model(
        "Linear Regression", lr_pipeline, X_train, X_test, y_train, y_test
    )
    results.append(lr_metrics)
    predictions["Linear_Regression_pred"] = lr_pred
    joblib.dump(lr_model, f"{MODELS_DIR}/linear_regression.joblib")


    preprocessor2 = build_preprocessing_pipeline(num_feats, cat_feats)
    rf_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor2),
        ("regressor", RandomForestRegressor(
            n_estimators=200, max_depth=12, min_samples_leaf=2,
            random_state=42, n_jobs=-1
        )),
    ])
    rf_model, rf_metrics, rf_pred = evaluate_model(
        "Random Forest", rf_pipeline, X_train, X_test, y_train, y_test
    )
    results.append(rf_metrics)
    predictions["Random_Forest_pred"] = rf_pred
    joblib.dump(rf_model, f"{MODELS_DIR}/random_forest.joblib")

    # ---- Save metrics ----
    with open(f"{OUTPUTS_DIR}/metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---- Save test predictions for dashboard plots ----
    pred_df = pd.DataFrame(predictions)
    pred_df.to_csv(f"{OUTPUTS_DIR}/test_predictions.csv", index=False)

    # ---- Save feature importance (Random Forest) ----
    rf_step = rf_model.named_steps["regressor"]
    feature_names = rf_model.named_steps["preprocessor"].get_feature_names_out()
    importances = pd.DataFrame({
        "feature": feature_names,
        "importance": rf_step.feature_importances_,
    }).sort_values("importance", ascending=False)
    importances.to_csv(f"{OUTPUTS_DIR}/feature_importance.csv", index=False)

    print(pd.DataFrame(results).to_string(index=False))
    print("\nModels and outputs saved successfully.")


if __name__ == "__main__":
    main()
