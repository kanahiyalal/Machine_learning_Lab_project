
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from preprocessing import engineer_features


st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide",
)

DATA_PATH = "data/used_car_data.csv"
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"


@st.cache_resource
def load_models():
    lr = joblib.load(f"{MODELS_DIR}/linear_regression.joblib")
    rf = joblib.load(f"{MODELS_DIR}/random_forest.joblib")
    return {"Linear Regression": lr, "Random Forest": rf}


@st.cache_data
def load_metrics():
    with open(f"{OUTPUTS_DIR}/metrics.json") as f:
        return pd.DataFrame(json.load(f))


@st.cache_data
def load_test_predictions():
    return pd.read_csv(f"{OUTPUTS_DIR}/test_predictions.csv")


@st.cache_data
def load_feature_importance():
    return pd.read_csv(f"{OUTPUTS_DIR}/feature_importance.csv")


@st.cache_data
def load_raw_data():
    return pd.read_csv(DATA_PATH)


models = load_models()
metrics_df = load_metrics()
test_preds = load_test_predictions()
feat_imp = load_feature_importance()
raw_df = load_raw_data()

BRANDS = sorted(raw_df["brand"].unique())
FUEL_TYPES = sorted(raw_df["fuel_type"].unique())
TRANSMISSIONS = sorted(raw_df["transmission"].unique())
CONDITIONS = ["Excellent", "Good", "Fair", "Poor"]


st.sidebar.title("🚗 Car Price Predictor")
page = st.sidebar.radio(
    "Navigate",
    ["Predict Price", "Model Comparison Dashboard", "About the Dataset"],
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Kanahiya Lal"
)

if page == "Predict Price":
    st.title("Predict a Used Car's Price")
    st.write(
        "Enter the car's details below and choose a model to generate an "
        "instant price prediction."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        brand = st.selectbox("Brand", BRANDS)
        age_years = st.slider("Age (years)", 0, 15, 5)
        mileage_km = st.number_input(
            "Mileage (km)", min_value=0, max_value=400000, value=80000, step=1000
        )

    with col2:
        fuel_type = st.selectbox("Fuel Type", FUEL_TYPES)
        transmission = st.selectbox("Transmission", TRANSMISSIONS)
        condition = st.selectbox("Condition", CONDITIONS, index=1)

    with col3:
        engine_size_l = st.slider("Engine Size (L)", 1.0, 4.0, 2.0, step=0.1)
        horsepower = st.slider("Horsepower", 60, 400, 150)
        owners = st.selectbox("Number of Previous Owners", [1, 2, 3, 4])
        accident_history = st.selectbox(
            "Accident History", ["No", "Yes"]
        )
        accident_history = 1 if accident_history == "Yes" else 0

    model_choice = st.radio(
        "Choose a model for prediction",
        list(models.keys()),
        horizontal=True,
    )

    if st.button("Predict Price", type="primary"):
        input_df = pd.DataFrame([{
            "brand": brand,
            "age_years": age_years,
            "mileage_km": mileage_km,
            "fuel_type": fuel_type,
            "transmission": transmission,
            "engine_size_l": engine_size_l,
            "horsepower": horsepower,
            "owners": owners,
            "accident_history": accident_history,
            "condition": condition,
        }])
        input_df = engineer_features(input_df)

        model = models[model_choice]
        prediction = model.predict(input_df)[0]
        prediction = max(prediction, 0)

        st.success(f"### Estimated Price: **${prediction:,.2f}** USD")
        st.caption(f"Prediction generated using: {model_choice} (trained locally)")

        st.markdown("#### Compare across both models")
        comparison = {}
        for name, m in models.items():
            comparison[name] = max(m.predict(input_df)[0], 0)
        comp_df = pd.DataFrame({
            "Model": list(comparison.keys()),
            "Predicted Price (USD)": [f"${v:,.2f}" for v in comparison.values()],
        })
        st.table(comp_df)


elif page == "Model Comparison Dashboard":
    st.title("Model Comparison Dashboard")
    st.write(
        "Both models were trained and evaluated on the **same dataset** "
        "and the **same train/test split** for a fair comparison."
    )

    st.markdown("### Performance Metrics")
    st.caption(
        "Since this is a regression task (predicting a continuous price), "
        "'Accuracy' is reported as the percentage of test predictions that "
        "fall within ±10% / ±15% of the actual price — a regression-appropriate "
        "stand-in for classification accuracy."
    )
    display_metrics = metrics_df.rename(columns={
        "model": "Model",
        "MAE": "MAE ($)",
        "RMSE": "RMSE ($)",
        "R2_Score": "R² Score",
        "Accuracy_within_10pct": "Accuracy (±10%)",
        "Accuracy_within_15pct": "Accuracy (±15%)",
        "Training_Time_sec": "Training Time (s)",
    })
    st.dataframe(display_metrics, use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### Error Comparison (Lower is Better)")
        fig, ax = plt.subplots(figsize=(5, 4))
        metrics_melted = metrics_df.melt(
            id_vars="model", value_vars=["MAE", "RMSE"],
            var_name="Metric", value_name="Value"
        )
        sns.barplot(data=metrics_melted, x="model", y="Value", hue="Metric", ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("Error (USD)")
        ax.set_title("MAE & RMSE by Model")
        st.pyplot(fig)

    with col2:
        st.markdown("#### R² Score & Accuracy (Higher is Better)")
        fig, ax = plt.subplots(figsize=(5, 4))
        acc_melted = metrics_df.melt(
            id_vars="model",
            value_vars=["Accuracy_within_10pct", "Accuracy_within_15pct"],
            var_name="Metric", value_name="Value"
        )
        sns.barplot(data=acc_melted, x="model", y="Value", hue="Metric", ax=ax)
        ax.set_xlabel("")
        ax.set_ylabel("% of predictions within tolerance")
        ax.set_title("Prediction Accuracy by Tolerance Band")
        st.pyplot(fig)

    st.markdown("### Predicted vs. Actual Price (Test Set)")
    st.caption(
        "This scatter plot is the regression equivalent of a confusion matrix: "
        "points on the diagonal line are perfect predictions; the tighter the "
        "spread, the better the model."
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col, name in zip(
        axes,
        ["Linear_Regression_pred", "Random_Forest_pred"],
        ["Linear Regression", "Random Forest"],
    ):
        ax.scatter(test_preds["y_test"], test_preds[col], alpha=0.4, s=18)
        lims = [0, max(test_preds["y_test"].max(), test_preds[col].max())]
        ax.plot(lims, lims, color="red", linestyle="--", linewidth=1)
        ax.set_xlabel("Actual Price (USD)")
        ax.set_ylabel("Predicted Price (USD)")
        ax.set_title(name)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("### Residual Distribution")
    st.caption("How far off were the predictions? Centered near zero is ideal.")
    fig, ax = plt.subplots(figsize=(8, 4))
    for col, name in zip(
        ["Linear_Regression_pred", "Random_Forest_pred"],
        ["Linear Regression", "Random Forest"],
    ):
        residuals = test_preds[col] - test_preds["y_test"]
        sns.kdeplot(residuals, label=name, ax=ax, fill=True, alpha=0.3)
    ax.axvline(0, color="black", linestyle="--", linewidth=1)
    ax.set_xlabel("Residual (Predicted - Actual), USD")
    ax.legend()
    st.pyplot(fig)

    st.markdown("### Feature Importance (Random Forest)")
    top_features = feat_imp.head(10).copy()
    top_features["feature"] = (
        top_features["feature"]
        .str.replace("num__", "", regex=False)
        .str.replace("cat__", "", regex=False)
    )
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.barplot(data=top_features, y="feature", x="importance", ax=ax, color="#2E75B6")
    ax.set_xlabel("Importance")
    ax.set_ylabel("")
    ax.set_title("Top 10 Most Influential Features")
    plt.tight_layout()
    st.pyplot(fig)


elif page == "About the Dataset":
    st.title("About the Dataset")
    st.write(
        f"This app uses a used-car dataset with **{len(raw_df)} records** "
        "and the following features: brand, age, mileage, fuel type, "
        "transmission, engine size, horsepower, ownership history, "
        "accident history, and condition. The target variable is the "
        "car's resale **price (USD)**."
    )

    st.markdown("#### Sample Records")
    st.dataframe(raw_df.head(10), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Price Distribution")
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.histplot(raw_df["price_usd"], bins=30, kde=True, ax=ax, color="#2E75B6")
        ax.set_xlabel("Price (USD)")
        st.pyplot(fig)

    with col2:
        st.markdown("#### Price by Brand")
        fig, ax = plt.subplots(figsize=(5, 4))
        order = raw_df.groupby("brand")["price_usd"].median().sort_values(ascending=False).index
        sns.boxplot(data=raw_df, x="brand", y="price_usd", order=order, ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_xlabel("")
        ax.set_ylabel("Price (USD)")
        st.pyplot(fig)

    st.markdown("#### Summary Statistics")
    st.dataframe(raw_df.describe(), use_container_width=True)
