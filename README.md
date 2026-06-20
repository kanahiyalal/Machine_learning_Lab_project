# Car Price Prediction — Machine Learning Web App

A web-based ML application that predicts the resale price of a used car
based on its brand, age, mileage, condition, and other attributes. Built
to satisfy a Machine Learning course project's mandatory requirements:

- ✅ Web interface (Streamlit)
- ✅ Two ML algorithms trained & evaluated on the same dataset (Linear
  Regression, Random Forest)
- ✅ No external prediction APIs — everything trained and run locally
- ✅ User input form that generates live predictions
- ✅ Model comparison dashboard (Accuracy, MAE/RMSE, R², Training Time,
  prediction-vs-actual plots, residual plots, feature importance)

---

## 1. Project Structure

```
car_price_project/
├── app.py                     # Streamlit web application (run this)
├── preprocessing.py            # Data cleaning & feature engineering
├── train_models.py             # Trains both models, saves metrics
├── requirements.txt
├── data/
│   ├── generate_dataset.py     # Script that generated the dataset
│   └── used_car_data.csv       # The dataset (1200 records)
├── models/
│   ├── linear_regression.joblib
│   └── random_forest.joblib
└── outputs/
    ├── metrics.json
    ├── test_predictions.csv
    └── feature_importance.csv
```

## 2. Setup

```bash
# (Recommended) create a virtual environment
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 3. Train the Models (already done — re-run only if you change the data)

```bash
python train_models.py
```

This trains Linear Regression and Random Forest on `data/used_car_data.csv`,
saves the trained pipelines to `models/`, and writes evaluation metrics to
`outputs/metrics.json`.

## 4. Run the Web App

```bash
streamlit run app.py
```

This opens the app in your browser (usually at `http://localhost:8501`).
The app has 3 pages, navigable from the sidebar:

1. **Predict Price** — enter car details, choose a model, get an instant
   predicted price (and a side-by-side comparison across both models).
2. **Model Comparison Dashboard** — performance metrics table, error
   charts, predicted-vs-actual scatter plots, residual distributions, and
   feature importance.
3. **About the Dataset** — dataset overview, sample records, and
   exploratory charts.

## 5. Regenerating the Dataset (optional)

If you want a different random dataset (e.g. more records, different
seed), edit and re-run:

```bash
python data/generate_dataset.py
python train_models.py
```

## 6. Using Your Own Dataset

To swap in a real dataset (e.g. from Kaggle), replace
`data/used_car_data.csv` with your file, making sure it has the same
column names used in `preprocessing.py` (`brand`, `age_years`,
`mileage_km`, `fuel_type`, `transmission`, `engine_size_l`,
`horsepower`, `owners`, `accident_history`, `condition`, `price_usd`) —
or update the column lists in `preprocessing.py` to match your columns.
Then re-run `train_models.py` and `streamlit run app.py`.

## 7. Notes on the "Accuracy" Metric

Car price prediction is a **regression** problem (predicting a
continuous number), not classification, so there's no single "correct
class" to score accuracy against directly. To satisfy the dashboard's
accuracy/precision/recall-style requirement in a way that's meaningful
for regression, this project reports:

- **MAE** (Mean Absolute Error) and **RMSE** (Root Mean Squared Error) —
  how far off predictions are, in dollars
- **R² Score** — how much of the price variation the model explains
- **Accuracy (±10%) / (±15%)** — percentage of predictions within 10%/15%
  of the true price, used as a regression-appropriate stand-in for
  classification accuracy
- **Predicted vs. Actual scatter plot** — the regression equivalent of a
  confusion matrix
- **Training Time** — wall-clock seconds to fit each model

This reasoning is also explained in Section 8 of the project report.
