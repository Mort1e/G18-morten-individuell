"""
Genererer manglende CSV-filer for figurer_nye.py:
  resultater_sku_mape.csv
  resultater_maaned.csv

Leser cached EDA og XGBoost-hyperparametre — re-kjører kun modellene.
ARIMA tar noen minutter for 105 SKU-er.
"""

import os
import sys
import warnings

warnings.filterwarnings("ignore")
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from analyse import (
    load_data, naive_forecast, holtwinters_forecast, arima_forecast,
    train_xgboost_global, predict_xgboost_sku,
    rmse, mae_metric, mape_metric,
    save_sku_mape_all_models, save_monthly_errors,
    DATA_FILE, TRAIN_END, TEST_PERIODS, OUTPUT_DIR,
)

print("Laster data og cached resultater...")
df = load_data(DATA_FILE)
eda_df = pd.read_csv(f"{OUTPUT_DIR}/eda_resultater.csv", index_col=0)

# Les beste XGBoost-hyperparametre fra cached CV (hopper over re-kalibrering)
cv_df = pd.read_csv(f"{OUTPUT_DIR}/xgboost_hyperparameter_cv.csv")
best_row = cv_df.sort_values("CV_RMSE").iloc[0]
best_params = {
    "n_estimators":  int(best_row["n_estimators"]),
    "learning_rate": float(best_row["learning_rate"]),
    "max_depth":     int(best_row["max_depth"]),
}
print(f"  Cached XGBoost-hyperparametre: {best_params}")

print("Trener global XGBoost...")
xgb_model = train_xgboost_global(df, best_params=best_params)

print("Kjører modeller (ARIMA tar noen minutter for 105 SKU-er)...")
results = {
    m: {"RMSE": [], "MAE": [], "MAPE": [], "preds": [], "actuals": []}
    for m in ["Naiv", "Holt-Winters", "ARIMA", "XGBoost"]
}

n = len(df)
for i, sku in enumerate(df.index):
    if (i + 1) % 15 == 0:
        print(f"  SKU {i+1}/{n}...")
    series = df.loc[sku].values.astype(float)
    train  = series[:TRAIN_END]
    actual = series[TRAIN_END:TRAIN_END + TEST_PERIODS]
    hw_form = eda_df.loc[sku, "HW_form"] if sku in eda_df.index else "additiv"

    forecasts = {
        "Naiv":         naive_forecast(train),
        "Holt-Winters": holtwinters_forecast(train, hw_form=hw_form),
        "ARIMA":        arima_forecast(train),
        "XGBoost":      predict_xgboost_sku(xgb_model, train),
    }
    for model_name, pred in forecasts.items():
        pred = np.array(pred[:TEST_PERIODS])
        results[model_name]["RMSE"].append(rmse(actual, pred))
        results[model_name]["MAE"].append(mae_metric(actual, pred))
        results[model_name]["MAPE"].append(mape_metric(actual, pred))
        results[model_name]["preds"].extend(list(pred))
        results[model_name]["actuals"].extend(list(actual))

# Les XGBoost individuell fra eksisterende CSV (ikke nødvendig å re-kjøre)
ind_df = pd.read_csv(f"{OUTPUT_DIR}/resultater_xgboost_individuell.csv")
results["XGBoost (individuell)"] = {
    "MAPE":    ind_df["MAPE"].tolist(),
    "preds":   [],
    "actuals": [],
}

print("\nLagrer CSV-filer...")
save_sku_mape_all_models(results, list(df.index), OUTPUT_DIR)
save_monthly_errors(results, OUTPUT_DIR)
print("\nFerdig! Kjør nå: python figurer_nye.py")
