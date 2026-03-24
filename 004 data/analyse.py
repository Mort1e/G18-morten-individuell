"""
LOG650 - Etterspørselsprognoseanalyse
Student: Morten Eidsvåg
Modeller: Naiv, Holt-Winters, ARIMA (SARIMA), XGBoost
Evalueringsmål: RMSE, MAE, MAPE (median over 105 SKU-er)
"""

import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import xgboost as xgb

warnings.filterwarnings("ignore")

# ── Konfigurasjon ──────────────────────────────────────────────────────────────
DATA_FILE     = "salgsdata_anonymized.csv"
TRAIN_END     = 36   # Jan 2022 – Des 2024
TEST_PERIODS  = 12   # Jan 2025 – Des 2025
SEASON_PERIOD = 12
OUTPUT_DIR    = "."

# ── 1. Last inn og klargjør data ───────────────────────────────────────────────
def load_data(filepath):
    df = pd.read_csv(filepath, sep=";", index_col=0)
    # Fjern siste kolonne (intern SKU-ID)
    df = df.iloc[:, :-1]
    # Konverter kolonnenavn til datetime-indeks
    months = pd.date_range(start="2022-01", periods=len(df.columns), freq="MS")
    df.columns = months
    # Konverter til numerisk, tving feil til NaN
    df = df.apply(pd.to_numeric, errors="coerce")
    # Fyll manglende verdier med 0 (ingen salg = 0)
    df = df.fillna(0)
    return df


# ── 2. Evalueringsmål ─────────────────────────────────────────────────────────
def rmse(actual, pred):
    return np.sqrt(mean_squared_error(actual, pred))

def mae(actual, pred):
    return mean_absolute_error(actual, pred)

def mape(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    mask = actual != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100


# ── 3. Modeller ────────────────────────────────────────────────────────────────
def naive_forecast(train):
    """Sesongnaiv: siste årets verdier som prognose."""
    return np.array(train[-SEASON_PERIOD:])

def holtwinters_forecast(train):
    """Holt-Winters eksponentiell glatting."""
    try:
        model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add",
            seasonal_periods=SEASON_PERIOD,
            initialization_method="estimated"
        )
        fit = model.fit(optimized=True, remove_bias=True)
        return fit.forecast(TEST_PERIODS)
    except Exception:
        try:
            model = ExponentialSmoothing(
                train,
                trend="add",
                seasonal=None,
                initialization_method="estimated"
            )
            fit = model.fit(optimized=True)
            return fit.forecast(TEST_PERIODS)
        except Exception:
            return np.full(TEST_PERIODS, np.mean(train))

def arima_forecast(train):
    """Auto ARIMA / SARIMA."""
    try:
        from pmdarima import auto_arima
        model = auto_arima(
            train,
            seasonal=True,
            m=SEASON_PERIOD,
            stepwise=True,
            suppress_warnings=True,
            error_action="ignore",
            max_p=3, max_q=3, max_P=2, max_Q=2
        )
        return model.predict(n_periods=TEST_PERIODS)
    except Exception:
        return np.full(TEST_PERIODS, np.mean(train))

def xgboost_forecast(all_series_train, all_series_test, sku_idx):
    """
    XGBoost global modell: trener på alle SKU-er samlet.
    Returnerer prognoser for én SKU (sku_idx).
    Kalles etter at global modell er trent.
    """
    pass  # Se build_xgboost_model() og predict_xgboost()

def build_xgboost_features(series, n_lags=12):
    """Lag-features, månedsnummer og trendindeks for én tidsserie."""
    rows = []
    for t in range(n_lags, len(series)):
        lags = [series[t - i] for i in range(1, n_lags + 1)]
        month = (t % 12) + 1
        trend = t
        rows.append(lags + [month, trend, series[t]])
    cols = [f"lag_{i}" for i in range(1, n_lags + 1)] + ["month", "trend", "target"]
    return pd.DataFrame(rows, columns=cols)

def train_xgboost_global(df_wide, n_lags=12):
    """Tren global XGBoost-modell på alle 105 SKU-er."""
    all_data = []
    for sku in df_wide.index:
        series = df_wide.loc[sku].values.astype(float)
        train_series = series[:TRAIN_END]
        feat_df = build_xgboost_features(train_series, n_lags)
        all_data.append(feat_df)
    combined = pd.concat(all_data, ignore_index=True)
    X = combined.drop("target", axis=1)
    y = combined["target"]
    model = xgb.XGBRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    model.fit(X, y)
    return model

def predict_xgboost_sku(model, train_series, n_lags=12):
    """Rullende prognose for én SKU med den globale modellen."""
    history = list(train_series.astype(float))
    preds = []
    for step in range(TEST_PERIODS):
        t = len(history)
        lags = [history[-(i)] for i in range(1, n_lags + 1)]
        month = (t % 12) + 1
        trend = t
        X = pd.DataFrame([lags + [month, trend]],
                         columns=[f"lag_{i}" for i in range(1, n_lags + 1)] + ["month", "trend"])
        pred = float(model.predict(X)[0])
        pred = max(0, pred)  # Ingen negative prognoser
        preds.append(pred)
        history.append(pred)
    return np.array(preds)


# ── 4. Kjør alle modeller ──────────────────────────────────────────────────────
def run_analysis(df):
    print("Trener global XGBoost-modell...")
    xgb_model = train_xgboost_global(df)

    results = {
        "Naiv":         {"RMSE": [], "MAE": [], "MAPE": []},
        "Holt-Winters": {"RMSE": [], "MAE": [], "MAPE": []},
        "ARIMA":        {"RMSE": [], "MAE": [], "MAPE": []},
        "XGBoost":      {"RMSE": [], "MAE": [], "MAPE": []},
    }

    n_skus = len(df)
    for i, sku in enumerate(df.index):
        if (i + 1) % 10 == 0:
            print(f"  Behandler SKU {i+1}/{n_skus}...")

        series = df.loc[sku].values.astype(float)
        train  = series[:TRAIN_END]
        actual = series[TRAIN_END:TRAIN_END + TEST_PERIODS]

        forecasts = {
            "Naiv":         naive_forecast(train),
            "Holt-Winters": holtwinters_forecast(train),
            "ARIMA":        arima_forecast(train),
            "XGBoost":      predict_xgboost_sku(xgb_model, train),
        }

        for model_name, pred in forecasts.items():
            pred = np.array(pred[:TEST_PERIODS])
            results[model_name]["RMSE"].append(rmse(actual, pred))
            results[model_name]["MAE"].append(mae(actual, pred))
            results[model_name]["MAPE"].append(mape(actual, pred))

    return results


# ── 5. Aggreger og vis resultater ──────────────────────────────────────────────
def summarize_results(results):
    rows = []
    for model_name, metrics in results.items():
        mape_vals = [v for v in metrics["MAPE"] if not np.isnan(v)]
        rows.append({
            "Modell":       model_name,
            "RMSE (median)": round(np.median(metrics["RMSE"]), 2),
            "MAE (median)":  round(np.median(metrics["MAE"]), 2),
            "MAPE (median)": round(np.median(mape_vals), 2) if mape_vals else np.nan,
            "MAPE (SKU-er brukt)": len(mape_vals),
        })
    summary = pd.DataFrame(rows).set_index("Modell")
    return summary


# ── 6. Diebold-Mariano test ────────────────────────────────────────────────────
def run_dm_test(df, results):
    """DM-test mellom Holt-Winters og ARIMA over alle SKU-er."""
    try:
        from dieboldmariano import dm_test as dm
        actuals, hw_preds, arima_preds = [], [], []
        for sku in df.index:
            series = df.loc[sku].values.astype(float)
            train  = series[:TRAIN_END]
            actual = series[TRAIN_END:TRAIN_END + TEST_PERIODS]
            hw_pred    = holtwinters_forecast(train)
            arima_pred = arima_forecast(train)
            actuals.extend(list(actual))
            hw_preds.extend(list(hw_pred[:TEST_PERIODS]))
            arima_preds.extend(list(arima_pred[:TEST_PERIODS]))
        stat, p = dm(actuals, hw_preds, arima_preds, h=1)
        print(f"\nDiebold-Mariano test (Holt-Winters vs ARIMA):")
        print(f"  DM-statistikk : {stat:.4f}")
        print(f"  p-verdi       : {p:.4f}")
        print(f"  Konklusjon    : {'Signifikant forskjell (p<0.05)' if p < 0.05 else 'Ingen signifikant forskjell'}")
        return stat, p
    except Exception as e:
        print(f"\nDM-test feilet: {e}")
        return None, None


# ── 7. Visualiseringer ────────────────────────────────────────────────────────
def plot_mape_boxplot(results, output_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    data = []
    labels = []
    for model_name, metrics in results.items():
        vals = [v for v in metrics["MAPE"] if not np.isnan(v)]
        data.append(vals)
        labels.append(model_name)
    ax.boxplot(data, labels=labels, patch_artist=True,
               boxprops=dict(facecolor="#d9eaf7"),
               medianprops=dict(color="darkblue", linewidth=2))
    ax.set_title("Figur 2: Fordeling av MAPE per modell (alle SKU-er)", fontsize=13)
    ax.set_ylabel("MAPE (%)")
    ax.set_ylim(0, None)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figur2_mape_boxplot.png", dpi=150)
    plt.close()
    print(f"Figur lagret: {output_dir}/figur2_mape_boxplot.png")

def plot_example_forecasts(df, output_dir):
    """Plott prognose vs. faktisk for én eksempel-SKU."""
    sku = df.index[1]  # Velg andre SKU som eksempel
    series = df.loc[sku].values.astype(float)
    train  = series[:TRAIN_END]
    actual = series[TRAIN_END:TRAIN_END + TEST_PERIODS]

    hw_pred    = holtwinters_forecast(train)
    naive_pred = naive_forecast(train)

    months_train = pd.date_range("2022-01", periods=TRAIN_END, freq="MS")
    months_test  = pd.date_range("2025-01", periods=TEST_PERIODS, freq="MS")

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(months_train, train, label="Historisk", color="black")
    ax.plot(months_test, actual, label="Faktisk (test)", color="black", linestyle="--")
    ax.plot(months_test, hw_pred[:TEST_PERIODS], label="Holt-Winters", color="steelblue")
    ax.plot(months_test, naive_pred, label="Naiv", color="gray", linestyle=":")
    ax.set_title(f"Figur 1: Eksempel prognose — SKU {sku}", fontsize=13)
    ax.set_ylabel("Volum (pakninger)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/figur1_eksempel_prognose.png", dpi=150)
    plt.close()
    print(f"Figur lagret: {output_dir}/figur1_eksempel_prognose.png")


# ── Hovedprogram ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("LOG650 — Etterspørselsprognoseanalyse")
    print("=" * 60)

    print("\nLaster inn data...")
    df = load_data(DATA_FILE)
    print(f"  {len(df)} SKU-er, {len(df.columns)} måneder")

    print("\nKjører modeller (dette tar noen minutter for ARIMA)...")
    results = run_analysis(df)

    print("\n" + "=" * 60)
    print("RESULTATER — Median over alle SKU-er")
    print("=" * 60)
    summary = summarize_results(results)
    print(summary.to_string())

    print("\nLagrer resultater til CSV...")
    summary.to_csv(f"{OUTPUT_DIR}/resultater_sammenligning.csv")

    print("\nKjører Diebold-Mariano test...")
    run_dm_test(df, results)

    print("\nLager visualiseringer...")
    plot_mape_boxplot(results, OUTPUT_DIR)
    plot_example_forecasts(df, OUTPUT_DIR)

    print("\nFerdig! Filer produsert:")
    print("  - resultater_sammenligning.csv")
    print("  - figur1_eksempel_prognose.png")
    print("  - figur2_mape_boxplot.png")
