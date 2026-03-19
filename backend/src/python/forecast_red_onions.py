import sys, json, pickle, os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "sarimax_import_red_onions.pkl")

def main():
    horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 7
    last_actual_date_str = sys.argv[2] if len(sys.argv) > 2 else None

    with open(MODEL_PATH, "rb") as f:
        results = pickle.load(f)

    #  Use DB-provided last date (most reliable)
    if last_actual_date_str and last_actual_date_str != "None":
        last_date = pd.to_datetime(last_actual_date_str, errors="coerce")
    else:
        last_date = pd.Timestamp.today().normalize()

    # fallback if parsing fails
    if pd.isna(last_date):
        last_date = pd.Timestamp.today().normalize()

    # exog forward fill (MVP)
    last_exog = results.model.exog[-1]
    future_exog = np.tile(last_exog, (horizon, 1))

    fc = results.get_forecast(steps=horizon, exog=future_exog)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)

    # next BUSINESS day after last actual date
    model_start = pd.bdate_range(last_date + pd.Timedelta(days=1), periods=1)[0]

    dates = pd.bdate_range(model_start, periods=horizon)

    forecast = []
    for i, d in enumerate(dates):
        forecast.append({
            "date": d.strftime("%Y-%m-%d"),
            "predicted": float(mean.iloc[i]),
            "lower": float(ci.iloc[i, 0]),
            "upper": float(ci.iloc[i, 1]),
        })

    vals = np.array([f["predicted"] for f in forecast], dtype=float)
    returns = np.diff(vals) / vals[:-1] if len(vals) > 1 else np.array([0.0])
    volatility = float(np.std(returns) * np.sqrt(min(7, len(vals))))

    print(json.dumps({
        "ok": True,
        "target": "Red Onions",
        "model": "SARIMAX",
        "lastActualDate": last_date.strftime("%Y-%m-%d"),
        "forecast": forecast,
        "volatility": volatility
    }, allow_nan=False))

if __name__ == "__main__":
    main()
