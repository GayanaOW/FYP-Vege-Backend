import sys, json, pickle, os
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "models")

# ===== FILE PATHS =====
SARIMAX_PATH = os.path.join(MODEL_DIR, "sarimax_big_onions.pkl")
LSTM_PATH = os.path.join(MODEL_DIR, "lstm_big_onions.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "res_scaler.pkl")
WINDOW_PATH = os.path.join(MODEL_DIR, "last_residual_window.pkl")

# ===== CONFIG =====
LSTM_WINDOW = 90

def main():
    try:
        horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 7
        last_actual_date_str = sys.argv[2] if len(sys.argv) > 2 else None

        # ===== LOAD MODELS =====
        with open(SARIMAX_PATH, "rb") as f:
            sarimax_res = pickle.load(f)

        lstm_model = load_model(LSTM_PATH)

        with open(SCALER_PATH, "rb") as f:
            scaler = pickle.load(f)

        with open(WINDOW_PATH, "rb") as f:
            last_window = pickle.load(f)

        # ===== HANDLE DATE =====
        if last_actual_date_str and last_actual_date_str != "None":
            last_date = pd.to_datetime(last_actual_date_str, errors="coerce")
        else:
            last_date = pd.Timestamp.today().normalize()

        if pd.isna(last_date):
            last_date = pd.Timestamp.today().normalize()

        # ===== FUTURE DATES (business days) =====
        future_dates = pd.bdate_range(
            last_date + pd.Timedelta(days=1),
            periods=horizon
        )

        # ===== SARIMAX FORECAST =====
        last_exog = sarimax_res.model.exog[-1]
        future_exog = np.tile(last_exog, (horizon, 1))

        sarimax_fc = sarimax_res.get_forecast(
            steps=horizon,
            exog=future_exog
        ).predicted_mean.values

        # ===== LSTM RESIDUAL FORECAST =====
        window = scaler.transform(np.array(last_window).reshape(-1, 1))

        residual_preds = []

        for _ in range(horizon):
            pred = lstm_model.predict(
                window.reshape(1, LSTM_WINDOW, 1),
                verbose=0
            )
            residual_preds.append(pred[0][0])
            window = np.vstack([window[1:], pred])

        residual_preds = scaler.inverse_transform(
            np.array(residual_preds).reshape(-1, 1)
        ).flatten()

        # ===== HYBRID = SARIMAX + LSTM =====
        hybrid = sarimax_fc + residual_preds

        # ===== BUILD RESPONSE =====
        forecast = []
        for i, d in enumerate(future_dates):
            forecast.append({
                "date": d.strftime("%Y-%m-%d"),
                "predicted": float(hybrid[i]),
                "lower": float(hybrid[i] * 0.95),  # simple CI approximation
                "upper": float(hybrid[i] * 1.05),
            })

        # ===== VOLATILITY =====
        vals = np.array(hybrid)
        if len(vals) >= 2:
            returns = np.diff(vals) / vals[:-1]
            volatility = float(np.std(returns))
        else:
            volatility = 0.0

        print(json.dumps({
            "ok": True,
            "model": "HYBRID_SARIMAX_LSTM",
            "forecast": forecast,
            "volatility": volatility
        }, allow_nan=False))

    except Exception as e:
        print(json.dumps({
            "ok": False,
            "error": str(e)
        }))
        sys.exit(1)

if __name__ == "__main__":
    main()