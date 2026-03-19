# import sys
# import json
# import os
# import pickle
# import numpy as np
# import pandas as pd
# from tensorflow.keras.models import load_model

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# MODEL_DIR = os.path.join(BASE_DIR, "models", "big_onions")

# VAR_PATH = os.path.join(MODEL_DIR, "bigOnion_var.pkl")
# LSTM_PATH = os.path.join(MODEL_DIR, "bigOnion_lstm.h5")
# SCALER_PATH = os.path.join(MODEL_DIR, "bigOnion_res_scaler.pkl")
# WINDOW_PATH = os.path.join(MODEL_DIR, "bigOnion_last_residual_window.pkl")
# META_PATH = os.path.join(MODEL_DIR, "bigOnion_meta.json")


# def main():
#     horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 7
#     last_actual_date_str = sys.argv[2] if len(sys.argv) > 2 else None

#     with open(VAR_PATH, "rb") as f:
#         var_res = pickle.load(f)

#     lstm_model = load_model(LSTM_PATH)

#     with open(SCALER_PATH, "rb") as f:
#         res_scaler = pickle.load(f)

#     with open(WINDOW_PATH, "rb") as f:
#         last_residual_window = pickle.load(f)

#     with open(META_PATH, "r") as f:
#         meta = json.load(f)

#     target = meta["target"]
#     features = meta["features"]
#     lstm_window = int(meta["lstm_window"])

#     if last_actual_date_str and last_actual_date_str != "None":
#         last_date = pd.to_datetime(last_actual_date_str, errors="coerce")
#     else:
#         last_date = pd.Timestamp.today().normalize()

#     if pd.isna(last_date):
#         last_date = pd.Timestamp.today().normalize()

#     # VAR base forecast
#     lag_order = var_res.k_ar
#     history = var_res.endog[-lag_order:]
#     var_fc = var_res.forecast(history, steps=horizon)

#     var_fc_df = pd.DataFrame(var_fc, columns=features)

#     # LSTM residual forecast
#     window = res_scaler.transform(np.array(last_residual_window).reshape(-1, 1))
#     residual_preds = []

#     for _ in range(horizon):
#         pred = lstm_model.predict(window.reshape(1, lstm_window, 1), verbose=0)
#         residual_preds.append(pred[0][0])
#         window = np.vstack([window[1:], pred])

#     residual_preds = res_scaler.inverse_transform(
#         np.array(residual_preds).reshape(-1, 1)
#     ).flatten()

#     # Hybrid forecast = VAR target + residual correction
#     hybrid_target = var_fc_df[target].values + residual_preds

#     future_dates = pd.bdate_range(
#         last_date + pd.Timedelta(days=1),
#         periods=horizon
#     )

#     forecast = []
#     for i, d in enumerate(future_dates):
#         pred = float(hybrid_target[i])
#         forecast.append({
#             "date": d.strftime("%Y-%m-%d"),
#             "predicted": pred,
#             "lower": float(pred * 0.95),
#             "upper": float(pred * 1.05),
#         })

#     vals = np.array([x["predicted"] for x in forecast], dtype=float)
#     returns = np.diff(vals) / vals[:-1] if len(vals) > 1 else np.array([0.0])
#     volatility = float(np.std(returns))

#     print(json.dumps({
#         "ok": True,
#         "target": target,
#         "model": "VAR_LSTM_HYBRID",
#         "lastActualDate": last_date.strftime("%Y-%m-%d"),
#         "forecast": forecast,
#         "volatility": volatility
#     }, allow_nan=False))


# if __name__ == "__main__":
#     main()





import sys
import json
import os
import pickle
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.optimizers import Adam

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "big_onions")

VAR_PATH = os.path.join(MODEL_DIR, "bigOnion_var.pkl")
LSTM_WEIGHTS_PATH = os.path.join(MODEL_DIR, "bigOnion_lstm.weights.h5")
SCALER_PATH = os.path.join(MODEL_DIR, "bigOnion_res_scaler.pkl")
WINDOW_PATH = os.path.join(MODEL_DIR, "bigOnion_last_residual_window.pkl")
META_PATH = os.path.join(MODEL_DIR, "bigOnion_meta.json")


def build_lstm_model(window: int):
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=(window, 1)),
        Dropout(0.2),
        LSTM(32),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1)
    ])
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="mse"
    )
    return model


def main():
    try:
        horizon = int(sys.argv[1]) if len(sys.argv) > 1 else 7
        last_actual_date_str = sys.argv[2] if len(sys.argv) > 2 else None

        with open(VAR_PATH, "rb") as f:
            var_res = pickle.load(f)

        with open(SCALER_PATH, "rb") as f:
            res_scaler = pickle.load(f)

        with open(WINDOW_PATH, "rb") as f:
            last_residual_window = pickle.load(f)

        with open(META_PATH, "r") as f:
            meta = json.load(f)

        target = meta["target"]
        features = meta["features"]
        lstm_window = int(meta["lstm_window"])

        lstm_model = build_lstm_model(lstm_window)
        lstm_model.load_weights(LSTM_WEIGHTS_PATH)

        if last_actual_date_str and last_actual_date_str != "None":
            last_date = pd.to_datetime(last_actual_date_str, errors="coerce")
        else:
            last_date = pd.Timestamp.today().normalize()

        if pd.isna(last_date):
            last_date = pd.Timestamp.today().normalize()

        # VAR base forecast
        lag_order = var_res.k_ar
        history = var_res.endog[-lag_order:]
        var_fc = var_res.forecast(history, steps=horizon)
        var_fc_df = pd.DataFrame(var_fc, columns=features)

        # LSTM residual forecast
        window = res_scaler.transform(np.array(last_residual_window).reshape(-1, 1))
        residual_preds = []

        for _ in range(horizon):
            pred = lstm_model.predict(window.reshape(1, lstm_window, 1), verbose=0)
            residual_preds.append(pred[0][0])
            window = np.vstack([window[1:], pred])

        residual_preds = res_scaler.inverse_transform(
            np.array(residual_preds).reshape(-1, 1)
        ).flatten()

        # Hybrid = VAR target + residual correction
        hybrid_target = var_fc_df[target].values + residual_preds

        future_dates = pd.bdate_range(
            last_date + pd.Timedelta(days=1),
            periods=horizon
        )

        forecast = []
        for i, d in enumerate(future_dates):
            pred = float(hybrid_target[i])
            forecast.append({
                "date": d.strftime("%Y-%m-%d"),
                "predicted": pred,
                "lower": float(pred * 0.95),
                "upper": float(pred * 1.05),
            })

        vals = np.array([x["predicted"] for x in forecast], dtype=float)
        returns = np.diff(vals) / vals[:-1] if len(vals) > 1 else np.array([0.0])
        volatility = float(np.std(returns))

        print(json.dumps({
            "ok": True,
            "target": target,
            "model": "VAR_LSTM_HYBRID",
            "lastActualDate": last_date.strftime("%Y-%m-%d"),
            "forecast": forecast,
            "volatility": volatility
        }, allow_nan=False))

    except Exception as e:
        print(json.dumps({
            "ok": False,
            "error": str(e)
        }, allow_nan=False))
        sys.exit(1)


if __name__ == "__main__":
    main()