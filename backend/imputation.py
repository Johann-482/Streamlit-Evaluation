import numpy as np
import pandas as pd

from backend.evaluation import is_gan_model
from backend.evaluation import (
    predict_gan,
    predict_seq_model,
    inverse_transform
)
from backend.preprocessing import create_windows


def impute_12_months(uploaded_df, model, scaler):

    df = uploaded_df.copy()

    required_cols = {"Year", "Month", "Precipitation"}
    if not required_cols.issubset(df.columns):
        raise ValueError("CSV must contain Year, Month, Precipitation")

    if len(df) != 12:
        raise ValueError("CSV must contain EXACTLY 12 rows (1 year of data)")

    df["Date"] = pd.to_datetime(df[["Year", "Month"]].assign(DAY=1))
    df = df.sort_values("Date").reset_index(drop=True)

    precip_raw = df["Precipitation"].replace("", np.nan).astype(float).values

    mask = (~np.isnan(precip_raw)).astype(np.float32)

    # Apply SAME transformation as training
    precip_log = np.log1p(np.nan_to_num(precip_raw, nan=0.0))
    precip_scaled = scaler.transform(precip_log.reshape(-1, 1))

    # Now fix missing positions to 0.5 (same as training)
    precip_scaled[mask == 0] = 0.5

    months = df["Month"].values

    month_sin = np.sin(2 * np.pi * months / 12).reshape(-1, 1)
    month_cos = np.cos(2 * np.pi * months / 12).reshape(-1, 1)

    # Correct feature order to match training
    X = np.concatenate([
        precip_scaled,
        month_sin,
        month_cos,
        mask.reshape(-1, 1),
    ], axis=1)

    # Correct windows
    X_windows, _ = create_windows(X)
    mask_windows, _ = create_windows(mask.reshape(-1,1))


    if is_gan_model(model):
        pred_scaled = predict_gan(model, X_windows, mask_windows)
    else:
        pred_scaled = predict_seq_model(model, X_windows, mask_windows)

    pred_real = inverse_transform(pred_scaled.reshape(-1, 1), scaler)

    final_output = precip_raw.copy()
    missing_idx = np.where(mask == 0)[0]
    final_output[missing_idx] = pred_real[missing_idx]

    df["Imputed_Precipitation"] = final_output

    return df