import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import config
import os
import sys
import joblib

def resource_path(relative_path):
    """Return absolute path to resource, in dev and PyInstaller"""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

"""1. Load and Prepare data"""

def load_and_prepare_data(uploaded_df=None):
    if uploaded_df is not None:
        df = uploaded_df.copy()
    else:
        if config.DATA_PATH is None:
            raise ValueError("No dataset provided. Upload a file through the UI.")
        df = pd.read_csv(config.DATA_PATH)

    # Ensure Year + Month columns exist
    if not {"Year", "Month", "Precipitation"}.issubset(df.columns):
        raise ValueError("CSV must contain Year, Month, and Precipitation columns.")

    df["Date"] = pd.to_datetime(df[["Year", "Month"]].assign(DAY=1))
    df = df.sort_values("Date").reset_index(drop=True)
    df["Precip_log"] = np.log1p(df["Precipitation"])

    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)

    return df[["Precip_log", "Month", "Month_sin", "Month_cos"]].copy()

"""2. Train/Test Split"""

def split_train_test(data_df, train_ratio=0.8):
    n_total = len(data_df)
    split_idx = int(n_total * train_ratio)

    return (
        data_df.iloc[:split_idx].copy(),
        data_df.iloc[split_idx:].copy()
    )

"""3. Scaling"""

def scale_datasets(train_df, test_df):
    scaler = MinMaxScaler()

    # Fit only on precipitation
    train_precip = train_df[["Precip_log"]]
    test_precip = test_df[["Precip_log"]]

    train_scaled_precip = scaler.fit_transform(train_precip)
    test_scaled_precip = scaler.transform(test_precip)

    # Keep month unscaled
    train_month = train_df[["Month_sin", "Month_cos"]].values
    test_month = test_df[["Month_sin", "Month_cos"]].values

    # Concatenate as before
    train_scaled = np.concatenate([train_scaled_precip, train_month], axis=1)
    test_scaled = np.concatenate([test_scaled_precip, test_month], axis=1)

    return train_scaled, test_scaled, scaler

"""4. MAR Masking"""

def apply_mar_mask(data, months, rate):

    rng = np.random.default_rng(config.SEED)

    masked = data.copy()

    # Extract features
    rainfall = data[:, 0]  # scaled log precipitation (0–1)
    month_sin = months[:, 0]
    month_cos = months[:, 1]

    # Convert to seasonal angle
    month_angle = np.arctan2(month_sin, month_cos)
    month_norm = (month_angle + np.pi) / (2 * np.pi)

    # --- TRUE MAR MECHANISM ---
    # Combine seasonality + observed rainfall
    logits = (
        2.0 * (month_norm - 0.5) +   # seasonal effect
        1.5 * (rainfall - 0.5)       # rainfall-dependent effect
    )

    prob_missing = 1 / (1 + np.exp(-logits))

    # Normalize to target missing rate
    prob_missing = prob_missing / (prob_missing.mean() + 1e-8)
    prob_missing = prob_missing * rate
    prob_missing = np.clip(prob_missing, 0, 1)

    # Sample missing positions
    random_draw = rng.random(prob_missing.shape)
    missing = random_draw < prob_missing

    # Apply mask ONLY to precipitation
    masked[missing, 0] = np.nan

    # Create mask
    mask = np.ones((data.shape[0], 1), dtype=np.float32)
    mask[missing, 0] = 0.0

    return masked, mask

"""5. Window Creation"""

def create_windows(arr):
    X, Y = [], []
    n = arr.shape[0]

    for i in range(n - config.WINDOW_SIZE + 1):
        X.append(arr[i:i + config.WINDOW_SIZE])
        Y.append(arr[i:i + config.WINDOW_SIZE])

    return np.array(X), np.array(Y)

"""6. Full Preprocessing for all Models"""

def preprocess_all(uploaded_df=None):

    import numpy as np
    import os
    import joblib
    import config

    # ----------------------------
    # Reproducibility
    # ----------------------------
    np.random.seed(config.SEED)

    # ----------------------------
    # Load + prepare data
    # ----------------------------
    data_df = load_and_prepare_data(uploaded_df)

    train_df, test_df = split_train_test(data_df)

    train_months = train_df[["Month_sin", "Month_cos"]].values
    test_months = test_df[["Month_sin", "Month_cos"]].values

    # ----------------------------
    # Load FIXED scaler (IMPORTANT)
    # ----------------------------
    scaler_path = resource_path("saved_models/scaler.pkl")
    scaler = joblib.load(scaler_path)

    # Apply scaler (DO NOT FIT)
    train_precip = train_df[["Precip_log"]]
    test_precip = test_df[["Precip_log"]]

    train_scaled_precip = scaler.transform(train_precip)
    test_scaled_precip = scaler.transform(test_precip)

    # Safety clamp (avoid out-of-range values)
    train_scaled_precip = np.clip(train_scaled_precip, 0, 1)
    test_scaled_precip = np.clip(test_scaled_precip, 0, 1)

    # Add seasonal features
    train_month = train_df[["Month_sin", "Month_cos"]].values
    test_month = test_df[["Month_sin", "Month_cos"]].values

    train_scaled = np.concatenate([train_scaled_precip, train_month], axis=1)
    test_scaled = np.concatenate([test_scaled_precip, test_month], axis=1)

    # ----------------------------
    # MASK HANDLING (EXE SAFE)
    # ----------------------------
    rate_tag = int(config.TEST_MISSING_RATE * 100)

    # Use writable directory (NOT _MEIPASS)
    mask_dir = os.path.join(os.getcwd(), "saved_masks")
    os.makedirs(mask_dir, exist_ok=True)

    train_mask_path = os.path.join(mask_dir, f"train_mask_{rate_tag}.npy")
    test_mask_path = os.path.join(mask_dir, f"test_mask_{rate_tag}.npy")

    use_saved = (
        os.path.exists(train_mask_path)
        and os.path.exists(test_mask_path)
    )

    # ----------------------------
    # LOAD OR GENERATE MASKS
    # ----------------------------
    if use_saved:
        train_mask = np.load(train_mask_path)
        test_mask = np.load(test_mask_path)

        # Validate shape (VERY IMPORTANT)
        if len(train_mask) != len(train_scaled) or len(test_mask) != len(test_scaled):
            use_saved = False

    if not use_saved:
        print(f"⚠️ Generating new masks for {rate_tag}%")

        train_masked, train_mask = apply_mar_mask(
            train_scaled,
            train_months,
            config.TRAIN_MISSING_RATE
        )

        test_masked, test_mask = apply_mar_mask(
            test_scaled,
            test_months,
            config.TEST_MISSING_RATE
        )

        np.save(train_mask_path, train_mask)
        np.save(test_mask_path, test_mask)

    else:
        print(f"✅ Using saved masks for {rate_tag}%")

        train_masked = train_scaled.copy()
        test_masked = test_scaled.copy()

        train_masked[train_mask.squeeze() == 0, 0] = np.nan
        test_masked[test_mask.squeeze() == 0, 0] = np.nan

    # ----------------------------
    # FILL MISSING VALUES
    # ----------------------------
    train_filled = np.nan_to_num(train_masked, nan=0.5)
    test_filled = np.nan_to_num(test_masked, nan=0.5)

    # ----------------------------
    # COMBINE WITH MASK
    # ----------------------------
    train_combined = np.concatenate([train_filled, train_mask], axis=1)
    test_combined = np.concatenate([test_filled, test_mask], axis=1)

    # ----------------------------
    # WINDOWING
    # ----------------------------
    X_train_masked, _ = create_windows(train_combined)
    Y_train, _ = create_windows(train_scaled)

    X_test, _ = create_windows(test_combined)
    Y_test, _ = create_windows(test_scaled)

    train_mask_windows, _ = create_windows(train_mask)
    test_mask_windows, _ = create_windows(test_mask)

    # ----------------------------
    # RETURN STRUCTURED DATA
    # ----------------------------
    return {
        "X_train_masked": X_train_masked,
        "Y_train": Y_train,
        "X_test": X_test,
        "Y_test": Y_test,
        "train_scaled": train_scaled,
        "test_scaled": test_scaled,
        "scaler": scaler,
        "train_mask": train_mask,
        "test_mask": test_mask,
        "train_masked": train_filled,
        "test_masked": test_filled,
        "train_mask_windows": train_mask_windows,
        "test_mask_windows": test_mask_windows
    }