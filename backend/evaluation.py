import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import os
import sys

import config

def get_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

MODEL_DIR = get_path("saved_models")
MASK_DIR = get_path("saved_masks")

CUSTOM_OBJECTS = {
    "mse": tf.keras.losses.MeanSquaredError,
    "mae": tf.keras.losses.MeanAbsoluteError,
    "MeanSquaredError": tf.keras.losses.MeanSquaredError,
    "MeanAbsoluteError": tf.keras.losses.MeanAbsoluteError,
}

def is_e2e_gan(model):
    return len(model.input_shape[0]) == 3


def safe_load_model(path, custom_objects):
    if not os.path.exists(path):
        print(f"[WARNING] Model not found: {path}")
        return None

    return keras.models.load_model(
        path,
        compile=False,
        safe_mode=False,
        custom_objects=custom_objects
    )

def is_gan_model(model):
    return isinstance(model.input, list) and len(model.input) == 3

def reconstruct_series(windows):

    import numpy as np

    windows = np.array(windows)

    if windows.ndim == 3:
        windows = windows[:, :, 0]
    elif windows.ndim != 2:
        raise ValueError(f"Unexpected window shape: {windows.shape}")

    n_windows, window_size = windows.shape
    series_len = n_windows + window_size - 1

    # 🔥 Center-weighting (LESS smoothing bias)
    center = window_size // 2
    weights = np.array([
        1.0 / (1 + abs(j - center)) for j in range(window_size)
    ])

    # Normalize weights
    weights = weights / weights.sum()

    # Storage
    values = [[] for _ in range(series_len)]
    weight_store = [[] for _ in range(series_len)]

    # Collect predictions
    for i in range(n_windows):
        for j in range(window_size):
            t = i + j
            values[t].append(windows[i, j])
            weight_store[t].append(weights[j])

    # 🔥 Weighted median approximation via weighted mean
    series = np.zeros(series_len)

    for t in range(series_len):

        if len(values[t]) == 0:
            series[t] = 0.0  # safety fallback
            continue

        v = np.array(values[t])
        w = np.array(weight_store[t])

        # Normalize weights per timestep
        w = w / (w.sum() + 1e-8)

        # Weighted average (less smoothing than simple mean)
        series[t] = np.sum(v * w)

    return series


def inverse_transform(series, scaler):

    series = series.reshape(-1, 1)

    # 🔥 SAFETY CLIP (prevents WGAN explosion)
    series = np.clip(series, 0, 1)

    # Step 1: reverse MinMax scaling
    log_vals = scaler.inverse_transform(series)

    # Step 2: reverse log1p transform
    real_vals = np.expm1(log_vals)

    return real_vals.flatten()


def compute_metrics_window(y_true, y_pred, mask_windows):

    # Only evaluate missing values
    missing = (mask_windows.squeeze(-1) == 0)

    true_vals = y_true[:, :, 0][missing]
    pred_vals = y_pred[missing]

    rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))
    mae = np.mean(np.abs(true_vals - pred_vals))

    # MASE (use full true series flattened)
    flat_true = y_true[:, :, 0].flatten()
    naive_error = np.mean(np.abs(flat_true[1:] - flat_true[:-1]))

    mase = mae / (naive_error + 1e-8)

    return rmse, mae, mase

def naive_mae(true_series):
    diffs = np.abs(true_series[1:] - true_series[:-1])
    return np.mean(diffs)


def mase(true_vals, pred_vals, train_series, missing_positions=None):
    model_mae = np.mean(np.abs(true_vals - pred_vals))
    naive_error = np.mean(np.abs(train_series[1:] - train_series[:-1]))
    return model_mae / (naive_error + 1e-8)

def compute_relative_improvement(results, baseline_key):

    baseline = results[baseline_key]
    rel_improvement = {}

    numeric_metrics = {
        k: v for k, v in baseline.items()
        if isinstance(v, (int, float))
    }

    for model_name, metrics in results.items():

        if model_name == baseline_key:
            continue

        rel_improvement[model_name] = {}

        for metric in numeric_metrics.keys():

            rel_improvement[model_name][metric] = (
                (baseline[metric] - metrics[metric]) /
                (baseline[metric] + 1e-8) * 100
            )

    return rel_improvement


def predict_seq_model(model, X_test, mask_windows):

    preds = model.predict(X_test)       # (batch, window, 1)
    preds = preds.squeeze(-1)           # (batch, window)

    mask = mask_windows.squeeze(-1)     # (batch, window)
    observed = X_test[:, :, 0]          # rainfall only

    final = mask * observed + (1 - mask) * preds

    return final, preds


def predict_gan(generator, X_test, test_mask_windows):

    window_preds = []

    for i, window in enumerate(X_test):
        rng = np.random.default_rng(config.SEED + i)

        rainfall_window = window[:, 0]

        if is_e2e_gan(generator):

            month_window = window[:, 2:]

            cond = np.concatenate(
                [rainfall_window.reshape(-1,1), month_window],
                axis=1
            )

            cond = cond.reshape(1, config.WINDOW_SIZE, cond.shape[1])

            mask = test_mask_windows[i].reshape(1, config.WINDOW_SIZE, 1)
            mask = np.repeat(mask, cond.shape[2], axis=2)

            noise = rng.normal(0, 1, (1, config.GAN_E2E_NOISE_DIM))

            pred = generator.predict([cond, mask, noise], verbose=0)

            pred = pred[:, :, 0].flatten()

        else:
            expected_shape = generator.input_shape[0]
            expected_dim = expected_shape[1]

            if expected_dim == config.WINDOW_SIZE:
                cond = rainfall_window.reshape(1, config.WINDOW_SIZE)

            else:
                num_features_expected = expected_dim // config.WINDOW_SIZE
                cond_window = window[:, :num_features_expected]
                cond = cond_window.reshape(1, -1)

            mask = test_mask_windows[i].reshape(1, config.WINDOW_SIZE)
            noise = rng.normal(0, 1, (1, config.GAN_NOISE_DIM))

            pred = generator.predict([cond, mask, noise], verbose=0)

            # Output handling
            if pred.ndim == 2 and pred.shape[1] > config.WINDOW_SIZE:
                num_features = pred.shape[1] // config.WINDOW_SIZE
                pred = pred.reshape(1, config.WINDOW_SIZE, num_features)

            if pred.ndim == 3:
                pred = pred[:, :, 0]

            pred = pred.flatten()

        window_preds.append(pred)

    window_preds = np.array(window_preds)

    observed = X_test[:, :, 0]
    mask = test_mask_windows.squeeze(-1)

    final = mask * observed + (1 - mask) * window_preds

    return final, window_preds
