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

    windows = np.array(windows)

    # Ensure 2D shape (n_windows, window_size)
    if windows.ndim == 3:
        windows = windows[:, :, 0]
    elif windows.ndim != 2:
        raise ValueError(f"Unexpected window shape: {windows.shape}")

    n_windows = windows.shape[0]
    window_size = windows.shape[1]

    series_len = n_windows + window_size - 1

    series = np.zeros(series_len)
    counts = np.zeros(series_len)

    for i in range(n_windows):
        for j in range(window_size):
            series[i + j] += float(windows[i, j])
            counts[i + j] += 1

    return series / (counts + 1e-8)


def inverse_transform(series, scaler):

    series = series.reshape(-1, 1)

    # Step 1: reverse MinMax scaling
    log_vals = scaler.inverse_transform(series)

    # Step 2: reverse log1p transform
    real_vals = np.expm1(log_vals)

    return real_vals.flatten()


def compute_metrics(true_series, pred_series, missing_positions):

    true_vals = true_series[missing_positions]
    pred_vals = pred_series[missing_positions]

    rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))
    mae = np.mean(np.abs(true_vals - pred_vals))
    mase_val = mase(true_vals, pred_vals, true_series, missing_positions)

    return rmse, mae, mase_val

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

    return reconstruct_series(final)


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

    return reconstruct_series(final)
