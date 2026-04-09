import streamlit as st
import pandas as pd
import numpy as np
import os
import tensorflow as tf
import config
import sys
import webbrowser
import tempfile

from backend.preprocessing import (create_windows, preprocess_all)
from backend.evaluation import (
    predict_gan,
    predict_seq_model,
    inverse_transform,
    reconstruct_series,
    compute_metrics_window,
    compute_relative_improvement,
    is_gan_model
)
from backend.rnn_model import CyclicGate
from backend.cnn_model import ReflectionPadding1D

from frontend.visualization import (
    plot_imputation_scatter,
    show_family_metrics_and_improvement
)

def compute_metrics_switchable(y_true, y_pred, mask_windows, scaler, mode="scaled"):
    """
    Compute metrics either in:
    - 'scaled' (log + normalized)
    - 'real' (mm scale)
    """

    missing = (mask_windows.squeeze(-1) == 0)

    true_vals = y_true[:, :, 0][missing]
    pred_vals = y_pred[missing]

    # ----------------------------
    # SWITCH: REAL SPACE
    # ----------------------------
    if mode == "real":
        true_vals = inverse_transform(true_vals, scaler)
        pred_vals = inverse_transform(pred_vals, scaler)

    # ----------------------------
    # METRICS
    # ----------------------------
    rmse = np.sqrt(np.mean((true_vals - pred_vals) ** 2))
    mae = np.mean(np.abs(true_vals - pred_vals))

    # MASE (always computed in same space as values)
    flat_true = y_true[:, :, 0].flatten()

    if mode == "real":
        flat_true = inverse_transform(flat_true, scaler)

    naive_error = np.mean(np.abs(flat_true[1:] - flat_true[:-1]))

    mase = mae / (naive_error + 1e-8)

    return rmse, mae, mase

def masked_huber_loss():
    def loss(y_true, y_pred):
        y_true_precip = y_true[..., :1]
        mask = y_true[..., -1:]
        missing = 1.0 - mask
        error = tf.keras.losses.huber(y_true_precip, y_pred)
        error = tf.expand_dims(error, axis=-1)
        return tf.reduce_mean(error * missing)
    return loss

def get_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)
def safe_load_keras_model(path, custom_objects):
    try:
        # First attempt (normal load)
        model = tf.keras.models.load_model(
            path,
            compile=False,
            custom_objects=custom_objects
        )
        return model

    except Exception as e:
        print(f"[Retry Load] Standard load failed: {e}")

        # 🔥 Rebuild manually (critical for LSTM issue)
        from backend.rnn_model import build_baseline_seq2seq

        # You MUST match training config
        model = build_baseline_seq2seq(num_features=3)  # adjust if needed

        model.load_weights(path)
        return model
    
SAVE_DIR = get_path("saved_models")
os.makedirs(SAVE_DIR, exist_ok=True)

st.set_page_config(page_title="Rainfall Imputation Evaluation", layout="wide")

def load_uploaded_models(uploaded_files):

    models = {}

    custom_objects = {
        "CyclicGate": CyclicGate,
        "ReflectionPadding1D": ReflectionPadding1D,
        "loss": masked_huber_loss(),
        "Huber": tf.keras.losses.Huber
    }

    for file in uploaded_files:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            model = safe_load_keras_model(tmp_path, custom_objects)

            model_name = file.name.replace(".keras", "").replace(".h5", "")
            models[model_name] = model

        except Exception as e:
            st.error(f"Failed to load {file.name}: {e}")

    return models

# ----------------------------
# SESSION STATE INIT (SAFE INIT FUNCTION)
# ----------------------------
def init_state():
    defaults = {
        "data_ready": False,
        "models_loaded": False,
        "preds": {},
        "metrics": {},
        "models": {},
        "true_series": None,
        "missing_positions": None
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ----------------------------
# SIDEBAR (CONTROLS)
# ----------------------------
st.sidebar.title("Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV (Year, Month, Precipitation)", type=["csv"]
)
uploaded_models = st.sidebar.file_uploader(
    "Upload trained models (.keras / .h5)",
    type=["keras", "h5"],
    accept_multiple_files=True
)

missing_rate_choice = st.sidebar.radio(
    "Missing Rate",
    ["15%", "25%", "50%"],
    index=1
)
metric_mode = st.sidebar.radio(
    "Metric Mode",
    ["Scaled (log-based)", "Real (mm)"],
    index=0
)

missing_rate_map = {"15%": 0.15, "25%": 0.25, "50%": 0.50}
selected_rate = missing_rate_map[missing_rate_choice]

# ----------------------------
# DATA RESET ON NEW UPLOAD
# ----------------------------
if uploaded_file is not None:
    if "last_file" not in st.session_state or st.session_state.last_file != uploaded_file.name:
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state.last_file = uploaded_file.name
        init_state()  # IMPORTANT: reinitialize after clear

# ----------------------------
# PREPROCESS BUTTON
# ----------------------------
if st.sidebar.button("Run Evaluation Setup"):
    df = pd.read_csv(uploaded_file)

    config.TEST_MISSING_RATE = selected_rate
    config.TRAIN_MISSING_RATE = selected_rate

    data = preprocess_all(df)

    st.session_state.X_test = data["X_test"]
    st.session_state.Y_test = data["Y_test"]
    st.session_state.test_mask = data["test_mask"]
    st.session_state.scaler = data["scaler"]

    st.session_state.test_mask_windows, _ = create_windows(data["test_mask"])

    # Ground truth
    true_series = reconstruct_series(data["Y_test"])
    true_series = inverse_transform(true_series, data["scaler"])

    st.session_state.true_series = true_series
    missing_positions = np.where(data["test_mask"] == 0)[0]

    # 🔥 ALIGN WITH RECONSTRUCTED SERIES LENGTH
    max_len = len(true_series)  # or pred_series (same length ideally)
    st.session_state.missing_positions = missing_positions[missing_positions < max_len]

    st.session_state.data_ready = True
    st.session_state.models_loaded = False
    st.session_state.preds = {}
    st.session_state.metrics = {}

# ----------------------------
# LOAD MODELS + PRECOMPUTE
# ----------------------------
if st.sidebar.button("Load & Precompute Models"):
    if not st.session_state.data_ready:
        st.warning("Run evaluation setup first.")
    else:
        with st.spinner("Loading models..."):
            if not uploaded_models:
                st.warning("Please upload at least one model.")
            else:
                with st.spinner("Loading uploaded models..."):
                    models = load_uploaded_models(uploaded_models)

        rate_tag = int(selected_rate * 100)

        preds = {}
        metrics = {}

        for name, model in models.items():
            if is_gan_model(model):
                final_windows, raw_preds = predict_gan(
                    model,
                    st.session_state.X_test,
                    st.session_state.test_mask_windows
                )
            else:
                final_windows, raw_preds = predict_seq_model(
                    model,
                    st.session_state.X_test,
                    st.session_state.test_mask_windows
                )

            # ✅ UNBIASED METRICS (WINDOW LEVEL)
            mode = "scaled" if metric_mode == "Scaled (log-based)" else "real"

            rmse, mae, mase = compute_metrics_switchable(
                st.session_state.Y_test,
                raw_preds,
                st.session_state.test_mask_windows,
                st.session_state.scaler,
                mode=mode
            )

            metrics[name] = {"RMSE": rmse, "MAE": mae, "MASE": mase}

            # ----------------------------
            # 🔵 RECONSTRUCTION (ONLY FOR PLOTTING)
            # ----------------------------
            reconstructed = reconstruct_series(final_windows)
            reconstructed = inverse_transform(reconstructed, st.session_state.scaler)

            preds[name] = reconstructed

            metrics[name] = {"RMSE": rmse, "MAE": mae, "MASE": mase}

        st.session_state.models = models
        st.session_state.preds = preds
        st.session_state.metrics = metrics
        st.session_state.models_loaded = True

# ----------------------------
# MODEL FILTER
# ----------------------------
filter_option = st.sidebar.radio(
    "Model Display",
    [
        "All",
        "RNN",
        "CNN",
        "GAN",
        "Baselines",
        "Custom"
    ]
)

selected_models = []

if st.session_state.models_loaded:
    models = st.session_state.models

    if filter_option == "All":
        selected_models = list(models.keys())

    elif filter_option == "RNN":
        selected_models = [k for k in models if "rnn" in k.lower()]

    elif filter_option == "CNN":
        selected_models = [k for k in models if "cnn" in k.lower()]

    elif filter_option == "GAN":
        selected_models = [k for k in models if "gan" in k.lower()]

    elif filter_option == "Baselines":
        selected_models = [k for k in models if "baseline" in k.lower()]

    elif filter_option == "Custom":
        selected_models = st.sidebar.multiselect(
            "Select models",
            list(models.keys()),
            max_selections=9
        )

selected_models = selected_models[:9]

# ----------------------------
# MAIN DISPLAY
# ----------------------------
st.title("Rainfall Imputation Evaluation")

# ----------------------------
# MAIN DISPLAY LOGIC (3 PHASES)
# ----------------------------

# CASE 1: File uploaded but NOT evaluated yet → show raw ground truth
if uploaded_file is not None and not st.session_state.data_ready:
    df_preview = pd.read_csv(uploaded_file)

    if "Precipitation" in df_preview.columns:
        st.subheader("Raw Ground Truth (No Missing Mask Yet)")
        st.line_chart(df_preview["Precipitation"])

# CASE 2: After evaluation setup → show GT + missing points only
elif st.session_state.data_ready and not st.session_state.models_loaded:
    st.subheader("Ground Truth with Missing Points")

    to_plot = {"Ground Truth": st.session_state.true_series}

    plot_imputation_scatter(
        st.session_state.true_series,
        to_plot,
        st.session_state.missing_positions
    )

    st.write("Missing Count:", len(st.session_state.missing_positions))

# CASE 3: Models loaded → ALWAYS show plot
elif st.session_state.models_loaded:
    st.subheader("Imputation Results")

    to_plot = {"Ground Truth": st.session_state.true_series}

    # Add selected model predictions (if any)
    for name in selected_models:
        if name in st.session_state.preds:
            to_plot[name] = st.session_state.preds[name]

    plot_imputation_scatter(
        st.session_state.true_series,
        to_plot,
        st.session_state.missing_positions
    )

    # Show metrics ONLY if models selected
    if len(selected_models) > 0:
        df = pd.DataFrame({k: st.session_state.metrics[k] for k in selected_models}).T

        st.subheader(f"Metrics ({metric_mode})")
        st.dataframe(df)

        show_family_metrics_and_improvement(df.to_dict("index"), compute_relative_improvement)
