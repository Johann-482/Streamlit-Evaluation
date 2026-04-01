import sys
import os
import warnings
import config
import joblib
import pandas as pd
import tensorflow as tf
from backend.preprocessing import preprocess_all
from backend.rnn_train import train_baseline_rnn, train_cyclic_rnn, train_pso_rnn
from backend.cnn_train import train_baseline_cnn, train_smoothed_cnn, train_pso_cnn
from backend.gan_train import train_baseline_gan, train_e2e_gan, train_wgan_gp

def resource_path(relative_path):
    """Return absolute path to resource, in dev and PyInstaller"""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

def get_output_dir():
    # Always save beside the EXE (user-accessible)
    if getattr(sys, 'frozen', False):
        return os.path.join(os.path.dirname(sys.executable), "saved_models")
    else:
        return os.path.join(os.getcwd(), "saved_models")

SAVE_DIR = get_output_dir()
os.makedirs(SAVE_DIR, exist_ok=True)

def main(csv_path, missing_rate):
    print(missing_rate)
    config.TRAIN_MISSING_RATE = float(missing_rate)
    config.TEST_MISSING_RATE = float(missing_rate)

    print("Loading dataset...")
    df = pd.read_csv(csv_path)

    data = preprocess_all(df)
    scaler = data["scaler"]

    rate = int(config.TRAIN_MISSING_RATE * 100)

    scaler_path = os.path.join(SAVE_DIR, f"scaler.pkl")
    joblib.dump(scaler, scaler_path)

    print("Scaler saved:", scaler_path)

    X_train = data["X_train_masked"]
    Y_train = data["Y_train"]
    mask_windows = data["train_mask_windows"]

    print("Training RNN models...", flush=True)

    rnn_models = {
        "baseline_rnn": train_baseline_rnn(X_train, Y_train, mask_windows),
        "cyclic_rnn": train_cyclic_rnn(X_train, Y_train, mask_windows),
        "pso_rnn": train_pso_rnn(X_train, Y_train, mask_windows)
    }

    print("Training CNN models...", flush=True)

    cnn_models = {
        "baseline_cnn": train_baseline_cnn(X_train, Y_train, mask_windows),
        "smoothed_cnn": train_smoothed_cnn(X_train, Y_train, mask_windows),
        "pso_cnn": train_pso_cnn(X_train, Y_train, mask_windows)
    }

    print("Training GAN models...", flush=True)
    gan_models = {
        "baseline_gan": train_baseline_gan(data),
        "e2e_gan": train_e2e_gan(data),
        "wgan_gp": train_wgan_gp(data)
    }

    print("Saving models...")

    rate = int(config.TRAIN_MISSING_RATE * 100)

    for name, model in {**rnn_models, **cnn_models, **gan_models}.items():
        save_name = f"{name}_{rate}.keras"
        save_path = os.path.join(SAVE_DIR, save_name)

        model.save(save_path)

        print(f"Saved: {save_name}")

    print("Training completed successfully.")

if __name__ == "__main__":
    csv_path = sys.argv[1]
    missing_rate = float(sys.argv[2])

    main(csv_path, missing_rate)