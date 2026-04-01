import gradio as gr
import pandas as pd
import joblib
import tempfile
import os
import threading
import time
import sys

from tensorflow import keras
from backend.imputation import impute_12_months
from backend.rnn_model import CyclicGate

def resource_path(relative_path):
    """Get correct absolute path for PyInstaller bundled files."""
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

# ================= LOAD SCALER =================
scaler_path = resource_path("saved_models/scaler.pkl")
scaler = joblib.load(scaler_path)

# ================= MODEL STORAGE =================
loaded_models = {}

# ================= FUNCTIONS =================

def load_models(files):
    global loaded_models
    loaded_models.clear()

    if not files:
        return gr.update(choices=[], value=None)

    model_names = []

    for i, file in enumerate(files):
        try:
            # 🔥 In Gradio v4, file is already a path
            file_path = file.name if hasattr(file, "name") else str(file)
            print("Uploaded filename raw repr:", repr(file.name))

            # Load model directly
            model = keras.models.load_model(
                file_path,
                compile=False,
                custom_objects={"CyclicGate": CyclicGate}
            )

            # Use real filename
            name = os.path.basename(file_path)

            loaded_models[name] = model
            model_names.append(name)

        except Exception as e:
            print(f"Error loading model {i}: {e}")

    if not model_names:
        return gr.update(
            choices=["❌ Failed to load models (check terminal)"],
            value=None
        )

    return gr.update(
        choices=model_names,
        value=model_names[0]
    )


def run_imputation(csv_file, model_name):
    if csv_file is None:
        return "❌ Please upload a CSV file.", None

    if model_name is None or model_name not in loaded_models:
        return "❌ Please select a valid model.", None

    try:
        df = pd.read_csv(csv_file.name)
        model = loaded_models[model_name]

        result_df = impute_12_months(df, model, scaler)

        output_path = os.path.join(tempfile.gettempdir(), "imputed_output.csv")
        result_df.to_csv(output_path, index=False)

        return "✅ Imputation completed!", output_path

    except Exception as e:
        return f"❌ Error: {str(e)}", None


def exit_app():
    def killer():
        time.sleep(0.2)
        os._exit(0)

    threading.Thread(target=killer).start()
    return "❌ Application terminated."


# ================= UI =================

with gr.Blocks(title="Imputation App") as app:

    gr.Markdown("## 🧩 Rainfall Imputation Tool")
    gr.Markdown("Upload trained models and a CSV with EXACTLY 12 months.")

    with gr.Row():

        # LEFT PANEL
        with gr.Column(scale=1):

            model_files = gr.File(
                label="📂 Upload Model(s) (.keras)",
                file_types=[".keras"],
                file_count="multiple"
            )

            model_dropdown = gr.Dropdown(
                label="🤖 Select Model",
                choices=[]
            )

            csv_input = gr.File(
                label="📂 Upload CSV",
                file_types=[".csv"]
            )

            run_button = gr.Button("🚀 Run Imputation", variant="primary")
            exit_button = gr.Button("❌ Exit Application", variant="secondary")

        # RIGHT PANEL
        with gr.Column(scale=2):

            status_output = gr.Textbox(
                label="📜 Status",
                lines=10,
                interactive=False
            )

            file_output = gr.File(label="📥 Download Imputed CSV")

    # 🔥 FIXED DROPDOWN UPDATE
    model_files.change(
        fn=load_models,
        inputs=model_files,
        outputs=model_dropdown
    )

    run_button.click(
        fn=run_imputation,
        inputs=[csv_input, model_dropdown],
        outputs=[status_output, file_output]
    )

    exit_button.click(
        exit_app,
        outputs=status_output,
        js="window.close()"
    )


# ================= RUN =================
if __name__ == "__main__":
    app.queue().launch(
        server_name="127.0.0.1",
        inbrowser=True
    )