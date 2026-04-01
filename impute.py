import gradio as gr
import pandas as pd
import joblib
import tempfile

from tensorflow import keras
from backend.imputation import impute_12_months

# ================= LOAD SCALER =================
scaler = joblib.load("saved_models/scaler.pkl")

# ================= MODEL STORAGE =================
loaded_models = {}

# ================= FUNCTIONS =================

def load_models(files):
    global loaded_models
    loaded_models.clear()

    if not files:
        return gr.Dropdown(choices=[], value=None)

    model_names = []

    for i, file in enumerate(files):
        try:
            # Save temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".keras") as tmp:
                tmp.write(file.read())
                tmp_path = tmp.name

            # Load model
            model = keras.models.load_model(tmp_path, compile=False)

            # Use clean display name
            name = f"Model {i+1}"
            loaded_models[name] = model
            model_names.append(name)

        except Exception as e:
            print(f"Error loading model {i}: {e}")

    # Force dropdown update properly
    return gr.Dropdown(
        choices=model_names,
        value=model_names[0] if model_names else None
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

        output_path = "imputed_output.csv"
        result_df.to_csv(output_path, index=False)

        return "✅ Imputation completed!", output_path

    except Exception as e:
        return f"❌ Error: {str(e)}", None


# ================= UI =================

with gr.Blocks() as app:

    gr.Markdown("## 🧩 Imputation Tool (12 Months)")
    gr.Markdown("Upload a CSV with EXACTLY 12 months (1 year).")

    model_files = gr.File(
        label="Upload Model(s) (.keras)",
        file_types=[".keras"],
        file_count="multiple"
    )

    model_dropdown = gr.Dropdown(
        label="Select Model",
        choices=[]
    )

    # 🔥 FIX: Proper dropdown update
    model_files.change(
        fn=load_models,
        inputs=model_files,
        outputs=model_dropdown
    )

    csv_input = gr.File(
        label="Upload CSV for Imputation",
        file_types=[".csv"]
    )

    run_button = gr.Button("Run Imputation")

    status_output = gr.Textbox(label="Status")
    file_output = gr.File(label="Download Imputed CSV")

    run_button.click(
        fn=run_imputation,
        inputs=[csv_input, model_dropdown],
        outputs=[status_output, file_output]
    )

# ================= RUN =================
if __name__ == "__main__":
    app.launch()