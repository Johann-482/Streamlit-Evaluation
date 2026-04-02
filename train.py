import gradio as gr
import pandas as pd
import sys
import os
import tempfile
from collections import deque
from training_pipeline import main as training_main, SAVE_DIR
import threading
import contextlib
import time
import warnings
import tensorflow as tf
import socket

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")
tf.get_logger().setLevel('ERROR')

stop_training_flag = {"stop": False}


def exit_app():
    stop_training_flag["stop"] = True  # stop training if running

    def killer():
        time.sleep(0.2)
        os._exit(0)  # hard-kill the entire Python process immediately

    threading.Thread(target=killer).start()
    return "❌ Application terminated."


class StreamToQueue:
    def __init__(self):
        self.buffer = []

    def write(self, message):
        if message.strip():
            self.buffer.append(message)

    def flush(self):
        pass

def run_training(csv_file, missing_rate):

    stop_training_flag["stop"] = False

    if csv_file is None:
        yield "❌ No CSV file uploaded."
        return

    temp_dir = tempfile.gettempdir()
    temp_csv_path = os.path.join(temp_dir, "training_input.csv")
    pd.read_csv(csv_file.name).to_csv(temp_csv_path, index=False)

    log_lines = deque(maxlen=50)

    def update_log(line):
        log_lines.append(line)
        return "\n".join(log_lines)

    yield update_log("🔥 Starting training...\n")

    stream = StreamToQueue()

    def train():
        try:
            with contextlib.redirect_stdout(stream):
                training_main(temp_csv_path, missing_rate)
        except Exception as e:
            stream.write(f"❌ ERROR: {str(e)}")

    thread = threading.Thread(target=train)
    thread.start()

    last_index = 0

    while thread.is_alive():

        # 🔴 STOP CONDITION
        if stop_training_flag["stop"]:
            yield update_log("🛑 Training stopped by user.")
            return

        if len(stream.buffer) > last_index:
            for i in range(last_index, len(stream.buffer)):
                yield update_log(stream.buffer[i])
            last_index = len(stream.buffer)
        else:
            time.sleep(0.5)

    for line in stream.buffer:
        yield update_log(line)

    yield update_log(f"✅ Training completed!")
    yield update_log(f"📁 Models saved at: {SAVE_DIR}")


def stop_training():
    stop_training_flag["stop"] = True
    return "🛑 Stopping training..."

with gr.Blocks(title="Model Training App") as demo:

    gr.Markdown("## 🚀 Rainfall Model Training App")

    with gr.Row():

        with gr.Column(scale=1):
            csv_input = gr.File(
                label="📂 Upload CSV",
                file_types=[".csv"]
            )

            missing_rate = gr.Radio(
                choices=[0.15, 0.25, 0.50],
                value=0.15,
                label="⚙️ Missing Rate"
            )

            train_btn = gr.Button("🚀 Train Models", variant="primary")
            stop_btn = gr.Button("🛑 Stop Training", variant="stop")
            exit_btn = gr.Button("❌ Exit Application", variant="secondary")

        with gr.Column(scale=2):
            log_output = gr.Textbox(
                label="📜 Training Logs",
                lines=25,
                interactive=False
            )

    train_btn.click(
        run_training,
        inputs=[csv_input, missing_rate],
        outputs=log_output
    )

    stop_btn.click(
        stop_training,
        outputs=log_output
    )

    exit_btn.click(
        exit_app,
        outputs=log_output,
        js="window.close()"
    )

def find_free_port(default=7860):
    """Return a free port. If default is free, use it; otherwise pick another."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", default))
            return default
        except OSError:
            s.bind(("127.0.0.1", 0))  # random free port
            return s.getsockname()[1]
        
if __name__ == "__main__":
    port = find_free_port(7860)

    demo.queue().launch(
        server_name="127.0.0.1",
        server_port=port,
        inbrowser=True
    )