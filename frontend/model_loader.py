import os
import tensorflow as tf
from backend.rnn_model import CyclicGate
from backend.cnn_model import ReflectionPadding1D
import sys

def get_path(relative_path):
    if hasattr(sys, '_MEIPASS'):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)

SAVE_DIR = get_path("saved_models")


def load_all_models():

    models = {}

    custom_objects = {
        "CyclicGate": CyclicGate,
        "ReflectionPadding1D": ReflectionPadding1D
    }

    for file in os.listdir(SAVE_DIR):

        if file.endswith(".keras"):

            model_name = file.replace(".keras", "")

            models[model_name] = tf.keras.models.load_model(
                os.path.join(SAVE_DIR, file),
                compile=False,
                custom_objects=custom_objects,
                safe_mode=False
            )

    return models