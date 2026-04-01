import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Layer, Conv1D, AveragePooling1D, Add
import numpy as np
import time
import config

class PSOTracker:
    def __init__(self):
        self.start_time = time.time()
        self.history = []

    def on_iteration_end(self, iteration, best_score):
        elapsed = time.time() - self.start_time
        self.history.append(best_score)
        print(
            f"[PSO-CNN] Iteration {iteration} completed | "
            f"Best Val RMSE: {best_score:.5f} | "
            f"Elapsed: {elapsed/60:.2f} min"
        )

    def on_finish(self, best_params, best_score):
        total_time = time.time() - self.start_time
        print("\n" + "="*50)
        print("[PSO-CNN] OPTIMIZATION FINISHED (CNN Baseline)")
        print(f"Best Validation RMSE: {best_score:.5f}")
        print("Best Parameters:")
        print(f"  filters = {int(best_params[0])}")
        print(f"  kernel_size = {int(best_params[1])}")
        print(f"  learning_rate = {10 ** best_params[2]:.6f}")
        print(f"Total PSO Time: {total_time/60:.2f} minutes")
        print("="*50)


class PSOEarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = np.inf
        self.counter = 0

    def check(self, current_best):
        if self.best - current_best > self.min_delta:
            self.best = current_best
            self.counter = 0
        else:
            self.counter += 1

        return self.counter >= self.patience

class ReflectionPadding1D(Layer):
    def __init__(self, padding=1, **kwargs):
        super().__init__(**kwargs)
        self.padding = padding

    def call(self, inputs):
        return tf.pad(
            inputs,
            [[0, 0], [self.padding, self.padding], [0, 0]],
            mode="REFLECT"
        )

def smoothing_layer():
    return keras.Sequential([
        AveragePooling1D(pool_size=3, strides=1, padding='same')
    ])

def build_baseline_cnn(
    num_features,
    filters=32,
    kernel_size=3,
    learning_rate=None
):
    model = keras.Sequential([
        keras.Input(shape=(config.WINDOW_SIZE, num_features)),
        Conv1D(filters, kernel_size, padding='same', activation='relu'),
        Conv1D(filters, kernel_size, padding='same', activation='relu'),
        Conv1D(1, 1, padding='same', activation='linear')
    ])

    lr = learning_rate if learning_rate is not None else config.LEARNING_RATE

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.Huber(delta=1.0)
    )

    return model


def build_smoothed_cnn(num_features):

    inputs = keras.Input(
        shape=(config.WINDOW_SIZE, num_features)
    )

    x = Conv1D(32, 3, padding="same", activation="relu")(inputs)
    x = Conv1D(32, 3, padding="same", activation="relu")(x)

    smooth = AveragePooling1D(
        pool_size=3,
        strides=1,
        padding="same"
    )(x)

    x = Add()([x, smooth])

    outputs = Conv1D(1, 1, padding="same", activation="linear")(x)

    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=config.LEARNING_RATE
        ),
        loss=keras.losses.Huber(delta=1.0)
    )

    return model