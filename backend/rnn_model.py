import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
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
            f"[PSO-RNN] Iteration {iteration} completed | "
            f"Best Val RMSE: {best_score:.5f} | "
            f"Elapsed: {elapsed/60:.2f} min"
        )

    def on_finish(self, best_params, best_score):
        total_time = time.time() - self.start_time
        print("\n" + "="*50)
        print("[PSO-RNN] OPTIMIZATION FINISHED")
        print(f"Best Validation RMSE: {best_score:.5f}")
        print("Best Parameters:")
        print(f"  units1 = {int(best_params[0])}")
        print(f"  units2 = {int(best_params[1])}")
        print(f"  learning_rate = {10 ** best_params[2]:.6f}")
        print(f"Total PSO Time: {total_time/60:.2f} minutes")
        print("="*50)

class PSOEarlyStopping:
    def __init__(self, patience=3, min_delta=1e-4):
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

class CyclicGate(layers.Layer):
    def __init__(self, init_cycle_length, **kwargs):
        super(CyclicGate, self).__init__(**kwargs)
        self.init_cycle_length = float(init_cycle_length)

    def build(self, input_shape):

        self.amplitude = self.add_weight(
            name="amplitude",
            shape=(1,),
            initializer="zeros",
            trainable=True
        )

        self.phase = self.add_weight(
            name="phase",
            shape=(1,),
            initializer="zeros",
            trainable=True
        )

        self.bias = self.add_weight(
            name="bias",
            shape=(1,),
            initializer="zeros",
            trainable=True
        )

        self.log_cycle = self.add_weight(
            name="log_cycle",
            shape=(1,),
            initializer=keras.initializers.Constant(np.log(self.init_cycle_length)),
            trainable=True
        )

        super().build(input_shape)

    def call(self, x):

        t = tf.cast(tf.range(start=0, limit=tf.shape(x)[1]), tf.float32)

        cycle_length = tf.exp(self.log_cycle)

        cycle = tf.sin(2 * np.pi * t / cycle_length + self.phase)

        gate = tf.sigmoid(self.amplitude * cycle + self.bias)

        gate = tf.reshape(gate, (1, -1, 1))

        return x * (1.0 + gate)

    # 🔥 REQUIRED FIX
    def get_config(self):
        config = super().get_config()
        config.update({
            "init_cycle_length": self.init_cycle_length,
        })
        return config

    # 🔥 (Optional but recommended)
    @classmethod
    def from_config(cls, config):
        return cls(**config)

def build_baseline_seq2seq(num_features, units1=64, units2=32, learning_rate=None):

    inp = keras.Input(
        shape=(config.WINDOW_SIZE, num_features)
    )

    x = layers.LSTM(units1, return_sequences=True)(inp)
    x = layers.LSTM(units2, return_sequences=True)(x)

    out = layers.TimeDistributed(
        layers.Dense(1)
    )(x)

    model = keras.Model(inp, out)

    lr = learning_rate if learning_rate is not None else config.LEARNING_RATE

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.Huber(delta=1.0)
    )

    return model


def build_cyclic_seq2seq(num_features):

    inp = keras.Input(
        shape=(config.WINDOW_SIZE, num_features)
    )

    x = layers.LSTM(64, return_sequences=True)(inp)
    x = layers.LSTM(32, return_sequences=True)(x)

    x = CyclicGate(
        init_cycle_length=config.WINDOW_SIZE
    )(x)

    out = layers.TimeDistributed(
        layers.Dense(1)
    )(x)

    model = keras.Model(inp, out)

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=config.LEARNING_RATE
        ),
        loss=keras.losses.Huber(delta=1.0)
    )

    return model