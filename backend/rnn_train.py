import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np

import config
from backend.preprocessing import preprocess_all
from backend.rnn_model import (
    build_baseline_seq2seq,
    build_cyclic_seq2seq,
    PSOTracker,
    PSOEarlyStopping
)
RNN_BOUNDS = {
    "units1": (32, 128),
    "units2": (16, 64),
    "log_lr": (-4, -2)
}

def masked_huber_loss():

    def loss(y_true, y_pred):

        y_true_precip = y_true[..., :1]
        mask = y_true[..., -1:]

        missing = 1.0 - mask

        error = tf.keras.losses.huber(y_true_precip, y_pred)
        # 🔥 CRITICAL FIX
        error = tf.expand_dims(error, axis=-1)

        return tf.reduce_mean(error * missing)
    

    return loss

def pso_objective_rnn(params, X_train, Y_train, mask_windows):

    units1 = int(params[0])
    units2 = int(params[1])
    lr = 10 ** params[2]

    model = build_baseline_seq2seq(
        num_features=X_train.shape[-1],
        units1=units1,
        units2=units2,
        learning_rate=lr
    )
    Y_train_with_mask = np.concatenate([Y_train, mask_windows], axis=-1)

    model.compile(
        optimizer=model.optimizer,
        loss=masked_huber_loss()
    )

    model.fit(
        X_train,
        Y_train_with_mask,
        epochs=10,
        batch_size=config.BATCH_SIZE,
        verbose=0
    )

    pred = model.predict(X_train, verbose=0)

    # Extract precipitation only
    y_true_precip = Y_train[..., :1]

    missing = 1 - mask_windows

    rmse = np.sqrt(np.mean(((pred - y_true_precip) * missing) ** 2))

    return rmse

def train_baseline_rnn(X_train, Y_train, mask_windows):
    print("\n===== [RNN-BASE] Training Started =====", flush=True)

    model = build_baseline_seq2seq(num_features=X_train.shape[-1])
    Y_train_with_mask = np.concatenate([Y_train, mask_windows], axis=-1)

    model.compile(
        optimizer=model.optimizer,
        loss=masked_huber_loss()
    )

    for epoch in range(config.EPOCHS):
        history = model.fit(
            X_train,
            Y_train_with_mask,
            epochs=1,
            batch_size=config.BATCH_SIZE,
            verbose=0
        )

        loss = history.history["loss"][0]

        print(
            f"[RNN-BASE] Epoch {epoch+1}/{config.EPOCHS} | Loss: {loss:.6f}",
            flush=True
        )

    print("[RNN-BASE] Training finished\n", flush=True)

    return model

def train_cyclic_rnn(X_train, Y_train, mask_windows):
    print("\n===== [RNN-CYCLIC] Training Started =====", flush=True)

    model = build_cyclic_seq2seq(num_features=X_train.shape[-1])
    Y_train_with_mask = np.concatenate([Y_train, mask_windows], axis=-1)

    model.compile(
        optimizer=model.optimizer,
        loss=masked_huber_loss()
    )

    for epoch in range(config.EPOCHS):
        history = model.fit(
            X_train,
            Y_train_with_mask,
            epochs=1,
            batch_size=config.BATCH_SIZE,
            verbose=0
        )

        loss = history.history["loss"][0]

        print(
            f"[RNN-CYCLIC] Epoch {epoch+1}/{config.EPOCHS} | Loss: {loss:.6f}",
            flush=True
        )

    print("[RNN-CYCLIC] Training finished\n", flush=True)

    return model

def train_pso_rnn(X_train, Y_train, mask_windows):

    tracker = PSOTracker()
    early = PSOEarlyStopping(patience=3)

    particles = np.zeros((config.N_PARTICLES, 3))

    particles[:, 0] = np.random.uniform(*RNN_BOUNDS["units1"], config.N_PARTICLES)
    particles[:, 1] = np.random.uniform(*RNN_BOUNDS["units2"], config.N_PARTICLES)
    particles[:, 2] = np.random.uniform(*RNN_BOUNDS["log_lr"], config.N_PARTICLES)
    velocities = np.zeros_like(particles)

    personal_best = particles.copy()
    personal_best_scores = np.full(config.N_PARTICLES, np.inf)

    global_best = None
    global_best_score = np.inf

    for iteration in range(config.N_ITERATIONS):
        print(f"[PSO-RNN] Iteration {iteration+1}/{config.N_ITERATIONS}", flush=True)
        for i in range(config.N_PARTICLES):
            
            score = pso_objective_rnn(particles[i], X_train, Y_train, mask_windows)
            units1_i = int(particles[i][0])
            units2_i = int(particles[i][1])
            lr_i = 10 ** particles[i][2]

            print(
                f"  [PSO-RNN] Particle {i+1}/{config.N_PARTICLES} | "
                f"units1={units1_i}, units2={units2_i}, lr={lr_i:.6f} | "
                f"RMSE={score:.6f}",
                flush=True
            )

            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best[i] = particles[i].copy()

            if score < global_best_score:
                print(f"   [PSO-RNN] New global best: {score:.6f}", flush=True)
                global_best_score = score
                global_best = particles[i].copy()

        print(
            f"[PSO-RNN] Iteration {iteration+1}/{config.N_ITERATIONS} | "
            f"Best RMSE: {global_best_score:.6f}",
            flush=True
        )
        tracker.on_iteration_end(iteration, global_best_score)

        for i in range(config.N_PARTICLES):
            velocities[i] = (
                config.W * velocities[i]
                + config.C1 * np.random.rand() * (personal_best[i] - particles[i])
                + config.C2 * np.random.rand() * (global_best - particles[i])
            )
            particles[i] += velocities[i]
            particles[i, 0] = np.clip(particles[i, 0], *RNN_BOUNDS["units1"])
            particles[i, 1] = np.clip(particles[i, 1], *RNN_BOUNDS["units2"])
            particles[i, 2] = np.clip(particles[i, 2], *RNN_BOUNDS["log_lr"])

        if early.check(global_best_score):
            print("[PSO-RNN] Early stopping activated", flush=True)
            break

    units1 = int(global_best[0])
    units2 = int(global_best[1])
    lr = 10 ** global_best[2]

    print(f"[PSO-RNN] Optimization Finished | Best RMSE: {global_best_score:.6f}", flush=True)
    print(f"[PSO-RNN] Best params: units1={units1}, units2={units2}, lr={lr:.6f}", flush=True)

    tracker.on_finish(global_best, global_best_score)

    model = build_baseline_seq2seq(
        num_features=X_train.shape[-1],
        units1=units1,
        units2=units2,
        learning_rate=lr
    )
    Y_train_with_mask = np.concatenate([Y_train, mask_windows], axis=-1)

    model.compile(
        optimizer=model.optimizer,
        loss=masked_huber_loss()
    )
    print(
        f"[PSO-RNN] Training final model | "
        f"units1={units1}, units2={units2}, lr={lr:.6f}",
        flush=True
    )

    for epoch in range(config.EPOCHS):
        history = model.fit(
            X_train,
            Y_train_with_mask,
            epochs=1,
            batch_size=config.BATCH_SIZE,
            verbose=0
        )

        loss = history.history["loss"][0]

        print(
            f"[PSO-RNN] Final Training Epoch {epoch+1}/{config.EPOCHS} | Loss: {loss:.6f}",
            flush=True
        )

    return model

if __name__ == "__main__":
    data = preprocess_all()

    X_train = data["X_train_masked"]
    Y_train = data["Y_train"]
    mask_windows = data["train_mask_windows"]

    train_baseline_rnn(X_train, Y_train, mask_windows)
    train_cyclic_rnn(X_train, Y_train, mask_windows)
    train_pso_rnn(X_train, Y_train, mask_windows)