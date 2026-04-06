import tensorflow as tf
from tensorflow import keras
import tensorflow.keras.backend as K
import numpy as np

import config
from backend.preprocessing import preprocess_all
from backend.cnn_model import (
    build_baseline_cnn,
    build_smoothed_cnn,
    PSOTracker,
    PSOEarlyStopping
)
CNN_BOUNDS = {
    "filters": (16, 128),
    "kernel_size": (2, 5),
    "log_lr": (-4, -2)
}

def masked_huber_loss():

    def loss(y_true, y_pred):

        # Extract components
        y_true_precip = y_true[..., :1]   # rainfall
        mask = y_true[..., -1:]           # observed (1) / missing (0)

        missing = 1.0 - mask

        # Base Huber loss
        error = tf.keras.losses.huber(y_true_precip, y_pred)
        error = tf.expand_dims(error, axis=-1)

        # 🔥 KEY: rainfall-based weighting
        # Higher rainfall → higher penalty
        weights = 1.0 + 3.0 * y_true_precip

        # Apply mask + weights
        weighted_error = error * missing * weights

        return tf.reduce_mean(weighted_error)

    return loss


def pso_objective_cnn(params, X_train, Y_train, mask_windows):

    filters = int(params[0])
    kernel_size = max(2, int(round(params[1])))
    lr = 10 ** params[2]

    model = build_baseline_cnn(
        num_features=X_train.shape[-1],
        filters=filters,
        kernel_size=kernel_size,
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

    # precipitation only
    y_true_precip = Y_train[..., :1]
    missing = 1 - mask_windows

    rmse = np.sqrt(np.mean(((pred - y_true_precip) * missing) ** 2))

    return rmse

def train_baseline_cnn(X_train, Y_train, mask_windows):

    print("\n===== [CNN-BASE] Training Started =====", flush=True)

    model = build_baseline_cnn(num_features=X_train.shape[-1])

    Y_train_with_mask = np.concatenate([Y_train, mask_windows], axis=-1)

    model.compile(
        optimizer=model.optimizer,
        loss=masked_huber_loss()
    )

    missing_weight = 1 - mask_windows

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
            f"[CNN-BASE] Epoch {epoch+1}/{config.EPOCHS} | Loss: {loss:.6f}",
            flush=True
        )

    return model

def train_smoothed_cnn(X_train, Y_train, mask_windows):

    print("\n===== [CNN-SMOOTHED] Training Started =====", flush=True)
    model = build_smoothed_cnn(num_features=X_train.shape[-1])

    Y_train_with_mask = np.concatenate([Y_train, mask_windows], axis=-1)

    model.compile(
        optimizer=model.optimizer,
        loss=masked_huber_loss()
    )

    missing_weight = 1 - mask_windows

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
            f"[CNN-SMOOTHED] Epoch {epoch+1}/{config.EPOCHS} | Loss: {loss:.6f}",
            flush=True
        )
    return model

def train_pso_cnn(X_train, Y_train, mask_windows):

    tracker = PSOTracker()
    early = PSOEarlyStopping(patience=3)

    particles = np.zeros((config.N_PARTICLES, 3))

    particles[:, 0] = np.random.uniform(*CNN_BOUNDS["filters"], config.N_PARTICLES)
    particles[:, 1] = np.random.uniform(*CNN_BOUNDS["kernel_size"], config.N_PARTICLES)
    particles[:, 2] = np.random.uniform(*CNN_BOUNDS["log_lr"], config.N_PARTICLES)
    velocities = np.zeros_like(particles)

    personal_best = particles.copy()
    personal_best_scores = np.full(config.N_PARTICLES, np.inf)

    global_best = None
    global_best_score = np.inf

    for iteration in range(config.N_ITERATIONS):

        print(f"[PSO-CNN] Iteration {iteration+1}/{config.N_ITERATIONS}", flush=True)

        for i in range(config.N_PARTICLES):

            score = pso_objective_cnn(
                particles[i], X_train, Y_train, mask_windows
            )

            filters_i = int(particles[i][0])
            kernel_i = int(round(particles[i][1]))
            lr_i = 10 ** particles[i][2]

            print(
                f"  [PSO-CNN] Particle {i+1}/{config.N_PARTICLES} | "
                f"filters={filters_i}, kernel={kernel_i}, lr={lr_i:.6f} | "
                f"RMSE={score:.6f}",
                flush=True
            )

            if score < personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best[i] = particles[i].copy()

            if score < global_best_score:
                print(f"   [PSO-CNN] New global best: {score:.6f}", flush=True)
                global_best_score = score
                global_best = particles[i].copy()

        print(
            f"[PSO-CNN] Iteration {iteration+1} | Best RMSE: {global_best_score:.6f}",
            flush=True
        )

        tracker.on_iteration_end(iteration, global_best_score)

        # ✅ UPDATE PARTICLES (use DIFFERENT variable name!)
        for j in range(config.N_PARTICLES):

            velocities[j] = (
                config.W * velocities[j]
                + config.C1 * np.random.rand() * (personal_best[j] - particles[j])
                + config.C2 * np.random.rand() * (global_best - particles[j])
            )

            particles[j] += velocities[j]

            particles[j, 0] = np.clip(particles[j, 0], *CNN_BOUNDS["filters"])
            particles[j, 1] = np.clip(particles[j, 1], *CNN_BOUNDS["kernel_size"])
            particles[j, 2] = np.clip(particles[j, 2], *CNN_BOUNDS["log_lr"])

        # ✅ EARLY STOPPING
        if early.check(global_best_score):
            print("[PSO-CNN] Early stopping triggered", flush=True)
            break

    print(f"[PSO-CNN] Optimization Finished | Best RMSE: {global_best_score:.6f}", flush=True)
    tracker.on_finish(global_best, global_best_score)

    filters = int(global_best[0])
    kernel_size = max(2, int(round(global_best[1])))
    lr = 10 ** global_best[2]

    print(
        f"[PSO-CNN] Training final model | "
        f"filters={filters}, kernel={kernel_size}, lr={lr:.6f}",
        flush=True
    )

    model = build_baseline_cnn(
        num_features=X_train.shape[-1],
        filters=filters,
        kernel_size=kernel_size,
        learning_rate=lr
    )

    Y_train_with_mask = np.concatenate([Y_train, mask_windows], axis=-1)

    model.compile(
        optimizer=model.optimizer,
        loss=masked_huber_loss()
    )

    missing_weight = 1 - mask_windows

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
            f"[PSO-CNN] Final Training Epoch {epoch+1}/{config.EPOCHS} | Loss: {loss:.6f}",
            flush=True
        )

    return model

if __name__ == "__main__":
    data = preprocess_all()

    X_train = data["X_train_masked"]
    Y_train = data["Y_train"]
    mask_windows = data["train_mask_windows"]

    train_baseline_cnn(X_train, Y_train, mask_windows)
    train_smoothed_cnn(X_train, Y_train, mask_windows)
    train_pso_cnn(X_train, Y_train, mask_windows)