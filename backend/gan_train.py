import numpy as np
import tensorflow as tf
from tensorflow import keras
import config
import os
import pandas as pd

from backend.preprocessing import preprocess_all, create_windows
from backend.gan_model import (
    build_baseline_generator,
    build_e2e_generator,
    build_baseline_discriminator,
    build_e2e_discriminator,
    build_wgan_critic,
    build_gan
)

SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)
batch_size = config.BATCH_SIZE


def gradient_penalty(critic, real, fake, condition, mask):
    """Compute WGAN-GP gradient penalty."""
    batch = tf.shape(real)[0]

    # Correct WGAN-GP alpha (one alpha per sample)
    alpha = tf.random.uniform((batch, 1), 0., 1.)
    alpha = tf.broadcast_to(alpha, tf.shape(real))

    # Correct interpolation
    interp = real * alpha + fake * (1 - alpha)

    with tf.GradientTape() as tape:
        tape.watch(interp)
        pred = critic([interp, condition, mask])

    grad = tape.gradient(pred, interp)
    grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad), axis=1) + 1e-8)

    return tf.reduce_mean((grad_norm - 1.0) ** 2)


def train_baseline_gan(data):

    X_train = data["X_train_masked"]
    Y_train = data["Y_train"]
    train_mask = data["train_mask"]

    window_size = config.WINDOW_SIZE
    num_features = Y_train.shape[-1]
    flat_dim = window_size * num_features

    mask_windows, _ = create_windows(train_mask)
    mask_windows = mask_windows.reshape(-1, window_size)

    # Prepare inputs FIRST
    cond = X_train.reshape(X_train.shape[0], -1)
    mask = mask_windows.astype(np.float32)
    real = Y_train.reshape(Y_train.shape[0], -1)

    # Get cond_dim AFTER cond exists
    cond_dim = cond.shape[1]

    # Build models AFTER dimensions are known
    generator = build_baseline_generator(
        flat_dim,
        window_size,
        num_features,
        cond_dim
    )
    discriminator = build_baseline_discriminator(
        flat_dim,
        window_size,
        num_features,
        cond_dim
    )

    # Optimizers
    g_opt = keras.optimizers.Adam(config.LEARNING_RATE, beta_1=0.5)
    d_opt = keras.optimizers.Adam(config.LEARNING_RATE, beta_1=0.5)

    bce = keras.losses.BinaryCrossentropy()

    dataset_size = cond.shape[0]

    print("\n===== Training Baseline GAN =====", flush=True)

    for epoch in range(config.EPOCHS):

        idx = np.random.permutation(dataset_size)

        for i in range(0, dataset_size, config.BATCH_SIZE):

            batch_idx = idx[i:i + config.BATCH_SIZE]

            cond_batch = tf.convert_to_tensor(cond[batch_idx], dtype=tf.float32)
            mask_batch = tf.convert_to_tensor(mask[batch_idx], dtype=tf.float32)
            real_batch = tf.convert_to_tensor(real[batch_idx], dtype=tf.float32)

            noise = tf.random.normal((cond_batch.shape[0], config.GAN_NOISE_DIM))

            # -------------------
            # Train Discriminator
            # -------------------
            with tf.GradientTape() as tape_d:

                fake = generator([cond_batch, mask_batch, noise], training=True)

                d_real = discriminator([real_batch, cond_batch, mask_batch], training=True)
                d_fake = discriminator([fake, cond_batch, mask_batch], training=True)

                d_loss = (
                    bce(tf.ones_like(d_real) * 0.9, d_real) +
                    bce(tf.zeros_like(d_fake), d_fake)
                ) / 2

            grads = tape_d.gradient(d_loss, discriminator.trainable_weights)
            d_opt.apply_gradients(zip(grads, discriminator.trainable_weights))

            # -------------------
            # Train Generator
            # -------------------
            with tf.GradientTape() as tape_g:

                fake = generator([cond_batch, mask_batch, noise], training=True)
                d_fake = discriminator([fake, cond_batch, mask_batch], training=False)

                gan_loss = bce(tf.ones_like(d_fake) * 0.9, d_fake)

                precip_indices = np.arange(0, flat_dim, num_features)

                real_seq = tf.reshape(real_batch, (-1, window_size, num_features))
                fake_seq = tf.reshape(fake, (-1, window_size, num_features))

                real_precip = real_seq[:, :, :1]
                fake_precip = fake_seq[:, :, :1]

                mask_precip = tf.reshape(mask_batch, (-1, window_size, 1))

                missing = 1.0 - mask_precip

                # 🔥 Rainfall-aware weighting
                weights = 1.0 + 1.0 * real_precip

                l1_loss = tf.reduce_mean(
                    tf.abs(real_precip - fake_precip) * missing * weights
                )

                g_loss = gan_loss + config.GAN_LAMBDA_L1 * l1_loss

            grads = tape_g.gradient(g_loss, generator.trainable_weights)
            g_opt.apply_gradients(zip(grads, generator.trainable_weights))

        print(
            f"[GAN-BASE] Epoch {epoch+1}/{config.EPOCHS} | "
            f"D Loss: {d_loss.numpy():.4f} | "
            f"G Loss: {g_loss.numpy():.4f}",
            flush=True
        )

    return generator


def train_e2e_gan(data):
    seq = data["train_scaled"]
    seq_masked = data["train_masked"]
    mask_seq = data["train_mask"]

    window = config.WINDOW_SIZE
    n_features = seq.shape[-1]
    n = len(seq) - window + 1

    generator = build_e2e_generator(window, n_features)
    discriminator = build_e2e_discriminator(window, n_features)

    g_opt = keras.optimizers.Adam(1e-4, beta_1=0.5)
    d_opt = keras.optimizers.Adam(1e-4, beta_1=0.5)

    bce = keras.losses.BinaryCrossentropy()

    print("\n===== Training E2E GAN =====", flush=True)

    for epoch in range(config.EPOCHS):

        idx_all = np.random.permutation(n)

        for i in range(0, n, config.BATCH_SIZE):

            batch_idx = idx_all[i:i + config.BATCH_SIZE]
            B = len(batch_idx)

            # -------------------
            # Build batch
            # -------------------
            real = np.stack([seq[j:j+window] for j in batch_idx], axis=0)

            # Condition = masked precip + month
            precip_masked = np.stack([seq_masked[j:j+window, 0] for j in batch_idx], axis=0)[..., None]
            month = np.stack([seq[j:j+window, 1:] for j in batch_idx], axis=0)
            cond = np.concatenate([precip_masked, month], axis=-1)   # (B, window, n_features)

            # Precipitation mask: ensure shape (B, window, 1)
            mask_precip = np.stack([mask_seq[j:j+window] for j in batch_idx], axis=0)
            if mask_precip.ndim == 3:
                # Already (B, window, 1)
                pass
            elif mask_precip.ndim == 2:
                # Expand to (B, window, 1)
                mask_precip = mask_precip[:, :, None]
            else:
                raise ValueError(f"Unexpected mask_precip shape: {mask_precip.shape}")

            # Month mask: always observed, shape (B, window, month_features)
            mask_month = np.ones_like(month)

            # Concatenate along last axis → (B, window, n_features)
            mask = np.concatenate([mask_precip, mask_month], axis=-1)



            # Convert to tensors
            real = tf.convert_to_tensor(real, tf.float32)
            cond = tf.convert_to_tensor(cond, tf.float32)
            mask = tf.convert_to_tensor(mask, tf.float32)

            noise = tf.random.normal((B, config.GAN_E2E_NOISE_DIM))

            # -------------------
            # Train Discriminator
            # -------------------
            with tf.GradientTape() as tape_d:
                noise = tf.random.normal((B, config.GAN_E2E_NOISE_DIM))
                
                fake = generator([cond, mask, noise], training=True)
                d_real = discriminator([real, cond, mask], training=True)
                d_fake = discriminator([fake, cond, mask], training=True)
                d_loss = (
                    bce(tf.ones_like(d_real) * 0.9, d_real) + 
                    bce(tf.zeros_like(d_fake), d_fake)
                ) / 2

            # -------------------
            # Train Generator
            # -------------------
            noise = tf.random.normal((B, config.GAN_E2E_NOISE_DIM))
            with tf.GradientTape() as tape_g:
                fake = generator([cond, mask, noise], training=True)
                d_fake = discriminator([fake, cond, mask], training=False)
                adv_loss = bce(tf.ones_like(d_fake) * 0.9, d_fake)
                real_precip = real[:, :, :1]
                fake_precip = fake[:, :, :1]
                mask_precip = mask[:, :, :1]
                missing = 1.0 - mask_precip
                weights = 1.0 + 3.0 * real_precip

                l1_loss = tf.reduce_mean(
                    tf.abs(real_precip - fake_precip) * missing * weights
                )
                g_loss = adv_loss + config.GAN_LAMBDA_L1 * l1_loss

            grads = tape_g.gradient(g_loss, generator.trainable_variables)
            g_opt.apply_gradients(zip(grads, generator.trainable_variables))

        print(
            f"[GAN-E2E] Epoch {epoch+1}/{config.EPOCHS} | "
            f"D Loss: {d_loss.numpy():.4f} | "
            f"G Loss: {g_loss.numpy():.4f}",
            flush=True
        )

    return generator


def train_wgan_gp(data):
    X_train = data["X_train_masked"]
    Y_train = data["Y_train"]
    mask_windows = data["train_mask_windows"]

    # Remove mask from condition
    # X_train = [precip_filled, month_sin, month_cos, mask]

    cond_seq = X_train[..., :3]   # keep only precip + month features
    cond = cond_seq.reshape(cond_seq.shape[0], -1)
    real = Y_train.reshape(Y_train.shape[0], -1)
    mask = mask_windows.squeeze(-1)   # (B, window)
    cond_dim = cond.shape[1]
    
    # Baseline generator expects flat = WINDOW_SIZE
    window_size = config.WINDOW_SIZE
    num_features = Y_train.shape[-1]
    flat_dim = window_size * num_features
    dataset_size = cond.shape[0]
    batch_size = config.BATCH_SIZE

    generator = build_baseline_generator(
        flat_dim,
        window_size,
        num_features,
        cond_dim
    )


    critic = build_wgan_critic(
        flat_dim,
        window_size,
        num_features,
        cond_dim
    )


    g_opt = keras.optimizers.Adam(config.LEARNING_RATE, beta_1=0.5, beta_2=0.9)
    c_opt = keras.optimizers.Adam(config.LEARNING_RATE, beta_1=0.5, beta_2=0.9)

    print("\n===== Training WGAN-GP =====", flush=True)

    for epoch in range(config.EPOCHS):

        # ---------------------
        # Train Critic
        # ---------------------

        for _ in range(config.GAN_N_CRITIC):
            idx = np.random.randint(0, dataset_size, batch_size)

            cond_batch = cond[idx]
            real_batch = real[idx]
            mask_batch = mask[idx]

            noise = tf.random.normal((batch_size, config.GAN_NOISE_DIM))

            with tf.GradientTape() as tape:
                cond_batch = tf.convert_to_tensor(cond_batch, dtype=tf.float32)
                mask_batch = tf.convert_to_tensor(mask_batch, dtype=tf.float32)
                real_batch = tf.convert_to_tensor(real_batch, dtype=tf.float32)

                fake = generator([cond_batch, mask_batch, noise], training=True)

                real_out = critic([real_batch, cond_batch, mask_batch], training=True)
                fake_out = critic([fake, cond_batch, mask_batch], training=True)

                gp = gradient_penalty(
                    critic,
                    real=real_batch,
                    fake=fake,
                    condition=cond_batch,
                    mask=mask_batch
                )

                loss_c = tf.reduce_mean(fake_out) - tf.reduce_mean(real_out)
                loss_c += config.GAN_LAMBDA_GP * gp

            grads = tape.gradient(loss_c, critic.trainable_variables)
            c_opt.apply_gradients(zip(grads, critic.trainable_variables))

        # ---------------------
        # Train Generator
        # ---------------------

        idx = np.random.randint(0, dataset_size, batch_size)

        cond_batch = tf.convert_to_tensor(cond[idx], dtype=tf.float32)
        mask_batch = tf.convert_to_tensor(mask[idx], dtype=tf.float32)
        real_batch = tf.convert_to_tensor(real[idx], dtype=tf.float32)

        noise = tf.random.normal((batch_size, config.GAN_NOISE_DIM))

        with tf.GradientTape() as tape:

            fake = generator([cond_batch, mask_batch, noise], training=True)

            fake_out = critic([fake, cond_batch, mask_batch], training=True)

            # adversarial loss
            adv_loss = -tf.reduce_mean(fake_out)

            precip_indices = np.arange(0, window_size * num_features, num_features)

            real_seq = tf.reshape(real_batch, (-1, window_size, num_features))
            fake_seq = tf.reshape(fake, (-1, window_size, num_features))

            real_precip = real_seq[:, :, :1]
            fake_precip = fake_seq[:, :, :1]

            mask_precip = tf.reshape(mask_batch, (-1, window_size, 1))

            missing = 1.0 - mask_precip

            weights = 1.0 + 1.0 * real_precip

            l1_loss = tf.reduce_mean(
                tf.abs(real_precip - fake_precip) * missing * weights
            )

            loss_g = adv_loss + config.GAN_LAMBDA_L1 * l1_loss

        grads = tape.gradient(loss_g, generator.trainable_variables)
        g_opt.apply_gradients(zip(grads, generator.trainable_variables))

        print(
            f"[WGAN-GP] Epoch {epoch+1}/{config.EPOCHS} | "
            f"Critic Loss: {loss_c.numpy():.4f} | "
            f"G Loss: {loss_g.numpy():.4f}",
            flush=True
        )

    return generator


if __name__ == "__main__":

    df = pd.read_csv(config.DATA_PATH)
    data = preprocess_all(df)

    train_baseline_gan(data)
    train_e2e_gan(data)
    train_wgan_gp(data)
