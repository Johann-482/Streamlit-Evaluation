import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import config

"""# Flatten / Reshape Helpers

# Generators
"""
def build_baseline_generator(flat_dim, window_size, n_features, cond_dim):

    cond = keras.Input(shape=(cond_dim,))
    mask = keras.Input(shape=(window_size,))
    noise = keras.Input(shape=(config.GAN_NOISE_DIM,))

    x = layers.Concatenate()([cond, mask, noise])

    x = layers.Dense(config.GAN_G_HIDDEN, activation='relu')(x)

    x = layers.Dense(config.GAN_G_HIDDEN, activation='relu')(x)

    gen_raw = layers.Dense(flat_dim, activation='sigmoid')(x)

    # Reshape
    cond_seq = layers.Reshape((window_size, -1))(cond)
    gen_seq = layers.Reshape((window_size, n_features))(gen_raw)

    # Extract precipitation
    # Keep everything 3D: (B, window, 1)
    precip = layers.Lambda(lambda x: x[:, :, :1])(cond_seq)
    gen_precip = layers.Lambda(lambda x: x[:, :, :1])(gen_seq)

    # Expand mask → (B, window, 1)
    mask_precip = layers.Reshape((window_size, 1))(mask)

    inv_mask_precip = layers.Lambda(lambda x: 1.0 - x)(mask_precip)

    preserved_precip = layers.Multiply()([mask_precip, precip])
    generated_precip = layers.Multiply()([inv_mask_precip, gen_precip])

    final_precip = layers.Add()([preserved_precip, generated_precip])

    # Rebuild full output
    month = layers.Lambda(lambda x: x[:, :, 1:])(cond_seq)

    out_seq = layers.Concatenate(axis=2)([
        final_precip,
        month[:, :, :n_features-1]  # match Y features
    ])

    out = layers.Reshape((flat_dim,))(out_seq)

    return keras.Model([cond, mask, noise], out, name="Baseline_Generator")
def build_e2e_generator(window_size, n_features):

    cond = keras.Input(shape=(window_size, n_features))
    mask = keras.Input(shape=(window_size, n_features))
    noise = keras.Input(shape=(config.GAN_E2E_NOISE_DIM,))

    # Broadcast noise
    noise_proj = layers.Dense(window_size * 8, activation='relu')(noise)
    noise_proj = layers.Reshape((window_size, 8))(noise_proj)

    x = layers.Concatenate()([cond, mask, noise_proj])

    x = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)

    x = layers.GRU(64, return_sequences=True)(x)

    x = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
    gen_raw = layers.Conv1D(n_features, 1, activation='sigmoid')(x)

    # Split features
    precip = layers.Lambda(lambda x: x[:, :, :1])(cond)
    month = layers.Lambda(lambda x: x[:, :, 1:])(cond)

    gen_precip = layers.Lambda(lambda x: x[:, :, :1])(gen_raw)

    # Extract mask for precipitation ONLY
    mask_precip = layers.Lambda(lambda x: x[:, :, :1])(mask)

    inv_mask_precip = layers.Lambda(lambda x: 1.0 - x)(mask_precip)

    # Apply mask ONLY to precipitation
    preserved_precip = layers.Multiply()([mask_precip, precip])
    generated_precip = layers.Multiply()([inv_mask_precip, gen_precip])

    final_precip = layers.Add()([preserved_precip, generated_precip])

    # Concatenate back with month (unchanged)
    out = layers.Concatenate(axis=2)([final_precip, month])

    return keras.Model([cond, mask, noise], out, name="E2E_Generator")

"""# Discriminators"""

def build_baseline_discriminator(flat_dim, window_size, n_features, cond_dim):

    candidate = keras.Input(shape=(flat_dim,))
    condition = keras.Input(shape=(cond_dim,))
    mask = keras.Input(shape=(window_size,))

    # Reshape everything to sequences
    candidate_seq = layers.Reshape((window_size, n_features))(candidate)
    condition_seq = layers.Reshape((window_size, -1))(condition)

    # Extract precipitation (B, window, 1)
    candidate_precip = layers.Lambda(lambda x: x[:, :, :1])(candidate_seq)

    mask_precip = layers.Reshape((window_size, 1))(mask)
    missing = layers.Lambda(lambda x: 1.0 - x)(mask_precip)

    candidate_missing = layers.Multiply()([candidate_precip, missing])

    x = layers.Concatenate(axis=2)([
        candidate_missing,
        condition_seq,
        mask_precip
    ])

    x = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    return keras.Model([candidate, condition, mask], out)

def build_e2e_discriminator(window_size, n_features):

    candidate = keras.Input(shape=(window_size, n_features))
    condition = keras.Input(shape=(window_size, n_features))
    mask = keras.Input(shape=(window_size, n_features))

    # Extract precipitation only
    candidate_precip = layers.Lambda(lambda x: x[:, :, :1])(candidate)
    mask_precip = layers.Lambda(lambda x: x[:, :, :1])(mask)

    missing = layers.Lambda(lambda x: 1.0 - x)(mask_precip)
    candidate_missing = layers.Multiply()([candidate_precip, missing])

    x = layers.Concatenate()([candidate_missing, condition, mask_precip])

    x = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(64, activation='relu')(x)
    out = layers.Dense(1, activation='sigmoid')(x)

    return keras.Model(
        [candidate, condition, mask],
        out,
        name="E2E_Discriminator"
    )

"""# WGAN-GP Critic"""

def build_wgan_critic(flat_dim, window_size, n_features, cond_dim):

    # Inputs
    candidate = keras.Input(shape=(flat_dim,))     # 36
    condition = keras.Input(shape=(cond_dim,))     # 48
    mask = keras.Input(shape=(window_size,))       # 12

    # Reshape candidate → (B, window, n_features)
    candidate_seq = layers.Reshape((window_size, n_features))(candidate)

    # Reshape condition → (B, window, cond_features)
    condition_seq = layers.Reshape((window_size, -1))(condition)

    # Extract precipitation ONLY
    candidate_precip = layers.Lambda(lambda x: x[:, :, 0])(candidate_seq)

    # Mask → (B, window, 1)
    mask_precip = layers.Reshape((window_size, 1))(mask)

    # cast to float32 (fixes dtype mismatch)
    mask_precip = layers.Lambda(lambda x: tf.cast(x, tf.float32))(mask_precip)

    missing = layers.Lambda(lambda x: 1.0 - x)(mask_precip)

    # candidate_precip is (B,12) → expand to (B,12,1)
    candidate_precip = layers.Lambda(lambda x: tf.expand_dims(x, -1))(candidate_precip)

    candidate_missing = layers.Multiply()([candidate_precip, missing])

    # Concatenate inputs
    x = layers.Concatenate()([
        candidate_missing,     # (B, window, 1)
        condition_seq,         # (B, window, cond_features)
        mask_precip            # (B, window, 1)
    ])

    # Critic network
    x = layers.Conv1D(32, 3, padding='same', activation='relu')(x)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(x)
    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(config.GAN_D_HIDDEN, activation='relu')(x)
    x = layers.Dense(config.GAN_D_HIDDEN // 2, activation='relu')(x)

    out = layers.Dense(1)(x)  # linear output (WGAN)

    return keras.Model([candidate, condition, mask], out, name="WGAN_Critic")


"""# Builds"""

def build_gan(model_type, window_size, n_features, cond_dim=None):

    flat_dim = window_size * n_features

    if model_type == "Baseline_GAN":
        generator = build_baseline_generator(
            flat_dim, window_size, n_features, cond_dim
        )
        discriminator = build_baseline_discriminator(
            flat_dim, window_size, n_features, cond_dim
        )
        return generator, discriminator

    elif model_type == "E2E_GAN":
        generator = build_e2e_generator(window_size, n_features)
        discriminator = build_e2e_discriminator(window_size, n_features)
        return generator, discriminator

    elif model_type == "WGAN_GP":
        generator = build_baseline_generator(
            flat_dim, window_size, n_features, cond_dim
        )
        critic = build_wgan_critic(
            flat_dim, window_size, n_features, cond_dim
        )
        return generator, critic

    else:
        raise ValueError(f"Unknown model type: {model_type}")