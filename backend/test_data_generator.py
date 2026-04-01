import numpy as np
import pandas as pd


def generate_test_csv(
    file_path,
    missing_rate=0.25,
    seed=42,
    save_path="generated_test.csv"
):

    rng = np.random.default_rng()

    # --- Load dataset ---
    df = pd.read_csv(file_path)

    required_cols = {"Year", "Month", "Precipitation"}
    if not required_cols.issubset(df.columns):
        raise ValueError("CSV must contain Year, Month, Precipitation")

    # --- Sort ---
    df["Date"] = pd.to_datetime(df[["Year", "Month"]].assign(DAY=1))
    df = df.sort_values("Date").reset_index(drop=True)

    # --- Pick random year ---
    years = df["Year"].unique()
    selected_year = rng.choice(years)

    df_year = df[df["Year"] == selected_year].copy()

    if len(df_year) != 12:
        raise ValueError(f"Year {selected_year} does not have exactly 12 months")

    df_year = df_year.sort_values("Month").reset_index(drop=True)

    # --- Save original for comparison ---
    df_full = df_year.copy()

    # --- MAR MASKING ---
    values = df_year["Precipitation"].values.astype(float)

    # Log transform (same as training)
    log_vals = np.log1p(values)

    months = df_year["Month"].values

    month_sin = np.sin(2 * np.pi * months / 12)
    month_cos = np.cos(2 * np.pi * months / 12)

    month_angle = np.arctan2(month_sin, month_cos)
    month_norm = (month_angle + np.pi) / (2 * np.pi)

    logits = 3.0 * (month_norm - 0.5)
    prob_missing = 1 / (1 + np.exp(-logits))

    # Match desired missing rate
    prob_missing = prob_missing / (prob_missing.mean() + 1e-8)
    prob_missing = prob_missing * missing_rate
    prob_missing = np.minimum(prob_missing, 1.0)

    random_draw = rng.random(prob_missing.shape)
    missing_mask = random_draw < prob_missing

    # Apply missing
    values_with_nan = values.copy()
    values_with_nan[missing_mask] = np.nan

    df_missing = df_year.copy()
    df_missing["Precipitation"] = values_with_nan

    # --- Save CSV ---
    df_missing.to_csv(save_path, index=False)

    print(f"✅ Generated test CSV saved to: {save_path}")
    print(f"📅 Selected Year: {selected_year}")
    print(f"📉 Missing Rate (approx): {missing_mask.mean():.2f}")

    return df_full, df_missing