from backend.test_data_generator import generate_test_csv

# Path to your ORIGINAL dataset (20 years data)
file_path = "Rainfall_data.csv"

# Choose missing rate: 0.15, 0.25, 0.50
df_full, df_missing = generate_test_csv(
    file_path=file_path,
    missing_rate=0.25,
    save_path="test_25_missing.csv"
)

print("\n=== ORIGINAL DATA (GROUND TRUTH) ===")
print(df_full)

print("\n=== WITH MISSING VALUES ===")
print(df_missing)