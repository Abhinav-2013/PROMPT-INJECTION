import pandas as pd

df = pd.read_csv("dataset/prompt_injection_dataset.csv")

print("=" * 80)
print("DATASET INFORMATION")
print("=" * 80)

print("\nNumber of rows:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head().to_string())

print("\nLabel distribution:")
print(df["label"].value_counts())

print("\nCategories:")
print(df["category"].value_counts(dropna=False))

print("\nSeverity:")
print(df["severity"].value_counts(dropna=False))

print("\nSource:")
print(df["source"].value_counts(dropna=False))

print("\nAugmented:")
print(df["augmented"].value_counts(dropna=False))