from datasets import load_dataset
import os

os.makedirs("dataset", exist_ok=True)

print("Downloading prompt injection dataset...")

dataset = load_dataset(
    "neuralchemy/prompt-injection-benign-dataset"
)

df = dataset["train"].to_pandas()

df.to_csv(
    "dataset/prompt_injection_dataset.csv",
    index=False
)

print("\nDataset downloaded successfully!")
print("Rows:", len(df))
print("Columns:")
print(df.columns.tolist())