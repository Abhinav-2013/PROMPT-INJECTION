import pandas as pd
import numpy as np
from preprocessing.pipeline import PreprocessingPipeline

INPUT_FILE = "dataset/prompt_injection_dataset.csv"
FEATURE_FILE = "dataset/processed/features.csv"
EMBEDDING_FILE = "dataset/processed/embeddings.npy"

def main():
    print("=" * 80)
    print("ATHS FULL DATASET PREPROCESSING")
    print("=" * 80)

    df = pd.read_csv(INPUT_FILE)

    print(f"\nTotal samples: {len(df)}")

    pipeline = PreprocessingPipeline(
        enable_embeddings=True
    )

    processed_rows = []
    embeddings = []

    for index, row in df.iterrows():
        prompt = str(row["text"])

        print(
            f"Processing {index + 1}/{len(df)}",
            end="\r"
        )

        result = pipeline.process_text(prompt)

        embedding = result["semantic_embedding"]

        if embedding is not None:
            embeddings.append(embedding)
        else:
            embeddings.append(np.zeros(384))

        processed_rows.append({
            "text": prompt,
            "label": int(row["label"]),
            "category": row["category"],
            "source": row["source"],
            "severity": row["severity"],
            "group_id": row["group_id"],
            "augmented": row["augmented"],
            "tags": row["tags"],
            "word_count": result["nlp"]["word_count"],
            "sentence_count": result["nlp"]["sentence_count"],
            "unique_word_count": result["nlp"]["unique_word_count"],
            "lexical_diversity": result["nlp"]["lexical_diversity"],
            "instruction_density": result["security_features"]["instruction_density"],
            "obfuscation_score": result["obfuscation"]["obfuscation_score"],
            "instruction_override": result["security_features"]["instruction_override"],
            "system_prompt_reference": result["security_features"]["system_prompt_reference"],
            "role_change_attempt": result["security_features"]["role_change_attempt"],
            "data_extraction_attempt": result["security_features"]["data_extraction_attempt"],
            "tool_manipulation": result["security_features"]["tool_manipulation"],
            "policy_bypass": result["security_features"]["policy_bypass"],
            "prompt_leakage": result["security_features"]["prompt_leakage"],
            "sensitive_data_reference": result["security_features"]["sensitive_data_reference"],
            "imperative_signal": result["security_features"]["imperative_signal"]
        })

    features_df = pd.DataFrame(processed_rows)

    features_df.to_csv(
        FEATURE_FILE,
        index=False
    )

    embeddings_array = np.array(embeddings)

    np.save(
        EMBEDDING_FILE,
        embeddings_array
    )

    print("\n\n" + "=" * 80)
    print("FULL DATASET PROCESSING COMPLETE")
    print("=" * 80)

    print(f"\nSamples processed: {len(features_df)}")
    print(f"Feature file: {FEATURE_FILE}")
    print(f"Embedding file: {EMBEDDING_FILE}")
    print(f"Embedding shape: {embeddings_array.shape}")

    print("\nLabel distribution:")
    print(features_df["label"].value_counts())

if __name__ == "__main__":
    main()