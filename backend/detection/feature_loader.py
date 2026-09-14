"""
ATHS Feature Loader

Loads the already-preprocessed ATHS dataset:

    dataset/processed/features.csv
    dataset/processed/embeddings.npy

This module does NOT perform preprocessing and does NOT train a model.
It only:
    1. Loads the processed CSV
    2. Loads the semantic embeddings
    3. Selects the ML feature columns
    4. Loads the target labels
    5. Loads group IDs for group-aware splitting
    6. Validates that CSV rows and embeddings are aligned
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

FEATURES_PATH = PROJECT_ROOT / "dataset" / "processed" / "features.csv"
EMBEDDINGS_PATH = PROJECT_ROOT / "dataset" / "processed" / "embeddings.npy"


# -------------------------------------------------------------------------
# Columns from the processed features.csv
# -------------------------------------------------------------------------

ML_FEATURE_COLUMNS: List[str] = [
    "word_count",
    "sentence_count",
    "unique_word_count",
    "lexical_diversity",
    "instruction_density",
    "obfuscation_score",
    "instruction_override",
    "system_prompt_reference",
    "role_change_attempt",
    "data_extraction_attempt",
    "tool_manipulation",
    "policy_bypass",
    "prompt_leakage",
    "sensitive_data_reference",
    "imperative_signal",
]

TARGET_COLUMN = "label"
GROUP_COLUMN = "group_id"

# These are metadata/text columns and are deliberately NOT used
# as direct ML features.
METADATA_COLUMNS: List[str] = [
    "text",
    "category",
    "source",
    "severity",
    "group_id",
    "augmented",
    "tags",
]


# -------------------------------------------------------------------------
# Loader class
# -------------------------------------------------------------------------

class ATHSFeatureLoader:
    """
    Loads and validates the already-preprocessed ATHS data.
    """

    def __init__(
        self,
        features_path: Optional[Path] = None,
        embeddings_path: Optional[Path] = None,
    ) -> None:

        self.features_path = (
            Path(features_path)
            if features_path is not None
            else FEATURES_PATH
        )

        self.embeddings_path = (
            Path(embeddings_path)
            if embeddings_path is not None
            else EMBEDDINGS_PATH
        )

        self.df: Optional[pd.DataFrame] = None
        self.embeddings: Optional[np.ndarray] = None

    # ---------------------------------------------------------------------
    # Load CSV
    # ---------------------------------------------------------------------

    def load_features_csv(self) -> pd.DataFrame:
        """Load features.csv and validate required columns."""

        if not self.features_path.exists():
            raise FileNotFoundError(
                f"features.csv was not found at:\n{self.features_path}"
            )

        df = pd.read_csv(self.features_path)

        required_columns = (
            ML_FEATURE_COLUMNS
            + [TARGET_COLUMN, GROUP_COLUMN]
        )

        missing_columns = [
            column
            for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                "features.csv is missing required columns:\n"
                + "\n".join(f"  - {column}" for column in missing_columns)
            )

        if df.empty:
            raise ValueError("features.csv is empty.")

        self.df = df

        return df

    # ---------------------------------------------------------------------
    # Load embeddings
    # ---------------------------------------------------------------------

    def load_embeddings(self) -> np.ndarray:
        """Load embeddings.npy and validate its shape."""

        if not self.embeddings_path.exists():
            raise FileNotFoundError(
                f"embeddings.npy was not found at:\n{self.embeddings_path}"
            )

        embeddings = np.load(self.embeddings_path)

        if embeddings.ndim != 2:
            raise ValueError(
                "embeddings.npy must be a 2-dimensional array. "
                f"Found shape: {embeddings.shape}"
            )

        self.embeddings = embeddings

        return embeddings

    # ---------------------------------------------------------------------
    # Validate CSV / embeddings alignment
    # ---------------------------------------------------------------------

    def validate_alignment(self) -> None:
        """
        Verify that:
            number of CSV rows == number of embeddings
        """

        if self.df is None:
            raise RuntimeError(
                "features.csv has not been loaded yet."
            )

        if self.embeddings is None:
            raise RuntimeError(
                "embeddings.npy has not been loaded yet."
            )

        csv_rows = len(self.df)
        embedding_rows = self.embeddings.shape[0]

        if csv_rows != embedding_rows:
            raise ValueError(
                "CSV and embeddings are not aligned.\n"
                f"features.csv rows : {csv_rows}\n"
                f"embeddings rows   : {embedding_rows}"
            )

    # ---------------------------------------------------------------------
    # Validate embedding dimension
    # ---------------------------------------------------------------------

    def validate_embedding_dimension(
        self,
        expected_dimension: int = 384,
    ) -> None:
        """Verify that embeddings have the expected dimension."""

        if self.embeddings is None:
            raise RuntimeError(
                "embeddings.npy has not been loaded yet."
            )

        actual_dimension = self.embeddings.shape[1]

        if actual_dimension != expected_dimension:
            raise ValueError(
                "Unexpected embedding dimension.\n"
                f"Expected : {expected_dimension}\n"
                f"Found    : {actual_dimension}"
            )

    # ---------------------------------------------------------------------
    # Validate labels
    # ---------------------------------------------------------------------

    def validate_labels(self) -> None:
        """Validate the ground-truth label column."""

        if self.df is None:
            raise RuntimeError(
                "features.csv has not been loaded yet."
            )

        labels = self.df[TARGET_COLUMN]

        if labels.isna().any():
            raise ValueError(
                "The label column contains missing values."
            )

        unique_labels = set(labels.unique())

        if not unique_labels.issubset({0, 1}):
            raise ValueError(
                "The label column must contain only 0 and 1.\n"
                f"Found labels: {sorted(unique_labels)}"
            )

    # ---------------------------------------------------------------------
    # Validate numeric ML features
    # ---------------------------------------------------------------------

    def validate_ml_features(self) -> None:
        """Ensure all selected ML features are numeric and complete."""

        if self.df is None:
            raise RuntimeError(
                "features.csv has not been loaded yet."
            )

        feature_df = self.df[ML_FEATURE_COLUMNS]

        non_numeric_columns = [
            column
            for column in ML_FEATURE_COLUMNS
            if not pd.api.types.is_numeric_dtype(feature_df[column])
        ]

        if non_numeric_columns:
            raise ValueError(
                "The following ML feature columns are not numeric:\n"
                + "\n".join(
                    f"  - {column}"
                    for column in non_numeric_columns
                )
            )

        missing_values = feature_df.isna().sum()

        columns_with_missing = missing_values[
            missing_values > 0
        ]

        if not columns_with_missing.empty:
            details = "\n".join(
                f"  - {column}: {count}"
                for column, count
                in columns_with_missing.items()
            )

            raise ValueError(
                "Missing values were found in ML features:\n"
                + details
            )

    # ---------------------------------------------------------------------
    # Get ML features
    # ---------------------------------------------------------------------

    def get_tabular_features(self) -> np.ndarray:
        """
        Return the handcrafted/NLP/security features as a NumPy array.

        Shape:
            (number_of_rows, number_of_features)
        """

        if self.df is None:
            raise RuntimeError(
                "features.csv has not been loaded yet."
            )

        self.validate_ml_features()

        X = self.df[ML_FEATURE_COLUMNS].to_numpy(
            dtype=np.float32
        )

        return X

    # ---------------------------------------------------------------------
    # Get embeddings
    # ---------------------------------------------------------------------

    def get_embeddings(self) -> np.ndarray:
        """Return the 384-dimensional semantic embeddings."""

        if self.embeddings is None:
            raise RuntimeError(
                "embeddings.npy has not been loaded yet."
            )

        return self.embeddings.astype(np.float32)

    # ---------------------------------------------------------------------
    # Get labels
    # ---------------------------------------------------------------------

    def get_labels(self) -> np.ndarray:
        """Return the binary ground-truth labels."""

        if self.df is None:
            raise RuntimeError(
                "features.csv has not been loaded yet."
            )

        self.validate_labels()

        return self.df[TARGET_COLUMN].to_numpy(
            dtype=np.int64
        )

    # ---------------------------------------------------------------------
    # Get groups
    # ---------------------------------------------------------------------

    def get_groups(self) -> np.ndarray:
        """
        Return group IDs.

        These will later be used for group-aware train/validation/test
        splitting to reduce data leakage.
        """

        if self.df is None:
            raise RuntimeError(
                "features.csv has not been loaded yet."
            )

        groups = self.df[GROUP_COLUMN]

        if groups.isna().any():
            raise ValueError(
                "group_id contains missing values."
            )

        return groups.to_numpy()

    # ---------------------------------------------------------------------
    # Get text
    # ---------------------------------------------------------------------

    def get_texts(self) -> np.ndarray:
        """
        Return the original prompt text.

        Text is kept for explanations and dashboard output,
        not as a direct feature for the first tabular ML model.
        """

        if self.df is None:
            raise RuntimeError(
                "features.csv has not been loaded yet."
            )

        if "text" not in self.df.columns:
            raise ValueError(
                "The 'text' column is missing from features.csv."
            )

        return self.df["text"].fillna("").to_numpy()

    # ---------------------------------------------------------------------
    # Get metadata
    # ---------------------------------------------------------------------

    def get_metadata(self) -> pd.DataFrame:
        """Return useful non-ML metadata."""

        if self.df is None:
            raise RuntimeError(
                "features.csv has not been loaded yet."
            )

        available_columns = [
            column
            for column in METADATA_COLUMNS
            if column in self.df.columns
        ]

        return self.df[available_columns].copy()

    # ---------------------------------------------------------------------
    # Load everything
    # ---------------------------------------------------------------------

    def load(self) -> Dict[str, object]:
        """
        Load and validate the complete processed dataset.

        Returns:
            {
                "dataframe": pandas DataFrame,
                "tabular_features": NumPy array,
                "embeddings": NumPy array,
                "labels": NumPy array,
                "groups": NumPy array,
                "texts": NumPy array,
                "metadata": pandas DataFrame,
                "feature_names": list[str]
            }
        """

        self.load_features_csv()
        self.load_embeddings()

        self.validate_alignment()
        self.validate_embedding_dimension()
        self.validate_labels()
        self.validate_ml_features()

        X_tabular = self.get_tabular_features()
        X_embeddings = self.get_embeddings()
        y = self.get_labels()
        groups = self.get_groups()
        texts = self.get_texts()
        metadata = self.get_metadata()

        return {
            "dataframe": self.df,
            "tabular_features": X_tabular,
            "embeddings": X_embeddings,
            "labels": y,
            "groups": groups,
            "texts": texts,
            "metadata": metadata,
            "feature_names": ML_FEATURE_COLUMNS.copy(),
        }


# -------------------------------------------------------------------------
# Convenience function
# -------------------------------------------------------------------------

def load_aths_data() -> Dict[str, object]:
    """
    Convenience function used by later training/detection modules.
    """

    loader = ATHSFeatureLoader()

    return loader.load()


# -------------------------------------------------------------------------
# Standalone test
# -------------------------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("ATHS FEATURE LOADER TEST")
    print("=" * 70)

    loader = ATHSFeatureLoader()

    try:
        data = loader.load()

        print("\nData loaded successfully.")

        print(f"\nCSV shape:")
        print(f"  {data['dataframe'].shape}")

        print("\nTabular feature shape:")
        print(f"  {data['tabular_features'].shape}")

        print("\nEmbedding shape:")
        print(f"  {data['embeddings'].shape}")

        print("\nLabel shape:")
        print(f"  {data['labels'].shape}")

        print("\nNumber of groups:")
        print(f"  {len(np.unique(data['groups']))}")

        print("\nClass distribution:")
        unique, counts = np.unique(
            data["labels"],
            return_counts=True
        )

        for label, count in zip(unique, counts):
            print(f"  Label {label}: {count}")

        print("\nML feature columns:")
        for index, column in enumerate(
            data["feature_names"],
            start=1
        ):
            print(f"  {index:02d}. {column}")

        print("\nFirst tabular feature row:")
        print(f"  {data['tabular_features'][0]}")

        print("\nFirst embedding:")
        print(
            f"  first 10 values = "
            f"{data['embeddings'][0][:10]}"
        )

        print("\n" + "=" * 70)
        print("FEATURE LOADER TEST PASSED")
        print("=" * 70)

    except Exception as exc:
        print("\nFEATURE LOADER TEST FAILED")
        print(f"\nError: {exc}")

        raise