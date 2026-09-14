from pathlib import Path
from typing import Dict, List, Tuple

import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from data.feedback_store import ATHSFeedbackStore
from detection.feature_loader import ML_FEATURE_COLUMNS


# ============================================================
# ATHS DATASET MANAGER CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

FEATURES_PATH = (
    BASE_DIR
    / "dataset"
    / "processed"
    / "features.csv"
)

RETRAINING_DIR = (
    BASE_DIR
    / "dataset"
    / "retraining"
)

SPLIT_FILE = (
    RETRAINING_DIR
    / "fixed_split.json"
)

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# ATHS DATASET MANAGER
# ============================================================

class ATHSDatasetManager:

    def __init__(
        self,
        features_path: Path = FEATURES_PATH,
    ) -> None:

        print(
            "Initializing ATHS Dataset Manager..."
        )

        self.features_path = Path(
            features_path
        )

        self.feedback_store = (
            ATHSFeedbackStore()
        )

        RETRAINING_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        print(
            "ATHS Dataset Manager initialized."
        )

    # --------------------------------------------------------
    # LOAD ORIGINAL DATASET
    # --------------------------------------------------------

    def load_original_dataset(
        self,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:

        if not self.features_path.exists():

            raise FileNotFoundError(
                "Processed feature file not found: "
                f"{self.features_path}"
            )

        dataframe = pd.read_csv(
            self.features_path
        )

        required_columns = (
            ML_FEATURE_COLUMNS
            + ["label"]
        )

        missing_columns = [
            column
            for column in required_columns
            if column not in dataframe.columns
        ]

        if missing_columns:

            raise ValueError(
                "Missing required columns: "
                f"{missing_columns}"
            )

        X = dataframe[
            ML_FEATURE_COLUMNS
        ].to_numpy(
            dtype=np.float32
        )

        y = dataframe[
            "label"
        ].to_numpy(
            dtype=int
        )

        if len(X) != len(y):

            raise ValueError(
                "Feature and label counts "
                "do not match."
            )

        if not set(
            np.unique(y)
        ).issubset({0, 1}):

            raise ValueError(
                "Labels must contain only "
                "0 and 1."
            )

        print()
        print(
            "ORIGINAL DATASET"
        )
        print("-" * 50)

        print(
            f"Rows      : {len(y)}"
        )

        print(
            f"Features  : {X.shape[1]}"
        )

        print(
            f"Benign    : "
            f"{np.sum(y == 0)}"
        )

        print(
            f"Malicious : "
            f"{np.sum(y == 1)}"
        )

        return (
            X,
            y,
            dataframe,
        )

    # --------------------------------------------------------
    # CREATE FIXED SPLIT
    # --------------------------------------------------------

    def create_fixed_split(
        self,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:

        indices = np.arange(
            len(y)
        )

        train_indices, test_indices = (
            train_test_split(
                indices,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=y,
            )
        )

        return (
            np.asarray(
                train_indices,
                dtype=int,
            ),
            np.asarray(
                test_indices,
                dtype=int,
            ),
        )

    # --------------------------------------------------------
    # SAVE FIXED SPLIT
    # --------------------------------------------------------

    def save_fixed_split(
        self,
        train_indices: np.ndarray,
        test_indices: np.ndarray,
    ) -> None:

        split_data = {
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "train_indices": (
                train_indices.tolist()
            ),
            "test_indices": (
                test_indices.tolist()
            ),
        }

        with open(
            SPLIT_FILE,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                split_data,
                file,
                indent=2,
            )

        print()
        print(
            "FIXED SPLIT SAVED"
        )
        print("-" * 50)

        print(
            f"Path : {SPLIT_FILE}"
        )

    # --------------------------------------------------------
    # LOAD EXISTING FIXED SPLIT
    # --------------------------------------------------------

    def load_fixed_split(
        self,
        dataset_size: int,
    ) -> Tuple[np.ndarray, np.ndarray]:

        if not SPLIT_FILE.exists():

            raise FileNotFoundError(
                f"Fixed split does not exist: "
                f"{SPLIT_FILE}"
            )

        with open(
            SPLIT_FILE,
            "r",
            encoding="utf-8",
        ) as file:

            split_data = json.load(
                file
            )

        train_indices = np.asarray(
            split_data["train_indices"],
            dtype=int,
        )

        test_indices = np.asarray(
            split_data["test_indices"],
            dtype=int,
        )

        # ----------------------------------------------------
        # VALIDATE STORED SPLIT
        # ----------------------------------------------------

        if len(train_indices) == 0:
            raise ValueError(
                "Stored training split is empty."
            )

        if len(test_indices) == 0:
            raise ValueError(
                "Stored test split is empty."
            )

        if (
            np.any(train_indices < 0)
            or np.any(
                train_indices >= dataset_size
            )
        ):
            raise ValueError(
                "Training indices are invalid "
                "for the current dataset."
            )

        if (
            np.any(test_indices < 0)
            or np.any(
                test_indices >= dataset_size
            )
        ):
            raise ValueError(
                "Test indices are invalid "
                "for the current dataset."
            )

        overlap = np.intersect1d(
            train_indices,
            test_indices,
        )

        if len(overlap) > 0:

            raise ValueError(
                "Training and test splits "
                "overlap."
            )

        combined = np.concatenate(
            [
                train_indices,
                test_indices,
            ]
        )

        if len(
            np.unique(combined)
        ) != dataset_size:

            raise ValueError(
                "Stored split does not cover "
                "the complete original dataset."
            )

        return (
            train_indices,
            test_indices,
        )

    # --------------------------------------------------------
    # GET FIXED SPLIT
    # --------------------------------------------------------

    def get_fixed_split(
        self,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:

        if SPLIT_FILE.exists():

            print()
            print(
                "LOADING EXISTING FIXED SPLIT"
            )
            print("-" * 50)

            train_indices, test_indices = (
                self.load_fixed_split(
                    len(y)
                )
            )

        else:

            print()
            print(
                "CREATING FIXED DATASET SPLIT"
            )
            print("-" * 50)

            train_indices, test_indices = (
                self.create_fixed_split(y)
            )

            self.save_fixed_split(
                train_indices,
                test_indices,
            )

        return (
            train_indices,
            test_indices,
        )

    # --------------------------------------------------------
    # LOAD CONFIRMED HUMAN FEEDBACK
    # --------------------------------------------------------

    def load_confirmed_feedback(
        self,
    ) -> List[Dict]:

        feedback = (
            self.feedback_store
            .get_labeled_data()
        )

        validated_feedback = []

        for record in feedback:

            if "text" not in record:
                continue

            if "human_label" not in record:
                continue

            text = str(
                record["text"]
            ).strip()

            if not text:
                continue

            label = int(
                record["human_label"]
            )

            if label not in (0, 1):
                continue

            validated_feedback.append(
                {
                    **record,
                    "text": text,
                    "human_label": label,
                }
            )

        print()
        print(
            "CONFIRMED HUMAN FEEDBACK"
        )
        print("-" * 50)

        print(
            f"Labeled samples : "
            f"{len(validated_feedback)}"
        )

        if validated_feedback:

            feedback_labels = np.asarray(
                [
                    record["human_label"]
                    for record
                    in validated_feedback
                ],
                dtype=int,
            )

            print(
                f"Benign          : "
                f"{np.sum(feedback_labels == 0)}"
            )

            print(
                f"Malicious       : "
                f"{np.sum(feedback_labels == 1)}"
            )

        return validated_feedback

    # --------------------------------------------------------
    # PREPARE COMPLETE DATASET
    # --------------------------------------------------------

    def prepare_dataset(self) -> Dict:

        (
            X,
            y,
            dataframe,
        ) = self.load_original_dataset()

        (
            train_indices,
            test_indices,
        ) = self.get_fixed_split(y)

        X_train = X[
            train_indices
        ]

        y_train = y[
            train_indices
        ]

        X_test = X[
            test_indices
        ]

        y_test = y[
            test_indices
        ]

        feedback = (
            self.load_confirmed_feedback()
        )

        print()
        print(
            "FIXED DATASET SPLIT"
        )
        print("-" * 50)

        print(
            f"Training samples : "
            f"{len(y_train)}"
        )

        print(
            f"Testing samples  : "
            f"{len(y_test)}"
        )

        print()
        print(
            "Training distribution:"
        )

        print(
            f"  Benign    : "
            f"{np.sum(y_train == 0)}"
        )

        print(
            f"  Malicious : "
            f"{np.sum(y_train == 1)}"
        )

        print()
        print(
            "Testing distribution:"
        )

        print(
            f"  Benign    : "
            f"{np.sum(y_test == 0)}"
        )

        print(
            f"  Malicious : "
            f"{np.sum(y_test == 1)}"
        )

        print()
        print(
            "DATASET SUMMARY"
        )
        print("-" * 50)

        summary = {
            "original_samples": int(
                len(y)
            ),
            "training_samples": int(
                len(y_train)
            ),
            "test_samples": int(
                len(y_test)
            ),
            "training_benign": int(
                np.sum(y_train == 0)
            ),
            "training_malicious": int(
                np.sum(y_train == 1)
            ),
            "test_benign": int(
                np.sum(y_test == 0)
            ),
            "test_malicious": int(
                np.sum(y_test == 1)
            ),
            "feedback_samples": int(
                len(feedback)
            ),
        }

        for key, value in summary.items():

            print(
                f"{key:<25}: {value}"
            )

        return {
            "X": X,
            "y": y,
            "dataframe": dataframe,

            "X_train": X_train,
            "y_train": y_train,

            "X_test": X_test,
            "y_test": y_test,

            "train_indices":
                train_indices,

            "test_indices":
                test_indices,

            "feedback":
                feedback,

            "summary":
                summary,
        }


# ============================================================
# STANDALONE TEST
# ============================================================

def main() -> None:

    print()
    print("=" * 60)
    print(
        "ATHS DATASET MANAGER TEST"
    )
    print("=" * 60)

    manager = (
        ATHSDatasetManager()
    )

    dataset = (
        manager.prepare_dataset()
    )

    print()
    print(
        "RETURNED DATA"
    )
    print("-" * 50)

    print(
        f"X_train shape : "
        f"{dataset['X_train'].shape}"
    )

    print(
        f"X_test shape  : "
        f"{dataset['X_test'].shape}"
    )

    print(
        f"y_train shape : "
        f"{dataset['y_train'].shape}"
    )

    print(
        f"y_test shape  : "
        f"{dataset['y_test'].shape}"
    )

    print(
        f"Feedback      : "
        f"{len(dataset['feedback'])}"
    )

    print(
        f"Split file    : "
        f"{SPLIT_FILE}"
    )

    # --------------------------------------------------------
    # REPRODUCIBILITY CHECK
    # --------------------------------------------------------

    dataset_again = (
        manager.prepare_dataset()
    )

    train_same = np.array_equal(
        dataset["train_indices"],
        dataset_again[
            "train_indices"
        ],
    )

    test_same = np.array_equal(
        dataset["test_indices"],
        dataset_again[
            "test_indices"
        ],
    )

    print()
    print(
        "REPRODUCIBILITY CHECK"
    )
    print("-" * 50)

    print(
        f"Training split identical : "
        f"{train_same}"
    )

    print(
        f"Test split identical     : "
        f"{test_same}"
    )

    if not train_same or not test_same:

        raise RuntimeError(
            "Fixed split reproducibility "
            "check failed."
        )

    print()
    print(
        "ATHS DATASET MANAGER TEST COMPLETE"
    )


if __name__ == "__main__":
    main()