"""
ATHS Model Retraining Pipeline

Retrains the Random Forest model using:

    1. Original training dataset
    2. Confirmed human feedback
    3. Eligible autonomous learning samples

The fixed test set is NEVER added to training.

Pipeline:

    Original training data
            +
    Human-confirmed feedback
            +
    Eligible self-learning samples
            |
            v
       Candidate model
            |
            v
      Fixed test evaluation
            |
            v
     Promotion / rejection
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from data.feedback_store import ATHSFeedbackStore
from detection.feature_loader import ML_FEATURE_COLUMNS
from preprocessing.pipeline import PreprocessingPipeline
from retraining.dataset_manager import ATHSDatasetManager


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

CANDIDATE_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "random_forest_candidate.joblib"
)

RANDOM_STATE = 42

# Human feedback threshold.
MIN_FEEDBACK_SAMPLES = 10

# Autonomous learning threshold.
#
# This is intentionally separate from the human-feedback
# threshold. Self-learning samples are only used when explicitly
# supplied to the retrainer.
MIN_LEARNING_SAMPLES = 50

# Maximum autonomous samples used in a single training cycle.
MAX_LEARNING_SAMPLES_PER_CYCLE = 250

# Random Forest configuration.
N_ESTIMATORS = 300
CLASS_WEIGHT = "balanced"


# ============================================================
# ATHS MODEL RETRAINER
# ============================================================

class ATHSModelRetrainer:
    """
    Creates a new Random Forest candidate model.

    Training sources:
        - original fixed training split
        - confirmed human feedback
        - optional eligible autonomous samples

    The class does NOT promote the candidate.

    Promotion remains the responsibility of model_evaluator.py.
    """

    def __init__(
        self,
        learning_samples: Optional[List[Dict[str, Any]]] = None,
        include_human_feedback: bool = True,
    ) -> None:
        print("Initializing ATHS Model Retrainer...")

        self.candidate_model_path = CANDIDATE_MODEL_PATH

        self.dataset_manager = ATHSDatasetManager()

        self.feedback_store = ATHSFeedbackStore()

        self.learning_samples = (
            learning_samples
            if learning_samples is not None
            else []
        )

        self.include_human_feedback = (
            include_human_feedback
        )

        print("ATHS Model Retrainer initialized.")

    # ========================================================
    # DATASET PREPARATION
    # ========================================================

    def prepare_dataset(self) -> Dict[str, Any]:
        """
        Load the original dataset and fixed train/test split.
        """
        dataset = self.dataset_manager.prepare_dataset()

        return dataset

    # ========================================================
    # FEATURE PREPROCESSING
    # ========================================================

    @staticmethod
    def _build_feature_vector(
        processed: Dict[str, Any],
    ) -> np.ndarray:
        """
        Convert preprocessing output into the exact 15-feature
        vector expected by the Random Forest.
        """

        feature_sources = {
            **processed["nlp"],
            **processed["obfuscation"],
            **processed["security_features"],
        }

        missing_features = [
            column
            for column in ML_FEATURE_COLUMNS
            if column not in feature_sources
        ]

        if missing_features:
            raise ValueError(
                "Missing ML features during retraining: "
                f"{missing_features}"
            )

        vector = np.asarray(
            [
                feature_sources[column]
                for column in ML_FEATURE_COLUMNS
            ],
            dtype=np.float32,
        )

        if vector.shape != (
            len(ML_FEATURE_COLUMNS),
        ):
            raise ValueError(
                "Invalid feature vector shape. "
                f"Expected {(len(ML_FEATURE_COLUMNS),)}, "
                f"received {vector.shape}."
            )

        return vector

    # ========================================================
    # HUMAN FEEDBACK
    # ========================================================

    def preprocess_human_feedback(
        self,
        feedback: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert confirmed human feedback into ML features.
        """

        if not feedback:
            return (
                np.empty(
                    (
                        0,
                        len(ML_FEATURE_COLUMNS),
                    ),
                    dtype=np.float32,
                ),
                np.empty(
                    (0,),
                    dtype=int,
                ),
            )

        print()
        print("PREPROCESSING HUMAN-LABELED PROMPTS")
        print("-" * 50)

        pipeline = PreprocessingPipeline(
            enable_embeddings=False
        )

        features: List[np.ndarray] = []
        labels: List[int] = []

        for index, record in enumerate(
            feedback,
            start=1,
        ):
            text = str(
                record.get("text", "")
            ).strip()

            if not text:
                continue

            human_label = int(
                record["human_label"]
            )

            if human_label not in (0, 1):
                continue

            processed = pipeline.process_text(
                text
            )

            feature_vector = (
                self._build_feature_vector(
                    processed
                )
            )

            features.append(
                feature_vector
            )

            labels.append(
                human_label
            )

            print(
                f"  Processed "
                f"{index}/{len(feedback)} "
                f"| label={human_label}"
            )

        if not features:
            return (
                np.empty(
                    (
                        0,
                        len(ML_FEATURE_COLUMNS),
                    ),
                    dtype=np.float32,
                ),
                np.empty(
                    (0,),
                    dtype=int,
                ),
            )

        return (
            np.asarray(
                features,
                dtype=np.float32,
            ),
            np.asarray(
                labels,
                dtype=int,
            ),
        )

    # ========================================================
    # AUTONOMOUS LEARNING SAMPLES
    # ========================================================

    def preprocess_learning_samples(
        self,
        samples: List[Dict[str, Any]],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert eligible autonomous learning samples into
        training features.

        Only samples with valid pseudo-labels 0/1 are accepted.

        Embeddings are deliberately not used for Random Forest
        retraining because the production RF uses the 15 engineered
        tabular features.
        """

        if not samples:
            return (
                np.empty(
                    (
                        0,
                        len(ML_FEATURE_COLUMNS),
                    ),
                    dtype=np.float32,
                ),
                np.empty(
                    (0,),
                    dtype=int,
                ),
            )

        print()
        print("PREPROCESSING AUTONOMOUS LEARNING SAMPLES")
        print("-" * 50)

        pipeline = PreprocessingPipeline(
            enable_embeddings=False
        )

        features: List[np.ndarray] = []
        labels: List[int] = []

        limited_samples = samples[
            :MAX_LEARNING_SAMPLES_PER_CYCLE
        ]

        for index, record in enumerate(
            limited_samples,
            start=1,
        ):
            text = str(
                record.get("text", "")
            ).strip()

            if not text:
                continue

            label_value = record.get(
                "pseudo_label",
                record.get("label"),
            )

            if label_value is None:
                continue

            label = int(label_value)

            if label not in (0, 1):
                continue

            processed = pipeline.process_text(
                text
            )

            feature_vector = (
                self._build_feature_vector(
                    processed
                )
            )

            features.append(
                feature_vector
            )

            labels.append(
                label
            )

            confidence = record.get(
                "confidence"
            )

            print(
                f"  Processed "
                f"{index}/{len(limited_samples)} "
                f"| label={label}"
                f" | confidence={confidence}"
            )

        if not features:
            return (
                np.empty(
                    (
                        0,
                        len(ML_FEATURE_COLUMNS),
                    ),
                    dtype=np.float32,
                ),
                np.empty(
                    (0,),
                    dtype=int,
                ),
            )

        return (
            np.asarray(
                features,
                dtype=np.float32,
            ),
            np.asarray(
                labels,
                dtype=int,
            ),
        )

    # ========================================================
    # COMBINE DATA
    # ========================================================

    @staticmethod
    def combine_training_data(
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_feedback: np.ndarray,
        y_feedback: np.ndarray,
        X_learning: Optional[np.ndarray] = None,
        y_learning: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Combine original training data with feedback and
        autonomous learning samples.

        X_test is never passed to this method.
        """

        arrays_x = [X_train]
        arrays_y = [y_train]

        if (
            X_feedback is not None
            and len(X_feedback) > 0
        ):
            arrays_x.append(X_feedback)
            arrays_y.append(y_feedback)

        if (
            X_learning is not None
            and len(X_learning) > 0
        ):
            arrays_x.append(X_learning)
            arrays_y.append(y_learning)

        X_combined = np.vstack(
            arrays_x
        ).astype(
            np.float32
        )

        y_combined = np.concatenate(
            arrays_y
        ).astype(
            int
        )

        return (
            X_combined,
            y_combined,
        )

    # ========================================================
    # TRAIN MODEL
    # ========================================================

    def train_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
    ) -> RandomForestClassifier:
        """
        Train the candidate Random Forest.
        """

        if len(X_train) == 0:
            raise ValueError(
                "Training dataset is empty."
            )

        if len(
            np.unique(y_train)
        ) < 2:
            raise ValueError(
                "Training dataset must contain "
                "both benign and malicious classes."
            )

        print()
        print("TRAINING RANDOM FOREST CANDIDATE")
        print("-" * 50)

        model = RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            class_weight=CLASS_WEIGHT,
            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train,
        )

        print(
            f"Trees       : {N_ESTIMATORS}"
        )

        print(
            f"Samples     : {len(y_train)}"
        )

        print(
            f"Features    : {X_train.shape[1]}"
        )

        return model

    # ========================================================
    # EVALUATE CANDIDATE
    # ========================================================

    @staticmethod
    def evaluate_model(
        model: RandomForestClassifier,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluate the candidate exclusively on the fixed test set.
        """

        predictions = model.predict(
            X_test
        )

        probabilities = model.predict_proba(
            X_test
        )[:, 1]

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        precision = precision_score(
            y_test,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            zero_division=0,
        )

        try:
            roc_auc = roc_auc_score(
                y_test,
                probabilities,
            )
        except ValueError:
            roc_auc = 0.0

        matrix = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1],
        )

        tn, fp, fn, tp = (
            matrix.ravel()
        )

        fpr = (
            fp / (fp + tn)
            if (fp + tn) > 0
            else 0.0
        )

        fnr = (
            fn / (fn + tp)
            if (fn + tp) > 0
            else 0.0
        )

        metrics = {
            "accuracy": float(
                accuracy
            ),
            "precision": float(
                precision
            ),
            "recall": float(
                recall
            ),
            "f1": float(
                f1
            ),
            "roc_auc": float(
                roc_auc
            ),
            "fpr": float(
                fpr
            ),
            "fnr": float(
                fnr
            ),
            "confusion_matrix": (
                matrix.tolist()
            ),
        }

        print()
        print("CANDIDATE MODEL EVALUATION")
        print("-" * 50)

        print(
            f"Accuracy  : {accuracy:.4f}"
        )
        print(
            f"Precision : {precision:.4f}"
        )
        print(
            f"Recall    : {recall:.4f}"
        )
        print(
            f"F1        : {f1:.4f}"
        )
        print(
            f"ROC-AUC   : {roc_auc:.4f}"
        )
        print(
            f"FPR       : {fpr:.4f}"
        )
        print(
            f"FNR       : {fnr:.4f}"
        )

        print()
        print("Confusion Matrix:")
        print(matrix)

        return metrics

    # ========================================================
    # SAVE CANDIDATE
    # ========================================================

    def save_candidate_model(
        self,
        model: RandomForestClassifier,
    ) -> None:
        """
        Save the newly trained model as a candidate.

        It does NOT replace the production model.
        """

        self.candidate_model_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        joblib.dump(
            model,
            self.candidate_model_path,
        )

        print()
        print("CANDIDATE MODEL SAVED")
        print("-" * 50)
        print(
            self.candidate_model_path
        )

    # ========================================================
    # COMPLETE RETRAINING
    # ========================================================

    def retrain(self) -> Dict[str, Any]:
        """
        Execute a complete candidate retraining cycle.
        """

        dataset = self.prepare_dataset()

        X_train = dataset[
            "X_train"
        ]

        y_train = dataset[
            "y_train"
        ]

        X_test = dataset[
            "X_test"
        ]

        y_test = dataset[
            "y_test"
        ]

        human_feedback = []

        if self.include_human_feedback:
            human_feedback = dataset.get(
                "feedback",
                [],
            )

        # ----------------------------------------------------
        # Human feedback requirement
        #
        # For a purely autonomous cycle, learning_samples can
        # be supplied without human feedback.
        # ----------------------------------------------------

        if (
            not human_feedback
            and not self.learning_samples
        ):
            print()
            print(
                "No human feedback or autonomous "
                "learning samples available."
            )

            return {
                "status": "no_learning_data",
                "human_feedback_samples": 0,
                "learning_samples": 0,
            }

        # ----------------------------------------------------
        # Preprocess human feedback
        # ----------------------------------------------------

        (
            X_feedback,
            y_feedback,
        ) = self.preprocess_human_feedback(
            human_feedback
        )

        # ----------------------------------------------------
        # Preprocess autonomous samples
        # ----------------------------------------------------

        (
            X_learning,
            y_learning,
        ) = self.preprocess_learning_samples(
            self.learning_samples
        )

        # ----------------------------------------------------
        # Safety validation
        # ----------------------------------------------------

        if (
            len(X_feedback) == 0
            and len(X_learning) == 0
        ):
            return {
                "status": "no_valid_learning_data",
                "human_feedback_samples": 0,
                "learning_samples": 0,
            }

        # ----------------------------------------------------
        # Combine training data
        #
        # IMPORTANT:
        # X_test/y_test are never modified.
        # ----------------------------------------------------

        (
            X_train_combined,
            y_train_combined,
        ) = self.combine_training_data(
            X_train=X_train,
            y_train=y_train,
            X_feedback=X_feedback,
            y_feedback=y_feedback,
            X_learning=X_learning,
            y_learning=y_learning,
        )

        print()
        print("DATA ISOLATION CHECK")
        print("-" * 50)

        print(
            f"Original training : "
            f"{X_train.shape}"
        )

        print(
            f"Human feedback    : "
            f"{X_feedback.shape}"
        )

        print(
            f"Self-learning     : "
            f"{X_learning.shape}"
        )

        print(
            f"Combined training  : "
            f"{X_train_combined.shape}"
        )

        print(
            f"Fixed test         : "
            f"{X_test.shape}"
        )

        print(
            "Fixed test set is excluded "
            "from training."
        )

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model = self.train_model(
            X_train_combined,
            y_train_combined,
        )

        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------

        metrics = self.evaluate_model(
            model,
            X_test,
            y_test,
        )

        # ----------------------------------------------------
        # Save candidate
        # ----------------------------------------------------

        self.save_candidate_model(
            model
        )

        result = {
            "status": "candidate_created",
            "human_feedback_samples": int(
                len(X_feedback)
            ),
            "learning_samples": int(
                len(X_learning)
            ),
            "training_samples": int(
                len(y_train_combined)
            ),
            "test_samples": int(
                len(y_test)
            ),
            "metrics": metrics,
            "candidate_model": str(
                self.candidate_model_path
            ),
        }

        return result


# ============================================================
# STANDALONE TEST
# ============================================================

def main() -> None:
    print()
    print("=" * 60)
    print("ATHS MODEL RETRAINING TEST")
    print("=" * 60)

    retrainer = ATHSModelRetrainer()

    result = retrainer.retrain()

    print()
    print("RETRAINING RESULT")
    print("-" * 50)

    print(
        f"Status : {result['status']}"
    )

    if "human_feedback_samples" in result:
        print(
            "Human feedback : "
            f"{result['human_feedback_samples']}"
        )

    if "learning_samples" in result:
        print(
            "Self-learning  : "
            f"{result['learning_samples']}"
        )

    if "training_samples" in result:
        print(
            "Training samples : "
            f"{result['training_samples']}"
        )

    if "test_samples" in result:
        print(
            "Test samples     : "
            f"{result['test_samples']}"
        )

    if "metrics" in result:
        print()
        print(
            "Candidate F1     : "
            f"{result['metrics']['f1']:.4f}"
        )

        print(
            "Candidate Recall : "
            f"{result['metrics']['recall']:.4f}"
        )

        print(
            "Candidate FNR    : "
            f"{result['metrics']['fnr']:.4f}"
        )

    print()
    print("=" * 60)
    print(
        "ATHS MODEL RETRAINING TEST COMPLETE"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()