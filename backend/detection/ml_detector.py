"""
ATHS Machine Learning Threat Detector

This module trains and uses the ML models for the ATHS
Threat Detection Engine.

Models:
    1. Logistic Regression - baseline
    2. Random Forest       - nonlinear comparison

Input:
    detection/feature_loader.py

Data:
    dataset/processed/features.csv
    dataset/processed/embeddings.npy

Important:
    Semantic embeddings are intentionally NOT used in this file yet.
    They will be evaluated separately in the semantic detection stage.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
from numpy.typing import NDArray

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from detection.feature_loader import ATHSFeatureLoader


# -------------------------------------------------------------------------
# Paths
# -------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODELS_DIR = PROJECT_ROOT / "models"

LOGISTIC_MODEL_PATH = MODELS_DIR / "logistic_regression.joblib"
RANDOM_FOREST_MODEL_PATH = MODELS_DIR / "random_forest.joblib"


# -------------------------------------------------------------------------
# Dataset split configuration
# -------------------------------------------------------------------------

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

RANDOM_STATE = 42


# -------------------------------------------------------------------------
# ML Detector
# -------------------------------------------------------------------------

class MLThreatDetector:
    """
    Trains and evaluates the baseline ML threat detection models.
    """

    def __init__(
        self,
        random_state: int = RANDOM_STATE,
    ) -> None:

        self.random_state = random_state

        self.loader = ATHSFeatureLoader()

        self.logistic_model: Optional[Pipeline] = None
        self.random_forest_model: Optional[RandomForestClassifier] = None

        self.X_train: Optional[np.ndarray] = None
        self.X_validation: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None

        self.y_train: Optional[np.ndarray] = None
        self.y_validation: Optional[np.ndarray] = None
        self.y_test: Optional[np.ndarray] = None

        self.groups_train: Optional[np.ndarray] = None
        self.groups_validation: Optional[np.ndarray] = None
        self.groups_test: Optional[np.ndarray] = None

        self.feature_names = None

    # ---------------------------------------------------------------------
    # Load dataset
    # ---------------------------------------------------------------------

    def load_dataset(self) -> None:
        """
        Load the processed ATHS dataset.

        For this first ML stage, only the 15 handcrafted/NLP/security
        features are used.

        The 384-dimensional semantic embeddings are loaded and validated
        by the feature loader, but are not used by these baseline models.
        """

        data = self.loader.load()

        X = data["tabular_features"]
        y = data["labels"]
        groups = data["groups"]

        self.feature_names = data["feature_names"]

        print("=" * 70)
        print("ATHS DATASET")
        print("=" * 70)

        print(f"Samples              : {len(X)}")
        print(f"Tabular features     : {X.shape[1]}")
        print(f"Embedding dimension  : {data['embeddings'].shape[1]}")
        print(f"Benign samples       : {np.sum(y == 0)}")
        print(f"Malicious samples    : {np.sum(y == 1)}")
        print(f"Unique groups        : {len(np.unique(groups))}")

        self._split_dataset(X, y, groups)

    # ---------------------------------------------------------------------
    # Dataset split
    # ---------------------------------------------------------------------

    def _split_dataset(
        self,
        X: np.ndarray,
        y: np.ndarray,
        groups: np.ndarray,
    ) -> None:
        """
        Split data into:

            70% training
            15% validation
            15% test

        Stratification preserves the benign/malicious class distribution.

        group_id is retained with every split so that it can be used later
        for leakage analysis and more advanced splitting if required.
        """

        if not np.isclose(
            TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE,
            1.0,
        ):
            raise ValueError(
                "TRAIN_SIZE + VALIDATION_SIZE + TEST_SIZE must equal 1."
            )

        # -------------------------------------------------------------
        # First split:
        #
        # 70% train
        # 30% temporary
        # -------------------------------------------------------------

        (
            X_train,
            X_temp,
            y_train,
            y_temp,
            groups_train,
            groups_temp,
        ) = train_test_split(
            X,
            y,
            groups,
            test_size=(VALIDATION_SIZE + TEST_SIZE),
            stratify=y,
            random_state=self.random_state,
        )

        # -------------------------------------------------------------
        # Second split:
        #
        # Temporary 30%:
        #     50% validation
        #     50% test
        #
        # Therefore:
        #     15% validation
        #     15% test
        # -------------------------------------------------------------

        (
            X_validation,
            X_test,
            y_validation,
            y_test,
            groups_validation,
            groups_test,
        ) = train_test_split(
            X_temp,
            y_temp,
            groups_temp,
            test_size=0.50,
            stratify=y_temp,
            random_state=self.random_state,
        )

        self.X_train = X_train
        self.X_validation = X_validation
        self.X_test = X_test

        self.y_train = y_train
        self.y_validation = y_validation
        self.y_test = y_test

        self.groups_train = groups_train
        self.groups_validation = groups_validation
        self.groups_test = groups_test

        print("\nDataset split:")
        print(
            f"  Training   : {len(self.X_train)} "
            f"({len(self.X_train) / len(X) * 100:.1f}%)"
        )
        print(
            f"  Validation : {len(self.X_validation)} "
            f"({len(self.X_validation) / len(X) * 100:.1f}%)"
        )
        print(
            f"  Test       : {len(self.X_test)} "
            f"({len(self.X_test) / len(X) * 100:.1f}%)"
        )

        print("\nClass distribution:")
        print(
            f"  Training   - benign: {np.sum(y_train == 0)}, "
            f"malicious: {np.sum(y_train == 1)}"
        )
        print(
            f"  Validation - benign: {np.sum(y_validation == 0)}, "
            f"malicious: {np.sum(y_validation == 1)}"
        )
        print(
            f"  Test       - benign: {np.sum(y_test == 0)}, "
            f"malicious: {np.sum(y_test == 1)}"
        )

    # ---------------------------------------------------------------------
    # Train Logistic Regression
    # ---------------------------------------------------------------------

    def train_logistic_regression(self) -> Pipeline:
        """
        Train Logistic Regression baseline.

        StandardScaler is included inside the Pipeline so that scaling is
        fitted only on the training data.
        """

        if self.X_train is None or self.y_train is None:
            raise RuntimeError(
                "Dataset has not been loaded. "
                "Call load_dataset() first."
            )

        self.logistic_model = Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

        self.logistic_model.fit(
            self.X_train,
            self.y_train,
        )

        return self.logistic_model

    # ---------------------------------------------------------------------
    # Train Random Forest
    # ---------------------------------------------------------------------

    def train_random_forest(
        self,
    ) -> RandomForestClassifier:
        """
        Train Random Forest classifier.
        """

        if self.X_train is None or self.y_train is None:
            raise RuntimeError(
                "Dataset has not been loaded. "
                "Call load_dataset() first."
            )

        self.random_forest_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=self.random_state,
            n_jobs=-1,
        )

        self.random_forest_model.fit(
            self.X_train,
            self.y_train,
        )

        return self.random_forest_model

    # ---------------------------------------------------------------------
    # Evaluate model
    # ---------------------------------------------------------------------

    def evaluate_model(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        model_name: str,
    ) -> Dict:
        """
        Evaluate a trained model.

        Metrics:
            accuracy
            precision
            recall
            F1
            ROC-AUC
            confusion matrix
        """

        predictions = model.predict(X)

        probabilities = model.predict_proba(X)[:, 1]

        accuracy = accuracy_score(
            y,
            predictions,
        )

        precision = precision_score(
            y,
            predictions,
            zero_division=0,
        )

        recall = recall_score(
            y,
            predictions,
            zero_division=0,
        )

        f1 = f1_score(
            y,
            predictions,
            zero_division=0,
        )

        roc_auc = roc_auc_score(
            y,
            probabilities,
        )

        matrix = confusion_matrix(
            y,
            predictions,
        )

        # Confusion matrix:
        #
        # [[TN, FP],
        #  [FN, TP]]
        #
        # False Negative Rate:
        # FN / (FN + TP)
        #
        # False Positive Rate:
        # FP / (FP + TN)

        true_negative = matrix[0, 0]
        false_positive = matrix[0, 1]
        false_negative = matrix[1, 0]
        true_positive = matrix[1, 1]

        false_positive_rate = (
            false_positive / (false_positive + true_negative)
            if (false_positive + true_negative) > 0
            else 0.0
        )

        false_negative_rate = (
            false_negative / (false_negative + true_positive)
            if (false_negative + true_positive) > 0
            else 0.0
        )

        metrics = {
            "model": model_name,
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "roc_auc": float(roc_auc),
            "false_positive_rate": float(false_positive_rate),
            "false_negative_rate": float(false_negative_rate),
            "confusion_matrix": matrix.tolist(),
            "true_negative": int(true_negative),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "true_positive": int(true_positive),
        }

        print("\n" + "=" * 70)
        print(model_name)
        print("=" * 70)

        print(f"Accuracy          : {accuracy:.4f}")
        print(f"Precision         : {precision:.4f}")
        print(f"Recall            : {recall:.4f}")
        print(f"F1 Score          : {f1:.4f}")
        print(f"ROC-AUC           : {roc_auc:.4f}")
        print(f"False Positive Rate: {false_positive_rate:.4f}")
        print(f"False Negative Rate: {false_negative_rate:.4f}")

        print("\nConfusion Matrix:")
        print(matrix)

        print("\nClassification Report:")
        print(
            classification_report(
                y,
                predictions,
                target_names=[
                    "benign",
                    "malicious",
                ],
                zero_division=0,
            )
        )

        return metrics

    # ---------------------------------------------------------------------
    # Evaluate validation set
    # ---------------------------------------------------------------------

    def evaluate_validation(self) -> Dict[str, Dict]:
        """
        Evaluate both models on the validation set.

        The validation set is used for model comparison and future
        threshold tuning.
        """

        if self.X_validation is None or self.y_validation is None:
            raise RuntimeError(
                "Validation data is not available."
            )

        if self.logistic_model is None:
            raise RuntimeError(
                "Logistic Regression has not been trained."
            )

        if self.random_forest_model is None:
            raise RuntimeError(
                "Random Forest has not been trained."
            )

        logistic_metrics = self.evaluate_model(
            self.logistic_model,
            self.X_validation,
            self.y_validation,
            "Logistic Regression - Validation",
        )

        random_forest_metrics = self.evaluate_model(
            self.random_forest_model,
            self.X_validation,
            self.y_validation,
            "Random Forest - Validation",
        )

        return {
            "logistic_regression": logistic_metrics,
            "random_forest": random_forest_metrics,
        }

    # ---------------------------------------------------------------------
    # Evaluate final test set
    # ---------------------------------------------------------------------

    def evaluate_test(self) -> Dict[str, Dict]:
        """
        Evaluate both models on the untouched test set.

        This should only be called after model selection/validation.
        """

        if self.X_test is None or self.y_test is None:
            raise RuntimeError(
                "Test data is not available."
            )

        if self.logistic_model is None:
            raise RuntimeError(
                "Logistic Regression has not been trained."
            )

        if self.random_forest_model is None:
            raise RuntimeError(
                "Random Forest has not been trained."
            )

        logistic_metrics = self.evaluate_model(
            self.logistic_model,
            self.X_test,
            self.y_test,
            "Logistic Regression - Test",
        )

        random_forest_metrics = self.evaluate_model(
            self.random_forest_model,
            self.X_test,
            self.y_test,
            "Random Forest - Test",
        )

        return {
            "logistic_regression": logistic_metrics,
            "random_forest": random_forest_metrics,
        }

    # ---------------------------------------------------------------------
    # Predict using selected model
    # ---------------------------------------------------------------------

    def predict(
        self,
        X: NDArray[np.float32],
        model_name: str = "random_forest",
    ) -> Dict[str, Any]:
        """
        Predict the probability of a malicious prompt.

        Parameters:
            X:
                Array containing the 15 tabular features.

            model_name:
                "logistic_regression"
                or
                "random_forest"
        """

        if model_name == "logistic_regression":

            if self.logistic_model is None:
                raise RuntimeError(
                    "Logistic Regression has not been trained."
                )

            model = self.logistic_model

        elif model_name == "random_forest":

            if self.random_forest_model is None:
                raise RuntimeError(
                    "Random Forest has not been trained."
                )

            model = self.random_forest_model

        else:
            raise ValueError(
                "Unknown model name. Use "
                "'logistic_regression' or 'random_forest'."
            )

        X = np.asarray(
            X,
            dtype=np.float32,
        )

        if X.ndim == 1:
            X = X.reshape(1, -1)

        probabilities = model.predict_proba(X)

        predictions = model.predict(X)

        return {
            "label": int(predictions[0]),
            "probability_benign": float(probabilities[0][0]),
            "probability_malicious": float(probabilities[0][1]),
        }

    # ---------------------------------------------------------------------
    # Save models
    # ---------------------------------------------------------------------

    def save_models(self) -> None:
        """
        Save trained models into the models/ directory.
        """

        MODELS_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        if self.logistic_model is None:
            raise RuntimeError(
                "Logistic Regression has not been trained."
            )

        if self.random_forest_model is None:
            raise RuntimeError(
                "Random Forest has not been trained."
            )

        joblib.dump(
            self.logistic_model,
            LOGISTIC_MODEL_PATH,
        )

        joblib.dump(
            self.random_forest_model,
            RANDOM_FOREST_MODEL_PATH,
        )

        print("\nModels saved:")
        print(f"  {LOGISTIC_MODEL_PATH}")
        print(f"  {RANDOM_FOREST_MODEL_PATH}")

    # ---------------------------------------------------------------------
    # Load saved models
    # ---------------------------------------------------------------------

    def load_models(self) -> None:
        """
        Load previously trained models.
        """

        if not LOGISTIC_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Logistic Regression model not found:\n"
                f"{LOGISTIC_MODEL_PATH}"
            )

        if not RANDOM_FOREST_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Random Forest model not found:\n"
                f"{RANDOM_FOREST_MODEL_PATH}"
            )

        self.logistic_model = joblib.load(
            LOGISTIC_MODEL_PATH
        )

        self.random_forest_model = joblib.load(
            RANDOM_FOREST_MODEL_PATH
        )

        print("Saved models loaded successfully.")

    # ---------------------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------------------

    def get_random_forest_feature_importance(self) -> Dict[str, float]:
        """
        Return Random Forest feature importance.

        This will later be used by the XAI layer.
        """

        if self.random_forest_model is None:
            raise RuntimeError(
                "Random Forest has not been trained."
            )

        importances = (
            self.random_forest_model.feature_importances_
        )

        importance_dict = dict(
            zip(
                self.feature_names,
                importances,
            )
        )

        return dict(
            sorted(
                importance_dict.items(),
                key=lambda item: item[1],
                reverse=True,
            )
        )


# -------------------------------------------------------------------------
# Standalone training test
# -------------------------------------------------------------------------

def main() -> None:

    print("\n")
    print("=" * 70)
    print("ATHS MACHINE LEARNING THREAT DETECTOR")
    print("=" * 70)

    detector = MLThreatDetector()

    # -------------------------------------------------------------
    # 1. Load processed data
    # -------------------------------------------------------------

    detector.load_dataset()

    # -------------------------------------------------------------
    # 2. Train baseline model
    # -------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TRAINING LOGISTIC REGRESSION")
    print("=" * 70)

    detector.train_logistic_regression()

    # -------------------------------------------------------------
    # 3. Train nonlinear model
    # -------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("TRAINING RANDOM FOREST")
    print("=" * 70)

    detector.train_random_forest()

    # -------------------------------------------------------------
    # 4. Validation evaluation
    # -------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("VALIDATION EVALUATION")
    print("=" * 70)

    validation_results = detector.evaluate_validation()

    # -------------------------------------------------------------
    # 5. Select the better model based on validation F1
    # -------------------------------------------------------------

    logistic_f1 = validation_results[
        "logistic_regression"
    ]["f1"]

    random_forest_f1 = validation_results[
        "random_forest"
    ]["f1"]

    if random_forest_f1 >= logistic_f1:
        selected_model = "random_forest"
    else:
        selected_model = "logistic_regression"

    print("\nSelected model based on validation F1:")
    print(f"  {selected_model}")

    # -------------------------------------------------------------
    # 6. Final test evaluation
    # -------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("FINAL TEST EVALUATION")
    print("=" * 70)

    test_results = detector.evaluate_test()

    # -------------------------------------------------------------
    # 7. Save models
    # -------------------------------------------------------------

    detector.save_models()

    # -------------------------------------------------------------
    # 8. Feature importance
    # -------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("RANDOM FOREST FEATURE IMPORTANCE")
    print("=" * 70)

    importance = (
        detector.get_random_forest_feature_importance()
    )

    for feature, value in importance.items():
        print(f"{feature:35s}: {value:.6f}")

    # -------------------------------------------------------------
    # 9. Example prediction
    # -------------------------------------------------------------

    print("\n")
    print("=" * 70)
    print("EXAMPLE PREDICTION")
    print("=" * 70)

    example_features = detector.X_test[0]

    prediction = detector.predict(
        example_features,
        model_name=selected_model,
    )

    print(f"Model                 : {selected_model}")
    print(f"Predicted label       : {prediction['label']}")
    print(
        f"Benign probability    : "
        f"{prediction['probability_benign']:.4f}"
    )
    print(
        f"Malicious probability : "
        f"{prediction['probability_malicious']:.4f}"
    )

    print("\n")
    print("=" * 70)
    print("ATHS ML DETECTOR TEST COMPLETE")
    print("=" * 70)


# -------------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------------

if __name__ == "__main__":
    main()