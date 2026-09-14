"""
ATHS XAI Explainer

Uses SHAP to explain the Random Forest threat detector.

The explainer:
1. Loads the already-trained Random Forest model.
2. Uses the existing processed feature dataset.
3. Calculates SHAP values.
4. Identifies features that increase or decrease malicious prediction.
5. Produces a structured explanation.

No model training or dataset preprocessing is performed here.
"""

from pathlib import Path
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
import shap


class ATHSXAIExplainer:
    """SHAP-based explainability for the ATHS Random Forest detector."""

    FEATURE_COLUMNS = [
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

    def __init__(
        self,
        model_path: str = "models/random_forest.joblib",
        features_path: str = "dataset/processed/features.csv",
    ):
        """Load the trained model and feature dataset."""

        self.project_root = Path(__file__).resolve().parent.parent

        self.model_path = self.project_root / model_path
        self.features_path = self.project_root / features_path

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Random Forest model not found: {self.model_path}"
            )

        if not self.features_path.exists():
            raise FileNotFoundError(
                f"Feature dataset not found: {self.features_path}"
            )

        print("Initializing ATHS XAI Explainer...")

        self.model = joblib.load(self.model_path)

        self.data = pd.read_csv(self.features_path)

        missing_columns = [
            column
            for column in self.FEATURE_COLUMNS
            if column not in self.data.columns
        ]

        if missing_columns:
            raise ValueError(
                "Missing ML feature columns: "
                + ", ".join(missing_columns)
            )

        self.X = self.data[self.FEATURE_COLUMNS].astype(
            np.float32
        )

        # TreeExplainer is designed for tree-based models
        # such as Random Forest.
        self.explainer = shap.TreeExplainer(self.model)

        print("ATHS XAI Explainer initialized.")
        print(f"  Model          : {self.model_path.name}")
        print(f"  Feature rows   : {len(self.X)}")
        print(f"  Feature count  : {len(self.FEATURE_COLUMNS)}")
        print("  Explainer      : SHAP TreeExplainer")

    # ------------------------------------------------------------------
    # SHAP value handling
    # ------------------------------------------------------------------

    def _get_malicious_shap_values(
        self,
        X: pd.DataFrame,
    ) -> np.ndarray:
        """
        Return SHAP values corresponding to the malicious class.

        SHAP output format can differ between versions, so this
        method handles both common formats.
        """

        shap_values = self.explainer.shap_values(X)

        # Older / traditional TreeExplainer format:
        #
        # list[
        #     class_0_values,
        #     class_1_values
        # ]
        if isinstance(shap_values, list):

            if len(shap_values) == 2:
                return np.asarray(shap_values[1])

            return np.asarray(shap_values[0])

        shap_values = np.asarray(shap_values)

        # Newer SHAP format may be:
        #
        # samples × features × classes
        #
        if shap_values.ndim == 3:

            if shap_values.shape[-1] == 2:
                return shap_values[:, :, 1]

            if shap_values.shape[1] == 2:
                return shap_values[:, 1, :]

        return shap_values

    # ------------------------------------------------------------------
    # Single prediction explanation
    # ------------------------------------------------------------------

    def explain(
        self,
        X: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Explain one feature vector.

        Parameters
        ----------
        X:
            One ATHS feature vector containing exactly 15 features.

        Returns
        -------
        Dictionary containing prediction and feature contributions.
        """

        X = np.asarray(X, dtype=np.float32)

        if X.ndim == 1:
            X = X.reshape(1, -1)

        if X.shape[1] != len(self.FEATURE_COLUMNS):
            raise ValueError(
                f"Expected {len(self.FEATURE_COLUMNS)} features, "
                f"received {X.shape[1]}."
            )

        X_df = pd.DataFrame(
            X,
            columns=self.FEATURE_COLUMNS,
        )

        prediction = int(self.model.predict(X)[0])

        probabilities = self.model.predict_proba(X)[0]

        probability_benign = float(probabilities[0])
        probability_malicious = float(probabilities[1])

        shap_values = self._get_malicious_shap_values(X_df)

        values = shap_values[0]

        feature_contributions = []

        for feature_name, feature_value, shap_value in zip(
            self.FEATURE_COLUMNS,
            X[0],
            values,
        ):

            feature_contributions.append(
                {
                    "feature": feature_name,
                    "value": float(feature_value),
                    "shap_value": float(shap_value),
                    "direction": (
                        "increases_threat"
                        if shap_value > 0
                        else "decreases_threat"
                        if shap_value < 0
                        else "neutral"
                    ),
                }
            )

        # Most influential features by absolute SHAP value.
        feature_contributions.sort(
            key=lambda item: abs(item["shap_value"]),
            reverse=True,
        )

        return {
            "prediction": prediction,
            "probability_benign": probability_benign,
            "probability_malicious": probability_malicious,
            "feature_contributions": feature_contributions,
            "top_features": feature_contributions[:5],
        }

    # ------------------------------------------------------------------
    # Dataset sample explanation
    # ------------------------------------------------------------------

    def explain_dataset_row(
        self,
        index: int,
    ) -> Dict[str, Any]:
        """Explain one existing row from the processed dataset."""

        if index < 0 or index >= len(self.X):
            raise IndexError(
                f"Dataset index must be between 0 and {len(self.X) - 1}."
            )

        result = self.explain(
            self.X.iloc[index].to_numpy()
        )

        result["dataset_index"] = index

        if "text" in self.data.columns:
            result["text"] = str(
                self.data.iloc[index]["text"]
            )

        if "label" in self.data.columns:
            result["actual_label"] = int(
                self.data.iloc[index]["label"]
            )

        return result

    # ------------------------------------------------------------------
    # Human-readable output
    # ------------------------------------------------------------------

    @staticmethod
    def print_explanation(
        result: Dict[str, Any],
    ) -> None:
        """Print a readable XAI explanation."""

        print("\nXAI EXPLANATION")
        print("=" * 60)

        if "dataset_index" in result:
            print(
                "Dataset index      :",
                result["dataset_index"],
            )

        if "text" in result:
            print(
                "Text               :",
                result["text"],
            )

        if "actual_label" in result:
            print(
                "Actual label       :",
                result["actual_label"],
            )

        print(
            "Predicted label    :",
            result["prediction"],
        )

        print(
            "Benign probability :",
            f"{result['probability_benign']:.4f}",
        )

        print(
            "Malicious probability:",
            f"{result['probability_malicious']:.4f}",
        )

        print("\nTOP FEATURE CONTRIBUTIONS")

        for item in result["top_features"]:

            print(
                f"  {item['feature']:<30} "
                f"value={item['value']:.4f}  "
                f"SHAP={item['shap_value']:+.4f}  "
                f"{item['direction']}"
            )

    # ------------------------------------------------------------------
    # Standalone test
    # ------------------------------------------------------------------

    def run_test(self) -> None:
        """Run benign and malicious dataset-row tests."""

        print("\nBENIGN SAMPLE TEST")

        benign_indices = self.data.index[
            self.data["label"] == 0
        ].tolist()

        if not benign_indices:
            raise RuntimeError(
                "No benign samples found in dataset."
            )

        benign_result = self.explain_dataset_row(
            benign_indices[0]
        )

        self.print_explanation(benign_result)

        print("\nMALICIOUS SAMPLE TEST")

        malicious_indices = self.data.index[
            self.data["label"] == 1
        ].tolist()

        if not malicious_indices:
            raise RuntimeError(
                "No malicious samples found in dataset."
            )

        malicious_result = self.explain_dataset_row(
            malicious_indices[0]
        )

        self.print_explanation(malicious_result)


def main():
    """Standalone XAI test."""

    print("ATHS XAI EXPLAINER TEST")

    explainer = ATHSXAIExplainer()

    explainer.run_test()

    print("\nATHS XAI EXPLAINER TEST COMPLETE")


if __name__ == "__main__":
    main()