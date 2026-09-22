"""Unified, failure-tolerant explanations for one ATHS detection result."""

from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np

from detection.feature_loader import ML_FEATURE_COLUMNS
from hypothesis.hypothesis_engine import HypothesisEngine
from xai.explainer import ATHSXAIExplainer


class ATHSUnifiedExplainer:
    """Combine model, rule, semantic, fusion, and hypothesis evidence."""

    def __init__(self) -> None:
        project_root = Path(__file__).resolve().parent.parent
        self.random_forest = ATHSXAIExplainer()
        self.logistic_regression = joblib.load(
            project_root / "models" / "logistic_regression.joblib"
        )
        self.hypothesis_engine = HypothesisEngine()

    @staticmethod
    def _logistic_explanation(feature_vector: np.ndarray, model: Any) -> Dict[str, Any]:
        values = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
        probabilities = model.predict_proba(values)[0]
        prediction = int(model.predict(values)[0])
        scaler = model.named_steps.get("scaler")
        classifier = model.named_steps.get("classifier")
        scaled = scaler.transform(values)[0]
        coefficients = np.asarray(classifier.coef_[0], dtype=float)
        contributions = []

        for feature, value, scaled_value, coefficient in zip(
            ML_FEATURE_COLUMNS, values[0], scaled, coefficients
        ):
            contribution = float(scaled_value * coefficient)
            contributions.append({
                "feature": feature,
                "value": float(value),
                "scaled_value": float(scaled_value),
                "coefficient": float(coefficient),
                "contribution": contribution,
                "direction": (
                    "increases_threat" if contribution > 0
                    else "decreases_threat" if contribution < 0
                    else "neutral"
                ),
            })

        contributions.sort(
            key=lambda item: abs(item["contribution"]), reverse=True
        )
        return {
            "prediction": prediction,
            "probability_benign": float(probabilities[0]),
            "probability_malicious": float(probabilities[1]),
            "intercept": float(classifier.intercept_[0]),
            "feature_contributions": contributions,
            "top_features": contributions[:5],
        }

    def explain(
        self,
        analysis: Dict[str, Any],
        feature_vector: np.ndarray,
    ) -> Dict[str, Any]:
        """Build an explanation without running any detector again."""
        explanation: Dict[str, Any] = {
            "status": "success",
            "random_forest": None,
            "logistic_regression": None,
            "rules": analysis.get("rules", {}),
            "semantic": analysis.get("semantic", {}),
            "threat_fusion": analysis.get("fusion", {}),
            "hypothesis": None,
            "errors": [],
        }

        try:
            explanation["random_forest"] = self.random_forest.explain(feature_vector)
        except Exception as exc:
            explanation["errors"].append({"source": "random_forest", "message": str(exc)})

        try:
            explanation["logistic_regression"] = self._logistic_explanation(
                feature_vector, self.logistic_regression
            )
        except Exception as exc:
            explanation["errors"].append({"source": "logistic_regression", "message": str(exc)})

        try:
            explanation["hypothesis"] = self.hypothesis_engine.analyze(analysis)
        except Exception as exc:
            explanation["errors"].append({"source": "hypothesis", "message": str(exc)})

        if explanation["errors"]:
            explanation["status"] = "partial"
        return explanation
