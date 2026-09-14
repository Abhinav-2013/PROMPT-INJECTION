"""
ATHS Threat Fusion Engine

Combines ML, rule-based, and semantic threat signals into a final
threat score and security decision.

Decision levels:
    ALLOW
    REVIEW
    BLOCK
"""

from typing import Any, Dict


class ThreatFusion:
    """Combines multiple threat signals into a final threat score."""

    ML_WEIGHT = 0.50
    RULE_WEIGHT = 0.35
    SEMANTIC_WEIGHT = 0.15

    ALLOW_THRESHOLD = 0.30
    BLOCK_THRESHOLD = 0.75

    # A very high-confidence ML prediction is independently sufficient
    # to block. This prevents a 1.00 ML threat score from being diluted
    # into REVIEW when supporting detectors have no matching evidence.
    HIGH_CONFIDENCE_ML_BLOCK = 0.90

    def __init__(
        self,
        ml_weight: float = ML_WEIGHT,
        rule_weight: float = RULE_WEIGHT,
        semantic_weight: float = SEMANTIC_WEIGHT,
        allow_threshold: float = ALLOW_THRESHOLD,
        block_threshold: float = BLOCK_THRESHOLD,
        high_confidence_ml_block: float = HIGH_CONFIDENCE_ML_BLOCK,
    ):
        self.ml_weight = ml_weight
        self.rule_weight = rule_weight
        self.semantic_weight = semantic_weight
        self.allow_threshold = allow_threshold
        self.block_threshold = block_threshold
        self.high_confidence_ml_block = high_confidence_ml_block
        self._validate_configuration()

    def _validate_configuration(self):
        weights = [self.ml_weight, self.rule_weight, self.semantic_weight]

        if any(weight < 0 for weight in weights):
            raise ValueError("Fusion weights cannot be negative.")

        weight_sum = sum(weights)
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(
                "Fusion weights must sum to 1.0. "
                f"Current sum: {weight_sum:.4f}"
            )

        if not 0.0 <= self.allow_threshold <= 1.0:
            raise ValueError("allow_threshold must be between 0 and 1.")

        if not 0.0 <= self.block_threshold <= 1.0:
            raise ValueError("block_threshold must be between 0 and 1.")

        if self.allow_threshold >= self.block_threshold:
            raise ValueError(
                "allow_threshold must be lower than block_threshold."
            )

        if not 0.0 <= self.high_confidence_ml_block <= 1.0:
            raise ValueError(
                "high_confidence_ml_block must be between 0 and 1."
            )

    @staticmethod
    def _validate_score(score: float, name: str) -> float:
        try:
            score = float(score)
        except (TypeError, ValueError):
            raise ValueError(f"{name} must be a numeric value.")

        if not 0.0 <= score <= 1.0:
            raise ValueError(
                f"{name} must be between 0 and 1. Received: {score}"
            )

        return score

    def calculate_threat_score(
        self,
        ml_probability: float,
        rule_score: float,
        semantic_score: float,
    ) -> float:
        ml_probability = self._validate_score(ml_probability, "ml_probability")
        rule_score = self._validate_score(rule_score, "rule_score")
        semantic_score = self._validate_score(semantic_score, "semantic_score")

        threat_score = (
            self.ml_weight * ml_probability
            + self.rule_weight * rule_score
            + self.semantic_weight * semantic_score
        )

        return round(threat_score, 4)

    def get_decision(
        self,
        threat_score: float,
        ml_probability: float = 0.0,
        rule_score: float = 0.0,
        semantic_score: float = 0.0,
    ) -> str:
        """
        Determine the final decision.

        A normal weighted score of >= BLOCK_THRESHOLD blocks.

        Additionally, a very high-confidence ML malicious probability
        independently blocks. This is deliberately restricted to a
        high threshold so ordinary REVIEW cases are not automatically
        converted to BLOCK.
        """
        threat_score = self._validate_score(threat_score, "threat_score")
        ml_probability = self._validate_score(
            ml_probability, "ml_probability"
        )
        rule_score = self._validate_score(rule_score, "rule_score")
        semantic_score = self._validate_score(
            semantic_score, "semantic_score"
        )

        if ml_probability >= self.high_confidence_ml_block:
            return "BLOCK"

        if threat_score < self.allow_threshold:
            return "ALLOW"

        if threat_score >= self.block_threshold:
            return "BLOCK"

        return "REVIEW"

    def fuse(
        self,
        ml_probability: float,
        rule_score: float,
        semantic_score: float,
    ) -> Dict[str, Any]:
        ml_probability = self._validate_score(ml_probability, "ml_probability")
        rule_score = self._validate_score(rule_score, "rule_score")
        semantic_score = self._validate_score(semantic_score, "semantic_score")

        threat_score = self.calculate_threat_score(
            ml_probability=ml_probability,
            rule_score=rule_score,
            semantic_score=semantic_score,
        )

        decision = self.get_decision(
            threat_score=threat_score,
            ml_probability=ml_probability,
            rule_score=rule_score,
            semantic_score=semantic_score,
        )

        return {
            "ml_probability": round(ml_probability, 4),
            "rule_score": round(rule_score, 4),
            "semantic_score": round(semantic_score, 4),
            "threat_score": threat_score,
            "decision": decision,
            "weights": {
                "ml": self.ml_weight,
                "rules": self.rule_weight,
                "semantic": self.semantic_weight,
            },
            "thresholds": {
                "allow": self.allow_threshold,
                "block": self.block_threshold,
                "high_confidence_ml_block": self.high_confidence_ml_block,
            },
        }


def print_result(title: str, result: Dict):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    print(f"ML probability : {result['ml_probability']:.4f}")
    print(f"Rule score     : {result['rule_score']:.4f}")
    print(f"Semantic score : {result['semantic_score']:.4f}")
    print(f"Threat score   : {result['threat_score']:.4f}")
    print(f"Decision       : {result['decision']}")


def main():
    fusion = ThreatFusion()

    tests = [
        ("BENIGN", 0.10, 0.00, 0.10),
        ("REVIEW", 0.55, 0.40, 0.50),
        ("HIGH ML ONLY", 1.00, 0.00, 0.00),
        ("MALICIOUS", 0.95, 0.90, 0.85),
    ]

    print("=" * 70)
    print("ATHS THREAT FUSION TEST")
    print("=" * 70)

    for title, ml, rules, semantic in tests:
        print_result(title, fusion.fuse(ml, rules, semantic))

    print("\n" + "=" * 70)
    print("ATHS THREAT FUSION TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
