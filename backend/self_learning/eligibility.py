import hashlib
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class LearningEligibility:
    eligible: bool
    pseudo_label: Optional[int]
    confidence: float
    reason: str


class ATHSLearningEligibility:
    """
    Conservative pseudo-labeling policy.

    The ML prediction alone is never considered sufficient evidence.

    A sample can enter autonomous learning only when there is
    sufficiently strong cross-detector evidence.
    """

    MIN_MALICIOUS_ML = 0.90
    MIN_BENIGN_ML = 0.10

    MIN_RULE_SUPPORT = 0.70
    MIN_SEMANTIC_SUPPORT = 0.70
    MIN_SEMANTIC_SIMILARITY = 0.85

    def evaluate(
        self,
        result: Dict[str, Any],
    ) -> LearningEligibility:

        ml_probability = float(
            result["ml"]["probability"]
        )

        rule_score = float(
            result["rules"]["score"]
        )

        semantic_score = float(
            result["semantic"].get("score", 0.0)
        )

        semantic_similarity = float(
            result["semantic"].get("similarity", 0.0)
        )

        semantic_malicious = bool(
            result["semantic"].get(
                "is_malicious_match",
                False,
            )
        )

        detector_votes = result.get(
            "detector_votes",
            {},
        )

        vote_count = int(
            detector_votes.get(
                "total",
                0,
            )
        )

        decision = str(
            result.get(
                "decision",
                "REVIEW",
            )
        )

        # Never automatically learn from ambiguous cases.
        if decision == "REVIEW":
            return LearningEligibility(
                eligible=False,
                pseudo_label=None,
                confidence=0.0,
                reason="review_decision_excluded",
            )

        # ---------------------------------------------------------
        # STRONG MALICIOUS SAMPLE
        # ---------------------------------------------------------

        malicious_support = (
            rule_score >= self.MIN_RULE_SUPPORT
            or (
                semantic_malicious
                and (
                    semantic_score >= self.MIN_SEMANTIC_SUPPORT
                    or semantic_similarity
                    >= self.MIN_SEMANTIC_SIMILARITY
                )
            )
        )

        if (
            ml_probability >= self.MIN_MALICIOUS_ML
            and malicious_support
        ):
            confidence = min(
                1.0,
                (
                    0.60 * ml_probability
                    + 0.25 * rule_score
                    + 0.15 * max(
                        semantic_score,
                        semantic_similarity,
                    )
                ),
            )

            if vote_count >= 2:
                confidence = min(
                    1.0,
                    confidence + 0.05,
                )

            return LearningEligibility(
                eligible=True,
                pseudo_label=1,
                confidence=confidence,
                reason=(
                    "strong_malicious_detector_agreement"
                ),
            )

        # ---------------------------------------------------------
        # STRONG BENIGN SAMPLE
        # ---------------------------------------------------------

        if (
            ml_probability <= self.MIN_BENIGN_ML
            and rule_score == 0.0
            and not semantic_malicious
            and semantic_score
            < self.MIN_SEMANTIC_SUPPORT
        ):
            return LearningEligibility(
                eligible=True,
                pseudo_label=0,
                confidence=1.0 - ml_probability,
                reason=(
                    "strong_benign_detector_agreement"
                ),
            )

        # ---------------------------------------------------------
        # THREE-DETECTOR AGREEMENT
        # ---------------------------------------------------------

        if (
            vote_count == 3
            and ml_probability >= 0.80
        ):
            return LearningEligibility(
                eligible=True,
                pseudo_label=1,
                confidence=min(
                    1.0,
                    ml_probability,
                ),
                reason=(
                    "three_detector_malicious_agreement"
                ),
            )

        if (
            vote_count == 0
            and ml_probability <= 0.05
        ):
            return LearningEligibility(
                eligible=True,
                pseudo_label=0,
                confidence=min(
                    1.0,
                    1.0 - ml_probability,
                ),
                reason=(
                    "three_detector_benign_agreement"
                ),
            )

        return LearningEligibility(
            eligible=False,
            pseudo_label=None,
            confidence=0.0,
            reason=(
                "insufficient_cross_detector_confidence"
            ),
        )

    @staticmethod
    def text_hash(text: str) -> str:
        normalized = " ".join(
            str(text)
            .strip()
            .lower()
            .split()
        )

        return hashlib.sha256(
            normalized.encode("utf-8")
        ).hexdigest()