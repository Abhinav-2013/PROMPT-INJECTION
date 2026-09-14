from typing import Any, Dict, List

from data.feedback_store import ATHSFeedbackStore
from self_learning.eligibility import (
    ATHSLearningEligibility,
)


class ATHSLearningManager:
    """
    Controls which detection results are allowed into
    autonomous learning.
    """

    RETRAIN_TRIGGER_SAMPLES = 50
    MAX_SAMPLES_PER_CYCLE = 100

    def __init__(
        self,
        feedback_store=None,
        eligibility=None,
    ) -> None:

        self.store = (
            feedback_store
            or ATHSFeedbackStore()
        )

        self.eligibility = (
            eligibility
            or ATHSLearningEligibility()
        )

    def register_prediction(
        self,
        prediction_id: int,
        result: Dict[str, Any],
    ) -> Dict[str, Any]:

        eligibility = (
            self.eligibility.evaluate(result)
        )

        if not eligibility.eligible:
            pending = (
                self.store
                .count_pending_learning_samples()
            )

            return {
                "eligible": False,
                "pseudo_label": None,
                "confidence": 0.0,
                "reason": eligibility.reason,
                "pending_samples": pending,
                "retraining_due": False,
            }

        text = str(
            result["text"]
        )

        sample_id = (
            self.store.save_learning_candidate(
                prediction_id=prediction_id,
                text_hash=self.eligibility.text_hash(
                    text
                ),
                pseudo_label=int(
                    eligibility.pseudo_label
                ),
                confidence=eligibility.confidence,
                eligibility_reason=(
                    eligibility.reason
                ),
                ml_probability=float(
                    result["ml"]["probability"]
                ),
                rule_score=float(
                    result["rules"]["score"]
                ),
                semantic_score=float(
                    result["semantic"]["score"]
                ),
                detector_votes=int(
                    result["detector_votes"]["total"]
                ),
                decision=str(
                    result["decision"]
                ),
            )
        )

        pending = (
            self.store
            .count_pending_learning_samples()
        )

        return {
            "eligible": sample_id is not None,
            "pseudo_label": eligibility.pseudo_label,
            "confidence": round(
                eligibility.confidence,
                4,
            ),
            "reason": eligibility.reason,
            "sample_id": sample_id,
            "pending_samples": pending,
            "retraining_due": (
                pending
                >= self.RETRAIN_TRIGGER_SAMPLES
            ),
        }

    def get_training_batch(
        self,
    ) -> List[Dict[str, Any]]:

        return (
            self.store
            .get_pending_learning_samples(
                self.MAX_SAMPLES_PER_CYCLE
            )
        )