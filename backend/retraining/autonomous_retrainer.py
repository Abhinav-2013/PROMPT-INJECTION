import threading
from typing import Any, Dict

from data.feedback_store import (
    ATHSFeedbackStore,
)

from self_learning.manager import (
    ATHSLearningManager,
)

from retraining.retrain_model import (
    ATHSModelRetrainer,
)

from retraining.model_evaluator import (
    ATHSModelEvaluator,
)

from retraining.model_promoter import (
    ATHSModelPromoter,
)


class ATHSAutonomousRetrainer:

    _lock = threading.Lock()

    def __init__(self):

        self.store = (
            ATHSFeedbackStore()
        )

        self.learning_manager = (
            ATHSLearningManager(
                self.store
            )
        )

    def maybe_retrain(
        self,
    ) -> Dict[str, Any]:

        samples = (
            self.store
            .get_pending_learning_samples(
                self.learning_manager
                .MAX_SAMPLES_PER_CYCLE
            )
        )

        required = (
            self.learning_manager
            .RETRAIN_TRIGGER_SAMPLES
        )

        if len(samples) < required:

            return {
                "status":
                    "threshold_not_reached",
                "pending_samples":
                    len(samples),
                "required_samples":
                    required,
            }

        # Prevent two API requests from
        # retraining simultaneously.
        if not self._lock.acquire(
            blocking=False
        ):

            return {
                "status":
                    "retraining_in_progress",
                "pending_samples":
                    len(samples),
            }

        try:

            sample_ids = [
                int(sample["id"])
                for sample in samples
            ]

            # Train candidate.
            retrainer = (
                ATHSModelRetrainer(
                    learning_samples=samples,
                    include_human_feedback=True,
                )
            )

            retraining_result = (
                retrainer.retrain()
            )

            if (
                retraining_result["status"]
                != "candidate_created"
            ):

                return {
                    "status":
                        "retraining_failed",
                    "retraining":
                        retraining_result,
                }

            # Evaluate against untouched fixed test set.
            evaluator = (
                ATHSModelEvaluator()
            )

            evaluation = (
                evaluator.evaluate()
            )

            comparison = (
                evaluation["comparison"]
            )

            # -------------------------------------------------
            # PROMOTE
            # -------------------------------------------------

            if comparison["promote"]:

                promotion = (
                    ATHSModelPromoter()
                    .promote()
                )

                self.store.mark_learning_samples_processed(
                    sample_ids
                )

                return {
                    "status": "promoted",
                    "retraining":
                        retraining_result,
                    "evaluation":
                        evaluation,
                    "promotion":
                        promotion,
                    "processed_samples":
                        len(sample_ids),
                }

            # -------------------------------------------------
            # REJECT
            # -------------------------------------------------

            self.store.mark_learning_samples_processed(
                sample_ids
            )

            return {
                "status": "rejected",
                "retraining":
                    retraining_result,
                "evaluation":
                    evaluation,
                "processed_samples":
                    len(sample_ids),
            }

        finally:
            self._lock.release()