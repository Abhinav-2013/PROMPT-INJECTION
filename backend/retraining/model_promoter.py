from datetime import datetime
from pathlib import Path
import shutil
import tempfile

import joblib


BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

CURRENT_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "random_forest.joblib"
)

CANDIDATE_MODEL_PATH = (
    BASE_DIR
    / "models"
    / "random_forest_candidate.joblib"
)

BACKUP_DIR = (
    BASE_DIR
    / "models"
    / "backups"
)


class ATHSModelPromoter:
    """
    Promotes a candidate only after the evaluator has
    explicitly approved it.
    """

    def promote(self):

        if not CANDIDATE_MODEL_PATH.exists():
            raise FileNotFoundError(
                "Candidate model not found: "
                f"{CANDIDATE_MODEL_PATH}"
            )

        candidate = joblib.load(
            CANDIDATE_MODEL_PATH
        )

        if not hasattr(candidate, "predict"):
            raise ValueError(
                "Candidate is not a valid classifier."
            )

        if not hasattr(
            candidate,
            "predict_proba",
        ):
            raise ValueError(
                "Candidate does not support probabilities."
            )

        CURRENT_MODEL_PATH.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        BACKUP_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        timestamp = (
            datetime.now()
            .strftime("%Y%m%d_%H%M%S")
        )

        backup_path = (
            BACKUP_DIR
            / f"random_forest_{timestamp}.joblib"
        )

        if CURRENT_MODEL_PATH.exists():
            shutil.copy2(
                CURRENT_MODEL_PATH,
                backup_path,
            )

        # Write replacement beside the production
        # model and then atomically replace it.
        with tempfile.NamedTemporaryFile(
            suffix=".joblib",
            prefix="rf_promote_",
            dir=str(
                CURRENT_MODEL_PATH.parent
            ),
            delete=False,
        ) as temp_file:

            temp_path = Path(
                temp_file.name
            )

        try:
            joblib.dump(
                candidate,
                temp_path,
            )

            verified = joblib.load(
                temp_path
            )

            if not hasattr(
                verified,
                "predict",
            ):
                raise ValueError(
                    "Promoted model verification failed."
                )

            temp_path.replace(
                CURRENT_MODEL_PATH
            )

        finally:
            if temp_path.exists():
                temp_path.unlink()

        return {
            "status": "promoted",
            "current_model": str(
                CURRENT_MODEL_PATH
            ),
            "backup_model": (
                str(backup_path)
                if backup_path.exists()
                else None
            ),
        }