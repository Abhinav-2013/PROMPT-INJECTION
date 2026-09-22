"""
Prompt Injection Detection System API

FastAPI backend for:

    - Prompt analysis
    - Threat detection
    - Human feedback
    - Controlled autonomous learning
    - SHAP explainability
    - Document scanning
    - System health

Internal project name:
    ATHS

Visible API/service name:
    Prompt Injection Detection System
"""

from __future__ import annotations

from typing import Any, Dict, cast, List
from pathlib import Path
import os
import tempfile
import threading
import traceback

import numpy as np

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    HTTPException,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from data.feedback_store import ATHSFeedbackStore
from detection.engine import ATHSThreatDetectionEngine
from detection.feature_loader import ML_FEATURE_COLUMNS
from preprocessing.pipeline import PreprocessingPipeline
from document.document_scanner import ATHSDocumentScanner
from xai.explainer import ATHSXAIExplainer
from xai.unified_explainer import ATHSUnifiedExplainer
from retraining.retrain_model import (
    ATHSModelRetrainer,
)
from retraining.model_evaluator import ATHSModelEvaluator


# =========================================================
# CONFIGURATION
# =========================================================

MAX_DOCUMENT_SIZE = 20 * 1024 * 1024

# Self-learning is deliberately conservative.

AUTO_LEARNING_ENABLED = True

# Minimum number of eligible autonomous samples required
# before a background candidate retraining cycle starts.
AUTO_RETRAIN_MIN_SAMPLES = 50

# Do not allow unlimited automatic retraining.
AUTO_RETRAIN_MAX_SAMPLES_PER_CYCLE = 250

# Confidence requirements.
STRONG_MALICIOUS_ML = 0.90
STRONG_BENIGN_ML = 0.10

STRONG_RULE_SCORE = 0.70
STRONG_SEMANTIC_SCORE = 0.85

REVIEW_THRESHOLD = 0.50


# =========================================================
# FASTAPI APPLICATION
# =========================================================

app = FastAPI(
    title="Prompt Injection Detection System",
    description=(
        "API for detecting prompt injection threats using "
        "machine learning, rules, semantic similarity, "
        "threat fusion, hypothesis generation, and "
        "controlled autonomous learning."
    ),
    version="1.0.0",
)


# =========================================================
# CORS
# =========================================================

configured_origins = os.getenv("CORS_ALLOWED_ORIGINS", "")
allowed_origins = [
    origin.strip().rstrip("/")
    for origin in configured_origins.split(",")
    if origin.strip()
]

for default_origin in [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://frontend-production-400c.up.railway.app",
]:
    if default_origin not in allowed_origins:
        allowed_origins.append(default_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# REQUEST MODELS
# =========================================================

class AnalyzeRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Prompt to analyze.",
    )


class FeedbackRequest(BaseModel):
    prediction_id: int = Field(
        ...,
        description="Prediction being reviewed.",
    )

    human_label: int = Field(
        ...,
        ge=0,
        le=1,
        description="0 = benign, 1 = malicious.",
    )

    attack_category: str | None = Field(
        default=None,
        description="Optional attack category.",
    )


# =========================================================
# INITIALIZE COMPONENTS
# =========================================================

print(
    "Initializing Prompt Injection Detection System API..."
)

engine = ATHSThreatDetectionEngine()

preprocessor = PreprocessingPipeline(
    enable_embeddings=True
)

feedback_store = ATHSFeedbackStore()

document_scanner = ATHSDocumentScanner(
    chunk_size=1500
)

xai_explainer = ATHSXAIExplainer()
unified_explainer = ATHSUnifiedExplainer()

# Prevent simultaneous background retraining jobs.
_retraining_lock = threading.Lock()

# Runtime state only.
_learning_state: Dict[str, Any] = {
    "enabled": AUTO_LEARNING_ENABLED,
    "running": False,
    "scheduled": False,
    "last_status": "idle",
    "last_result": None,
    # Automatic retraining pauses after a rejected candidate until
    # genuinely new eligible samples arrive.
    "auto_retrain_blocked": False,
    "blocked_pending_count": 0,
}


print(
    "Prompt Injection Detection System API initialized."
)


# =========================================================
# SHARED PREPROCESSING
# =========================================================

def build_feature_vector(
    text: str,
) -> tuple[
    Dict[str, Any],
    np.ndarray,
    np.ndarray,
]:
    """
    Run the same preprocessing used by /analyze.

    Returns:

        processed:
            Complete preprocessing output.

        feature_vector:
            15-dimensional ML feature vector.

        embedding:
            384-dimensional semantic embedding.
    """

    processed = cast(
        Dict[str, Any],
        cast(
            Any,
            preprocessor,
        ).process_text(text),
    )

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
            "Missing ML features: "
            f"{missing_features}"
        )

    feature_vector = np.asarray(
        [
            feature_sources[column]
            for column in ML_FEATURE_COLUMNS
        ],
        dtype=np.float32,
    )

    if feature_vector.shape != (
        len(ML_FEATURE_COLUMNS),
    ):
        raise ValueError(
            "Expected "
            f"{len(ML_FEATURE_COLUMNS)} ML features, "
            f"received {feature_vector.shape}."
        )

    embedding_data = processed.get(
        "semantic_embedding"
    )

    if not isinstance(
        embedding_data,
        list,
    ):
        raise ValueError(
            "Semantic embedding was not generated."
        )

    embedding = np.asarray(
        embedding_data,
        dtype=np.float32,
    )

    if embedding.shape != (384,):
        raise ValueError(
            "Expected 384-dimensional embedding, "
            f"received {embedding.shape}."
        )

    return (
        processed,
        feature_vector,
        embedding,
    )


# =========================================================
# AUTONOMOUS LEARNING ELIGIBILITY
# =========================================================

def determine_learning_eligibility(
    result: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Decide whether a prediction is safe enough to become an
    autonomous pseudo-labeled learning sample.

    IMPORTANT:

    The system does NOT automatically learn from every
    prediction.

    Strong malicious condition:

        ML >= 0.90
        AND
        (
            rule score >= 0.70
            OR
            semantic evidence is strongly malicious
        )

    OR:

        all three detectors agree.

    Strong benign condition:

        ML <= 0.10
        AND
        rule score == 0
        AND
        no malicious semantic match.

    REVIEW predictions are excluded.
    """

    ml = result.get(
        "ml",
        {}
    )

    rules = result.get(
        "rules",
        {}
    )

    semantic = result.get(
        "semantic",
        {}
    )

    decision = str(
        result.get(
            "decision",
            "REVIEW",
        )
    ).upper()

    ml_probability = float(
        ml.get(
            "probability",
            0.5,
        )
    )

    ml_label = int(
        ml.get(
            "label",
            0,
        )
    )

    rule_score = float(
        rules.get(
            "score",
            0.0,
        )
    )

    rule_threat = bool(
        rules.get(
            "is_threat",
            False,
        )
    )

    semantic_score = float(
        semantic.get(
            "score",
            semantic.get(
                "similarity_score",
                0.0,
            ),
        )
    )

    semantic_malicious = bool(
        semantic.get(
            "is_malicious_match",
            semantic.get(
                "malicious_match",
                False,
            ),
        )
    )

    semantic_high_confidence = bool(
        semantic.get(
            "high_confidence",
            False,
        )
    )

    # -----------------------------------------------------
    # Detector votes
    # -----------------------------------------------------

    ml_threat = (
        ml_probability >= REVIEW_THRESHOLD
    )

    rules_threat = (
        rule_score >= REVIEW_THRESHOLD
        or rule_threat
    )

    semantic_threat = (
        semantic_malicious
        and semantic_score >= STRONG_SEMANTIC_SCORE
    )

    votes = int(
        ml_threat
        + rules_threat
        + semantic_threat
    )

    # -----------------------------------------------------
    # Strong malicious agreement
    # -----------------------------------------------------

    strong_malicious = (
        ml_probability
        >= STRONG_MALICIOUS_ML
        and (
            rule_score
            >= STRONG_RULE_SCORE
            or (
                semantic_malicious
                and (
                    semantic_score
                    >= STRONG_SEMANTIC_SCORE
                    or semantic_high_confidence
                )
            )
        )
    )

    unanimous_malicious = (
        ml_probability
        >= STRONG_MALICIOUS_ML
        and rule_threat
        and semantic_malicious
        and semantic_score
        >= STRONG_SEMANTIC_SCORE
    )

    # -----------------------------------------------------
    # Strong benign agreement
    # -----------------------------------------------------

    strong_benign = (
        ml_probability
        <= STRONG_BENIGN_ML
        and rule_score <= 0.0
        and not semantic_malicious
    )

    # -----------------------------------------------------
    # Eligibility
    # -----------------------------------------------------

    eligible = False
    pseudo_label: int | None = None
    reason = "Prediction does not satisfy learning criteria."

    confidence = 0.0

    if (
        strong_malicious
        or unanimous_malicious
    ):
        eligible = True
        pseudo_label = 1

        confidence = max(
            ml_probability,
            rule_score,
            semantic_score,
        )

        if unanimous_malicious:
            reason = (
                "ML, rules, and semantic detectors "
                "agree on a strong malicious signal."
            )
        else:
            reason = (
                "Strong malicious ML confidence with "
                "independent rule or semantic evidence."
            )

    elif strong_benign:
        eligible = True
        pseudo_label = 0

        confidence = 1.0 - ml_probability

        reason = (
            "Strong benign ML confidence with no "
            "rule or malicious semantic evidence."
        )

    # REVIEW predictions are never accepted simply because
    # their final score is high/low.
    if decision == "REVIEW":
        eligible = False
        pseudo_label = None
        confidence = 0.0
        reason = (
            "REVIEW prediction excluded from autonomous learning."
        )

    return {
        "eligible": eligible,
        "pseudo_label": pseudo_label,
        "confidence": float(
            confidence
        ),
        "reason": reason,
        "detector_votes": votes,
        "ml_probability": ml_probability,
        "rule_score": rule_score,
        "semantic_score": semantic_score,
        "semantic_malicious": semantic_malicious,
    }


# =========================================================
# SAVE AUTONOMOUS LEARNING SAMPLE
# =========================================================

def save_learning_candidate(
    text: str,
    result: Dict[str, Any],
    eligibility: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Persist an eligible prediction as an autonomous learning
    candidate.

    The feedback store is responsible for persistence,
    deduplication, and lifecycle management.
    """

    if not eligibility["eligible"]:
        return {
            "stored": False,
            "reason": eligibility["reason"],
        }

    pseudo_label = eligibility[
        "pseudo_label"
    ]

    if pseudo_label not in (0, 1):
        return {
            "stored": False,
            "reason": "Invalid pseudo-label.",
        }

    evidence = {
        "ml_probability": float(
            result["ml"]["probability"]
        ),
        "rule_score": float(
            result["rules"]["score"]
        ),
        "semantic_score": float(
            result["semantic"].get(
                "score",
                result["semantic"].get(
                    "similarity_score",
                    0.0,
                ),
            )
        ),
        "decision": result.get(
            "decision"
        ),
        "severity": result.get(
            "severity"
        ),
        "detector_votes": result.get(
            "detector_votes",
            {},
        ),
        "matched_rules": result[
            "rules"
        ].get(
            "matched_rules",
            [],
        ),
        "semantic_malicious": result[
            "semantic"
        ].get(
            "is_malicious_match",
            result["semantic"].get(
                "malicious_match",
                False,
            ),
        ),
    }

    try:
        learning_id = (
            feedback_store.save_learning_candidate(
                text=text,
                pseudo_label=int(
                    pseudo_label
                ),
                confidence=float(
                    eligibility["confidence"]
                ),
                source="autonomous_detection",
                evidence=evidence,
            )
        )

        return {
            "stored": True,
            "learning_id": learning_id,
            "pseudo_label": int(
                pseudo_label
            ),
            "confidence": float(
                eligibility["confidence"]
            ),
            "reason": eligibility["reason"],
        }

    except Exception as exc:
        # Autonomous learning must never make /analyze fail.
        print(
            "Autonomous learning candidate storage failed:",
            exc,
        )

        return {
            "stored": False,
            "reason": (
                "Learning candidate could not be stored."
            ),
        }


# =========================================================
# AUTONOMOUS RETRAINING
# =========================================================

def run_autonomous_retraining() -> None:
    """
    Run a candidate retraining cycle in the background.

    This function does NOT promote the model.

    It creates a candidate and then runs the existing fixed-test
    evaluator. Promotion remains governed by model_evaluator.py.
    """

    if not AUTO_LEARNING_ENABLED:
        return

    if not _retraining_lock.acquire(
        blocking=False
    ):
        print(
            "Autonomous retraining already running."
        )
        _learning_state["scheduled"] = False
        return

    try:
        _learning_state[
            "running"
        ] = True

        _learning_state[
            "last_status"
        ] = "running"

        print()
        print("=" * 60)
        print("AUTONOMOUS RETRAINING STARTED")
        print("=" * 60)

        samples = (
            feedback_store.get_pending_learning_samples(
                limit=(
                    AUTO_RETRAIN_MAX_SAMPLES_PER_CYCLE
                )
            )
        )

        if len(samples) < AUTO_RETRAIN_MIN_SAMPLES:
            print(
                "Not enough eligible learning samples."
            )

            _learning_state[
                "last_status"
            ] = "waiting_for_samples"

            return

        print(
            f"Learning samples selected: "
            f"{len(samples)}"
        )

        retrainer = ATHSModelRetrainer(
            learning_samples=samples,
            include_human_feedback=True,
        )

        result = retrainer.retrain()

        # -----------------------------------------------------
        # Evaluate candidate on the untouched fixed test set.
        # Promotion is performed only when every gate passes.
        # -----------------------------------------------------

        if result.get("status") == "candidate_created":
            evaluator = ATHSModelEvaluator()
            evaluation_result = evaluator.evaluate_and_promote()

            promotion = evaluation_result.get(
                "promotion",
                {},
            )

            result["evaluation"] = evaluation_result
            result["promotion"] = promotion

            if promotion.get("status") == "promoted":
                # Samples are considered successfully incorporated only
                # after the candidate has passed evaluation and promotion.
                try:
                    learning_ids = [
                        sample.get("id")
                        for sample in samples
                        if sample.get("id") is not None
                    ]

                    if learning_ids:
                        feedback_store.mark_learning_samples_processed(
                            learning_ids
                        )
                except Exception as exc:
                    print(
                        "Warning: could not mark learning "
                        f"samples processed: {exc}"
                    )

                _learning_state["last_status"] = "promoted"
                _learning_state["auto_retrain_blocked"] = False
                _learning_state["blocked_pending_count"] = 0
            else:
                # Rejected candidates remain pending so their samples are
                # not silently consumed by an unsuccessful learning cycle.
                # Prevent automatic retraining from repeatedly using the
                # same rejected batch. New samples must arrive first.
                _learning_state["last_status"] = "rejected"
                _learning_state["auto_retrain_blocked"] = True
                _learning_state["blocked_pending_count"] = len(samples)

        else:
            _learning_state["last_status"] = result.get(
                "status",
                "completed",
            )

        _learning_state["last_result"] = result

        print()
        print(
            "AUTONOMOUS RETRAINING COMPLETE"
        )
        print(
            f"Status: {_learning_state['last_status']}"
        )

    except Exception as exc:
        print(
            "AUTONOMOUS RETRAINING FAILED:",
            exc,
        )

        traceback.print_exc()

        _learning_state[
            "last_status"
        ] = "error"

        _learning_state[
            "last_result"
        ] = {
            "status": "error",
            "error": str(exc),
        }

    finally:
        _learning_state[
            "running"
        ] = False
        _learning_state[
            "scheduled"
        ] = False

        _retraining_lock.release()


# =========================================================
# ROOT
# =========================================================

@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "system": "Prompt Injection Detection System",
        "status": "running",
        "service": (
            "Prompt Injection Threat Detection API"
        ),
        "version": "1.0.0",
        "autonomous_learning": {
            "enabled": AUTO_LEARNING_ENABLED,
            "minimum_samples": (
                AUTO_RETRAIN_MIN_SAMPLES
            ),
        },
    }


@app.head("/")
def root_head() -> None:
    """Support Render and other health probes that use HEAD."""
    return None


# =========================================================
# HEALTH
# =========================================================

@app.get("/health")
def health() -> Dict[str, Any]:
    """
    Health endpoint used by the frontend.

    It reports the API as healthy only if the initialized
    backend components are available.
    """

    return {
        "status": "healthy",
        "service": (
            "Prompt Injection Detection System"
        ),
        "autonomous_learning": {
            "enabled": AUTO_LEARNING_ENABLED,
            "running": _learning_state[
                "running"
            ],
            "scheduled": _learning_state[
                "scheduled"
            ],
            "last_status": _learning_state[
                "last_status"
            ],
        },
    }


# =========================================================
# ANALYZE PROMPT
# =========================================================

@app.post("/analyze")
def analyze_prompt(
    request: AnalyzeRequest,
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:

    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Prompt text cannot be empty.",
        )

    try:
        # -------------------------------------------------
        # 1. Preprocess
        # -------------------------------------------------

        (
            processed,
            feature_vector,
            embedding,
        ) = build_feature_vector(
            text
        )

        # -------------------------------------------------
        # 2. Detection engine
        # -------------------------------------------------

        result = engine.analyze(
            text=text,
            feature_vector=feature_vector,
            embedding=embedding,
        )

        try:
            result["explainability"] = unified_explainer.explain(
                analysis=result,
                feature_vector=feature_vector,
            )

            hypothesis = result["explainability"].get("hypothesis")
            if isinstance(hypothesis, dict):
                result.update({
                    key: hypothesis[key]
                    for key in (
                        "hypothesis",
                        "description",
                        "confidence",
                        "recommended_action",
                        "hypotheses",
                        "evidence",
                    )
                    if key in hypothesis
                })
        except Exception as exc:
            # Explanations must never prevent a detection response.
            result["explainability"] = {
                "status": "failed",
                "errors": [{"source": "unified", "message": str(exc)}],
            }

        # -------------------------------------------------
        # 3. Store prediction
        # -------------------------------------------------

        prediction_id = (
            feedback_store.save_prediction(
                text=text,
                ml_probability=float(
                    result["ml"]["probability"]
                ),
                rule_score=float(
                    result["rules"]["score"]
                ),
                semantic_score=float(
                    result["semantic"].get(
                        "score",
                        result["semantic"].get(
                            "similarity_score",
                            0.0,
                        ),
                    )
                ),
                threat_score=float(
                    result["threat_score"]
                ),
                decision=str(
                    result["decision"]
                ),
                severity=str(
                    result["severity"]
                ),
            )
        )

        result[
            "prediction_id"
        ] = prediction_id

        # -------------------------------------------------
        # 4. Autonomous learning eligibility
        # -------------------------------------------------

        eligibility = (
            determine_learning_eligibility(
                result
            )
        )

        learning_result = (
            save_learning_candidate(
                text=text,
                result=result,
                eligibility=eligibility,
            )
        )

        # -------------------------------------------------
        # 5. Check retraining threshold
        # -------------------------------------------------

        pending_count = 0

        try:
            pending_count = (
                feedback_store.count_pending_learning_samples()
            )
        except Exception as exc:
            print(
                "Could not count pending learning samples:",
                exc,
            )

        retraining_scheduled = False

        # A rejected batch must not continuously retrigger. Unblock only
        # after the pending count increases because genuinely new samples
        # have been accepted.
        if (
            _learning_state["auto_retrain_blocked"]
            and pending_count
            > _learning_state["blocked_pending_count"]
        ):
            _learning_state["auto_retrain_blocked"] = False
            _learning_state["blocked_pending_count"] = 0
            _learning_state["last_status"] = "idle"

        # Set scheduled before queueing the task so concurrent /analyze
        # requests cannot enqueue duplicate background jobs.
        if (
            AUTO_LEARNING_ENABLED
            and pending_count >= AUTO_RETRAIN_MIN_SAMPLES
            and not _learning_state["running"]
            and not _learning_state["scheduled"]
            and not _learning_state["auto_retrain_blocked"]
        ):
            _learning_state["scheduled"] = True
            background_tasks.add_task(
                run_autonomous_retraining
            )
            retraining_scheduled = True

        # -------------------------------------------------
        # 6. Add learning information to response
        # -------------------------------------------------

        result[
            "self_learning"
        ] = {
            "enabled": AUTO_LEARNING_ENABLED,
            "eligible": eligibility[
                "eligible"
            ],
            "pseudo_label": eligibility[
                "pseudo_label"
            ],
            "confidence": eligibility[
                "confidence"
            ],
            "reason": eligibility[
                "reason"
            ],
            "candidate_stored": learning_result.get(
                "stored",
                False,
            ),
            "pending_samples": pending_count,
            "retraining_threshold": (
                AUTO_RETRAIN_MIN_SAMPLES
            ),
            "retraining_scheduled": (
                retraining_scheduled
            ),
        }

        return result

    except HTTPException:
        raise

    except Exception as exc:
        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=(
                "Prompt analysis failed: "
                f"{str(exc)}"
            ),
        )


# =========================================================
# SHAP EXPLAINABILITY
# =========================================================

@app.post("/explain")
def explain_prompt(
    request: AnalyzeRequest,
) -> Dict[str, Any]:

    text = request.text.strip()

    if not text:
        raise HTTPException(
            status_code=400,
            detail="Prompt text cannot be empty.",
        )

    try:
        (
            processed,
            feature_vector,
            embedding,
        ) = build_feature_vector(
            text
        )

        explanation = (
            xai_explainer.explain(
                feature_vector
            )
        )

        logistic_explanation = (
            unified_explainer._logistic_explanation(
                feature_vector,
                unified_explainer.logistic_regression,
            )
        )

        return {
            "status": "success",
            "prompt": text,
            "feature_count": int(
                feature_vector.shape[0]
            ),
            "embedding_dimension": int(
                embedding.shape[0]
            ),
            "xai": explanation,
            "logistic_regression": logistic_explanation,
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "SHAP explanation failed: "
                f"{str(exc)}"
            ),
        )


# =========================================================
# HUMAN FEEDBACK
# =========================================================

@app.post("/feedback")
def submit_feedback(
    request: FeedbackRequest,
) -> Dict[str, Any]:

    try:
        feedback_store.add_feedback(
            prediction_id=request.prediction_id,
            human_label=request.human_label,
            attack_category=request.attack_category,
        )

        return {
            "status": "success",
            "prediction_id": (
                request.prediction_id
            ),
            "human_label": (
                request.human_label
            ),
            "attack_category": (
                request.attack_category
            ),
            "message": (
                "Human feedback stored successfully."
            ),
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                "Failed to save feedback: "
                f"{str(exc)}"
            ),
        )


# =========================================================
# DOCUMENT SCANNING
# =========================================================

@app.post("/documents/scan")
async def scan_documents(
    files: List[UploadFile] | None = File(default=None),
    file: UploadFile | None = File(default=None),
) -> Dict[str, Any]:
    """
    Scan one or more documents.

    The endpoint accepts both:
      - `files`: the preferred multi-file field
      - `file`: the legacy single-file field

    This keeps the backend compatible with the current frontend while
    supporting multiple uploads. Every document is scanned independently.
    PDF results preserve page numbers.
    """
    upload_files: List[UploadFile] = []

    if files:
        upload_files.extend(files)

    if file is not None:
        upload_files.append(file)

    if not upload_files:
        raise HTTPException(
            status_code=400,
            detail="No files provided. Upload one or more PDF, DOCX, or TXT files.",
        )

    allowed_extensions = {".pdf", ".docx", ".txt"}
    documents: List[Dict[str, Any]] = []

    for file in upload_files:
        filename = file.filename or ""

        if not filename:
            documents.append({
                "filename": "",
                "status": "error",
                "error": "No file provided.",
            })
            continue

        extension = Path(filename).suffix.lower()

        if extension not in allowed_extensions:
            documents.append({
                "filename": filename,
                "status": "error",
                "error": (
                    "Unsupported file type. Supported formats: "
                    "PDF, DOCX, TXT."
                ),
            })
            continue

        temp_path: str | None = None

        try:
            contents = await file.read()

            if len(contents) > MAX_DOCUMENT_SIZE:
                documents.append({
                    "filename": filename,
                    "file_type": extension.lstrip("."),
                    "file_size_bytes": len(contents),
                    "status": "error",
                    "error": "File exceeds the 20 MB limit.",
                })
                continue

            with tempfile.NamedTemporaryFile(
                suffix=extension,
                delete=False,
            ) as temp_file:
                temp_file.write(contents)
                temp_path = temp_file.name

            scan_result = document_scanner.scan(temp_path)
            # The scanner uses a temporary path internally. Preserve the
            # original client filename in every user-facing response.
            scan_result["file_name"] = filename

            documents.append({
                "filename": filename,
                "file_type": extension.lstrip("."),
                "file_size_bytes": len(contents),
                "status": "success",
                "result": scan_result,
            })

        except Exception as exc:
            print(
                f"Document scanning failed for {filename}: {exc}"
            )
            traceback.print_exc()

            documents.append({
                "filename": filename,
                "file_type": extension.lstrip("."),
                "status": "error",
                "error": f"Document scanning failed: {str(exc)}",
            })

        finally:
            if temp_path is not None:
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                except OSError:
                    pass

    successful = [
        document
        for document in documents
        if document.get("status") == "success"
    ]
    failed = [
        document
        for document in documents
        if document.get("status") == "error"
    ]

    decisions = [
        document["result"]["document_decision"]
        for document in successful
        if "result" in document
    ]

    if "BLOCK" in decisions:
        overall_decision = "BLOCK"
    elif "REVIEW" in decisions:
        overall_decision = "REVIEW"
    else:
        overall_decision = "ALLOW"

    return {
        "status": "success",
        "document_count": len(upload_files),
        "successful_documents": len(successful),
        "failed_documents": len(failed),
        "overall_decision": overall_decision,
        "documents": documents,
    }


# =========================================================
# AUTONOMOUS LEARNING STATUS
# =========================================================

@app.get("/learning/status")
def learning_status() -> Dict[str, Any]:
    """
    Return the current autonomous-learning state.

    This endpoint does not trigger retraining.
    """

    try:
        pending_samples = (
            feedback_store.count_pending_learning_samples()
        )
    except Exception:
        pending_samples = 0

    try:
        total_learning_samples = (
            feedback_store.count_learning_samples()
        )
    except Exception:
        total_learning_samples = 0

    try:
        statistics = (
            feedback_store.get_statistics()
        )
    except Exception:
        statistics = {}

    return {
        "status": "success",
        "enabled": AUTO_LEARNING_ENABLED,
        "running": _learning_state[
            "running"
        ],
        "scheduled": _learning_state[
            "scheduled"
        ],
        "auto_retrain_blocked": _learning_state[
            "auto_retrain_blocked"
        ],
        "blocked_pending_count": _learning_state[
            "blocked_pending_count"
        ],
        "last_status": _learning_state[
            "last_status"
        ],
        "pending_learning_samples": (
            pending_samples
        ),
        "total_learning_samples": (
            total_learning_samples
        ),
        "retraining_threshold": (
            AUTO_RETRAIN_MIN_SAMPLES
        ),
        "last_result": _learning_state[
            "last_result"
        ],
        "statistics": statistics,
    }


# =========================================================
# MANUAL AUTONOMOUS RETRAIN TRIGGER
# =========================================================

@app.post("/learning/retrain")
def trigger_learning_retrain(
    background_tasks: BackgroundTasks,
) -> Dict[str, Any]:
    """
    Manually trigger the same controlled autonomous learning
    cycle used by the background system.

    It creates a candidate only.
    It does not bypass model evaluation/promotion gates.
    """

    if not AUTO_LEARNING_ENABLED:
        raise HTTPException(
            status_code=403,
            detail=(
                "Autonomous learning is disabled."
            ),
        )

    if _learning_state[
        "running"
    ]:
        return {
            "status": "already_running",
            "message": (
                "Autonomous retraining is already running."
            ),
        }

    try:
        pending_samples = (
            feedback_store.count_pending_learning_samples()
        )
    except Exception:
        pending_samples = 0

    if (
        pending_samples
        < AUTO_RETRAIN_MIN_SAMPLES
    ):
        return {
            "status": "insufficient_samples",
            "pending_samples": pending_samples,
            "required_samples": (
                AUTO_RETRAIN_MIN_SAMPLES
            ),
            "message": (
                "Not enough eligible learning samples."
            ),
        }

    # Manual retraining is an explicit operator action and may retry a
    # previously rejected batch.
    _learning_state["auto_retrain_blocked"] = False
    _learning_state["blocked_pending_count"] = 0

    if _learning_state["scheduled"]:
        return {
            "status": "already_scheduled",
            "pending_samples": pending_samples,
            "message": (
                "Autonomous retraining is already scheduled."
            ),
        }

    _learning_state["scheduled"] = True
    background_tasks.add_task(
        run_autonomous_retraining
    )

    return {
        "status": "scheduled",
        "pending_samples": pending_samples,
        "message": (
            "Autonomous candidate retraining scheduled."
        ),
    }


# =========================================================
# RUN DIRECTLY
# =========================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.app:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )