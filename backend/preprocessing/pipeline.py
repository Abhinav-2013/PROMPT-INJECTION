from .text_normalizer import (
    normalize_text,
    security_lowercase,
    detect_zero_width
)

from .obfuscation import extract_obfuscation_features
from .nlp_preprocessor import extract_nlp_features
from .security_features import extract_security_features
from .semantic_encoder import SemanticEncoder
from .context_preprocessor import ContextPreprocessor, SecurityContext


class PreprocessingPipeline:

    def __init__(
        self,
        enable_embeddings: bool = False,
        embedding_model: str = (
            "sentence-transformers/all-MiniLM-L6-v2"
        )
    ):
        self.enable_embeddings = enable_embeddings

        self.encoder = None

        if enable_embeddings:
            self.encoder = SemanticEncoder(
                embedding_model
            )

    def process_text(self, text: str) -> dict:
        if text is None:
            text = ""

        if not isinstance(text, str):
            text = str(text)

        original = text

        # ---------------------------------------------------------
        # 1. NORMALIZATION
        # ---------------------------------------------------------

        zero_width = detect_zero_width(original)

        normalized = normalize_text(original)

        model_text = security_lowercase(
            original
        )

        # ---------------------------------------------------------
        # 2. NLP PREPROCESSING
        # ---------------------------------------------------------

        nlp_features = extract_nlp_features(
            normalized
        )

        # ---------------------------------------------------------
        # 3. OBFUSCATION DETECTION
        # ---------------------------------------------------------

        obfuscation_features = extract_obfuscation_features(
            original,
            zero_width
        )

        # ---------------------------------------------------------
        # 4. SECURITY FEATURES
        # ---------------------------------------------------------

        security_features = extract_security_features(
            model_text
        )

        # ---------------------------------------------------------
        # 5. SEMANTIC REPRESENTATION
        # ---------------------------------------------------------

        embedding = None

        if self.enable_embeddings:
            try:
                embedding = self.encoder.encode(
                    model_text
                )
            except Exception as error:
                embedding = {
                    "error": str(error)
                }

        # ---------------------------------------------------------
        # FINAL PREPROCESSED OBJECT
        # ---------------------------------------------------------

        return {
            "original_text": original,

            "normalized_text": normalized,

            "model_text": model_text,

            "nlp": nlp_features,

            "obfuscation": obfuscation_features,

            "security_features": security_features,

            "semantic_embedding": embedding
        }

    def process_context(
        self,
        context: SecurityContext
    ) -> dict:

        context_processor = ContextPreprocessor(
            self.process_text
        )

        return context_processor.process(
            context
        )
