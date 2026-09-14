"""
ATHS Semantic / Embedding Threat Detector

Uses the existing 384-dimensional embeddings generated during
the preprocessing stage.

The detector compares a new prompt embedding against the
stored dataset embeddings using cosine similarity.

Semantic threat evidence is based on:
1. Nearest-neighbor similarity.
2. The labels of the strongest similar examples.
3. Malicious-neighbor evidence rather than similarity alone.
"""

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from detection.feature_loader import ATHSFeatureLoader


PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

DEFAULT_SIMILARITY_THRESHOLD = 0.70

# Number of nearest examples considered when determining
# semantic threat evidence.
DEFAULT_TOP_K = 5

# High-confidence semantic evidence.
HIGH_CONFIDENCE_SIMILARITY = 0.85


class EmbeddingThreatDetector:
    """
    Semantic threat detector based on cosine similarity.

    Instead of relying only on the single closest example,
    the detector examines the top-K nearest examples and
    determines whether malicious examples dominate the
    semantic neighborhood.
    """

    def __init__(
        self,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
    ):
        self.loader = ATHSFeatureLoader()

        self.similarity_threshold = similarity_threshold
        self.top_k = top_k

        self.embeddings: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.texts: Optional[np.ndarray] = None

        self.normalized_embeddings: Optional[np.ndarray] = None

        self._load_embeddings()

    # ---------------------------------------------------------
    # DATA LOADING
    # ---------------------------------------------------------

    def _load_embeddings(self):
        """
        Load the existing embeddings and corresponding labels/text.
        """

        data = self.loader.load()

        self.embeddings = data["embeddings"]
        self.labels = data["labels"]
        self.texts = data["texts"]

        if self.embeddings is None:
            raise RuntimeError(
                "Embedding data could not be loaded."
            )

        if self.labels is None:
            raise RuntimeError(
                "Embedding labels could not be loaded."
            )

        if self.texts is None:
            raise RuntimeError(
                "Embedding texts could not be loaded."
            )

        if len(self.embeddings) != len(self.labels):
            raise ValueError(
                "Embedding and label counts do not match."
            )

        if len(self.embeddings) != len(self.texts):
            raise ValueError(
                "Embedding and text counts do not match."
            )

        if self.top_k < 1:
            raise ValueError(
                "top_k must be at least 1."
            )

        self.top_k = min(
            self.top_k,
            len(self.embeddings),
        )

        # Normalize once during initialization instead of
        # recalculating normalization for every query.
        self.normalized_embeddings = (
            self._normalize_embeddings(self.embeddings)
        )

        print("Semantic detector data loaded.")
        print(
            f"  Embedding shape : "
            f"{self.embeddings.shape}"
        )
        print(
            f"  Label shape     : "
            f"{self.labels.shape}"
        )
        print(
            f"  Text count      : "
            f"{len(self.texts)}"
        )
        print(
            f"  Top-K           : "
            f"{self.top_k}"
        )

    # ---------------------------------------------------------
    # NORMALIZATION
    # ---------------------------------------------------------

    @staticmethod
    def _normalize_embeddings(
        embeddings: np.ndarray,
    ) -> np.ndarray:
        """
        L2-normalize embeddings.

        Normalized dot product is equivalent to cosine similarity.
        """

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        norms = np.linalg.norm(
            embeddings,
            axis=1,
            keepdims=True,
        )

        norms = np.maximum(
            norms,
            1e-12,
        )

        return embeddings / norms

    # ---------------------------------------------------------
    # SIMILARITY
    # ---------------------------------------------------------

    def calculate_similarities(
        self,
        query_embedding: np.ndarray,
    ) -> np.ndarray:
        """
        Calculate cosine similarity between a query embedding
        and every stored dataset embedding.
        """

        if self.normalized_embeddings is None:
            raise RuntimeError(
                "Embeddings have not been loaded."
            )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(
                1,
                -1,
            )

        if query_embedding.ndim != 2:
            raise ValueError(
                "Query embedding must be a 1D or 2D array."
            )

        if query_embedding.shape[0] != 1:
            raise ValueError(
                "Exactly one query embedding is expected."
            )

        if (
            query_embedding.shape[1]
            != self.normalized_embeddings.shape[1]
        ):
            raise ValueError(
                "Embedding dimension mismatch: "
                f"expected "
                f"{self.normalized_embeddings.shape[1]}, "
                f"got {query_embedding.shape[1]}"
            )

        normalized_query = self._normalize_embeddings(
            query_embedding
        )

        similarities = np.dot(
            self.normalized_embeddings,
            normalized_query.T,
        ).flatten()

        return similarities

    # ---------------------------------------------------------
    # TOP-K NEIGHBORS
    # ---------------------------------------------------------

    def get_top_k_neighbors(
        self,
        similarities: np.ndarray,
    ) -> list[Dict[str, Any]]:
        """
        Return the top-K most semantically similar examples.
        """

        if self.labels is None or self.texts is None:
            raise RuntimeError(
                "Labels or texts have not been loaded."
            )

        k = min(
            self.top_k,
            len(similarities),
        )

        # argpartition is faster than sorting the entire
        # dataset when only the top-K examples are required.
        candidate_indices = np.argpartition(
            similarities,
            -k,
        )[-k:]

        candidate_indices = candidate_indices[
            np.argsort(
                similarities[candidate_indices]
            )[::-1]
        ]

        neighbors = []

        for index in candidate_indices:
            index = int(index)

            neighbors.append(
                {
                    "index": index,
                    "similarity": float(
                        similarities[index]
                    ),
                    "label": int(
                        self.labels[index]
                    ),
                    "text": str(
                        self.texts[index]
                    ),
                }
            )

        return neighbors

    # ---------------------------------------------------------
    # DETECTION
    # ---------------------------------------------------------

    def detect(
        self,
        query_embedding: NDArray[np.float32],
    ) -> Dict[str, Any]:
        """
        Analyze a query embedding.

        The detector examines the top-K nearest examples.

        Semantic threat evidence is produced when:
        - the strongest match is malicious and exceeds the
          similarity threshold, OR
        - multiple strong malicious neighbors support the result.
        """

        similarities = self.calculate_similarities(
            query_embedding
        )

        neighbors = self.get_top_k_neighbors(
            similarities
        )

        best_neighbor = neighbors[0]

        best_similarity = float(
            best_neighbor["similarity"]
        )

        matched_label = int(
            best_neighbor["label"]
        )

        matched_index = int(
            best_neighbor["index"]
        )

        matched_text = str(
            best_neighbor["text"]
        )

        # -----------------------------------------------------
        # MALICIOUS NEIGHBOR ANALYSIS
        # -----------------------------------------------------

        malicious_neighbors = [
            neighbor
            for neighbor in neighbors
            if neighbor["label"] == 1
        ]

        strong_malicious_neighbors = [
            neighbor
            for neighbor in malicious_neighbors
            if neighbor["similarity"]
            >= self.similarity_threshold
        ]

        malicious_neighbor_count = len(
            malicious_neighbors
        )

        strong_malicious_neighbor_count = len(
            strong_malicious_neighbors
        )

        malicious_similarity = 0.0

        if malicious_neighbors:
            malicious_similarity = max(
                neighbor["similarity"]
                for neighbor in malicious_neighbors
            )

        # -----------------------------------------------------
        # SEMANTIC THREAT DECISION
        # -----------------------------------------------------

        strongest_match_is_malicious = (
            matched_label == 1
            and best_similarity
            >= self.similarity_threshold
        )

        multiple_malicious_neighbors = (
            strong_malicious_neighbor_count >= 2
            and malicious_similarity
            >= self.similarity_threshold
        )

        is_threat = (
            strongest_match_is_malicious
            or multiple_malicious_neighbors
        )

        # -----------------------------------------------------
        # SEMANTIC THREAT SCORE
        # -----------------------------------------------------

        if is_threat:
            semantic_threat_score = malicious_similarity
        else:
            semantic_threat_score = 0.0

        # -----------------------------------------------------
        # HIGH-CONFIDENCE SIGNAL
        # -----------------------------------------------------

        high_confidence = (
            is_threat
            and malicious_similarity
            >= HIGH_CONFIDENCE_SIMILARITY
        )

        return {
            "similarity_score": round(
                best_similarity,
                4,
            ),
            "similarity_threshold": (
                self.similarity_threshold
            ),
            "is_threat": is_threat,

            "matched_label": matched_label,
            "matched_index": matched_index,
            "matched_text": matched_text,

            "top_k": self.top_k,
            "top_neighbors": neighbors,

            "malicious_neighbor_count": (
                malicious_neighbor_count
            ),
            "strong_malicious_neighbor_count": (
                strong_malicious_neighbor_count
            ),
            "malicious_similarity": round(
                malicious_similarity,
                4,
            ),

            "semantic_threat_score": round(
                semantic_threat_score,
                4,
            ),

            "high_confidence": high_confidence,
        }


# ---------------------------------------------------------
# STANDALONE TEST
# ---------------------------------------------------------

def main():
    """
    Standalone semantic detector test.

    Tests:
    1. A benign stored embedding.
    2. A malicious stored embedding.
    3. Top-K neighborhood information.
    """

    print("=" * 70)
    print("ATHS EMBEDDING THREAT DETECTOR TEST")
    print("=" * 70)

    detector = EmbeddingThreatDetector(
        similarity_threshold=DEFAULT_SIMILARITY_THRESHOLD,
        top_k=DEFAULT_TOP_K,
    )

    # ---------------------------------------------------------
    # TEST 1: BENIGN SELF-MATCH
    # ---------------------------------------------------------

    query_index = 0

    query_embedding = detector.embeddings[
        query_index
    ]

    result = detector.detect(
        query_embedding
    )

    print("\n" + "=" * 70)
    print("BENIGN SELF-MATCH TEST")
    print("=" * 70)

    print(
        f"Query index       : "
        f"{query_index}"
    )

    print(
        f"Query label       : "
        f"{detector.labels[query_index]}"
    )

    print(
        f"Similarity score  : "
        f"{result['similarity_score']:.4f}"
    )

    print(
        f"Threshold         : "
        f"{result['similarity_threshold']:.4f}"
    )

    print(
        f"Matched label     : "
        f"{result['matched_label']}"
    )

    print(
        f"Malicious neighbors: "
        f"{result['malicious_neighbor_count']}"
    )

    print(
        f"Strong malicious  : "
        f"{result['strong_malicious_neighbor_count']}"
    )

    print(
        f"Semantic score    : "
        f"{result['semantic_threat_score']:.4f}"
    )

    print(
        f"Is threat         : "
        f"{result['is_threat']}"
    )

    print(
        f"High confidence   : "
        f"{result['high_confidence']}"
    )

    print("\nTop neighbors:")

    for neighbor in result["top_neighbors"]:
        print(
            f"  [{neighbor['index']}] "
            f"similarity={neighbor['similarity']:.4f} "
            f"label={neighbor['label']}"
        )

    # ---------------------------------------------------------
    # TEST 2: MALICIOUS SELF-MATCH
    # ---------------------------------------------------------

    malicious_indices = np.where(
        detector.labels == 1
    )[0]

    if len(malicious_indices) > 0:

        malicious_index = int(
            malicious_indices[0]
        )

        malicious_query = detector.embeddings[
            malicious_index
        ]

        malicious_result = detector.detect(
            malicious_query
        )

        print("\n" + "=" * 70)
        print("MALICIOUS SELF-MATCH TEST")
        print("=" * 70)

        print(
            f"Query index       : "
            f"{malicious_index}"
        )

        print(
            f"Query label       : "
            f"{detector.labels[malicious_index]}"
        )

        print(
            f"Similarity score  : "
            f"{malicious_result['similarity_score']:.4f}"
        )

        print(
            f"Matched label     : "
            f"{malicious_result['matched_label']}"
        )

        print(
            f"Malicious neighbors: "
            f"{malicious_result['malicious_neighbor_count']}"
        )

        print(
            f"Strong malicious  : "
            f"{malicious_result['strong_malicious_neighbor_count']}"
        )

        print(
            f"Semantic score    : "
            f"{malicious_result['semantic_threat_score']:.4f}"
        )

        print(
            f"Is threat         : "
            f"{malicious_result['is_threat']}"
        )

        print(
            f"High confidence   : "
            f"{malicious_result['high_confidence']}"
        )

        print("\nMatched text:")
        print(
            f"  {malicious_result['matched_text']}"
        )

        print("\nTop neighbors:")

        for neighbor in malicious_result[
            "top_neighbors"
        ]:
            print(
                f"  [{neighbor['index']}] "
                f"similarity={neighbor['similarity']:.4f} "
                f"label={neighbor['label']}"
            )

    print("\n" + "=" * 70)
    print("ATHS EMBEDDING DETECTOR TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()