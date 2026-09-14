"""
ATHS Document Scanner

Scans PDF, DOCX, and TXT documents for prompt-injection threats.

Pipeline:

    Document
       ↓
    Text Extraction
       ↓
    Chunking
       ↓
    ATHS Preprocessing
       ↓
    ATHS Threat Detection Engine
       ↓
    Per-Chunk Results
       ↓
    Overall Document Decision
"""

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from detection.engine import ATHSThreatDetectionEngine
from detection.feature_loader import ML_FEATURE_COLUMNS
from preprocessing.pipeline import PreprocessingPipeline

from document.document_extractor import ATHSDocumentExtractor


class ATHSDocumentScanner:
    """Scan documents using the existing ATHS detection engine."""

    def __init__(self, chunk_size: int = 1500) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0.")

        print("Initializing ATHS Document Scanner...")

        self.chunk_size = chunk_size

        self.extractor = ATHSDocumentExtractor()

        self.pipeline = PreprocessingPipeline(
            enable_embeddings=True
        )

        self.engine = ATHSThreatDetectionEngine()

        print("ATHS Document Scanner initialized.")

    # =============================================================
    # TEXT CHUNKING
    # =============================================================

    def _split_text(self, text: str) -> List[str]:
        """
        Split extracted document text into chunks.

        The scanner uses character-based chunks so that large
        documents do not have to be processed as one huge prompt.
        """

        text = text.strip()

        if not text:
            return []

        chunks = []

        start = 0

        while start < len(text):
            end = start + self.chunk_size

            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            start = end

        return chunks

    # =============================================================
    # FEATURE VECTOR
    # =============================================================

    def _build_feature_vector(
        self,
        processed: Dict[str, Any],
    ) -> np.ndarray:
        """
        Build the 15-dimensional feature vector required
        by ATHS Threat Detection Engine.
        """

        feature_sources = {
            **processed["nlp"],
            **processed["obfuscation"],
            **processed["security_features"],
        }

        feature_vector = np.asarray(
            [
                feature_sources[column]
                for column in ML_FEATURE_COLUMNS
            ],
            dtype=np.float32,
        )

        return feature_vector

    # =============================================================
    # EMBEDDING
    # =============================================================

    def _get_embedding(
        self,
        processed: Dict[str, Any],
    ) -> np.ndarray:
        """
        Retrieve and validate the semantic embedding.
        """

        embedding_data = processed.get(
            "semantic_embedding"
        )

        if not isinstance(embedding_data, list):
            raise RuntimeError(
                "Semantic embedding was not generated. "
                "Make sure sentence-transformers is installed "
                "and the embedding model is available."
            )

        embedding = np.asarray(
            embedding_data,
            dtype=np.float32,
        )

        if embedding.ndim != 1:
            raise ValueError(
                "Semantic embedding must be 1-dimensional."
            )

        if len(embedding) != 384:
            raise ValueError(
                "Semantic embedding dimension mismatch. "
                f"Expected 384, got {len(embedding)}."
            )

        return embedding

    # =============================================================
    # CHUNK ANALYSIS
    # =============================================================

    def _analyze_chunk(
        self,
        chunk: str,
        chunk_index: int,
    ) -> Dict[str, Any]:
        """
        Run one document chunk through the complete ATHS pipeline.
        """

        processed = self.pipeline.process_text(
            chunk
        )

        feature_vector = self._build_feature_vector(
            processed
        )

        embedding = self._get_embedding(
            processed
        )

        result = self.engine.analyze(
            text=chunk,
            feature_vector=feature_vector,
            embedding=embedding,
        )

        return {
            "chunk_index": chunk_index,
            "text": chunk,
            "decision": result["decision"],
            "severity": result["severity"],
            "threat_score": result["threat_score"],
            "ml_probability": result["ml"]["probability"],
            "rule_score": result["rules"]["score"],
            "semantic_score": result["semantic"]["score"],
            "categories": result["rules"]["categories"],
            "matched_rules": result["rules"]["matched_rules"],
            "detector_votes": result["detector_votes"],
            "full_result": result,
        }

    # =============================================================
    # OVERALL DOCUMENT DECISION
    # =============================================================

    def _get_document_decision(
        self,
        results: List[Dict[str, Any]],
    ) -> str:
        """
        Determine the overall document decision.

        Priority:

            BLOCK
              ↓
            REVIEW
              ↓
            ALLOW
        """

        if not results:
            return "ALLOW"

        decisions = {
            result["decision"]
            for result in results
        }

        if "BLOCK" in decisions:
            return "BLOCK"

        if "REVIEW" in decisions:
            return "REVIEW"

        return "ALLOW"

    def _get_document_severity(
        self,
        results: List[Dict[str, Any]],
    ) -> str:
        """
        Determine the highest severity found in the document.
        """

        severity_rank = {
            "LOW": 1,
            "MEDIUM": 2,
            "HIGH": 3,
        }

        highest = "LOW"

        for result in results:
            severity = result["severity"]

            if severity_rank.get(
                severity,
                1,
            ) > severity_rank[highest]:
                highest = severity

        return highest

    # =============================================================
    # COMPLETE DOCUMENT SCAN
    # =============================================================

    def scan(
        self,
        file_path: str,
    ) -> Dict[str, Any]:
        """
        Scan a complete document while preserving page numbers.

        Each page is chunked independently, so every threat result can
        be traced back to the page where it was found.
        """
        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Document not found: {file_path}"
            )

        extracted = self.extractor.extract(str(path))

        if not extracted["success"]:
            return {
                "file_name": path.name,
                "file_type": extracted["file_type"],
                "success": False,
                "page_count": extracted.get("page_count", 1),
                "extracted_pages": extracted.get("extracted_pages", 0),
                "chunk_count": 0,
                "document_decision": "REVIEW",
                "document_severity": "MEDIUM",
                "threat_chunks": 0,
                "detected_pages": [],
                "results": [],
                "error": (
                    "No readable text could be extracted "
                    "from the document."
                ),
            }

        # The extractor now provides page-aware content. Keep a safe
        # fallback for older extractor responses.
        pages = extracted.get("pages")
        if not isinstance(pages, list) or not pages:
            pages = [{
                "page_number": 1,
                "text": extracted.get("text", ""),
            }]

        results: List[Dict[str, Any]] = []

        for page in pages:
            page_number = int(page.get("page_number", 1))
            page_text = str(page.get("text", "") or "").strip()

            if not page_text:
                continue

            chunks = self._split_text(page_text)

            for local_index, chunk in enumerate(chunks, start=1):
                result = self._analyze_chunk(
                    chunk=chunk,
                    chunk_index=len(results) + 1,
                )
                result["page_number"] = page_number
                result["page_chunk_index"] = local_index
                results.append(result)

        document_decision = self._get_document_decision(results)
        document_severity = self._get_document_severity(results)

        threat_results = [
            result
            for result in results
            if result["decision"] in {"REVIEW", "BLOCK"}
        ]

        detected_pages = sorted({
            int(result["page_number"])
            for result in results
            if result["decision"] == "BLOCK"
        })

        review_pages = sorted({
            int(result["page_number"])
            for result in results
            if result["decision"] == "REVIEW"
        })

        return {
            "file_name": path.name,
            "file_type": extracted["file_type"],
            "success": True,
            "page_count": extracted.get("page_count", len(pages)),
            "extracted_pages": extracted.get(
                "extracted_pages",
                len([p for p in pages if str(p.get("text", "")).strip()]),
            ),
            "character_count": len(extracted.get("text", "")),
            "chunk_count": len(results),
            "document_decision": document_decision,
            "document_severity": document_severity,
            "threat_chunks": len(threat_results),
            "detected_pages": detected_pages,
            "review_pages": review_pages,
            "results": results,
        }


# =============================================================
# STANDALONE TEST
# =============================================================

def main() -> None:
    """
    Standalone scanner test.

    This test only verifies initialization and text chunking.
    A real document path can be supplied later.
    """

    print()
    print("=" * 70)
    print("ATHS DOCUMENT SCANNER TEST")
    print("=" * 70)

    scanner = ATHSDocumentScanner(
        chunk_size=1500
    )

    test_text = (
        "This is a normal resume section. "
        "Python, Java, SQL, machine learning, "
        "and data analysis."
    )

    chunks = scanner._split_text(
        test_text
    )

    print()
    print("CHUNKING TEST")
    print(f"Input characters : {len(test_text)}")
    print(f"Chunks created   : {len(chunks)}")

    for index, chunk in enumerate(
        chunks,
        start=1,
    ):
        print()
        print(f"CHUNK {index}")
        print("-" * 40)
        print(chunk)

    print()
    print("=" * 70)
    print("ATHS DOCUMENT SCANNER TEST COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()