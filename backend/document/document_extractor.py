"""
ATHS Document Extractor

Extracts readable text from supported document formats so that
the existing ATHS preprocessing and threat-detection pipeline
can analyze the extracted content.

Supported formats:
    - PDF
    - DOCX
    - TXT
"""

from pathlib import Path
from typing import Dict, Any

from pypdf import PdfReader
from docx import Document


class ATHSDocumentExtractor:
    """Extract text from supported document files."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}

    def __init__(self) -> None:
        print("ATHS Document Extractor initialized.")

    def is_supported(self, file_path: str) -> bool:
        """
        Check whether the supplied file has a supported extension.
        """
        extension = Path(file_path).suffix.lower()
        return extension in self.SUPPORTED_EXTENSIONS

    def extract(self, file_path: str) -> Dict[str, Any]:
        """
        Extract text from a supported document.

        Returns:
            {
                "file_name": str,
                "file_type": str,
                "text": str,
                "page_count": int,
                "success": bool
            }
        """

        path = Path(file_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Document not found: {file_path}"
            )

        extension = path.suffix.lower()

        if extension not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {extension}. "
                f"Supported types: {sorted(self.SUPPORTED_EXTENSIONS)}"
            )

        if extension == ".pdf":
            return self._extract_pdf(path)

        if extension == ".docx":
            return self._extract_docx(path)

        if extension == ".txt":
            return self._extract_txt(path)

        raise ValueError("Unsupported document format.")

    def _extract_pdf(self, path: Path) -> Dict[str, Any]:
        """Extract text from a PDF while preserving page boundaries."""

        reader = PdfReader(str(path))

        pages = []
        successful_pages = 0

        for page_number, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""

            text = text.strip()

            if text:
                successful_pages += 1

            pages.append(
                f"\n--- PAGE {page_number} ---\n{text}"
            )

        full_text = "\n".join(pages).strip()

        return {
            "file_name": path.name,
            "file_type": "pdf",
            "text": full_text,
            "page_count": len(reader.pages),
            "extracted_pages": successful_pages,
            "pages": [
                {
                    "page_number": page_number,
                    "text": page_text,
                }
                for page_number, page_text in enumerate(
                    [p.extract_text() or "" for p in reader.pages],
                    start=1,
                )
            ],
            "success": bool(full_text),
        }

    def _extract_docx(self, path: Path) -> Dict[str, Any]:
        """Extract text from a DOCX document."""

        document = Document(str(path))

        paragraphs = []

        for paragraph in document.paragraphs:
            text = paragraph.text.strip()

            if text:
                paragraphs.append(text)

        full_text = "\n".join(paragraphs).strip()

        return {
            "file_name": path.name,
            "file_type": "docx",
            "text": full_text,
            "page_count": 1,
            "extracted_pages": 1 if full_text else 0,
            "pages": [{"page_number": 1, "text": full_text}],
            "success": bool(full_text),
        }

    def _extract_txt(self, path: Path) -> Dict[str, Any]:
        """Extract text from a plain text file."""

        text = path.read_text(
            encoding="utf-8",
            errors="replace",
        ).strip()

        return {
            "file_name": path.name,
            "file_type": "txt",
            "text": text,
            "page_count": 1,
            "extracted_pages": 1 if text else 0,
            "pages": [{"page_number": 1, "text": text}],
            "success": bool(text),
        }


def main() -> None:
    """Standalone extractor test."""

    print()
    print("ATHS DOCUMENT EXTRACTOR TEST")
    print("=" * 40)

    extractor = ATHSDocumentExtractor()

    print("Supported formats:")
    for extension in sorted(extractor.SUPPORTED_EXTENSIONS):
        print(f"  {extension}")

    print()
    print("Testing extension detection:")

    test_files = [
        "resume.pdf",
        "resume.docx",
        "notes.txt",
        "image.png",
    ]

    for file_name in test_files:
        supported = extractor.is_supported(file_name)
        print(f"  {file_name:<20} : {supported}")

    print()
    print("ATHS DOCUMENT EXTRACTOR TEST COMPLETE")


if __name__ == "__main__":
    main()