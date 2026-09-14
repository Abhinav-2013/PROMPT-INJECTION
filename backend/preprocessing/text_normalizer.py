import re
import unicodedata


ZERO_WIDTH_CHARS = {
    "\u200b",  # Zero Width Space
    "\u200c",  # Zero Width Non-Joiner
    "\u200d",  # Zero Width Joiner
    "\ufeff",  # Zero Width No-Break Space
}


def validate_text(text: str) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return text


def detect_zero_width(text: str) -> bool:
    return any(char in text for char in ZERO_WIDTH_CHARS)


def normalize_unicode(text: str) -> str:
    # NFKC converts compatibility forms into a consistent representation.
    return unicodedata.normalize("NFKC", text)


def remove_zero_width(text: str) -> str:
    return "".join(
        char for char in text
        if char not in ZERO_WIDTH_CHARS
    )


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def normalize_text(text: str) -> str:
    text = validate_text(text)
    text = normalize_unicode(text)
    text = remove_zero_width(text)
    text = normalize_whitespace(text)
    return text


def security_lowercase(text: str) -> str:
    return normalize_text(text).lower()
