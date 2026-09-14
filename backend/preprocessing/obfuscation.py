import base64
import binascii
import re


def detect_excessive_spacing(text: str) -> bool:
    """
    Detect character-by-character or highly separated text.

    Example:
        i g n o r e p r e v i o u s
    """
    return bool(
        re.search(r"(?:[A-Za-z]\s+){4,}[A-Za-z]", text)
    )


def detect_base64_like(text: str) -> bool:
    """
    Detect tokens that are structurally Base64-like and can actually
    be decoded. This is only a signal; Base64 is not automatically malicious.
    """
    pattern = re.compile(r"^[A-Za-z0-9+/]{20,}={0,2}$")

    for token in text.split():
        if not pattern.fullmatch(token):
            continue

        try:
            decoded = base64.b64decode(token, validate=True)
            if len(decoded) >= 8:
                return True
        except (ValueError, binascii.Error):
            pass

    return False


def detect_hex_like(text: str) -> bool:
    tokens = re.findall(r"\b[0-9a-fA-F]{16,}\b", text)
    return any(len(token) % 2 == 0 for token in tokens)


def detect_url_encoding(text: str) -> bool:
    return bool(re.search(r"%[0-9a-fA-F]{2}", text))


def detect_html_entities(text: str) -> bool:
    return bool(
        re.search(r"&#(?:x[0-9a-fA-F]+|\d+);", text)
    )


def detect_repeated_delimiters(text: str) -> bool:
    return bool(re.search(r"([\-_=*#])\1{5,}", text))


def extract_obfuscation_features(
    original_text: str,
    zero_width_detected: bool
) -> dict:

    signals = {
        "zero_width_detected": int(zero_width_detected),
        "excessive_spacing": int(
            detect_excessive_spacing(original_text)
        ),
        "base64_like": int(
            detect_base64_like(original_text)
        ),
        "hex_like": int(
            detect_hex_like(original_text)
        ),
        "url_encoding": int(
            detect_url_encoding(original_text)
        ),
        "html_entities": int(
            detect_html_entities(original_text)
        ),
        "repeated_delimiters": int(
            detect_repeated_delimiters(original_text)
        ),
    }

    # Weighted signal, not a final maliciousness decision.
    weights = {
        "zero_width_detected": 0.25,
        "excessive_spacing": 0.20,
        "base64_like": 0.20,
        "hex_like": 0.10,
        "url_encoding": 0.10,
        "html_entities": 0.05,
        "repeated_delimiters": 0.10,
    }

    score = sum(
        weights[name] for name, value in signals.items()
        if value
    )

    signals["obfuscation_score"] = round(
        min(score, 1.0), 3
    )

    return signals
