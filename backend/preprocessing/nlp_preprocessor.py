import re
from collections import Counter


# Words are kept in the text. For prompt-injection detection,
# removing stopwords can destroy useful instruction semantics.
STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then",
    "is", "are", "was", "were", "to", "of", "in", "on",
    "for", "with", "as", "by", "at", "from", "this", "that",
    "it", "be", "can", "you", "your", "i", "me", "my"
}


def tokenize(text: str) -> list[str]:
    """
    Security-friendly tokenization.

    Keeps words, numbers and useful symbols separately instead of
    aggressively deleting punctuation.
    """
    return re.findall(
        r"https?://\S+|www\.\S+|[A-Za-z]+(?:['-][A-Za-z]+)*|\d+(?:\.\d+)?|[^\w\s]",
        text
    )


def word_tokens(text: str) -> list[str]:
    return [
        token.lower()
        for token in tokenize(text)
        if re.fullmatch(r"[A-Za-z]+(?:['-][A-Za-z]+)*", token)
    ]


def sentence_tokenize(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


def remove_stopwords_for_analysis(tokens: list[str]) -> list[str]:
    """
    Optional analytical representation only.

    DO NOT use this representation as the main model input because
    instruction semantics may depend on words such as 'not', 'ignore', etc.
    """
    return [
        token for token in tokens
        if token.lower() not in STOPWORDS
    ]


def get_pos_tags(text: str):
    """
    Optional spaCy POS tagging.

    This function is deliberately optional so the main pipeline can run
    without spaCy. Install spaCy and an English model if POS tags are needed.
    """
    try:
        import spacy
    except ImportError:
        return None

    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        return None

    doc = nlp(text)

    return [
        {
            "token": token.text,
            "pos": token.pos_,
            "tag": token.tag_,
            "dependency": token.dep_
        }
        for token in doc
    ]


def extract_nlp_features(text: str) -> dict:
    tokens = tokenize(text)
    words = word_tokens(text)
    sentences = sentence_tokenize(text)

    lower_words = [word.lower() for word in words]
    frequencies = Counter(lower_words)

    repeated_words = {
        word: count
        for word, count in frequencies.items()
        if count > 1
    }

    return {
        "tokens": tokens,
        "word_tokens": words,
        "sentences": sentences,
        "token_count": len(tokens),
        "word_count": len(words),
        "sentence_count": len(sentences),
        "unique_word_count": len(set(lower_words)),
        "lexical_diversity": round(
            len(set(lower_words)) / max(len(lower_words), 1),
            4
        ),
        "repeated_words": repeated_words,
        "stopword_removed_tokens": remove_stopwords_for_analysis(words),
    }
