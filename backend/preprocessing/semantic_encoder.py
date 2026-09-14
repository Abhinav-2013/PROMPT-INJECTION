class SemanticEncoder:
    """
    Optional semantic representation using Sentence Transformers.

    Model:
        all-MiniLM-L6-v2

    It is loaded lazily so the rest of the preprocessing pipeline can run
    without downloading a model.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    ):
        self.model_name = model_name
        self.model = None

    def load(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(self.model_name)

    def encode(self, text: str) -> list[float]:
        self.load()

        embedding = self.model.encode(
            text,
            normalize_embeddings=True
        )

        return embedding.tolist()

    def similarity(self, text_a: str, text_b: str) -> float:
        self.load()

        embeddings = self.model.encode(
            [text_a, text_b],
            normalize_embeddings=True
        )

        return float(embeddings[0] @ embeddings[1])


def generate_embedding(
    text: str,
    enabled: bool = False
):
    if not enabled:
        return None

    encoder = SemanticEncoder()
    return encoder.encode(text)
