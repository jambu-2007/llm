"""Sentence-transformer model loading and text embedding helpers."""

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAMES = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "paraphrase-multilingual-MiniLM-L12-v2": (
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    ),
}


@lru_cache(maxsize=len(MODEL_NAMES))
def _load_model(model_name: str) -> SentenceTransformer:
    """Load a supported model once and reuse it for later calls."""
    if model_name not in MODEL_NAMES:
        choices = ", ".join(MODEL_NAMES)
        raise ValueError(f"Unknown model '{model_name}'. Choose one of: {choices}")

    return SentenceTransformer(MODEL_NAMES[model_name], device="cpu")


def embed(texts: list[str] | tuple[str, ...], model_name: str) -> np.ndarray:
    """Return one sentence embedding per input text as a NumPy array."""
    model = _load_model(model_name)
    if not texts:
        return np.empty((0, model.get_sentence_embedding_dimension()), dtype=np.float32)

    vectors = model.encode(
        list(texts),
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)
