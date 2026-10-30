"""Small NumPy implementations of cosine similarity and result ranking."""

import numpy as np


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """Calculate cosine similarity, returning zero if either vector is zero."""
    a = np.asarray(vector_a, dtype=float).reshape(-1)
    b = np.asarray(vector_b, dtype=float).reshape(-1)
    if a.shape != b.shape:
        raise ValueError("Both vectors must have the same number of elements.")

    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    return float(np.dot(a, b) / denominator)


def rank(
    query_vec: np.ndarray, corpus_vecs: np.ndarray
) -> tuple[list[int], list[float]]:
    """Return corpus indices and scores sorted from most to least similar."""
    query = np.asarray(query_vec, dtype=float).reshape(-1)
    corpus = np.asarray(corpus_vecs, dtype=float)
    if corpus.ndim != 2 or corpus.shape[1] != query.size:
        raise ValueError("Corpus vectors must be a 2D array matching the query size.")
    if corpus.shape[0] == 0:
        return [], []

    denominators = np.linalg.norm(corpus, axis=1) * np.linalg.norm(query)
    scores = np.zeros(corpus.shape[0], dtype=float)
    nonzero = denominators != 0
    scores[nonzero] = (corpus[nonzero] @ query) / denominators[nonzero]

    # Stable sorting keeps the original order when scores are tied.
    indices = np.argsort(-scores, kind="stable")
    return indices.tolist(), scores[indices].tolist()
