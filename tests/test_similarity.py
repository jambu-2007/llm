"""Tests for the NumPy cosine similarity helpers."""

import numpy as np
import pytest
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

from similarity import cosine_similarity, rank


def test_cosine_similarity_matches_sklearn() -> None:
    """The manual calculation should match scikit-learn for nonzero vectors."""
    vector_a = np.array([1.0, 2.0, 3.0])
    vector_b = np.array([3.0, 2.0, 1.0])
    expected = sklearn_cosine_similarity([vector_a], [vector_b])[0, 0]
    assert cosine_similarity(vector_a, vector_b) == pytest.approx(expected)


def test_identical_nonzero_vectors_have_similarity_one() -> None:
    vector = np.array([2.0, -1.0, 4.0])
    assert cosine_similarity(vector, vector) == pytest.approx(1.0)


def test_orthogonal_vectors_have_similarity_zero() -> None:
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == 0.0


def test_zero_vectors_are_safe() -> None:
    assert cosine_similarity(np.zeros(3), np.array([1.0, 2.0, 3.0])) == 0.0
    indices, scores = rank(
        np.zeros(2), np.array([[1.0, 0.0], [0.0, 0.0]])
    )
    assert indices == [0, 1]
    assert scores == [0.0, 0.0]
