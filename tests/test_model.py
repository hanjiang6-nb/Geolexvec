import numpy as np
import pytest

from geolexvec.model import (
    contextual_entity_similarity,
    fuse_scores,
    phrase_group_score,
    sentence_cosine_matrix,
)


def test_sentence_cosine_uses_normalized_sentence_vectors():
    questions = np.asarray([[2.0, 0.0], [1.0, 1.0]], dtype=np.float32)
    evidence = np.asarray([[4.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    result = sentence_cosine_matrix(questions, evidence)
    assert result[0].tolist() == pytest.approx([1.0, 0.0])
    assert result[1].tolist() == pytest.approx([2**-0.5, 2**-0.5])


def test_contextual_entity_similarity_uses_unfiltered_average_maximum():
    query_vectors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    evidence_vectors = np.asarray(
        [[1.0, 0.0], [0.0, 0.8], [0.6, 0.0]], dtype=np.float32
    )
    assert contextual_entity_similarity(query_vectors, evidence_vectors) == pytest.approx(1.0)


def test_phrase_and_three_module_fusion():
    phrase = phrase_group_score([["Li", "lithium"], ["kaolinite"]], "Li and kaolinite")
    assert phrase == 1.0
    value = fuse_scores(0.8, 0.6, phrase, [0.67, 0.22, 0.11])
    assert float(value) == pytest.approx(0.778)


def test_phrase_matching_strips_resource_whitespace():
    assert phrase_group_score([[" feldspar"]], "feldspar is present") == 1.0
