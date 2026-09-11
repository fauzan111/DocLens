import pytest

from doclens.embed.text_embedder import TextEmbedder


def test_text_embedder_does_not_load_model_on_init():
    embedder = TextEmbedder()
    assert embedder._model is None


@pytest.mark.slow
def test_embed_query_and_passages_are_semantically_comparable():
    import numpy as np

    embedder = TextEmbedder()
    query_vector = np.array(embedder.embed_query("torque specification for flange bolts"))
    relevant = np.array(
        embedder.embed_passages(["The flange bolts require 45 Nm of torque."])[0]
    )
    irrelevant = np.array(
        embedder.embed_passages(["The office is closed on public holidays."])[0]
    )

    relevant_similarity = float(np.dot(query_vector, relevant))
    irrelevant_similarity = float(np.dot(query_vector, irrelevant))

    assert relevant_similarity > irrelevant_similarity
