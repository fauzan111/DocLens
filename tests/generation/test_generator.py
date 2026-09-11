import os

import pytest

from doclens.generation.generator import GeminiGenerator
from doclens.retrieval.models import Chunk


def _chunk() -> Chunk:
    return Chunk(chunk_id="c1", doc_id="doc1", doc_title="Pump Manual", page_number=3,
                 chunk_index=0, text="The flange bolts require a torque of 45 Nm.",
                 source_url="https://example.com/manual.pdf")


def test_generator_does_not_configure_on_init():
    generator = GeminiGenerator(api_key="unused-for-this-test")
    assert generator._model is None


def test_generate_with_no_chunks_returns_ungrounded_answer_without_api_key():
    generator = GeminiGenerator(api_key=None)
    answer = generator.generate("What torque should be used?", chunks=[])

    assert answer.grounded is False
    assert answer.citations == []
    assert generator._model is None  # never attempted to load/configure


def test_generate_with_chunks_and_no_api_key_raises():
    generator = GeminiGenerator(api_key=None)
    with pytest.raises(RuntimeError):
        generator.generate("What torque should be used?", chunks=[_chunk()])


@pytest.mark.slow
def test_generate_with_real_api_key_produces_grounded_answer():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set; skipping live Gemini call")

    generator = GeminiGenerator(api_key=api_key)
    answer = generator.generate("What torque should the flange bolts be tightened to?", [_chunk()])

    assert answer.grounded is True
    assert len(answer.citations) == 1
    assert answer.citations[0].page_number == 3
    assert answer.text.strip() != ""
