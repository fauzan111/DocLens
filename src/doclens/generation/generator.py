import os

from doclens.generation.models import Answer, Citation
from doclens.retrieval.models import Chunk

NO_INFO_TEXT = "I don't have enough information to answer this question."


class GeminiGenerator:
    def __init__(self, model_name: str = "gemini-3.6-flash", api_key: str | None = None):
        self.model_name = model_name
        self.api_key = api_key if api_key is not None else os.environ.get("GEMINI_API_KEY")
        self._model = None

    def _ensure_loaded(self):
        if self._model is None:
            if not self.api_key:
                raise RuntimeError("GEMINI_API_KEY is not set")
            # google.generativeai is deprecated (end-of-life, no more updates); still works
            # as of gemini-3.6-flash. Follow-up: migrate to the google.genai SDK.
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)

    def generate(self, query: str, chunks: list[Chunk]) -> Answer:
        if not chunks:
            return Answer(text=NO_INFO_TEXT, citations=[], grounded=False)

        self._ensure_loaded()
        context = "\n\n".join(
            f"[Source: {chunk.doc_title}, page {chunk.page_number}]\n{chunk.text}"
            for chunk in chunks
        )
        prompt = (
            "Answer the question using ONLY the provided context. "
            "If the context does not contain the answer, say you don't know.\n\n"
            f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"
        )
        response = self._model.generate_content(prompt)
        citations = [
            Citation(doc_title=chunk.doc_title, page_number=chunk.page_number, source_url=chunk.source_url)
            for chunk in chunks
        ]
        return Answer(text=response.text.strip(), citations=citations, grounded=True)
