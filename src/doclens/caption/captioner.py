import os
from pathlib import Path

NO_VISUAL_CONTENT_SENTINEL = "NO_VISUAL_CONTENT"

CAPTION_PROMPT = (
    "Look at this page from a technical manual. If it contains a data table, "
    "diagram, wiring schematic, exploded parts view, or chart, describe it in "
    "detail: for tables, list the column headers and a representative sample "
    "of values; for diagrams, describe labeled parts, callouts, or "
    "connections. Be specific about numbers, labels, and units. If this page "
    "is plain text or prose with no meaningful table, diagram, or figure, "
    f"respond with exactly the single word {NO_VISUAL_CONTENT_SENTINEL} and "
    "nothing else."
)


def _parse_caption_response(text: str) -> str | None:
    stripped = text.strip()
    if stripped == NO_VISUAL_CONTENT_SENTINEL:
        return None
    return stripped


class PageCaptioner:
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

    def caption_page(self, image_path: Path) -> str | None:
        self._ensure_loaded()
        import PIL.Image

        image = PIL.Image.open(image_path)
        response = self._model.generate_content([CAPTION_PROMPT, image])
        return _parse_caption_response(response.text)
