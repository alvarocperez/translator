import unicodedata
from typing import Any


class TextNormalizer:
    _ZERO_WIDTH_SPACE = "\u200b"

    def normalize(self, text: Any) -> str:
        if not isinstance(text, str):
            return ""

        normalized_text = unicodedata.normalize("NFKC", text)
        cleaned_text = normalized_text.replace(self._ZERO_WIDTH_SPACE, "")

        return cleaned_text.strip()
