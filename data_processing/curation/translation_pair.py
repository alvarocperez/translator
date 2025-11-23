from dataclasses import dataclass
from typing import Any


@dataclass
class TranslationPair:
    source_text: str
    target_text: str
    source_language: str
    target_language: str
    source_dataset: str
    quality_score: float = 0.0
    exact_hash: str = ''
    fuzzy_hash_source: str = ''
    fuzzy_hash_target: str = ''
    language_valid: bool = True

    def to_dict(self, include_metadata: bool = False) -> dict[str, Any]:
        result = {
            'source_text': self.source_text,
            'target_text': self.target_text,
            'source_lang': self.source_language,
            'target_lang': self.target_language,
        }

        if include_metadata:
            result['source_dataset'] = self.source_dataset
            result['quality_score'] = round(self.quality_score, 2)

        return result
