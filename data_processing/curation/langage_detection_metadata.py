from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageDetectionMetadata:
    total_pairs: int
    valid_pairs: int
    invalid_pairs: int
    invalid_source_language: int
    invalid_target_language: int
    low_confidence_source: int
    low_confidence_target: int
