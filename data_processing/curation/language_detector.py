import logging
from pathlib import Path
from typing import Final, TypeVar

from tqdm import tqdm

from data_processing.curation.langage_detection_metadata import LanguageDetectionMetadata

logger = logging.getLogger(__name__)

T = TypeVar('T')


class LanguageDetector:
    _LABEL_PREFIX: Final[str] = '__label__'

    def __init__(self, model_path: Path, min_confidence: float = 0.9):
        self._min_confidence = min_confidence
        self._model = self._load_model(model_path)

    def _load_model(self, model_path: Path):
        import fasttext

        logger.info(f'Loading fastText model from {model_path}...')
        model = fasttext.load_model(str(model_path.absolute()))
        logger.info('fastText model loaded successfully')

        return model

    def validate_language_pairs(
        self, pairs: list[T], expected_source_lang: str, expected_target_lang: str
    ) -> tuple[list[T], LanguageDetectionMetadata]:
        logger.info('Validating language pairs...')
        logger.info(f'Expected source language: {expected_source_lang}')
        logger.info(f'Expected target language: {expected_target_lang}')
        logger.info(f'Minimum confidence: {self._min_confidence:.0%}')

        valid_pairs: list[T] = []
        invalid_source_lang_count = 0
        invalid_target_lang_count = 0
        low_confidence_source_count = 0
        low_confidence_target_count = 0

        for pair in tqdm(pairs, desc='Detecting languages'):
            source_language, source_confidence = self._detect_language(pair.source_text)
            target_language, target_confidence = self._detect_language(pair.target_text)

            is_source_valid = source_language == expected_source_lang and source_confidence >= self._min_confidence
            is_target_valid = target_language == expected_target_lang and target_confidence >= self._min_confidence

            if not is_source_valid:
                if source_language != expected_source_lang:
                    invalid_source_lang_count += 1
                else:
                    low_confidence_source_count += 1

            if not is_target_valid:
                if target_language != expected_target_lang:
                    invalid_target_lang_count += 1
                else:
                    low_confidence_target_count += 1

            if is_source_valid and is_target_valid:
                pair.language_valid = True
                valid_pairs.append(pair)
            else:
                pair.language_valid = False

        stats = LanguageDetectionMetadata(
            total_pairs=len(pairs),
            valid_pairs=len(valid_pairs),
            invalid_pairs=len(pairs) - len(valid_pairs),
            invalid_source_language=invalid_source_lang_count,
            invalid_target_language=invalid_target_lang_count,
            low_confidence_source=low_confidence_source_count,
            low_confidence_target=low_confidence_target_count,
        )

        self._log_stats(stats)

        return valid_pairs, stats

    def _detect_language(self, text: str) -> tuple[str, float]:
        cleaned_text = text.replace('\n', ' ')
        predictions = self._model.predict(cleaned_text, k=1)

        detected_language = predictions[0][0].replace(self._LABEL_PREFIX, '')
        confidence = float(predictions[1][0])

        return detected_language, confidence

    def _log_stats(self, stats: LanguageDetectionMetadata) -> None:
        logger.info('Language validation complete:')
        logger.info(f'  Total pairs: {stats.total_pairs:,}')
        logger.info(f'  Valid pairs: {stats.valid_pairs:,} ({stats.valid_pairs / stats.total_pairs * 100:.2f}%)')
        logger.info(f'  Invalid pairs: {stats.invalid_pairs:,} ({stats.invalid_pairs / stats.total_pairs * 100:.2f}%)')
        logger.info(f'  Invalid source language: {stats.invalid_source_language:,}')
        logger.info(f'  Invalid target language: {stats.invalid_target_language:,}')
        logger.info(f'  Low confidence source: {stats.low_confidence_source:,}')
        logger.info(f'  Low confidence target: {stats.low_confidence_target:,}')
