import json
import logging
from pathlib import Path
from typing import Any

from tqdm import tqdm

from data_processing.line_counter import LineCounter
from data_processing.text_normalizer import TextNormalizer
from data_processing.text_pair_validator import TextPairValidator

logger = logging.getLogger(__name__)


class ParallelCorpusProcessor:
    def __init__(
            self,
            source_file_path: Path,
            target_file_path: Path,
            output_file_path: Path,
            source_language: str,
            target_language: str,
            max_samples: int | None = None,
            validator_config: dict[str, Any] | None = None
    ) -> None:
        self._source_file_path = source_file_path
        self._target_file_path = target_file_path
        self._output_file_path = output_file_path
        self._source_language = source_language
        self._target_language = target_language
        self._max_samples = max_samples

        self._normalizer = TextNormalizer()
        self._line_counter = LineCounter()

        validator_params = validator_config or {}
        self._validator = TextPairValidator(**validator_params)

        self._processed_line_count = 0
        self._valid_pair_count = 0
        self._invalid_pair_count = 0

    def process(self) -> None:
        logger.info("=" * 80)
        logger.info("Starting parallel corpus processing")
        logger.info(f"Source file: {self._source_file_path}")
        logger.info(f"Target file: {self._target_file_path}")
        logger.info(f"Output file: {self._output_file_path}")
        logger.info(f"Language pair: {self._source_language} -> {self._target_language}")

        if self._max_samples is not None:
            logger.info(f"Sample limit: {self._max_samples:,}")

        logger.info("=" * 80)

        self._validate_input_files()
        self._prepare_output_directory()

        total_lines = self._line_counter.count_lines(self._source_file_path)

        self._process_parallel_files(total_lines)
        self._log_processing_summary()

    def _validate_input_files(self) -> None:
        if not self._source_file_path.exists():
            error_message = f"Source file not found: {self._source_file_path}"
            logger.error(error_message)
            raise FileNotFoundError(error_message)

        if not self._target_file_path.exists():
            error_message = f"Target file not found: {self._target_file_path}"
            logger.error(error_message)
            raise FileNotFoundError(error_message)

        logger.info("Input files validated successfully")

    def _prepare_output_directory(self) -> None:
        self._output_file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory prepared: {self._output_file_path.parent}")

    def _process_parallel_files(self, total_lines: int) -> None:
        logger.info("Starting line-by-line processing")

        with open(self._source_file_path, 'r', encoding='utf-8', buffering=8192 * 1024) as source_file, \
                open(self._target_file_path, 'r', encoding='utf-8', buffering=8192 * 1024) as target_file, \
                open(self._output_file_path, 'w', encoding='utf-8', buffering=8192 * 1024) as output_file:

            for source_line, target_line in tqdm(
                    zip(source_file, target_file),
                    total=total_lines,
                    desc="Processing pairs",
                    unit="lines"
            ):
                self._processed_line_count += 1

                normalized_source = self._normalizer.normalize(source_line)
                normalized_target = self._normalizer.normalize(target_line)

                if not self._validator.is_valid_pair(normalized_source, normalized_target):
                    self._invalid_pair_count += 1
                    continue

                translation_record = self._create_translation_record(
                    normalized_source,
                    normalized_target
                )

                self._write_record(output_file, translation_record)
                self._valid_pair_count += 1

                if self._has_reached_sample_limit():
                    logger.info(f"Reached sample limit of {self._max_samples:,} valid pairs")
                    break

    def _create_translation_record(
            self,
            source_text: str,
            target_text: str
    ) -> dict[str, str]:
        return {
            "source_text": source_text,
            "target_text": target_text,
            "source_lang": self._source_language,
            "target_lang": self._target_language
        }

    def _write_record(self, output_file: Any, record: dict[str, str]) -> None:
        json_line = json.dumps(record, ensure_ascii=False) + "\n"
        output_file.write(json_line)

    def _has_reached_sample_limit(self) -> bool:
        if self._max_samples is None:
            return False

        return self._valid_pair_count >= self._max_samples

    def _log_processing_summary(self) -> None:
        validity_rate = (
                    self._valid_pair_count / self._processed_line_count * 100) if self._processed_line_count > 0 else 0

        logger.info("=" * 80)
        logger.info("Processing Summary:")
        logger.info(f"  Lines processed: {self._processed_line_count:,}")
        logger.info(f"  Valid pairs: {self._valid_pair_count:,}")
        logger.info(f"  Invalid pairs: {self._invalid_pair_count:,}")
        logger.info(f"  Validity rate: {validity_rate:.2f}%")
        logger.info(f"  Dataset saved to: {self._output_file_path}")
        logger.info("=" * 80)
