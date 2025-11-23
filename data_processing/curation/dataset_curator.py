import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Final

from tqdm import tqdm

from data_processing.curation.curation_config import CurationConfig
from data_processing.curation.language_detector import LanguageDetector
from data_processing.curation.quality_scorer import QualityScorer
from data_processing.curation.text_deduplicator import TextDeduplicator
from data_processing.curation.translation_pair import TranslationPair

logger = logging.getLogger(__name__)


class DatasetCurator:
    _SEPARATOR: Final[str] = '=' * 80

    def __init__(self, config: CurationConfig):
        self._config = config
        self._quality_scorer = QualityScorer()
        self._text_deduplicator = TextDeduplicator(ngram_size=config.ngram_size_for_dedup)
        self._language_detector: LanguageDetector | None = None

        if config.use_language_detection and config.fasttext_model_path is not None:
            self._language_detector = LanguageDetector(
                model_path=config.fasttext_model_path, min_confidence=config.min_language_confidence
            )

        random.seed(config.random_seed)

    def curate(self) -> None:
        logger.info(self._SEPARATOR)
        logger.info('DATASET CURATION PIPELINE')
        logger.info(self._SEPARATOR)

        self._log_configuration()
        datasets = self._load_datasets()
        datasets = self._compute_metadata(datasets)
        datasets = self._deduplicate_within_datasets(datasets)
        datasets = self._filter_by_quality(datasets)
        if self._language_detector is not None:
            datasets = self._validate_languages(datasets)

        merged_pairs = self._merge_and_deduplicate_across_datasets(datasets)

        if self._config.source_distribution is not None:
            balanced_pairs = self._balance_sources(merged_pairs)
        else:
            balanced_pairs = merged_pairs

        train_pairs, validation_pairs, test_pairs = self._create_splits(balanced_pairs)
        self._save_datasets(train_pairs, validation_pairs, test_pairs)
        self._log_final_statistics(train_pairs, validation_pairs, test_pairs)

        logger.info(self._SEPARATOR)
        logger.info('CURATION COMPLETE')
        logger.info(self._SEPARATOR)

    def _load_datasets(self) -> dict[str, list[TranslationPair]]:
        logger.info(f'\n{self._SEPARATOR}')
        logger.info('STEP 1: LOADING DATASETS')
        logger.info(self._SEPARATOR)

        datasets: dict[str, list[TranslationPair]] = {}

        for dataset_name, file_path in self._config.input_datasets.items():
            if not file_path.exists():
                logger.warning(f'Dataset file not found: {file_path}, skipping...')
                continue

            logger.info(f'Loading {dataset_name} from {file_path.name}...')
            pairs = self._load_jsonl(file_path, dataset_name)
            datasets[dataset_name] = pairs
            logger.info(f'Loaded {len(pairs):,} pairs from {dataset_name}')

        total_loaded = sum(len(pairs) for pairs in datasets.values())
        logger.info(f'\nTotal loaded: {total_loaded:,} pairs from {len(datasets)} sources')

        return datasets

    def _load_jsonl(self, file_path: Path, source_name: str) -> list[TranslationPair]:
        pairs: list[TranslationPair] = []

        with open(file_path, encoding='utf-8') as file:
            for line in tqdm(file, desc=f'Reading {source_name}'):
                try:
                    data = json.loads(line.strip())
                    pair = TranslationPair(
                        source_text=data['source_text'],
                        target_text=data['target_text'],
                        source_language=data['source_lang'],
                        target_language=data['target_lang'],
                        source_dataset=source_name,
                    )
                    pairs.append(pair)
                except (json.JSONDecodeError, KeyError):
                    continue

        return pairs

    def _compute_metadata(self, datasets: dict[str, list[TranslationPair]]) -> dict[str, list[TranslationPair]]:
        logger.info(f'\n{self._SEPARATOR}')
        logger.info('STEP 2: COMPUTING HASHES AND QUALITY SCORES')
        logger.info(self._SEPARATOR)

        for dataset_name, pairs in datasets.items():
            logger.info(f'\nProcessing {dataset_name}...')

            for pair in tqdm(pairs, desc=f'Computing metadata for {dataset_name}'):
                pair.exact_hash = self._text_deduplicator.compute_exact_hash(pair.source_text, pair.target_text)
                pair.fuzzy_hash_source = self._text_deduplicator.compute_fuzzy_hash(pair.source_text)
                pair.fuzzy_hash_target = self._text_deduplicator.compute_fuzzy_hash(pair.target_text)

                metrics = self._quality_scorer.compute_quality_score(pair.source_text, pair.target_text)
                pair.quality_score = metrics.final_score

        logger.info('\nMetadata computation complete')
        return datasets

    def _deduplicate_within_datasets(
        self, datasets: dict[str, list[TranslationPair]]
    ) -> dict[str, list[TranslationPair]]:
        logger.info(f'\n{self._SEPARATOR}')
        logger.info('STEP 3: INTRA-DATASET DEDUPLICATION')
        logger.info(self._SEPARATOR)

        deduplicated_datasets: dict[str, list[TranslationPair]] = {}

        for dataset_name, pairs in datasets.items():
            logger.info(f'\n--- {dataset_name.upper()} ---')
            pairs = self._text_deduplicator.deduplicate_exact(pairs)
            pairs = self._text_deduplicator.deduplicate_fuzzy(pairs)
            deduplicated_datasets[dataset_name] = pairs

        total_after = sum(len(pairs) for pairs in deduplicated_datasets.values())
        logger.info(f'\nTotal after intra-dataset deduplication: {total_after:,} pairs')

        return deduplicated_datasets

    def _filter_by_quality(self, datasets: dict[str, list[TranslationPair]]) -> dict[str, list[TranslationPair]]:
        logger.info(f'\n{self._SEPARATOR}')
        logger.info('STEP 4: QUALITY FILTERING')
        logger.info(self._SEPARATOR)
        logger.info(f'Minimum quality score: {self._config.min_quality_score}')

        filtered_datasets: dict[str, list[TranslationPair]] = {}

        for dataset_name, pairs in datasets.items():
            logger.info(f'\n--- {dataset_name.upper()} ---')
            initial_count = len(pairs)

            filtered_pairs = [pair for pair in pairs if pair.quality_score >= self._config.min_quality_score]

            removed_count = initial_count - len(filtered_pairs)
            logger.info(f'Removed {removed_count:,} low-quality pairs ({removed_count / initial_count * 100:.2f}%)')
            logger.info(f'Remaining: {len(filtered_pairs):,}')

            if len(filtered_pairs) > 0:
                quality_scores = [p.quality_score for p in filtered_pairs]
                logger.info(
                    f'Quality scores: min={min(quality_scores):.1f}, '
                    f'max={max(quality_scores):.1f}, '
                    f'avg={sum(quality_scores) / len(quality_scores):.1f}'
                )

            filtered_datasets[dataset_name] = filtered_pairs

        total_after = sum(len(pairs) for pairs in filtered_datasets.values())
        logger.info(f'\nTotal after quality filtering: {total_after:,} pairs')

        return filtered_datasets

    def _validate_languages(self, datasets: dict[str, list[TranslationPair]]) -> dict[str, list[TranslationPair]]:
        logger.info(f'\n{self._SEPARATOR}')
        logger.info('STEP 5: LANGUAGE VALIDATION')
        logger.info(self._SEPARATOR)

        validated_datasets: dict[str, list[TranslationPair]] = {}

        for dataset_name, pairs in datasets.items():
            logger.info(f'\n--- {dataset_name.upper()} ---')

            valid_pairs, stats = self._language_detector.validate_language_pairs(
                pairs, self._config.expected_source_language, self._config.expected_target_language
            )

            validated_datasets[dataset_name] = valid_pairs

        total_after = sum(len(pairs) for pairs in validated_datasets.values())
        logger.info(f'\nTotal after language validation: {total_after:,} pairs')

        return validated_datasets

    def _merge_and_deduplicate_across_datasets(
        self, datasets: dict[str, list[TranslationPair]]
    ) -> list[TranslationPair]:
        logger.info(f'\n{self._SEPARATOR}')
        logger.info('STEP 6: CROSS-DATASET MERGE AND DEDUPLICATION')
        logger.info(self._SEPARATOR)

        all_pairs: list[TranslationPair] = []
        for pairs in datasets.values():
            all_pairs.extend(pairs)

        logger.info(f'Total pairs before cross-dataset dedup: {len(all_pairs):,}')

        all_pairs = self._text_deduplicator.deduplicate_exact(all_pairs)
        all_pairs = self._text_deduplicator.deduplicate_fuzzy(all_pairs)

        return all_pairs

    def _balance_sources(self, pairs: list[TranslationPair]) -> list[TranslationPair]:
        logger.info(f'\n{self._SEPARATOR}')
        logger.info('STEP 7: SOURCE BALANCING')
        logger.info(self._SEPARATOR)

        pairs_by_source: dict[str, list[TranslationPair]] = defaultdict(list)
        for pair in pairs:
            pairs_by_source[pair.source_dataset].append(pair)

        balanced_pairs: list[TranslationPair] = []

        for source_name, target_ratio in self._config.source_distribution.items():
            if source_name not in pairs_by_source:
                logger.warning(f'Source {source_name} not found in datasets')
                continue

            source_pairs = pairs_by_source[source_name]

            if self._config.max_total_samples is not None:
                target_count = int(self._config.max_total_samples * target_ratio)
            else:
                target_count = len(source_pairs)

            sorted_pairs = sorted(source_pairs, key=lambda p: p.quality_score, reverse=True)
            selected_pairs = sorted_pairs[:target_count]

            balanced_pairs.extend(selected_pairs)

            logger.info(
                f'{source_name}: selected {len(selected_pairs):,} of {len(source_pairs):,} '
                f'(target: {target_count:,}, ratio: {target_ratio:.1%})'
            )

        logger.info(f'\nTotal balanced: {len(balanced_pairs):,} pairs')

        return balanced_pairs

    def _create_splits(self, pairs: list[TranslationPair]) -> tuple[list[TranslationPair], ...]:
        logger.info(f'\n{self._SEPARATOR}')
        logger.info('STEP 8: CREATING SPLITS')
        logger.info(self._SEPARATOR)
        logger.info(
            f'Split ratios - Train: {self._config.train_ratio:.1%}, '
            f'Val: {self._config.validation_ratio:.1%}, '
            f'Test: {self._config.test_ratio:.1%}'
        )

        shuffled_pairs = pairs.copy()
        random.shuffle(shuffled_pairs)

        total_count = len(shuffled_pairs)
        train_size = int(total_count * self._config.train_ratio)
        validation_size = int(total_count * self._config.validation_ratio)

        train_pairs = shuffled_pairs[:train_size]
        validation_pairs = shuffled_pairs[train_size : train_size + validation_size]
        test_pairs = shuffled_pairs[train_size + validation_size :]

        logger.info(f'Train: {len(train_pairs):,} pairs')
        logger.info(f'Validation: {len(validation_pairs):,} pairs')
        logger.info(f'Test: {len(test_pairs):,} pairs')

        return train_pairs, validation_pairs, test_pairs

    def _save_datasets(
        self,
        train_pairs: list[TranslationPair],
        validation_pairs: list[TranslationPair],
        test_pairs: list[TranslationPair],
    ) -> None:
        logger.info(f'\n{self._SEPARATOR}')
        logger.info('STEP 9: SAVING DATASETS')
        logger.info(self._SEPARATOR)

        output_dir = self._config.output_directory
        output_dir.mkdir(parents=True, exist_ok=True)

        self._save_jsonl(train_pairs, output_dir / 'train.jsonl')
        self._save_jsonl(validation_pairs, output_dir / 'validation.jsonl')
        self._save_jsonl(test_pairs, output_dir / 'test.jsonl')

        sample_size = min(1000, len(train_pairs))
        self._save_jsonl(
            train_pairs[:sample_size], output_dir / 'train_sample_with_metadata.jsonl', include_metadata=True
        )

        logger.info(f'\nAll datasets saved to {output_dir.absolute()}')

    def _save_jsonl(self, pairs: list[TranslationPair], file_path: Path, include_metadata: bool = False) -> None:
        with open(file_path, 'w', encoding='utf-8') as file:
            for pair in pairs:
                json_line = json.dumps(pair.to_dict(include_metadata), ensure_ascii=False)
                file.write(json_line + '\n')

        logger.info(f'Saved {len(pairs):,} pairs to {file_path.name}')

    def _log_configuration(self) -> None:
        logger.info('\nConfiguration:')
        logger.info(f'  Input datasets: {len(self._config.input_datasets)}')
        logger.info(f'  Output directory: {self._config.output_directory}')
        logger.info(f'  Min quality score: {self._config.min_quality_score}')
        logger.info(f'    Model: {self._config.fasttext_model_path}')
        logger.info(f'    Min confidence: {self._config.min_language_confidence:.0%}')
        logger.info(f'  Random seed: {self._config.random_seed}')

    def _log_final_statistics(
        self,
        train_pairs: list[TranslationPair],
        validation_pairs: list[TranslationPair],
        test_pairs: list[TranslationPair],
    ) -> None:
        logger.info(f'\n{self._SEPARATOR}')
        logger.info('FINAL STATISTICS')
        logger.info(self._SEPARATOR)

        all_pairs = train_pairs + validation_pairs + test_pairs

        source_counts = Counter(pair.source_dataset for pair in all_pairs)
        logger.info('\nDistribution by source:')
        for source, count in source_counts.most_common():
            percentage = count / len(all_pairs) * 100
            logger.info(f'  {source}: {count:,} ({percentage:.1f}%)')

        quality_scores = [pair.quality_score for pair in all_pairs if pair.quality_score is not None]
        if len(quality_scores) > 0:
            logger.info('\nQuality statistics:')
            logger.info(f'  Min: {min(quality_scores):.1f}')
            logger.info(f'  Max: {max(quality_scores):.1f}')
            logger.info(f'  Avg: {sum(quality_scores) / len(quality_scores):.1f}')

        logger.info('\nSplit totals:')
        logger.info(f'  Train: {len(train_pairs):,}')
        logger.info(f'  Validation: {len(validation_pairs):,}')
        logger.info(f'  Test: {len(test_pairs):,}')
        logger.info(f'  TOTAL: {len(all_pairs):,}')

        logger.info(f'\n{self._SEPARATOR}')
        logger.info('✅ CURATION COMPLETED SUCCESSFULLY')
        logger.info(self._SEPARATOR)
