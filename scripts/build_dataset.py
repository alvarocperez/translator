import logging
from pathlib import Path
from typing import Final, NoReturn

from curation.curation_config import CurationConfig
from curation.dataset_curator import DatasetCurator

from data_processing.dataset_registry import DatasetRegistry
from data_processing.parallel_corpus_processor import ParallelCorpusProcessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


_CURATION_OUTPUT_DIR: Final[Path] = Path('dataset/curated')
_CURATION_MIN_QUALITY_SCORE: Final[float] = 70.0
_CURATION_MAX_TOTAL_SAMPLES: Final[int | None] = 5_000_000
_CURATION_USE_LANGUAGE_DETECTION: Final[bool] = True
_CURATION_FASTTEXT_MODEL_PATH: Final[Path] = Path('../models/lid.176.ftz')
_CURATION_SOURCE_DISTRIBUTION: Final[dict[str, float]] = {
    'ccaligned': 0.40,
    'dgt': 0.30,
    'eubookshop': 0.30,
}


def _process_dataset(dataset_name: str) -> bool:
    try:
        dataset_config = DatasetRegistry.get_dataset_config(dataset_name)

        logger.info(f"\n{'=' * 80}")
        logger.info(f'Processing dataset: {dataset_config.name}')
        logger.info(f"{'=' * 80}\n")

        processor = ParallelCorpusProcessor(
            source_file_path=dataset_config.get_source_file_path(),
            target_file_path=dataset_config.get_target_file_path(),
            output_file_path=dataset_config.get_output_file_path(),
            source_language=dataset_config.source_language,
            target_language=dataset_config.target_language,
            max_samples=dataset_config.max_samples,
            validator_config=dataset_config.validator_config,
        )

        processor.process()

        logger.info(f'✅ {dataset_config.name} processed successfully!\n')
        return True

    except Exception as exception:
        logger.error(f'❌ Failed to process {dataset_name}: {exception}\n')
        return False


def _curate_datasets() -> bool:
    logger.info(f"\n{'=' * 80}")
    logger.info('STARTING DATASET CURATION')
    logger.info(f"{'=' * 80}\n")

    input_datasets: dict[str, Path] = {}
    for dataset_name in DatasetRegistry.get_all_dataset_names():
        config = DatasetRegistry.get_dataset_config(dataset_name)
        output_path = config.get_output_file_path()

        if output_path.exists() is True:
            input_datasets[dataset_name] = output_path
        else:
            logger.warning(f'Processed dataset not found: {output_path}')

    if len(input_datasets) == 0:
        logger.error('No processed datasets found. Cannot proceed with curation.')
        return False

    logger.info(f'Found {len(input_datasets)} processed datasets for curation')

    try:
        curation_config = CurationConfig(
            input_datasets=input_datasets,
            output_directory=_CURATION_OUTPUT_DIR,
            min_quality_score=_CURATION_MIN_QUALITY_SCORE,
            use_language_detection=_CURATION_USE_LANGUAGE_DETECTION,
            fasttext_model_path=_CURATION_FASTTEXT_MODEL_PATH if _CURATION_USE_LANGUAGE_DETECTION else None,
            max_total_samples=_CURATION_MAX_TOTAL_SAMPLES,
            source_distribution=_CURATION_SOURCE_DISTRIBUTION,
        )

        curator = DatasetCurator(curation_config)
        curator.curate()

        logger.info('\n✅ Dataset curation completed successfully!')
        return True

    except Exception as exception:
        logger.error(f'❌ Failed to curate datasets: {exception}\n')
        return False


def main() -> NoReturn:
    dataset_names = DatasetRegistry.get_all_dataset_names()

    logger.info(f"\n{'=' * 80}")
    logger.info('DATASET PROCESSING PIPELINE')
    logger.info(f"{'=' * 80}\n")
    logger.info(f"Processing {len(dataset_names)} datasets: {', '.join(dataset_names)}\n")

    success_count = 0
    failure_count = 0

    for dataset_name in dataset_names:
        success = _process_dataset(dataset_name)

        if success is True:
            success_count += 1
        else:
            failure_count += 1

    logger.info(f"\n{'=' * 80}")
    logger.info('PROCESSING SUMMARY')
    logger.info(f"{'=' * 80}")
    logger.info(f'  Total datasets: {len(dataset_names)}')
    logger.info(f'  Successful: {success_count}')
    logger.info(f'  Failed: {failure_count}')
    logger.info(f"{'=' * 80}\n")

    if failure_count > 0:
        logger.warning(f'⚠️  {failure_count} dataset(s) failed processing')
        logger.warning('Proceeding with curation of successfully processed datasets\n')

    curation_success = _curate_datasets()

    logger.info(f"\n{'=' * 80}")
    logger.info('PIPELINE SUMMARY')
    logger.info(f"{'=' * 80}")
    logger.info(f"  Processing: {'✅ Success' if failure_count == 0 else f'⚠️  {failure_count} failures'}")
    logger.info(f"  Curation: {'✅ Success' if curation_success else '❌ Failed'}")
    logger.info(f"{'=' * 80}\n")

    if failure_count > 0 or curation_success is False:
        exit(1)

    exit(0)


if __name__ == '__main__':
    main()
