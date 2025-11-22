import logging
from typing import NoReturn

from data_processing.dataset_registry import DatasetRegistry
from data_processing.parallel_corpus_processor import ParallelCorpusProcessor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def _process_dataset(dataset_name: str) -> bool:
    try:
        dataset_config = DatasetRegistry.get_dataset_config(dataset_name)

        logger.info(f"\n{'=' * 80}")
        logger.info(f"Processing dataset: {dataset_config.name}")
        logger.info(f"{'=' * 80}\n")

        processor = ParallelCorpusProcessor(
            source_file_path=dataset_config.get_source_file_path(),
            target_file_path=dataset_config.get_target_file_path(),
            output_file_path=dataset_config.get_output_file_path(),
            source_language=dataset_config.source_language,
            target_language=dataset_config.target_language,
            max_samples=dataset_config.max_samples,
            validator_config=dataset_config.validator_config
        )

        processor.process()

        logger.info(f"✅ {dataset_config.name} processed successfully!\n")
        return True

    except Exception as exception:
        logger.error(f"❌ Failed to process {dataset_name}: {exception}\n")
        return False


def main() -> NoReturn:
    dataset_names = DatasetRegistry.get_all_dataset_names()

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
    logger.info("Processing Summary:")
    logger.info(f"  Total datasets: {len(dataset_names)}")
    logger.info(f"  Successful: {success_count}")
    logger.info(f"  Failed: {failure_count}")
    logger.info(f"{'=' * 80}")

    if failure_count > 0:
        exit(1)

    exit(0)


if __name__ == "__main__":
    main()
