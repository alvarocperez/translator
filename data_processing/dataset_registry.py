from pathlib import Path
from typing import Final

from data_processing.dataset_config import DatasetConfig

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
_BASE_RAW_DIR: Final[Path] = _PROJECT_ROOT / "dataset" / "raw"
_BASE_PROCESSED_DIR: Final[Path] = _PROJECT_ROOT / "dataset" / "processed"

_DATASET_CONFIGS: Final[dict[str, DatasetConfig]] = {
    "ccaligned": DatasetConfig(
        name="CCAligned",
        raw_directory=_BASE_RAW_DIR / "ccaligned",
        processed_directory=_BASE_PROCESSED_DIR / "ccaligned",
        source_filename="CCAligned.en-es.es",
        target_filename="CCAligned.en-es.en",
        output_filename="ccaligned_dataset.jsonl",
        source_language="en",
        target_language="es",
        max_samples=None,
        validator_config={
            "min_tokens": 3,
            "max_tokens": 150,
            "max_length_ratio": 2.5,
            "min_length_ratio": 0.4
        }
    ),
    "eubookshop": DatasetConfig(
        name="EUbookshop",
        raw_directory=_BASE_RAW_DIR / "eubookshop",
        processed_directory=_BASE_PROCESSED_DIR / "eubookshop",
        source_filename="EUbookshop.en-es.es",
        target_filename="EUbookshop.en-es.en",
        output_filename="eubookshop_dataset.jsonl",
        source_language="en",
        target_language="es",
        max_samples=None,
        validator_config={
            "min_tokens": 3,
            "max_tokens": 150,
            "max_length_ratio": 2.5,
            "min_length_ratio": 0.4
        }
    ),
    "dgt": DatasetConfig(
        name="DGT",
        raw_directory=_BASE_RAW_DIR / "DGT",
        processed_directory=_BASE_PROCESSED_DIR / "DGT",
        source_filename="DGT.en-es.es",
        target_filename="DGT.en-es.en",
        output_filename="dgt_dataset.jsonl",
        source_language="en",
        target_language="es",
        max_samples=None,
        validator_config={
            "min_tokens": 3,
            "max_tokens": 150,
            "max_length_ratio": 2.5,
            "min_length_ratio": 0.4
        }
    )
}


class DatasetRegistry:
    @staticmethod
    def get_dataset_config(dataset_name: str) -> DatasetConfig:
        dataset_config = _DATASET_CONFIGS.get(dataset_name.lower())

        if dataset_config is None:
            available_datasets = ", ".join(_DATASET_CONFIGS.keys())
            raise ValueError(
                f"Unknown dataset: {dataset_name}. "
                f"Available datasets: {available_datasets}"
            )

        return dataset_config

    @staticmethod
    def get_all_dataset_names() -> list[str]:
        return list(_DATASET_CONFIGS.keys())

    @staticmethod
    def get_all_dataset_configs() -> list[DatasetConfig]:
        return list(_DATASET_CONFIGS.values())
