from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    raw_directory: Path
    processed_directory: Path
    source_filename: str
    target_filename: str
    output_filename: str
    source_language: str
    target_language: str
    max_samples: int | None = None
    validator_config: dict[str, Any] | None = None

    def get_source_file_path(self) -> Path:
        return self.raw_directory / self.source_filename

    def get_target_file_path(self) -> Path:
        return self.raw_directory / self.target_filename

    def get_output_file_path(self) -> Path:
        return self.processed_directory / self.output_filename
