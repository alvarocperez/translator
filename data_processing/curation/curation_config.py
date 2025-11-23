from dataclasses import dataclass
from pathlib import Path


@dataclass
class CurationConfig:
    input_datasets: dict[str, Path]
    output_directory: Path
    expected_source_language: str
    expected_target_language: str
    min_quality_score: float = 70.0
    ngram_size_for_dedup: int = 5
    use_language_detection: bool = True
    fasttext_model_path: Path | None = None
    min_language_confidence: float = 0.9
    source_distribution: dict[str, float] | None = None
    max_total_samples: int | None = None
    train_ratio: float = 0.98
    validation_ratio: float = 0.01
    test_ratio: float = 0.01
    random_seed: int = 42

    def __post_init__(self):
        total_ratio = self.train_ratio + self.validation_ratio + self.test_ratio
        if abs(total_ratio - 1.0) > 0.01:
            raise ValueError(f'Split ratios must sum to 1.0, got {total_ratio}')

        if self.source_distribution is not None:
            distribution_sum = sum(self.source_distribution.values())
            if abs(distribution_sum - 1.0) > 0.01:
                raise ValueError(f'Source distribution must sum to 1.0, got {distribution_sum}')
