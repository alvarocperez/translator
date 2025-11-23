from dataclasses import dataclass


@dataclass(frozen=True)
class QualityMetrics:
    lexical_diversity_score: float
    alphabetic_ratio_score: float
    length_balance_score: float
    optimal_length_score: float
    repetition_penalty: float
    identity_penalty: float
    final_score: float
