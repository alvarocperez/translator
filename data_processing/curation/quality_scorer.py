from typing import Final

from data_processing.curation.quality_metrics import QualityMetrics


class QualityScorer:
    _MIN_UNIQUE_RATIO: Final[float] = 0.5
    _MODERATE_UNIQUE_RATIO: Final[float] = 0.7
    _MIN_ALPHA_RATIO: Final[float] = 0.6
    _MODERATE_ALPHA_RATIO: Final[float] = 0.75
    _MAX_LENGTH_RATIO: Final[float] = 2.5
    _MIN_LENGTH_RATIO: Final[float] = 0.4
    _MODERATE_MAX_LENGTH_RATIO: Final[float] = 2.0
    _MODERATE_MIN_LENGTH_RATIO: Final[float] = 0.5
    _MIN_WORDS_FOR_CONTEXT: Final[int] = 5
    _MAX_WORDS_OPTIMAL: Final[int] = 80
    _MIN_WORDS_OPTIMAL: Final[int] = 8
    _MAX_WORDS_IDEAL: Final[int] = 40
    _MAX_CONSECUTIVE_REPEATS: Final[int] = 2

    def compute_quality_score(self, source_text: str, target_text: str) -> QualityMetrics:
        source_words = source_text.split()
        target_words = target_text.split()

        lexical_diversity = self._compute_lexical_diversity_score(source_words, target_words)
        alphabetic_ratio = self._compute_alphabetic_ratio_score(source_text, target_text)
        length_balance = self._compute_length_balance_score(source_words, target_words)
        optimal_length = self._compute_optimal_length_score(source_words, target_words)
        repetition_penalty = self._compute_repetition_penalty(source_words, target_words)
        identity_penalty = self._compute_identity_penalty(source_text, target_text)

        final_score = max(
            0.0,
            min(
                100.0,
                100.0
                + lexical_diversity
                + alphabetic_ratio
                + length_balance
                + optimal_length
                + repetition_penalty
                + identity_penalty,
            ),
        )

        return QualityMetrics(
            lexical_diversity_score=lexical_diversity,
            alphabetic_ratio_score=alphabetic_ratio,
            length_balance_score=length_balance,
            optimal_length_score=optimal_length,
            repetition_penalty=repetition_penalty,
            identity_penalty=identity_penalty,
            final_score=final_score,
        )

    def _compute_lexical_diversity_score(self, source_words: list[str], target_words: list[str]) -> float:
        if len(source_words) == 0 or len(target_words) == 0:
            return -50.0

        source_unique_ratio = len(set(source_words)) / len(source_words)
        target_unique_ratio = len(set(target_words)) / len(target_words)

        if source_unique_ratio < self._MIN_UNIQUE_RATIO or target_unique_ratio < self._MIN_UNIQUE_RATIO:
            return -30.0
        if source_unique_ratio < self._MODERATE_UNIQUE_RATIO or target_unique_ratio < self._MODERATE_UNIQUE_RATIO:
            return -15.0

        return 0.0

    def _compute_alphabetic_ratio_score(self, source_text: str, target_text: str) -> float:
        if len(source_text) == 0 or len(target_text) == 0:
            return -50.0

        source_alpha_ratio = sum(c.isalpha() or c.isspace() for c in source_text) / len(source_text)
        target_alpha_ratio = sum(c.isalpha() or c.isspace() for c in target_text) / len(target_text)

        if source_alpha_ratio < self._MIN_ALPHA_RATIO or target_alpha_ratio < self._MIN_ALPHA_RATIO:
            return -25.0
        if source_alpha_ratio < self._MODERATE_ALPHA_RATIO or target_alpha_ratio < self._MODERATE_ALPHA_RATIO:
            return -10.0

        return 0.0

    def _compute_length_balance_score(self, source_words: list[str], target_words: list[str]) -> float:
        if len(target_words) == 0:
            return -50.0

        length_ratio = len(source_words) / max(1, len(target_words))

        if length_ratio > self._MAX_LENGTH_RATIO or length_ratio < self._MIN_LENGTH_RATIO:
            return -30.0
        if length_ratio > self._MODERATE_MAX_LENGTH_RATIO or length_ratio < self._MODERATE_MIN_LENGTH_RATIO:
            return -15.0

        return 0.0

    def _compute_optimal_length_score(self, source_words: list[str], target_words: list[str]) -> float:
        score = 0.0

        if len(source_words) < self._MIN_WORDS_FOR_CONTEXT or len(target_words) < self._MIN_WORDS_FOR_CONTEXT:
            score -= 15.0
        elif len(source_words) > self._MAX_WORDS_OPTIMAL or len(target_words) > self._MAX_WORDS_OPTIMAL:
            score -= 10.0

        if (
            self._MIN_WORDS_OPTIMAL <= len(source_words) <= self._MAX_WORDS_IDEAL
            and self._MIN_WORDS_OPTIMAL <= len(target_words) <= self._MAX_WORDS_IDEAL
        ):
            score += 10.0

        return score

    def _compute_repetition_penalty(self, source_words: list[str], target_words: list[str]) -> float:
        source_max_repeats = self._count_max_consecutive_repeats(source_words)
        target_max_repeats = self._count_max_consecutive_repeats(target_words)

        if source_max_repeats > self._MAX_CONSECUTIVE_REPEATS or target_max_repeats > self._MAX_CONSECUTIVE_REPEATS:
            return -20.0

        return 0.0

    def _compute_identity_penalty(self, source_text: str, target_text: str) -> float:
        if source_text.lower().strip() == target_text.lower().strip():
            return -50.0

        return 0.0

    def _count_max_consecutive_repeats(self, words: list[str]) -> int:
        if len(words) == 0:
            return 0

        max_repeat = 1
        current_repeat = 1

        for i in range(1, len(words)):
            if words[i] == words[i - 1]:
                current_repeat += 1
                max_repeat = max(max_repeat, current_repeat)
            else:
                current_repeat = 1

        return max_repeat
