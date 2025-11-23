import hashlib
import logging
from typing import Final, TypeVar

from tqdm import tqdm

logger = logging.getLogger(__name__)

T = TypeVar('T')


class TextDeduplicator:
    _DEFAULT_NGRAM_SIZE: Final[int] = 5

    def __init__(self, ngram_size: int = _DEFAULT_NGRAM_SIZE):
        self._ngram_size = ngram_size

    def deduplicate_exact(self, pairs: list[T]) -> list[T]:
        logger.info('Starting exact deduplication...')

        hash_to_best_pair: dict[str, T] = {}

        for pair in tqdm(pairs, desc='Processing exact dedup'):
            hash_value = pair.exact_hash

            if hash_value not in hash_to_best_pair or pair.quality_score > hash_to_best_pair[hash_value].quality_score:
                hash_to_best_pair[hash_value] = pair

        unique_pairs = list(hash_to_best_pair.values())
        duplicates_removed = len(pairs) - len(unique_pairs)

        logger.info(
            f'Exact deduplication complete: removed {duplicates_removed:,} duplicates '
            f'({duplicates_removed / len(pairs) * 100:.2f}%)'
        )
        logger.info(f'Remaining pairs: {len(unique_pairs):,}')

        return unique_pairs

    def deduplicate_fuzzy(self, pairs: list[T]) -> list[T]:
        logger.info('Starting fuzzy deduplication...')

        fuzzy_hash_to_best_pair: dict[str, T] = {}

        for pair in tqdm(pairs, desc='Processing fuzzy dedup'):
            fuzzy_composite_key = f'{pair.fuzzy_hash_source}|||{pair.fuzzy_hash_target}'

            if (
                fuzzy_composite_key not in fuzzy_hash_to_best_pair
                or pair.quality_score > fuzzy_hash_to_best_pair[fuzzy_composite_key].quality_score
            ):
                fuzzy_hash_to_best_pair[fuzzy_composite_key] = pair

        unique_pairs = list(fuzzy_hash_to_best_pair.values())
        duplicates_removed = len(pairs) - len(unique_pairs)

        logger.info(
            f'Fuzzy deduplication complete: removed {duplicates_removed:,} duplicates '
            f'({duplicates_removed / len(pairs) * 100:.2f}%)'
        )
        logger.info(f'Remaining pairs: {len(unique_pairs):,}')

        return unique_pairs

    @staticmethod
    def compute_exact_hash(source_text: str, target_text: str) -> str:
        normalized_pair = f'{source_text.lower().strip()}|||{target_text.lower().strip()}'
        return hashlib.md5(normalized_pair.encode('utf-8')).hexdigest()

    def compute_fuzzy_hash(self, text: str) -> str:
        normalized_text = text.lower().strip()
        words = normalized_text.split()

        if len(words) < self._ngram_size:
            return hashlib.md5(normalized_text.encode('utf-8')).hexdigest()

        ngrams = [' '.join(words[i : i + self._ngram_size]) for i in range(len(words) - self._ngram_size + 1)]

        sorted_ngrams = ''.join(sorted(ngrams))
        return hashlib.md5(sorted_ngrams.encode('utf-8')).hexdigest()
