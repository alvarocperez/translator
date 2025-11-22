import re


class TextPairValidator:
    _NOISY_PATTERN = re.compile(r"http[s]?://|www\.|[@#<>$%^*_=+\\\[\]{}~`|]")
    _UNWANTED_CHARS = re.compile(r"[^\w\s,.!?¿¡'\"()\-\u00C0-\u017F]")

    def __init__(
            self,
            min_tokens: int = 3,
            max_tokens: int = 150,
            max_length_ratio: float = 2.5,
            min_length_ratio: float = 0.4
    ) -> None:
        self._min_tokens = min_tokens
        self._max_tokens = max_tokens
        self._max_length_ratio = max_length_ratio
        self._min_length_ratio = min_length_ratio

    def is_valid_pair(self, source_text: str, target_text: str) -> bool:
        if not source_text or not target_text:
            return False

        source_tokens = source_text.split()
        target_tokens = target_text.split()

        if not self._is_valid_token_count(source_tokens, target_tokens):
            return False

        if not self._is_valid_length_ratio(source_tokens, target_tokens):
            return False

        if self._contains_noise(source_text) or self._contains_noise(target_text):
            return False

        if self._are_identical(source_text, target_text):
            return False

        return True

    def _is_valid_token_count(self, source_tokens: list[str], target_tokens: list[str]) -> bool:
        source_token_count = len(source_tokens)
        target_token_count = len(target_tokens)

        if source_token_count < self._min_tokens or target_token_count < self._min_tokens:
            return False

        if source_token_count > self._max_tokens or target_token_count > self._max_tokens:
            return False

        return True

    def _is_valid_length_ratio(self, source_tokens: list[str], target_tokens: list[str]) -> bool:
        length_ratio = len(source_tokens) / max(1, len(target_tokens))

        return self._min_length_ratio <= length_ratio <= self._max_length_ratio

    def _contains_noise(self, text: str) -> bool:
        if not text:
            return True

        if self._NOISY_PATTERN.search(text) is not None:
            return True

        if self._UNWANTED_CHARS.search(text) is not None:
            return True

        return False

    def _are_identical(self, source_text: str, target_text: str) -> bool:
        normalized_source = source_text.strip().lower()
        normalized_target = target_text.strip().lower()

        return normalized_source == normalized_target