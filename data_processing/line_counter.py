import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class LineCounter:
    def count_lines(self, filepath: Path) -> int:
        logger.info(f"Counting lines in {filepath.name}")

        line_count = 0
        with open(filepath, 'r', encoding='utf-8', buffering=8192 * 1024) as file:
            for _ in file:
                line_count += 1

        logger.info(f"Total lines found: {line_count:,}")
        return line_count
