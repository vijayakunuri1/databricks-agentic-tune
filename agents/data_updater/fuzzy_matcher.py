"""RapidFuzz wrapper for address/location fuzzy matching."""
from __future__ import annotations

import logging
from typing import Literal

from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

SupportLevel = Literal["L1", "L2", "skip"]


class FuzzyMatcher:
    """
    Wraps rapidfuzz to find the best match for a query string
    against a reference list (e.g. canonical city names).

    Scores from rapidfuzz are 0-100; this class normalizes to 0.0-1.0.
    """

    L1_THRESHOLD = 85.0
    L2_LOWER = 50.0

    def __init__(self, reference_data: list[str], scorer_name: str = "WRatio"):
        self.reference = reference_data
        self.scorer = self._resolve_scorer(scorer_name)

    def find_best_match(
        self, query: str, limit: int = 5, score_cutoff: float = 50.0
    ) -> list[tuple[str, float, int]]:
        """
        Returns: [(match_string, normalized_score_0_to_1, reference_index), ...]
        Sorted by score descending.
        """
        if not query or not self.reference:
            return []

        results = process.extract(
            query,
            self.reference,
            scorer=self.scorer,
            limit=limit,
            score_cutoff=score_cutoff,
        )
        # rapidfuzz extract returns (match, score_0_100, index)
        return [(m, s / 100.0, i) for m, s, i in results]

    def best_match(self, query: str) -> tuple[str | None, float]:
        """Returns (best_match_string, confidence_0_to_1). None if below skip threshold."""
        results = self.find_best_match(query, limit=1, score_cutoff=self.L2_LOWER)
        if not results:
            return None, 0.0
        match_str, confidence, _ = results[0]
        return match_str, confidence

    def classify_match(self, score_normalized: float) -> SupportLevel:
        score_pct = score_normalized * 100.0
        if score_pct >= self.L1_THRESHOLD:
            return "L1"
        if score_pct >= self.L2_LOWER:
            return "L2"
        return "skip"

    @staticmethod
    def _resolve_scorer(name: str):
        scorers = {
            "WRatio": fuzz.WRatio,
            "ratio": fuzz.ratio,
            "partial_ratio": fuzz.partial_ratio,
            "token_sort_ratio": fuzz.token_sort_ratio,
            "token_set_ratio": fuzz.token_set_ratio,
        }
        scorer = scorers.get(name)
        if scorer is None:
            logger.warning("Unknown scorer '%s', using WRatio.", name)
            return fuzz.WRatio
        return scorer
