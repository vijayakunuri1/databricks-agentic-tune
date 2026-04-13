"""Address normalization and correction pipeline."""
from __future__ import annotations

import re
import logging
from typing import Any

from agents.data_updater.fuzzy_matcher import FuzzyMatcher
from schemas.qc_results import CorrectionResult

logger = logging.getLogger(__name__)

# Standard street type abbreviation expansions
# Pattern: \b<abbr>\b\.?  — word boundary after abbreviation, then optionally consume a trailing dot
# This ensures "St." → "Street" and "St" → "Street" both work correctly.
_STREET_ABBREVS = {
    r"\bSt\b\.?": "Street",
    r"\bAve\b\.?": "Avenue",
    r"\bBlvd\b\.?": "Boulevard",
    r"\bDr\b\.?": "Drive",
    r"\bRd\b\.?": "Road",
    r"\bLn\b\.?": "Lane",
    r"\bCt\b\.?": "Court",
    r"\bPl\b\.?": "Place",
    r"\bTer\b\.?": "Terrace",
    r"\bCir\b\.?": "Circle",
    r"\bHwy\b\.?": "Highway",
    r"\bPkwy\b\.?": "Parkway",
    r"\bSq\b\.?": "Square",
    r"\bFwy\b\.?": "Freeway",
}

# US state name → abbreviation and vice-versa
_STATE_TO_ABBR = {
    "alabama": "AL","alaska": "AK","arizona": "AZ","arkansas": "AR","california": "CA",
    "colorado": "CO","connecticut": "CT","delaware": "DE","florida": "FL","georgia": "GA",
    "hawaii": "HI","idaho": "ID","illinois": "IL","indiana": "IN","iowa": "IA",
    "kansas": "KS","kentucky": "KY","louisiana": "LA","maine": "ME","maryland": "MD",
    "massachusetts": "MA","michigan": "MI","minnesota": "MN","mississippi": "MS",
    "missouri": "MO","montana": "MT","nebraska": "NE","nevada": "NV",
    "new hampshire": "NH","new jersey": "NJ","new mexico": "NM","new york": "NY",
    "north carolina": "NC","north dakota": "ND","ohio": "OH","oklahoma": "OK",
    "oregon": "OR","pennsylvania": "PA","rhode island": "RI","south carolina": "SC",
    "south dakota": "SD","tennessee": "TN","texas": "TX","utah": "UT","vermont": "VT",
    "virginia": "VA","washington": "WA","west virginia": "WV","wisconsin": "WI",
    "wyoming": "WY","district of columbia": "DC",
}
_ALL_STATES = list(_STATE_TO_ABBR.values()) + [s.title() for s in _STATE_TO_ABBR.keys()]
_ALL_STATE_NAMES = list(_STATE_TO_ABBR.keys())


class AddressCorrector:
    """
    Combines rule-based normalization and fuzzy matching to suggest corrections
    for individual address components.

    Correction priority:
    1. Exact match after normalization → confidence 1.0
    2. rapidfuzz match against reference list → confidence = fuzzy_score
    3. rule-based abbreviation expansion → confidence 0.80
    4. No match → confidence 0.0

    When a GeonamesIndex is provided, city lookups are scoped to the
    canonical city list for the given state, which dramatically improves
    precision (e.g. "Sacrmento" → "Sacramento" only in CA, not "Sacramento"
    in KY or PA).
    """

    def __init__(
        self,
        city_reference: list[str],
        state_reference: list[str] | None = None,
        zip_reference: list[str] | None = None,
        scorer_name: str = "WRatio",
        geonames_index: Any | None = None,
    ):
        self._base_city_reference = city_reference or []
        self._scorer_name = scorer_name
        self._geonames = geonames_index
        self._state_city_cache: dict[str, FuzzyMatcher] = {}

        self.city_matcher = FuzzyMatcher(self._base_city_reference, scorer_name)
        self.state_matcher = FuzzyMatcher(state_reference or _ALL_STATES, scorer_name)
        self.zip_matcher = FuzzyMatcher(zip_reference or [], scorer_name)

    def _city_matcher_for_state(self, state_code: str | None) -> FuzzyMatcher:
        """Return a state-scoped city matcher if GeoNames is available."""
        if not state_code or self._geonames is None:
            return self.city_matcher
        state_upper = state_code.upper()
        if state_upper not in self._state_city_cache:
            cities = self._geonames.cities_for_state(state_upper)
            self._state_city_cache[state_upper] = FuzzyMatcher(
                cities or self._base_city_reference, self._scorer_name
            )
        return self._state_city_cache[state_upper]

    def correct_city(self, value: str, state_hint: str | None = None) -> CorrectionResult:
        if not value:
            return CorrectionResult(original_value=value, corrected_value=None, confidence=0.0, method="none")

        normalized = value.strip().title()
        matcher = self._city_matcher_for_state(state_hint)
        best, confidence = matcher.best_match(normalized)
        if best and confidence >= 0.50:
            return CorrectionResult(
                original_value=value,
                corrected_value=best,
                confidence=confidence,
                method="fuzzy_match",
                candidates=self.city_matcher.find_best_match(normalized, limit=3),
            )
        return CorrectionResult(original_value=value, corrected_value=None, confidence=0.0, method="none")

    def correct_state(self, value: str) -> CorrectionResult:
        if not value:
            return CorrectionResult(original_value=value, corrected_value=None, confidence=0.0, method="none")

        # Exact: already a valid abbreviation
        stripped = value.strip().upper()
        if stripped in _STATE_TO_ABBR.values():
            return CorrectionResult(original_value=value, corrected_value=stripped, confidence=1.0, method="rule_based")

        # Exact: full name
        lower = value.strip().lower()
        if lower in _STATE_TO_ABBR:
            return CorrectionResult(
                original_value=value,
                corrected_value=_STATE_TO_ABBR[lower],
                confidence=1.0,
                method="rule_based",
            )

        # Fuzzy match full name
        best, confidence = self.state_matcher.best_match(value.strip().title())
        if best:
            # Normalize to abbreviation
            abbr = _STATE_TO_ABBR.get(best.lower(), best.upper()[:2])
            return CorrectionResult(
                original_value=value,
                corrected_value=abbr,
                confidence=confidence,
                method="fuzzy_match",
            )

        return CorrectionResult(original_value=value, corrected_value=None, confidence=0.0, method="none")

    def correct_zip(self, value: str, city_hint: str | None = None) -> CorrectionResult:
        if not value:
            return CorrectionResult(original_value=value, corrected_value=None, confidence=0.0, method="none")

        stripped = re.sub(r"[^0-9\-]", "", value.strip())

        # Valid 5-digit or 9-digit zip
        if re.match(r"^\d{5}(-\d{4})?$", stripped):
            return CorrectionResult(
                original_value=value,
                corrected_value=stripped,
                confidence=1.0,
                method="rule_based",
            )

        # Attempt zero-padding (e.g. "1234" → "01234")
        digits_only = re.sub(r"\D", "", stripped)
        if len(digits_only) == 4:
            padded = "0" + digits_only
            return CorrectionResult(
                original_value=value,
                corrected_value=padded,
                confidence=0.80,
                method="rule_based",
            )

        return CorrectionResult(original_value=value, corrected_value=None, confidence=0.0, method="none")

    def correct_street(self, value: str) -> CorrectionResult:
        if not value:
            return CorrectionResult(original_value=value, corrected_value=None, confidence=0.0, method="none")

        corrected = value.strip()
        changed = False
        for pattern, replacement in _STREET_ABBREVS.items():
            new_val = re.sub(pattern, replacement, corrected, flags=re.IGNORECASE)
            if new_val != corrected:
                corrected = new_val
                changed = True

        if changed:
            return CorrectionResult(
                original_value=value,
                corrected_value=corrected,
                confidence=0.80,
                method="rule_based",
            )
        return CorrectionResult(original_value=value, corrected_value=None, confidence=0.0, method="none")
