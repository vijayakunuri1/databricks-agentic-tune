"""Fetch and index US postal-code data from the GeoNames project.

GeoNames (https://www.geonames.org/) publishes free geographical data under
a Creative Commons Attribution 4.0 licence.  This module downloads the US
postal-codes file, parses it into a pandas DataFrame and builds in-memory
lookup indices used by the QC pipeline.
"""
from __future__ import annotations

import io
import logging
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_GEONAMES_URL = "https://download.geonames.org/export/zip/US.zip"

# Column names as documented at https://download.geonames.org/export/zip/readme.txt
_COLUMNS = [
    "country_code",
    "postal_code",
    "place_name",
    "admin_name1",   # state full name
    "admin_code1",   # state abbreviation
    "admin_name2",   # county
    "admin_code2",
    "admin_name3",
    "admin_code3",
    "latitude",
    "longitude",
    "accuracy",
]


class GeonamesIndex:
    """In-memory index over US postal-code reference data from GeoNames.

    The index supports:
    * ZIP → canonical city / state lookup
    * State → city list
    * Fuzzy city-name matching (via *rapidfuzz*)
    """

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    def __init__(self, df: pd.DataFrame) -> None:
        """Build the index from an already-loaded DataFrame.

        ``df`` must contain at least ``postal_code``, ``place_name`` and
        ``admin_code1`` columns (the schema produced by :meth:`download`).
        """
        self._df = df.copy()
        self._normalise_columns()
        self._build_indices()

    def _normalise_columns(self) -> None:
        """Ensure the expected column names exist, handling common aliases."""
        renames: dict[str, str] = {}
        cols_lower = {c.lower(): c for c in self._df.columns}
        for expected, aliases in (
            ("postal_code", ["postal_code", "postalcode", "zip", "zip_code"]),
            ("place_name", ["place_name", "placename", "city", "place"]),
            ("admin_code1", ["admin_code1", "admincode1", "state", "state_code"]),
            ("admin_name1", ["admin_name1", "adminname1", "state_name"]),
            ("latitude", ["latitude", "lat"]),
            ("longitude", ["longitude", "lng", "lon"]),
        ):
            if expected not in self._df.columns:
                for alias in aliases:
                    if alias.lower() in cols_lower:
                        renames[cols_lower[alias.lower()]] = expected
                        break
        if renames:
            self._df = self._df.rename(columns=renames)

        # Normalise postal_code to 5-digit zero-padded string
        if "postal_code" in self._df.columns:
            self._df["postal_code"] = (
                self._df["postal_code"].astype(str).str.strip().str.zfill(5)
            )

    def _build_indices(self) -> None:
        """Build fast-lookup dicts from the DataFrame."""
        # ZIP → first (primary) place name
        self._zip_to_primary: dict[str, str] = {}
        # ZIP → state code
        self._zip_to_state: dict[str, str] = {}
        # ZIP → full row dict (first occurrence)
        self._zip_to_info: dict[str, dict[str, Any]] = {}
        # state → set of unique city names
        self._state_to_cities: dict[str, list[str]] = {}

        for _, row in self._df.iterrows():
            zc = str(row.get("postal_code", ""))
            city = str(row.get("place_name", ""))
            state = str(row.get("admin_code1", ""))

            if zc and zc not in self._zip_to_primary:
                self._zip_to_primary[zc] = city
                self._zip_to_state[zc] = state
                self._zip_to_info[zc] = {
                    "canonical_city": city,
                    "state_code": state,
                    "state_name": str(row.get("admin_name1", "")),
                    "latitude": row.get("latitude"),
                    "longitude": row.get("longitude"),
                }

            if state:
                self._state_to_cities.setdefault(state, [])
                if city and city not in self._state_to_cities[state]:
                    self._state_to_cities[state].append(city)

    # ------------------------------------------------------------------
    # Factory class methods
    # ------------------------------------------------------------------

    @classmethod
    def download(cls, cache_path: str | None = None) -> "GeonamesIndex":
        """Download the US postal-codes ZIP from GeoNames and return an index.

        If *cache_path* points to an existing Parquet file the download is
        skipped and the cached version is used instead.
        """
        if cache_path and Path(cache_path).exists():
            logger.info("Loading cached GeoNames data from %s", cache_path)
            df = pd.read_parquet(cache_path)
            return cls(df)

        logger.info("Downloading GeoNames US postal codes from %s …", _GEONAMES_URL)
        resp = requests.get(_GEONAMES_URL, timeout=60)
        resp.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            # The archive contains a single TSV file named ``US.txt``
            with zf.open("US.txt") as fh:
                df = pd.read_csv(
                    fh,
                    sep="\t",
                    header=None,
                    names=_COLUMNS,
                    dtype=str,
                    keep_default_na=False,
                )

        # Convert numeric fields
        for col in ("latitude", "longitude"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False)
            logger.info("Cached GeoNames data to %s", cache_path)

        return cls(df)

    @classmethod
    def from_file(cls, path: str) -> "GeonamesIndex":
        """Load a previously cached Parquet file and return an index."""
        df = pd.read_parquet(path)
        return cls(df)

    # ------------------------------------------------------------------
    # Lookup methods
    # ------------------------------------------------------------------

    def state_for_zip(self, zip_code: str) -> str | None:
        """Return the two-letter state code for *zip_code*, or ``None``."""
        return self._zip_to_state.get(str(zip_code).strip().zfill(5))

    def canonical_city_for_zip(self, zip_code: str) -> str | None:
        """Return the canonical city name for *zip_code*, or ``None``."""
        return self._zip_to_primary.get(str(zip_code).strip().zfill(5))

    def cities_for_state(self, state_code: str) -> list[str]:
        """Return the list of known city names for *state_code*."""
        return self._state_to_cities.get(state_code.upper(), [])

    def lookup_zip(self, zip_code: str) -> dict[str, Any] | None:
        """Return a dict with canonical city, state, lat/lon for a ZIP."""
        return self._zip_to_info.get(str(zip_code).strip().zfill(5))

    def validate_city_zip(self, city: str, zip_code: str) -> dict[str, Any]:
        """Check whether *city* matches the canonical city for *zip_code*.

        Returns a dict with ``valid``, ``expected_city``, ``actual_city``,
        and ``confidence`` keys.
        """
        info = self.lookup_zip(zip_code)
        if info is None:
            return {
                "valid": False,
                "expected_city": None,
                "actual_city": city,
                "confidence": 0.0,
                "error": f"ZIP {zip_code} not found in GeoNames.",
            }

        expected = info["canonical_city"]
        normalised_city = city.strip().upper()
        normalised_expected = expected.strip().upper()

        if normalised_city == normalised_expected:
            return {
                "valid": True,
                "expected_city": expected,
                "actual_city": city,
                "confidence": 1.0,
            }

        # Use fuzzy matching for near-misses
        try:
            from rapidfuzz import fuzz

            score = fuzz.WRatio(normalised_city, normalised_expected) / 100.0
        except ImportError:
            score = 0.0

        return {
            "valid": score >= 0.90,
            "expected_city": expected,
            "actual_city": city,
            "confidence": round(score, 3),
        }

    def fuzzy_city_lookup(
        self,
        city: str,
        state_code: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Fuzzy-match *city* against reference city names.

        If *state_code* is given the search is restricted to cities in that
        state, which significantly improves precision.

        Returns a list of ``{"city": ..., "state_code": ..., "score": ...}``
        dicts sorted by descending score.
        """
        try:
            from rapidfuzz import fuzz, process
        except ImportError:
            logger.warning("rapidfuzz not installed – fuzzy_city_lookup returns empty.")
            return []

        if state_code:
            candidates = self.cities_for_state(state_code.upper())
        else:
            # Build a flat list of all unique cities
            seen: set[str] = set()
            candidates = []
            for cities in self._state_to_cities.values():
                for c in cities:
                    if c not in seen:
                        seen.add(c)
                        candidates.append(c)

        if not candidates:
            return []

        matches = process.extract(
            city,
            candidates,
            scorer=fuzz.WRatio,
            limit=min(limit, len(candidates)),
        )

        results: list[dict[str, Any]] = []
        for match_city, score, _idx in matches:
            # Determine state for this city (prefer the requested state)
            st = state_code.upper() if state_code else ""
            if not st:
                for s, cities in self._state_to_cities.items():
                    if match_city in cities:
                        st = s
                        break
            results.append({
                "city": match_city,
                "state_code": st,
                "score": round(score / 100.0, 3),
            })

        return results
