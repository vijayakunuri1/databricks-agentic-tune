"""MCP tools: GeoNames city lookup and ZIP code validation."""
from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

# Basic input guards: city name / ZIP should not contain SQL metacharacters
_CITY_RE = re.compile(r"^[a-zA-Z0-9 \'\-\.,]{1,128}$")
_ZIP_RE  = re.compile(r"^\d{5}$")
_STATE_RE = re.compile(r"^[A-Za-z]{2}$")


def register_geo_tools(mcp) -> None:
    """Register geo/address lookup tools on the FastMCP app."""

    @mcp.tool()
    def lookup_city(
        city_name: str,
        state_code: str = "",
        limit: int = 5,
    ) -> str:
        """
        Fuzzy-match a city name against the GeoNames US reference dataset.

        Use this to check whether a city name is a likely typo and get
        the canonical spelling. Optionally filter by state for higher accuracy.

        Args:
            city_name:  The city name to look up (e.g. 'Los Angles', 'Chcago')
            state_code: Optional US state abbreviation to filter results (e.g. 'CA', 'IL')
            limit:      Max number of candidate matches to return (default: 5)

        Returns:
            JSON with {query, state_code, matches: [{city, state_code, score}]}
            Score is 0.0–1.0 where 1.0 = exact match.
        """
        # --- Input validation ---
        if not _CITY_RE.match(city_name):
            return json.dumps({"error": "Invalid city_name: only letters, digits, spaces, hyphens, apostrophes, commas, dots allowed (max 128 chars)."})
        if state_code and not _STATE_RE.match(state_code):
            return json.dumps({"error": "Invalid state_code: must be a 2-letter US state abbreviation."})
        limit = max(1, min(int(limit), 20))

        # --- Rate limiting ---
        from configs.rate_limiting import check_geo_lookup, rate_limit_error
        allowed, retry_after = check_geo_lookup()
        if not allowed:
            from configs.security_logging import log_rate_limit_hit
            log_rate_limit_hit(client_ip="mcp-tool", tool="lookup_city")
            return json.dumps({"error": rate_limit_error(retry_after, "geo_lookup")})

        from mcp_server.pipeline_state import get_config

        cfg = get_config()
        geonames = cfg.get("geonames_index")

        if geonames is None:
            return json.dumps({"error": "GeoNames index not loaded. Call ingest_geonames first."})

        matches = geonames.fuzzy_city_lookup(city_name, state_code or None, limit=limit)
        return json.dumps({
            "query": city_name,
            "state_filter": state_code or "none",
            "matches": matches,
            "tip": "Score >= 0.85 → L1 auto-correct. Score 0.50-0.84 → L2 flag for review.",
        })

    @mcp.tool()
    def lookup_zip(zip_code: str) -> str:
        """
        Look up a US ZIP code in the GeoNames reference dataset.

        Returns the canonical city, state, and coordinates for the ZIP.
        Use this to:
        - Validate that a ZIP code exists
        - Check if the city in your data matches what GeoNames expects
        - Detect ZIP/state mismatches (e.g. a CA company with a TX ZIP)

        Args:
            zip_code: 5-digit US ZIP code (e.g. '90001', '10001')

        Returns:
            JSON with {zip_code, canonical_city, state_code, state_name, latitude, longitude}
            or {error: ...} if ZIP not found.
        """
        # --- Input validation ---
        if not _ZIP_RE.match(str(zip_code).strip()):
            return json.dumps({"error": "Invalid zip_code: must be a 5-digit US ZIP code."})

        # --- Rate limiting ---
        from configs.rate_limiting import check_geo_lookup, rate_limit_error
        allowed, retry_after = check_geo_lookup()
        if not allowed:
            from configs.security_logging import log_rate_limit_hit
            log_rate_limit_hit(client_ip="mcp-tool", tool="lookup_zip")
            return json.dumps({"error": rate_limit_error(retry_after, "geo_lookup")})

        from mcp_server.pipeline_state import get_config

        cfg = get_config()
        geonames = cfg.get("geonames_index")

        if geonames is None:
            return json.dumps({"error": "GeoNames index not loaded."})

        info = geonames.lookup_zip(zip_code)
        if info is None:
            return json.dumps({"zip_code": zip_code, "found": False, "error": "ZIP code not found in GeoNames."})

        return json.dumps({
            "zip_code": str(zip_code).zfill(5),
            "found": True,
            **info,
        })

    @mcp.tool()
    def validate_city_zip(city: str, zip_code: str) -> str:
        """
        Validate whether a city name matches what GeoNames expects for a ZIP code.

        This is the key check for the SEC EDGAR use case — companies sometimes
        have a city name that doesn't match their ZIP code, indicating a data entry
        error or an outdated address after a city reorganization.

        Args:
            city:     The city name in your data
            zip_code: The ZIP code in your data

        Returns:
            JSON with {valid, expected_city, actual_city, confidence, recommendation}
        """
        # --- Input validation ---
        if not _CITY_RE.match(city):
            return json.dumps({"error": "Invalid city: only letters, digits, spaces, hyphens, apostrophes, commas, dots allowed (max 128 chars)."})
        if not _ZIP_RE.match(str(zip_code).strip()):
            return json.dumps({"error": "Invalid zip_code: must be a 5-digit US ZIP code."})

        # --- Rate limiting ---
        from configs.rate_limiting import check_geo_lookup, rate_limit_error
        allowed, retry_after = check_geo_lookup()
        if not allowed:
            from configs.security_logging import log_rate_limit_hit
            log_rate_limit_hit(client_ip="mcp-tool", tool="validate_city_zip")
            return json.dumps({"error": rate_limit_error(retry_after, "geo_lookup")})

        from mcp_server.pipeline_state import get_config

        cfg = get_config()
        geonames = cfg.get("geonames_index")

        if geonames is None:
            return json.dumps({"error": "GeoNames index not loaded."})

        result = geonames.validate_city_zip(city, zip_code)
        result["recommendation"] = (
            "ACCEPT" if result["valid"]
            else f"SUGGEST_CORRECTION: change city to '{result['expected_city']}'"
        )
        return json.dumps(result)
