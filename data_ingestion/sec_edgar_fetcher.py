"""Fetch company registration data from the SEC EDGAR full-text search API.

The SEC EDGAR EFTS (full-text search) API is free and requires no API key,
but does require a ``User-Agent`` header identifying the caller.  The API is
rate-limited to **10 requests per second** — this module respects that limit
automatically.

Reference:
  https://efts.sec.gov/LATEST/search-index?q=*&dateRange=custom&startdt=2020-01-01&enddt=2024-12-31&forms=10-K
  https://www.sec.gov/search#/dateRange=custom
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests

logger = logging.getLogger(__name__)

_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
_REQUEST_DELAY = 0.12  # ~8 req/s to stay under the 10 req/s limit
_USER_AGENT = "databricks-agentic-tune/1.0 (SEC EDGAR research)"


class SECEdgarFetcher:
    """Download SEC EDGAR company registration data."""

    def __init__(self, user_agent: str = _USER_AGENT) -> None:
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": user_agent,
            "Accept-Encoding": "gzip, deflate",
        })

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_companies(
        self,
        limit: int = 500,
        us_only: bool = True,
        cache_path: str | None = None,
    ) -> pd.DataFrame:
        """Return a DataFrame of SEC EDGAR company registrations.

        Parameters
        ----------
        limit:
            Maximum number of companies to return.
        us_only:
            If ``True``, filter to US-based companies only (non-empty
            ``stateOrCountry`` that is a 2-letter US state code).
        cache_path:
            If given and the file exists, load from cache instead of
            downloading.  After downloading, the result is saved here.
        """
        if cache_path and Path(cache_path).exists():
            logger.info("Loading cached SEC EDGAR data from %s", cache_path)
            return pd.read_parquet(cache_path)

        df = self._download_company_data(limit=limit, us_only=us_only)

        if cache_path:
            Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False)
            logger.info("Cached SEC EDGAR data to %s (%d rows)", cache_path, len(df))

        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _download_company_data(self, limit: int, us_only: bool) -> pd.DataFrame:
        """Fetch the company tickers file, then enrich with address data."""
        logger.info("Fetching SEC EDGAR company tickers …")
        resp = self._get(_COMPANY_TICKERS_URL)
        tickers: dict[str, Any] = resp.json()

        rows: list[dict[str, Any]] = []
        entries = list(tickers.values())[:limit * 2]  # fetch extra to allow filtering

        for entry in entries:
            if len(rows) >= limit:
                break
            cik = str(entry.get("cik_str", "")).zfill(10)
            try:
                detail = self._fetch_submission(cik)
            except Exception:
                logger.debug("Skipping CIK %s (fetch failed)", cik)
                continue

            addresses = detail.get("addresses", {})
            biz = addresses.get("business", {}) or addresses.get("mailing", {}) or {}

            state_or_country = str(biz.get("stateOrCountry", "") or "")
            if us_only and (len(state_or_country) != 2 or not state_or_country.isalpha()):
                continue

            rows.append({
                "cik": str(entry.get("cik_str", "")),
                "name": detail.get("name", entry.get("title", "")),
                "ticker": entry.get("ticker", ""),
                "sic": detail.get("sic", ""),
                "sic_description": detail.get("sicDescription", ""),
                "street1": str(biz.get("street1", "") or ""),
                "street2": str(biz.get("street2", "") or ""),
                "city": str(biz.get("city", "") or ""),
                "state_or_country": state_or_country,
                "zip_code": str(biz.get("zipCode", "") or ""),
                # 'id' mirrors 'cik' — downstream code uses 'id' as the
                # primary key while 'cik' is the SEC-specific identifier.
                "id": str(entry.get("cik_str", "")),
            })

        logger.info("Fetched %d companies from SEC EDGAR.", len(rows))
        return pd.DataFrame(rows)

    def _fetch_submission(self, cik: str) -> dict[str, Any]:
        """Fetch the submissions JSON for a single CIK."""
        url = _SUBMISSIONS_URL.format(cik=cik.zfill(10))
        time.sleep(_REQUEST_DELAY)
        resp = self._get(url)
        return resp.json()

    def _get(self, url: str) -> requests.Response:
        resp = self._session.get(url, timeout=30)
        resp.raise_for_status()
        return resp
