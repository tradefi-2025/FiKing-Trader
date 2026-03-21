import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
from dotenv import load_dotenv

try:
    import refinitiv.data as rd
except ImportError:  # pragma: no cover - library not installed
    rd = None

# -----------------------------------------------------------------------------
# Environment & Logging
# -----------------------------------------------------------------------------

# Let the application control where .env lives; fall back to project root if present.
DEFAULT_ENV = Path(__file__).resolve().parents[2] / ".env"
if DEFAULT_ENV.exists():
    load_dotenv(DEFAULT_ENV)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

SUPPORTED_INTERVALS = [
    "1min",
    "5min",
    "10min",
    "30min",
    "60min",
    "1h",
    "1d",
    "7d",
    "1W",
    
]  # from Refinitiv docs[web:28][web:29]


def _build_inline_config() -> dict:
    """Build an inline refinitiv-data config from environment variables."""
    api_key = os.getenv("REFINITIV_API_KEY")
    username = os.getenv("REFINITIV_USERNAME")
    password = os.getenv("REFINITIV_PASSWORD")

    if not api_key:
        raise RuntimeError("REFINITIV_API_KEY not set")

    return {
        "sessions": {
            "default": "platform.rdp",
            "platform": {
                "rdp": {
                    "app-key": api_key,
                    "username": username,
                    "password": password,
                }
            },
        }
    }


@dataclass
class RefinitivService:
    """
    Refinitiv API client for historical OHLCV data using refinitiv-data.

    Designed to integrate cleanly with the MongoDB time-series service.
    """

    strict_errors: bool = False
    config_name: Optional[str] = None  # path to a config JSON; if None, build inline
    session_open: bool = False

    def __post_init__(self) -> None:
        if rd is None:
            raise ImportError("refinitiv.data library is not available")
        self._open_session()

    # -------------------------------------------------------------------------
    # Session lifecycle
    # -------------------------------------------------------------------------

    def _open_session(self) -> None:
        """Open a Refinitiv session, building a config file if needed."""
        if self.session_open:
            return

        try:
            if self.config_name:
                # Use an existing configuration file (recommended for production).[web:39][web:42]
                rd.open_session(config_name=self.config_name)
            else:
                # Build a minimal config JSON next to this module.
                cfg = _build_inline_config()
                cfg_path = Path(__file__).with_name("refinitiv-data.config.json")
                cfg_path.write_text(json.dumps(cfg))
                rd.open_session(config_name=str(cfg_path))

            self.session_open = True
            logger.info("✅ Refinitiv session opened")
        except Exception as exc:
            logger.error("❌ Failed to open Refinitiv session: %s", exc)
            if self.strict_errors:
                raise

    def close_session(self) -> None:
        """Close the Refinitiv default session."""
        if not self.session_open:
            return
        try:
            rd.close_session()
            self.session_open = False
            logger.info("🔌 Refinitiv session closed")
        except Exception as exc:
            logger.error("❌ Error closing Refinitiv session: %s", exc)
            if self.strict_errors:
                raise

    # Optional context manager usage
    def __enter__(self) -> "RefinitivService":
        if not self.session_open:
            self._open_session()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close_session()

    # -------------------------------------------------------------------------
    # Core OHLC fetch
    # -------------------------------------------------------------------------

    def get_ohlc_df(
        self,
        equity_ric: str,
        start: datetime,
        end: datetime,
        *,
        interval: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """
        Fetch OHLCV history for a single RIC between start and end.

        Args:
            equity_ric: RIC (e.g. "AAPL.O").
            start: Start datetime.
            end: End datetime.
            interval: Refinitiv interval string (e.g. "1min", "1d").[web:28][web:29]

        Returns:
            DataFrame with columns [timestamp, open, high, low, close, volume]
            sorted ascending, or None on error / no data.
        """
        if not self.session_open:
            logger.error("Refinitiv session not open")
            if self.strict_errors:
                raise RuntimeError("Refinitiv session not open")
            return None

        if interval not in SUPPORTED_INTERVALS:
            raise ValueError(f"Unsupported interval: {interval}")

        try:
            df = rd.get_history(
                universe=equity_ric.upper(),  # RICs are typically uppercase; ensure consistency
                fields=["OPEN_PRC", "HIGH_1", "LOW_1", "TRDPRC_1", "ACVOL_UNS"],
                start=start,  # datetimes are supported directly[web:28]
                end=end,
                interval=interval,
            )

            if df is None or df.empty:
                logger.warning("No data returned for %s", equity_ric)
                return None

            # Ensure expected columns are present.
            missing = [c for c in ["OPEN_PRC", "HIGH_1", "LOW_1", "TRDPRC_1", "ACVOL_UNS"] if c not in df.columns]
            if missing:
                msg = f"Missing expected fields {missing} for {equity_ric}"
                logger.error(msg)
                if self.strict_errors:
                    raise RuntimeError(msg)
                return None

            df_clean = pd.DataFrame(
                {
                    "timestamp": pd.to_datetime(df.index),
                    "open": df["OPEN_PRC"].astype(float),
                    "high": df["HIGH_1"].astype(float),
                    "low": df["LOW_1"].astype(float),
                    "close": df["TRDPRC_1"].astype(float),
                    "volume": df["ACVOL_UNS"].astype(int),
                }
            )

            return df_clean.sort_values("timestamp").reset_index(drop=True)

        except Exception as exc:
            logger.error("❌ Error fetching OHLC for %s: %s", equity_ric, exc)
            if self.strict_errors:
                raise
            return None

    # -------------------------------------------------------------------------
    # Higher-level helpers
    # -------------------------------------------------------------------------

    def _default_lookback_days_for_interval(self, interval: str) -> int:
        """
        Heuristic lookback in days per interval.

        Refinitiv intraday DB typically stores up to ~1 year of minute data,
        while daily data can go back much further.[web:81][web:85]
        """
        interval = interval.lower()

        # Intraday (minute/hour) – aim for 365 days; the backend will clip
        if interval in {"minute", "1min", "5min", "10min", "30min", "60min", "hourly", "1h"}:
            return 365

        # Daily / weekly – ask for several years
        if interval in {"daily", "1d", "1d", "7d", "weekly", "1w"}:
            return 365 * 5  # 5 years

        # Monthly / quarterly / yearly – longer window is fine
        if interval in {"monthly", "1m", "quarterly", "3m", "6m", "yearly", "12m", "1y"}:
            return 365 * 15  # 15 years

        # Fallback
        return 365

    def get_past_year_ohlc(
        self,
        equity: str,
        *,
        interval: str = "1min",
    ) -> Optional[pd.DataFrame]:
        """
        Fetch as much history as possible for a given interval, up to a
        reasonable lookback window per interval (e.g. ~1 year for 1min).

        Tries NASDAQ (".O") then NYSE (".N").
        """
        end = datetime.utcnow()
        days = self._default_lookback_days_for_interval(interval)
        start = end - timedelta(days=days)

        for suffix in [".O", ".N"]:
            ric = equity + suffix
            logger.info(
                "Attempting to fetch %s from %s to %s with interval=%s",
                ric,
                start.isoformat(timespec="seconds"),
                end.isoformat(timespec="seconds"),
                interval,
            )
            df = self.get_ohlc_df(ric, start, end, interval=interval)
            if df is not None and not df.empty:
                logger.info(
                    "✅ Successfully fetched %d rows for %s (interval=%s)",
                    len(df),
                    ric,
                    interval,
                )
                return df

        logger.warning("No OHLC data found for equity %s (interval=%s)", equity, interval)
        return None

    # -------------------------------------------------------------------------
    # Bridge to MongoDB time-series service
    # -------------------------------------------------------------------------

    def get_ohlc_df_for_mongo(
        self,
        equity: str,
        *,
        interval: str = "1d",
    ) -> Optional[pd.DataFrame]:
        """
        Fetch 1-year OHLCV and return a DataFrame in the exact shape expected
        by MongoDBService.push_timeseries_df: columns are
        [timestamp, open, high, low, close, volume].

        This is effectively an alias to get_past_year_ohlc, but named to make
        its purpose explicit in your ingestion pipeline.
        """
        return self.get_past_year_ohlc(equity, interval=interval)
    def get_dataset_all_equities(self) -> dict:
        """
        Fetch datasets for all equities defined in configs/entities.json.

        Returns a dict mapping equity symbols to their OHLCV DataFrames.
        """
        with open("configs/entities.json", "r") as f:
            equities = json.load(f)

        for interval in SUPPORTED_INTERVALS:
            datasets = {}
            for equity in equities:
                data={}
                logger.info(f"Fetching dataset for {equity} at interval {interval}...")
                df = self.get_ohlc_df_for_mongo(equity, interval=interval)
                if df is not None and not df.empty:
                    data[equity] = df
                    logger.info(f"✅ Fetched dataset for {equity} at interval {interval}")
                else:
                    logger.warning(f"No data found for {equity} at interval {interval}")
            datasets[interval] = data
        return datasets
    
    def download_and_store_all_equities(self) -> None:
        """
        Fetch datasets for all equities and store them in MongoDB.

        This is a high-level helper that combines get_dataset_all_equities with
        MongoDBService.push_timeseries_df for each equity and interval.
        """
        equities= json.load(open("configs/entities.json", "r"))
        frequencies=SUPPORTED_INTERVALS
        for frequency in frequencies:
            if not os.path.exists(f"docs/ts_dataset/{frequency}"):
                os.mkdir(f"docs/ts_dataset/{frequency}")
            for equity in equities:
                logger.info(f"Processing {equity} at interval {frequency}...")
                if  os.path.exists(f"docs/ts_dataset/{frequency}/{equity}.csv"):
                    logger.info(f"Dataset for {equity} at interval {frequency} already exists, skipping download.")
                    continue
                df = self.get_ohlc_df_for_mongo(equity, interval=frequency)
                if df is not None and not df.empty:
                    df.to_csv(f"docs/ts_dataset/{frequency}/{equity}.csv", index=False)
                    logger.info(f"✅ Stored dataset for {equity} at interval {frequency}")
                else:
                    logger.warning(f"No data to store for {equity} at interval {frequency}")

                
if __name__ == "__main__":
    # Example usage
    with RefinitivService() as service:
        # df = service.get_past_year_ohlc("AAPL")
        # if df is not None:
        #     print(df.head())
        # data=service.get_dataset_all_equities()
        service.download_and_store_all_equities()
