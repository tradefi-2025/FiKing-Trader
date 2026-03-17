"""
Refinitiv Price Data Service
Interface for fetching historical and real-time OHLC data using refinitiv-data library.

Environment Variables:
- REFINITIV_API_KEY: Your Refinitiv API key
- REFINITIV_HOSTNAME: Refinitiv Data Platform hostname (optional, defaults to platform.refinitiv.com)
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import json
# Load .env from project root
env_path = Path(__file__).resolve().parents[2] / ".env"
print(f"Looking for .env at: {env_path}")
print(f"File exists: {env_path.exists()}")
load_dotenv(env_path)
print(f"REFINITIV_API_KEY loaded: {bool(os.getenv('REFINITIV_API_KEY'))}")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


try:
    import refinitiv.data as rd
    # Configure Refinitiv from environment variables
    _refinitiv_config = {
        "sessions": {
            "default": "platform.rdp",
            "platform": {
                "rdp": {
                    "app-key": os.getenv("REFINITIV_API_KEY"),
                    "username": os.getenv("REFINITIV_USERNAME"),
                    "password": os.getenv("REFINITIV_PASSWORD")
                }
            }
        }
    }
except ImportError:
    rd = None  # Refinitiv not available
    _refinitiv_config = None






class RefinitivService:
    """Refinitiv API client using refinitiv-data library for OHLC data"""

    def __init__(self):

        self.session_open = False

        self._open_session()

    def _open_session(self) -> bool:
        """Open Refinitiv session using environment variables"""
        if rd is None:
            raise ImportError("Refinitiv library not available")
        # Write temp config file (Refinitiv requires file-based config)
        if not self.session_open:
            config_path = os.path.join(os.path.dirname(__file__), '.refinitiv-config.json')
            with open(config_path, 'w') as f:
                json.dump(_refinitiv_config, f)
            rd.open_session(config_name=config_path)
            self.session_open = True
            logger.info("✓ Refinitiv session opened")
        return True
    

    def get_ohlc(
        self, equity: str, start: datetime, end: datetime,frequency: str = "1min"
    ) -> Optional[pd.DataFrame]:
        """
        Get OHLC data for an equity between start and end dates.

        Args:
            equity: RIC code (e.g., "AAPL.O", "MSFT.O")
            start: Start datetime
            end: End datetime

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
            or None on error.
        """
        if not self.session_open:
            logger.error("Refinitiv session not open")
            return None

        try:
            df = rd.get_history(
                universe=equity,
                fields=["OPEN_PRC", "HIGH_1", "LOW_1", "TRDPRC_1", "ACVOL_UNS"],
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
                interval=frequency
            )
            print(f"Raw data for {equity}:\n{df.head()}")
            if df is None or df.empty:
                logger.warning(f"No data returned for {equity}")
                return None

            # Rename columns to standard format
            df_clean = pd.DataFrame({
                "timestamp": pd.to_datetime(df.index),
                "open": df.get("OPEN_PRC", 0).astype(float),
                "high": df.get("HIGH_1", 0).astype(float),
                "low": df.get("LOW_1", 0).astype(float),
                "close": df.get("TRDPRC_1", 0).astype(float),
                "volume": df.get("ACVOL_UNS", 0).astype(float),
            })
            
            return df_clean.sort_values("timestamp").reset_index(drop=True)

        except Exception as e:
            logger.error(f"Error fetching {equity}: {e}")
            return None

    def test_ohlc_fetch(self):
        """Test OHLC fetch for last 30 days"""
        start = datetime.now() - timedelta(days=30)
        end = datetime.now()
        df = self.get_ohlc("AAPL.O", start, end)
        if df is not None:
            logger.info(f"✓ Successfully fetched {len(df)} records for AAPL.O")
            print(df.head())
        else:
            logger.error("Failed to fetch OHLC data for AAPL.O")

    def get_past_year_ohlc(self, equity: str) -> Optional[pd.DataFrame]:
        """
        Get OHLC data for an equity over the past year.

        Args:
            equity: Ticker symbol (e.g., "AAPL", "MSFT") — suffix auto-detected

        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
            or None on error.
        """
        end = datetime.now()
        start = end - timedelta(days=365)
        
        # Try NASDAQ then NYSE
        for suffix in [".O", ".N"]:
            ric = equity + suffix
            logger.info(f"Attempting to fetch {ric}...")
            df = self.get_ohlc(ric, start, end)
            if df is not None and not df.empty:
                logger.info(f"✓ Successfully fetched data for {ric}")
                return df
        
        logger.warning(f"No data found for {equity}")
        return None

    def close_session(self):
        """Close Refinitiv session"""
        try:
            rd.close_session()
            self.session_open = False
            logger.info("✓ Refinitiv session closed")
        except Exception as e:
            logger.error(f"Error closing session: {e}")


if __name__ == "__main__":
    svc = RefinitivService()
    df = svc.get_past_year_ohlc("AAPL")
    if df is not None:
        logger.info(f"Retrieved {len(df)} records")
        print(df.head())
    else:
        logger.error("Failed to fetch data")
    svc.close_session()
