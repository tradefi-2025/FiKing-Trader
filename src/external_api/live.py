"""
Refinitiv Price Data Service
Simple interface for fetching historical OHLC data.

Environment Variables:
- REFINITIV_API_KEY: Your Refinitiv API key
- REFINITIV_USERNAME: Username (optional, for password flow)
- REFINITIV_PASSWORD: Password (optional, for password flow)
"""

import os
import logging
from datetime import datetime, timedelta
from typing import Optional
from pathlib import Path
import requests
import pandas as pd
from dotenv import load_dotenv

# Load .env from project root
env_path = Path(__file__).resolve().parents[2] / ".env"
print(f"Looking for .env at: {env_path}")
print(f"File exists: {env_path.exists()}")
load_dotenv(env_path)
print(f"REFINITIV_API_KEY loaded: {bool(os.getenv('REFINITIV_API_KEY'))}")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RefinitivService:
    """Simple Refinitiv API client for OHLC data"""

    def __init__(self):
        self.api_key = os.getenv('REFINITIV_API_KEY')
        self.username = os.getenv('REFINITIV_USERNAME')
        self.password = os.getenv('REFINITIV_PASSWORD')
        self.base_url = os.getenv('REFINITIV_BASE_URL', 'https://api.refinitiv.com')
        self.token = None

        if not self.api_key:
            raise ValueError("REFINITIV_API_KEY required in .env")

    def _authenticate(self) -> bool:
        """Get access token"""
        if self.token:
            return True

        auth_data = {
            "grant_type": "client_credentials",
            "client_id": self.api_key,
            "scope": "trapi"
        }
        if self.username and self.password:
            auth_data.update({
                "grant_type": "password",
                "username": self.username,
                "password": self.password,
            })

        try:
            resp = requests.post(
                f"{self.base_url}/auth/oauth2/v1/token",
                data=auth_data,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            if resp.status_code == 200:
                self.token = resp.json().get("access_token")
                return True
            logger.error(f"Auth failed: {resp.status_code} - {resp.text}")
        except Exception as e:
            logger.error(f"Auth error: {e}")
        return False

    def get_ohlc(
        self, equity: str, start: datetime, end: datetime
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
        if not self._authenticate():
            return None

        try:
            endpoint = f"{self.base_url}/data/historical-pricing/v1/views/interday-summaries/{equity}"
            params = {
                "start": start.strftime("%Y-%m-%d"),
                "end": end.strftime("%Y-%m-%d"),
                "fields": "TRDPRC_1,OPEN_PRC,HIGH_1,LOW_1,ACVOL_UNS",
                "interval": "P1D",
            }
            headers = {"Authorization": f"Bearer {self.token}"}

            resp = requests.get(endpoint, params=params, headers=headers)
            if resp.status_code != 200:
                logger.error(f"API error: {resp.status_code} - {resp.text}")
                return None

            data = resp.json().get("data", [])
            if not data:
                logger.warning(f"No data for {equity}")
                return None

            rows = []
            for row in data:
                rows.append({
                    "timestamp": pd.to_datetime(row.get("date")),
                    "open": float(row.get("OPEN_PRC", 0)),
                    "high": float(row.get("HIGH_1", 0)),
                    "low": float(row.get("LOW_1", 0)),
                    "close": float(row.get("TRDPRC_1", 0)),
                    "volume": float(row.get("ACVOL_UNS", 0)),
                })

            df = pd.DataFrame(rows)
            return df.sort_values("timestamp").reset_index(drop=True)

        except Exception as e:
            logger.error(f"Error fetching {equity}: {e}")
            return None
    def test_ohlc_fetch(self):
        start= datetime.now() - timedelta(days=30)
        end= datetime.now()
        df= self.get_ohlc("AAPL.O", start, end)
        if df is not None:
            print(df.head())
        else:
            print("Failed to fetch OHLC data for AAPL.O")


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
            df = self.get_ohlc(equity + suffix, start, end)
            if df is not None:
                return df
        return None


if __name__ == "__main__":
    svc = RefinitivService()
    df = svc.test_ohlc_fetch()