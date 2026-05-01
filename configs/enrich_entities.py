#!/usr/bin/env python3
"""
enrich_entities.py
------------------
Reads entities.json ({"ticker": index, ...}) and enriches each ticker with
metadata from yfinance. Saves results to entities_enriched.json.

Fields added per ticker:
    name, long_name, sector, industry, market_cap, currency, exchange,
    country, website, description, employees, pe_ratio, forward_pe,
    dividend_yield, beta, 52w_high, 52w_low, avg_volume, asset_type

Usage:
    pip install yfinance
    python enrich_entities.py [--checkpoint checkpoint.json] [--workers 8]
"""

import json
import os
import time
import argparse
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

import yfinance as yf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FIELDS_MAP = {
    "shortName":           "name",
    "longName":            "long_name",
    "sector":              "sector",
    "industry":            "industry",
    "marketCap":           "market_cap",
    "currency":            "currency",
    "exchange":            "exchange",
    "country":             "country",
    "website":             "website",
    "longBusinessSummary": "description",
    "fullTimeEmployees":   "employees",
    "trailingPE":          "pe_ratio",
    "forwardPE":           "forward_pe",
    "dividendYield":       "dividend_yield",
    "beta":                "beta",
    "fiftyTwoWeekHigh":    "52w_high",
    "fiftyTwoWeekLow":     "52w_low",
    "averageVolume":       "avg_volume",
    "quoteType":           "asset_type",
}


def fetch_ticker(symbol: str, retries: int = 3, delay: float = 1.0) -> dict:
    """Fetch info for a single ticker. Returns partial dict on failure."""
    for attempt in range(retries):
        try:
            info = yf.Ticker(symbol.upper()).info
            result = {}
            for yf_key, out_key in FIELDS_MAP.items():
                val = info.get(yf_key)
                if val not in (None, "None", "N/A", ""):
                    result[out_key] = val
            return result
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(delay * (attempt + 1))
            else:
                log.warning(f"Failed {symbol}: {e}")
                return {}
    return {}


def load_checkpoint(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def save_checkpoint(path: str, data: dict):
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Enrich equity tickers via yfinance.")
    parser.add_argument("--input",      default="configs/entities.json",          help="Input file (default: entities.json)")
    parser.add_argument("--output",     default="configs/entities_enriched.json", help="Output file (default: entities_enriched.json)")
    parser.add_argument("--checkpoint", default="configs/enrich_checkpoint.json", help="Checkpoint file for resuming (default: enrich_checkpoint.json)")
    parser.add_argument("--workers",    type=int, default=8,              help="Concurrent workers (default: 8, reduce if rate-limited)")
    parser.add_argument("--delay",      type=float, default=0.2,          help="Base delay between retries in seconds (default: 0.2)")
    parser.add_argument("--save-every", type=int, default=100,            help="Save checkpoint every N tickers (default: 100)")
    args = parser.parse_args()

    # Load input
    with open(args.input) as f:
        entities: dict = json.load(f)
    tickers = list(entities.keys())
    log.info(f"Loaded {len(tickers)} tickers from {args.input}")

    # Load checkpoint
    enriched: dict = load_checkpoint(args.checkpoint)
    remaining = [t for t in tickers if t not in enriched]
    log.info(f"Already enriched: {len(enriched)}, remaining: {len(remaining)}")

    def worker(ticker):
        data = fetch_ticker(ticker, delay=args.delay)
        return ticker, data

    completed = 0
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(worker, t): t for t in remaining}
        for future in as_completed(futures):
            ticker, data = future.result()
            enriched[ticker] = {
                "index": entities[ticker],
                **data
            }
            completed += 1
            if completed % args.save_every == 0:
                save_checkpoint(args.checkpoint, enriched)
                log.info(f"Progress: {len(enriched)}/{len(tickers)} ({100*len(enriched)/len(tickers):.1f}%)")

    # Final save — output is ordered by original index
    ordered = dict(sorted(enriched.items(), key=lambda x: x[1].get("index", 0)))
    with open(args.output, "w") as f:
        json.dump(ordered, f, indent=2)
    save_checkpoint(args.checkpoint, enriched)

    log.info(f"Done. Output saved to {args.output}")
    log.info(f"Enriched {sum(1 for v in enriched.values() if len(v) > 1)} tickers with data out of {len(tickers)}")


if __name__ == "__main__":
    main()
