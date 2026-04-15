import io
import os
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import torch
from src.utils.env import load_env
from pymongo import ASCENDING, DESCENDING, MongoClient, UpdateOne
from pymongo.collection import Collection
from pymongo.errors import BulkWriteError, ConnectionFailure

# -----------------------------------------------------------------------------
# Environment & Logging
# -----------------------------------------------------------------------------

load_env()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Data Models
# -----------------------------------------------------------------------------

@dataclass
class TimeSeriesData:
    """OHLCV candle for a single equity at a given timestamp."""
    equity: str
    frequency: str  # e.g. '1m', '5m', '15m', '1h', '1d'
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    additional_data: Optional[Dict[str, Any]] = None


# -----------------------------------------------------------------------------
# MongoDB Service
# -----------------------------------------------------------------------------

class MongoDBService:
    """High-level MongoDB access layer for agents, time series, and news."""

    SUPPORTED_FREQUENCIES: List[str] = [
    "minute",
    "1min",
    "5min",
    "10min",
    "30min",
    "60min",
    "hourly",
    "1h",
    "daily",
    "1d",
    "1D",
    "7D",
    "7d",
    "weekly",
    "1W",
    "monthly",
    "1M",
    "quarterly",
    "3M",
    "6M",
    "yearly",
    "12M",
    "1Y",
]  # from Refinitiv docs[web:28][web:29]

    def __init__(
        self,
        uri: Optional[str] = None,
        database: Optional[str] = None,
        *,
        strict_errors: bool = False,
    ) -> None:
        """
        Initialize MongoDB service.

        Args:
            uri: MongoDB connection URI (uses MONGO_URI if None).
            database: DB name (uses MONGO_DATABASE or 'fiking_trader' if None).
            strict_errors: If True, re-raise DB exceptions instead of returning fallbacks.
        """
        self.uri = uri or os.getenv("MONGO_URI", "mongodb://localhost:27017/")
        self.database_name = database or os.getenv("MONGO_DATABASE", "fiking_trader")
        self.strict_errors = strict_errors

        self.client: MongoClient = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
        self.db = self.client[self.database_name]

        self._connect_and_init()
        self.host=os.getenv("MONGO_HOST", "localhost")

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _connect_and_init(self) -> None:
        """Connect and run basic initialization (indexes)."""
        try:
            self.client.admin.command("ping")
            logger.info("✅ Connected to MongoDB: %s", self.database_name)
            self._setup_indexes()
        except Exception as exc:
            logger.error("❌ Failed to initialize MongoDB: %s", exc)
            if isinstance(exc, ConnectionFailure) or self.strict_errors:
                raise

    def _setup_indexes(self) -> None:
        """Create indexes used by this service."""
        try:
            # agents_weights indexes
            weights = self.db["agents_weights"]
            weights.create_index([("model_id", ASCENDING)], unique=True, name="model_id_unique")
            weights.create_index([("agent_name", ASCENDING)], name="agent_name_idx")
            weights.create_index([("equity", ASCENDING)], name="equity_idx")
            weights.create_index([("created_at", DESCENDING)], name="created_at_desc")

            # timeseries_* indexes
            for freq in self.SUPPORTED_FREQUENCIES:
                coll = self.db[f"timeseries_{freq}"]
                coll.create_index(
                    [("equity", ASCENDING), ("timestamp", ASCENDING)],
                    unique=True,
                    name="equity_timestamp_unique",
                )

            logger.info("✅ Database indexes ensured")
        except Exception as exc:
            logger.warning("⚠️ Error creating indexes: %s", exc)
            if self.strict_errors:
                raise

    @staticmethod
    def _normalize_candle_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a raw OHLCV MongoDB document to JSON-friendly numerics."""
        doc["_id"] = str(doc["_id"])
        doc["open"] = float(doc["open"])
        doc["high"] = float(doc["high"])
        doc["low"] = float(doc["low"])
        doc["close"] = float(doc["close"])
        doc["volume"] = int(doc.get("volume", 0))
        return doc

    def _get_timeseries_collection(self, frequency: str) -> Collection:
        if frequency not in self.SUPPORTED_FREQUENCIES:
            raise ValueError(f"Unsupported frequency: {frequency}")
        return self.db[f"timeseries_{frequency}"]

    # -------------------------------------------------------------------------
    # Agent Weights: PUSH / GET
    # -------------------------------------------------------------------------

    def push_agent_weights(
        self,
        model_id: str,
        agent_name: str,
        weights: Any,
        *,
        version: str = "v1",
        equity: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        performance_metrics: Optional[Dict[str, float]] = None,
    ) -> bool:
        """Upsert serialized model weights for an agent."""
        if self.host=="localhost":
            try:
                logger.warning("⚠️ You are saving agent weights to a local MongoDB instance. Make sure this is intentional.")
                os.makedirs("data/models", exist_ok=True)
                torch.save(weights, f"data/models/{model_id}_{version}.pt")
                logger.info("✅ Saved weights for agent %s to local file", model_id)
                return True
            except Exception as exc:
                logger.error("❌ Error saving agent weights locally: %s", exc)
                if self.strict_errors:
                    raise
                return False

        try:
            buf = io.BytesIO()
            torch.save(weights, buf)
            weights_bytes = buf.getvalue()

            now = datetime.utcnow()
            payload: Dict[str, Any] = {
                "model_id": model_id,
                "agent_name": agent_name,
                "version": version,
                "weights_data": weights_bytes,
                "metadata": metadata or {},
                "equity": equity,
                "performance_metrics": performance_metrics or {},
                "training_date": now,
                "updated_at": now,
            }

            result = self.db["agents_weights"].update_one(
                {"model_id": model_id},
                {"$set": payload, "$setOnInsert": {"created_at": now}},
                upsert=True,
            )
            ok = bool(result.upserted_id or result.modified_count)
            if ok:
                logger.info("✅ Saved weights for agent %s", model_id)
            else:
                logger.warning("⚠️ No changes when saving weights for agent %s", model_id)
            return ok

        except Exception as exc:
            logger.error("❌ Error saving agent weights: %s", exc)
            if self.strict_errors:
                raise
            return False

    def get_agent_weights(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Load agent model weights and metadata from the database."""
        if self.host=="localhost":
            logger.warning("⚠️ You are loading agent weights from a local MongoDB instance. Make sure this is intentional.")
            try:
                weights = torch.load(f"data/models/{model_id}_v1.pt", map_location=torch.device("cpu"))
                return {
                    "model_id": model_id,
                    "agent_name": model_id.split("_")[0],
                    "version": "v1",
                    "weights": weights,
                    "metadata": {},
                    "equity": None,
                    "performance_metrics": {},
                    "training_date": None,
                    "created_at": None,
                    "updated_at": None,
                }
            except FileNotFoundError:
                logger.error("❌ Local weights file not found for model_id: %s", model_id)
                return None
        try:
            doc = self.db["agents_weights"].find_one({"model_id": model_id})
            if not doc:
                logger.warning("⚠️ Agent not found: %s", model_id)
                return None

            weights = torch.load(
                io.BytesIO(doc["weights_data"]),
                map_location="cpu",
            )

            return {
                "model_id": doc["model_id"],
                "agent_name": doc["agent_name"],
                "version": doc["version"],
                "weights": weights,
                "metadata": doc.get("metadata", {}),
                "equity": doc.get("equity"),
                "performance_metrics": doc.get("performance_metrics", {}),
                "training_date": doc.get("training_date"),
                "created_at": doc.get("created_at"),
                "updated_at": doc.get("updated_at"),
            }

        except Exception as exc:
            logger.error("❌ Error loading agent weights: %s", exc)
            if self.strict_errors:
                raise
            return None

    def get_agents(self, equity: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all agents, optionally filtered by equity (without weights blobs)."""
        try:
            query: Dict[str, Any] = {}
            if equity:
                query["equity"] = equity

            cursor = (
                self.db["agents_weights"]
                .find(query, {"weights_data": 0})
                .sort("updated_at", DESCENDING)
            )

            agents: List[Dict[str, Any]] = []
            for doc in cursor:
                doc["_id"] = str(doc["_id"])
                agents.append(doc)
            return agents

        except Exception as exc:
            logger.error("❌ Error listing agents: %s", exc)
            if self.strict_errors:
                raise
            return []

    def delete_agent(self, model_id: str) -> bool:
        """Delete an agent's weights and metadata."""
        try:
            result = self.db["agents_weights"].delete_one({"model_id": model_id})
            if result.deleted_count:
                logger.info("✅ Deleted agent %s", model_id)
                return True
            logger.warning("⚠️ Agent not found for deletion: %s", model_id)
            return False
        except Exception as exc:
            logger.error("❌ Error deleting agent: %s", exc)
            if self.strict_errors:
                raise
            return False

    # -------------------------------------------------------------------------
    # News & Embeddings: GET
    # -------------------------------------------------------------------------

    def get_news(
        self,
        equity: str,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Fetch news articles for an equity within an optional date range."""
        try:
            query: Dict[str, Any] = {"Stock_symbol": equity.lower()}
            if start or end:
                query["Date_parsed"] = {}
                if start:
                    query["Date_parsed"]["$gte"] = start
                if end:
                    query["Date_parsed"]["$lte"] = end

            cursor = self.db["news_articles"].find(query).sort("Date_parsed", DESCENDING)
            if limit is not None:
                cursor = cursor.limit(limit)

            return [{**doc, "_id": str(doc["_id"])} for doc in cursor]

        except Exception as exc:
            logger.error("❌ Error fetching news: %s", exc)
            if self.strict_errors:
                raise
            return []

    def get_news_embeddings(
        self,
        equity: str,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Return pre-computed embeddings aligned to a time range.

        Returns:
            {
                "timestamps": List[datetime],
                "embeddings": torch.Tensor  # shape (N, dim) or empty (0,)
            }
        """
        articles = self.get_news(equity, start=start, end=end)
        print(len(articles))
        timestamps: List[datetime] = []
        embeddings: List[torch.Tensor] = []

        for doc in articles:
            emb = doc.get("embedding")
            if emb is None:
                continue

            ts = doc.get("Date_parsed") or doc.get("Date")
            if ts is None:
                continue

            timestamps.append(ts)
            embeddings.append(emb if isinstance(emb, torch.Tensor) else torch.tensor(emb))

        if not embeddings:
            return {"timestamps": [], "embeddings": torch.empty(0)}

        return {"timestamps": timestamps, "embeddings": torch.stack(embeddings)}

    # -------------------------------------------------------------------------
    # Time Series: PUSH / GET
    # -------------------------------------------------------------------------

    def push_timeseries_batch(
        self,
        equity: str,
        frequency: str,
        data_points: Iterable[TimeSeriesData],
        *,
        batch_size: int = 10_000,
    ) -> bool:
        """
        Upsert a batch of candles for a given equity/frequency.

        Uses bulk_write with unordered batches for high throughput.
        """
        coll = self._get_timeseries_collection(frequency)
        points = list(data_points)
        if not points:
            return True

        try:
            # Optional manual batching for very large backfills
            for i in range(0, len(points), batch_size):
                chunk = points[i : i + batch_size]
                ops = [
                    UpdateOne(
                        {"equity": equity, "timestamp": dp.timestamp},
                        {
                            "$set": {
                                "equity": equity,
                                "frequency": frequency,
                                "timestamp": dp.timestamp,
                                "open": float(dp.open),
                                "high": float(dp.high),
                                "low": float(dp.low),
                                "close": float(dp.close),
                                "volume": int(dp.volume),
                                **(dp.additional_data or {}),
                            }
                        },
                        upsert=True,
                    )
                    for dp in chunk
                ]
                try:
                    result = coll.bulk_write(ops, ordered=False)
                    logger.info(
                        "✅ %s/%s: %d new, %d updated (%d candles)",
                        equity,
                        frequency,
                        result.upserted_count,
                        result.modified_count,
                        len(ops),
                    )
                except BulkWriteError as bwe:
                    logger.error("❌ BulkWriteError in push_timeseries_batch: %s", bwe.details)
                    if self.strict_errors:
                        raise

            return True

        except Exception as exc:
            logger.error("❌ Error saving timeseries: %s", exc)
            if self.strict_errors:
                raise
            return False

    def push_timeseries_df(
        self,
        equity: str,
        frequency: str,
        df: pd.DataFrame,
        *,
        batch_size: int = 10_000,
    ) -> bool:
        """
        Convenience wrapper to push a pandas DataFrame of OHLCV candles.

        Expected columns: timestamp, open, high, low, close, volume.
        Index is ignored unless it is named 'timestamp'.
        """
        coll = self._get_timeseries_collection(frequency)  # validates frequency
        _ = coll  # silence unused; we just want validation here

        if "timestamp" not in df.columns:
            if df.index.name == "timestamp":
                df = df.reset_index()
            else:
                raise ValueError(
                    "DataFrame must contain 'timestamp' column or have index named 'timestamp'"
                )

        required = {"timestamp", "open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        # Normalize timestamps
        ts = pd.to_datetime(df["timestamp"], errors="raise")
        if getattr(ts.dt, "tz", None) is not None:
            ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)

        df = df.copy()
        df["timestamp"] = ts.dt.to_pydatetime()

        data_points = [
            TimeSeriesData(
                equity=equity,
                frequency=frequency,
                timestamp=row.timestamp,
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=int(row.volume),
            )
            for row in df[["timestamp", "open", "high", "low", "close", "volume"]].itertuples(
                index=False
            )
        ]

        return self.push_timeseries_batch(equity, frequency, data_points, batch_size=batch_size)

    def get_timeseries(
        self,
        equity: str,
        frequency: str,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve OHLCV candles for an equity/frequency, sorted ascending by timestamp."""
        try:
            coll = self._get_timeseries_collection(frequency)
            query: Dict[str, Any] = {"equity": equity}

            if start or end:
                query["timestamp"] = {}
                if start:
                    query["timestamp"]["$gte"] = start
                if end:
                    query["timestamp"]["$lte"] = end

            cursor = coll.find(query).sort("timestamp", ASCENDING)
            if limit is not None:
                cursor = cursor.limit(limit)

            return [self._normalize_candle_doc(doc) for doc in cursor]

        except Exception as exc:
            logger.error("❌ Error retrieving timeseries: %s", exc)
            if self.strict_errors:
                raise
            return []

    def get_latest_candle(
        self,
        equity: str,
        frequency: str = "1m",
    ) -> Optional[Dict[str, Any]]:
        """Return the most recent OHLCV candle for an equity/frequency."""
        try:
            coll = self._get_timeseries_collection(frequency)
            doc = coll.find_one({"equity": equity}, sort=[("timestamp", DESCENDING)])
            return self._normalize_candle_doc(doc) if doc else None
        except Exception as exc:
            logger.error("❌ Error getting latest candle: %s", exc)
            if self.strict_errors:
                raise
            return None

    def get_equities(self, frequency: str = "1d") -> List[str]:
        """List distinct equities that have data for a given frequency."""
        try:
            coll = self._get_timeseries_collection(frequency)
            equities = coll.distinct("equity")
            return sorted(equities)
        except Exception as exc:
            logger.error("❌ Error listing equities: %s", exc)
            if self.strict_errors:
                raise
            return []

    def delete_timeseries(
        self,
        equity: str,
        frequency: str,
        *,
        start: Optional[datetime] = None,
        end: Optional[datetime] = None,
    ) -> int:
        """Delete OHLCV candles for an equity/frequency in an optional date range."""
        try:
            coll = self._get_timeseries_collection(frequency)
            query: Dict[str, Any] = {"equity": equity}
            if start or end:
                query["timestamp"] = {}
                if start:
                    query["timestamp"]["$gte"] = start
                if end:
                    query["timestamp"]["$lte"] = end

            result = coll.delete_many(query)
            logger.info(
                "✅ Deleted %d records for %s at %s", result.deleted_count, equity, frequency
            )
            return result.deleted_count

        except Exception as exc:
            logger.error("❌ Error deleting timeseries: %s", exc)
            if self.strict_errors:
                raise
            return 0

    def get_timeseries_stats(self, equity: str, frequency: str) -> Dict[str, Any]:
        """Return simple stats (count, oldest, newest) for an equity/frequency."""
        try:
            coll = self._get_timeseries_collection(frequency)

            count = coll.count_documents({"equity": equity})
            oldest = coll.find_one({"equity": equity}, sort=[("timestamp", ASCENDING)])
            newest = coll.find_one({"equity": equity}, sort=[("timestamp", DESCENDING)])

            return {
                "equity": equity,
                "frequency": frequency,
                "total_records": count,
                "oldest_date": oldest["timestamp"] if oldest else None,
                "newest_date": newest["timestamp"] if newest else None,
            }

        except Exception as exc:
            logger.error("❌ Error getting timeseries stats: %s", exc)
            if self.strict_errors:
                raise
            return {}

    # -------------------------------------------------------------------------
    # Health / Utility
    # -------------------------------------------------------------------------

    def get_connection_status(self) -> Dict[str, Any]:
        """Return basic connection and DB stats."""
        try:
            self.client.admin.command("ping")
            stats = self.db.command("dbStats")
            return {
                "connected": True,
                "database": self.database_name,
                "collections": self.db.list_collection_names(),
                "size_mb": round(stats.get("dataSize", 0) / (1024 * 1024), 2),
                "supported_frequencies": list(self.SUPPORTED_FREQUENCIES),
            }
        except Exception as exc:
            logger.error("❌ Error getting connection status: %s", exc)
            if self.strict_errors:
                raise
            return {"connected": False, "error": str(exc)}

    def check_ready(self) -> Dict[str, Any]:
        """
        Verify DB connectivity and indexes.

        Returns:
            {
                "ok": bool,
                "issues": List[str],
                "details": Dict[str, Any]
            }
        """
        issues: List[str] = []
        details: Dict[str, Any] = {}

        try:
            self.client.admin.command("ping")
        except Exception as exc:
            return {"ok": False, "issues": [f"ping failed: {exc}"], "details": {}}

        # agents_weights index check
        try:
            info = self.db["agents_weights"].index_information()
            details["agents_weights_indexes"] = list(info.keys())
            has_unique_agent = any(
                idx.get("unique") is True and idx.get("key") == [("model_id", 1)]
                for idx in info.values()
            )
            if not has_unique_agent:
                issues.append("agents_weights missing unique index on model_id")
        except Exception as exc:
            issues.append(f"could not inspect agents_weights indexes: {exc}")

        # timeseries_* index check
        missing_ts_indexes: List[str] = []
        for freq in self.SUPPORTED_FREQUENCIES:
            coll_name = f"timeseries_{freq}"
            try:
                idx_info = self.db[coll_name].index_information()
                has_unique = any(
                    idx.get("unique") is True
                    and idx.get("key") == [("equity", 1), ("timestamp", 1)]
                    for idx in idx_info.values()
                )
                if not has_unique:
                    missing_ts_indexes.append(coll_name)
            except Exception as exc:
                issues.append(f"could not inspect {coll_name} indexes: {exc}")

        if missing_ts_indexes:
            issues.append(
                "missing unique (equity, timestamp) index on: " + ", ".join(missing_ts_indexes)
            )

        return {"ok": not issues, "issues": issues, "details": details}

    def close(self) -> None:
        """Close the underlying MongoDB client."""
        if self.client:
            self.client.close()
            logger.info("🔌 MongoDB connection closed")


def update_timeseries_dataset():
    from src.external_api.live import RefinitivService
    import json
    rd_service = RefinitivService()
    service = MongoDBService()
    equities = json.load(open("configs/entities.json", "r"))
    metaData = json.load(open("docs/ts_dataset/metaData.json", "r"))

    for frequency in MongoDBService.SUPPORTED_FREQUENCIES:
        for equity in equities:
       
            if metaData.get(equity,{}).get(frequency):
                logger.info(f"Dataset for {equity} at {frequency} is up to date. Skipping...")
                continue

            logger.info(f"Updating dataset for {equity} at {frequency}...")
            try:
                df = rd_service.get_ohlc_df_for_mongo(equity, interval=frequency)
                if df is not None and not df.empty:
                    service.push_timeseries_df(equity, frequency, df)
                    logger.info(f"✅ Updated dataset for {equity} at {frequency}")
                    
                else:
                    logger.warning(f"⚠️ No data to update for {equity} at {frequency}")

                metaData[equity][frequency] = True
            except Exception as exc:
                logger.error(f"❌ Error updating dataset for {equity} at {frequency}: {exc}")
        
    with open("docs/ts_dataset/metaData.json", "w") as f:
        json.dump(metaData, f, indent=4)

def update_listed_equities(equities,service, frequency: str = "1d") -> None:
    from src.external_api.live import RefinitivService

    rd= RefinitivService()
    for equity in equities:
        try:
            ts=rd.get_ohlc_df_for_mongo(equity, interval=frequency)
            if ts is not None and not ts.empty:
                service.push_timeseries_df(equity, frequency, ts)
                logger.info(f"✅ Updated dataset for {equity} at {frequency}")
            else:
                logger.warning(f"⚠️ No data to update for {equity} at {frequency}")

        except Exception as exc:
            logger.error(f"❌ Error updating dataset for {equity} at {frequency}: {exc}")
    
def get_all_ts_stats(service: MongoDBService, equities: List[str], frequencies: List[str]) -> Dict[str, Any]:
    stats = {}
    for equity in equities:
        stats[equity] = {}
        for freq in frequencies:
            stats[equity][freq] = service.get_timeseries_stats(equity, freq)
            print(f"Stats for {equity} at {freq}: {stats[equity][freq]}")
    return stats
if __name__ == "__main__":
    # update_timeseries_dataset()
    # equities = json.load(open("configs/entities.json", "r"))
    # frequencies = MongoDBService.SUPPORTED_FREQUENCIES
    # stats = get_all_ts_stats(service, list(equities.keys())[:200], frequencies)
    # json.dump(stats, open("timeseries_stats.json", "w"), indent=4, default=str)

    # srvice = MongoDBService()
    # update_listed_equities(["AAPL", "MSFT"], srvice, frequency="1min")
    # print(get_all_ts_stats(MongoDBService(), ["aapl", "MSFT"], ["1min","1d", "1h"]))
    service = MongoDBService()
    res=service.get_news_embeddings(equity='AAPL', start=None, end=None)
    print(res['embeddings'])