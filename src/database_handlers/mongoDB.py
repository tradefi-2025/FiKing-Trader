"""
MongoDB Database Handler
Manages agent weights storage and time series datasets for equities

Database Structure:
- agents_weights: Collection for storing trained model weights
- timeseries_{frequency}: Collections for time series data at different frequencies
  (e.g., timeseries_1m, timeseries_5m, timeseries_1h, timeseries_1d)

Environment Variables Required:
- MONGODB_URI: MongoDB connection string
- MONGODB_DATABASE: Database name (defaults to 'fiking_trader')
"""

import io
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

import pandas as pd
import torch

from pymongo import MongoClient, ASCENDING, DESCENDING, UpdateOne as MongoUpdateOne
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TimeSeriesData:
    """Data class for time series market data"""
    equity: str
    frequency: str  # e.g., '1m', '5m', '15m', '1h', '1d'
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    additional_data: Optional[Dict[str, Any]] = None
    

class MongoDBService:
    """Service for MongoDB database operations"""
    
    # Supported time frequencies
    
    SUPPORTED_FREQUENCIES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M']

    @staticmethod
    def _normalize_candle_doc(doc: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize a raw MongoDB OHLCV document into JSON-friendly numerics."""
        doc['_id'] = str(doc['_id'])
        doc['open'] = float(doc['open'])
        doc['high'] = float(doc['high'])
        doc['low'] = float(doc['low'])
        doc['close'] = float(doc['close'])
        doc['volume'] = int(doc.get('volume', 0))
        return doc
    
    def __init__(self, uri: str = None, database: str = None):
        """
        Initialize MongoDB service
        
        Args:
            uri: MongoDB connection URI (loads from MONGODB_URI env var if not provided)
            database: Database name (loads from MONGODB_DATABASE env var if not provided)
        """
        self.uri = uri or os.getenv('MONGODB_URI', 'mongodb://localhost:27017/')
        self.database_name = database or os.getenv('MONGODB_DATABASE', 'fiking_trader')
        
        self.client = None
        self.db = None
        self._connect()
    
    def _connect(self):
        """Establish connection to MongoDB"""
        try:
            self.client = MongoClient(self.uri, serverSelectionTimeoutMS=5000)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.database_name]
            logger.info(f"✅ Connected to MongoDB: {self.database_name}")
            self._setup_indexes()
        except ConnectionFailure as e:
            logger.error(f"❌ Failed to connect to MongoDB: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"❌ Unexpected error connecting to MongoDB: {str(e)}")
            raise
    
    def _setup_indexes(self):
        """Create indexes for optimized queries"""
        try:
            # Indexes for agent weights collection
            weights_collection = self.db['agents_weights']
            weights_collection.create_index([('agent_id', ASCENDING)], unique=True)
            weights_collection.create_index([('agent_name', ASCENDING)])
            weights_collection.create_index([('equity', ASCENDING)])
            weights_collection.create_index([('created_at', DESCENDING)])
            
            # Indexes for time series collections
            for freq in self.SUPPORTED_FREQUENCIES:
                ts_collection = self.db[f'timeseries_{freq}']
                # Unique compound index — one document per (equity, timestamp) candle.
                # ASCENDING on timestamp so range queries walk the index forward.
                ts_collection.create_index(
                    [('equity', ASCENDING), ('timestamp', ASCENDING)],
                    unique=True,
                    name='equity_timestamp_unique'
                )
                
            logger.info("✅ Database indexes created successfully")
        except Exception as e:
            logger.warning(f"⚠️ Error creating indexes: {str(e)}")
    
    # ==================== Agent Weights Methods ====================
    
    def save_agent_weights(self, 
                          agent_id: str,
                          agent_name: str,
                          weights: Any,
                          version: str = "v1",
                          equity: str = None,
                          metadata: Dict[str, Any] = None,
                          performance_metrics: Dict[str, float] = None) -> bool:
        """
        Save agent model weights to database
        
        Args:
            agent_id: Unique identifier for the agent
            agent_name: Name of the agent
            weights: Model weights (serialized via torch.save)
            version: Model version
            equity: Associated equity symbol
            metadata: Additional metadata
            performance_metrics: Training/evaluation metrics
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Serialize with torch.save — handles tensors, device maps, and
            # custom classes correctly. pickle alone can break on GPU tensors.
            buf = io.BytesIO()
            torch.save(weights, buf)
            weights_bytes = buf.getvalue()
            
            agent_data = {
                'agent_id': agent_id,
                'agent_name': agent_name,
                'version': version,
                'weights_data': weights_bytes,
                'metadata': metadata or {},
                'equity': equity,
                'performance_metrics': performance_metrics or {},
                'training_date': datetime.now(),
                'updated_at': datetime.now()
            }
            
            # Upsert operation
            result = self.db['agents_weights'].update_one(
                {'agent_id': agent_id},
                {'$set': agent_data, '$setOnInsert': {'created_at': datetime.now()}},
                upsert=True
            )
            
            if result.upserted_id or result.modified_count > 0:
                logger.info(f"✅ Saved weights for agent: {agent_id}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"❌ Error saving agent weights: {str(e)}")
            return False
    
    def load_agent_weights(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Load agent model weights from database
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            Dictionary containing weights and metadata, or None if not found
        """
        try:
            result = self.db['agents_weights'].find_one({'agent_id': agent_id})
            
            if result:
                # Deserialize with torch.load — mirrors torch.save above
                weights = torch.load(
                    io.BytesIO(result['weights_data']), weights_only=True
                )
                
                return {
                    'agent_id': result['agent_id'],
                    'agent_name': result['agent_name'],
                    'version': result['version'],
                    'weights': weights,
                    'metadata': result.get('metadata', {}),
                    'equity': result.get('equity'),
                    'performance_metrics': result.get('performance_metrics', {}),
                    'training_date': result.get('training_date'),
                    'created_at': result.get('created_at'),
                    'updated_at': result.get('updated_at')
                }
            
            logger.warning(f"⚠️ Agent not found: {agent_id}")
            return None
            
        except Exception as e:
            logger.error(f"❌ Error loading agent weights: {str(e)}")
            return None
    
    def list_agents(self, equity: str = None) -> List[Dict[str, Any]]:
        """
        List all agents, optionally filtered by equity
        
        Args:
            equity: Filter by equity symbol
            
        Returns:
            List of agent information dictionaries
        """
        try:
            query = {'equity': equity} if equity else {}
            
            results = self.db['agents_weights'].find(
                query,
                {'weights_data': 0}  # Exclude heavy weights data
            ).sort('updated_at', DESCENDING)
            
            agents = []
            for result in results:
                result['_id'] = str(result['_id'])  # Convert ObjectId to string
                agents.append(result)
            
            return agents
            
        except Exception as e:
            logger.error(f"❌ Error listing agents: {str(e)}")
            return []
    
    def delete_agent_weights(self, agent_id: str) -> bool:
        """
        Delete agent weights from database
        
        Args:
            agent_id: Unique identifier for the agent
            
        Returns:
            True if deleted, False otherwise
        """
        try:
            result = self.db['agents_weights'].delete_one({'agent_id': agent_id})
            
            if result.deleted_count > 0:
                logger.info(f"✅ Deleted agent: {agent_id}")
                return True
            
            logger.warning(f"⚠️ Agent not found: {agent_id}")
            return False
            
        except Exception as e:
            logger.error(f"❌ Error deleting agent: {str(e)}")
            return False
    
    # ==================== News / Embeddings Methods ====================

    def get_news(self, equity: str, start: datetime = None,
                 end: datetime = None, limit: int = None) -> List[Dict]:
        """
        Fetch news articles for an equity within a date range.
        Articles are sorted most-recent first.
        """
        query = {'Stock_symbol': equity.lower()}
        if start or end:
            query['Date_parsed'] = {}
            if start:
                query['Date_parsed']['$gte'] = start
            if end:
                query['Date_parsed']['$lte'] = end

        cursor = self.db['news_articles'].find(query).sort('Date_parsed', DESCENDING)
        if limit:
            cursor = cursor.limit(limit)
        return [{**doc, '_id': str(doc['_id'])} for doc in cursor]

    def get_news_embeddings(self, equity: str, start: datetime = None,
                            end: datetime = None) -> Dict[str, Any]:
        """
        Return pre-computed embeddings for news articles aligned to a time range.

        Returns a dict with:
            'timestamps' : List[datetime]   — one per article that has an embedding
            'embeddings' : torch.Tensor     — shape (N, embedding_dim), e.g. (N, 768)

        Zero-tensor placeholder is returned when no articles with embeddings exist.
        """
        articles = self.get_news(equity, start, end)
        timestamps, embeddings = [], []
        for doc in articles:
            emb = doc.get('embedding')
            if emb is not None:
                timestamps.append(doc.get('Date_parsed') or doc.get('Date'))
                embeddings.append(
                    emb if isinstance(emb, torch.Tensor) else torch.tensor(emb)
                )
        if embeddings:
            return {'timestamps': timestamps, 'embeddings': torch.stack(embeddings)}
        return {'timestamps': [], 'embeddings': torch.empty(0)}

    # ==================== Time Series Methods ====================
    
    def save_timeseries_data(self,
                             equity: str,
                             frequency: str,
                             data_points: List[TimeSeriesData]) -> bool:
        """
        Save candles for one equity at one frequency.

        Uses a single bulk_write call — all upserts go in ONE round-trip to
        MongoDB regardless of how many candles are in the batch.  A candle is
        uniquely identified by (equity, timestamp) so the operation is fully
        idempotent: re-running with the same data never creates duplicates.

        For large backfills (e.g. 1-minute bars over 1 year ≈ 130 k candles)
        this is orders of magnitude faster than one update_one() per candle.
        """
        if frequency not in self.SUPPORTED_FREQUENCIES:
            logger.error(f"❌ Unsupported frequency: {frequency}")
            return False
        if not data_points:
            return False

        try:
            operations = [
                MongoUpdateOne(
                    # Filter — the unique key for this candle
                    {'equity': equity, 'timestamp': dp.timestamp},
                    # Payload — create or overwrite all OHLCV fields
                    {'$set': {
                        'equity':    equity,
                        'frequency': frequency,
                        'timestamp': dp.timestamp,
                        'open':      dp.open,
                        'high':      dp.high,
                        'low':       dp.low,
                        'close':     dp.close,
                        'volume':    dp.volume,
                        **(dp.additional_data or {}),
                    }},
                    upsert=True,
                )
                for dp in data_points
            ]

            result = self.db[f'timeseries_{frequency}'].bulk_write(
                operations, ordered=False
            )
            logger.info(
                f"✅ {equity}/{frequency}: "
                f"{result.upserted_count} new, {result.modified_count} updated "
                f"({len(operations)} candles in 1 round-trip)"
            )
            return True

        except Exception as e:
            logger.error(f"❌ Error saving timeseries: {e}")
            return False

    def save_timeseries_df(self,
                           equity: str,
                           frequency: str,
                           df: pd.DataFrame) -> bool:
        """
        Convenience wrapper: save a pandas DataFrame of OHLCV candles.

        Expected columns: timestamp, open, high, low, close, volume
        The DataFrame index is ignored — use the 'timestamp' column.
        """
        if frequency not in self.SUPPORTED_FREQUENCIES:
            raise ValueError(f"Unsupported frequency: {frequency}")

        # Accept either a 'timestamp' column or a datetime-like index.
        if 'timestamp' not in df.columns:
            if df.index.name == 'timestamp':
                df = df.reset_index()
            else:
                raise ValueError(
                    "DataFrame must contain a 'timestamp' column or have index named 'timestamp'"
                )

        required = {'timestamp', 'open', 'high', 'low', 'close', 'volume'}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        # Normalize timestamps to python datetime objects.
        ts = pd.to_datetime(df['timestamp'], errors='raise')
        # If tz-aware, convert to UTC and strip tzinfo for PyMongo.
        if getattr(ts.dt, 'tz', None) is not None:
            ts = ts.dt.tz_convert('UTC').dt.tz_localize(None)

        df = df.copy()
        df['timestamp'] = ts.dt.to_pydatetime()

        data_points: List[TimeSeriesData] = []
        for row in df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].itertuples(index=False):
            data_points.append(
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
            )

        return self.save_timeseries_data(equity, frequency, data_points)

    def df_datastore(self, df: pd.DataFrame, frequency: str, equity: str) -> bool:
        """Store a DataFrame of OHLCV candles into MongoDB.

        This is a convenience wrapper around `save_timeseries_df` that matches the
        common call-site order: (df, frequency, equity).
        """
        return self.save_timeseries_df(equity=equity, frequency=frequency, df=df)
    
    def get_timeseries_data(self,
                           equity: str,
                           frequency: str,
                           start_date: datetime = None,
                           end_date: datetime = None,
                           limit: int = None) -> List[Dict[str, Any]]:
        """
        Retrieve time series data for an equity
        
        Args:
            equity: Equity symbol
            frequency: Time frequency
            start_date: Start date filter (optional)
            end_date: End date filter (optional)
            limit: Maximum number of records to return
            
        Returns:
            List of time series data dictionaries
        """
        if frequency not in self.SUPPORTED_FREQUENCIES:
            logger.error(f"❌ Unsupported frequency: {frequency}")
            return []
        
        try:
            collection_name = f'timeseries_{frequency}'
            collection = self.db[collection_name]
            
            # Build query
            query = {'equity': equity}
            
            if start_date or end_date:
                query['timestamp'] = {}
                if start_date:
                    query['timestamp']['$gte'] = start_date
                if end_date:
                    query['timestamp']['$lte'] = end_date
            
            # Execute query — sort ASCENDING so callers get chronological order
            cursor = collection.find(query).sort('timestamp', ASCENDING)

            if limit:
                cursor = cursor.limit(limit)

            results = []
            for doc in cursor:
                results.append(self._normalize_candle_doc(doc))

            return results
            
        except Exception as e:
            logger.error(f"❌ Error retrieving time series data: {str(e)}")
            return []
    
    def get_latest_price(self, equity: str, frequency: str = '1m') -> Optional[Dict[str, Any]]:
        """
        Get the most recent candle for an equity.
        Queries with DESCENDING sort so limit=1 gives the latest, not the oldest.
        """
        if frequency not in self.SUPPORTED_FREQUENCIES:
            return None
        try:
            doc = self.db[f'timeseries_{frequency}'].find_one(
                {'equity': equity},
                sort=[('timestamp', DESCENDING)],
            )
            if not doc:
                return None
            return self._normalize_candle_doc(doc)
        except Exception as e:
            logger.error(f"❌ Error getting latest price: {e}")
            return None
    
    def list_equities(self, frequency: str = '1d') -> List[str]:
        """
        List all equities that have data in the database
        
        Args:
            frequency: Time frequency to check
            
        Returns:
            List of equity symbols
        """
        if frequency not in self.SUPPORTED_FREQUENCIES:
            frequency = '1d'
        
        try:
            collection_name = f'timeseries_{frequency}'
            equities = self.db[collection_name].distinct('equity')
            return sorted(equities)
            
        except Exception as e:
            logger.error(f"❌ Error listing equities: {str(e)}")
            return []
    
    def delete_timeseries_data(self, 
                              equity: str,
                              frequency: str,
                              start_date: datetime = None,
                              end_date: datetime = None) -> int:
        """
        Delete time series data for an equity
        
        Args:
            equity: Equity symbol
            frequency: Time frequency
            start_date: Start date filter (optional, deletes all if not provided)
            end_date: End date filter (optional)
            
        Returns:
            Number of documents deleted
        """
        if frequency not in self.SUPPORTED_FREQUENCIES:
            logger.error(f"❌ Unsupported frequency: {frequency}")
            return 0
        
        try:
            collection_name = f'timeseries_{frequency}'
            collection = self.db[collection_name]
            
            # Build query
            query = {'equity': equity}
            
            if start_date or end_date:
                query['timestamp'] = {}
                if start_date:
                    query['timestamp']['$gte'] = start_date
                if end_date:
                    query['timestamp']['$lte'] = end_date
            
            result = collection.delete_many(query)
            logger.info(f"✅ Deleted {result.deleted_count} records for {equity} at {frequency}")
            
            return result.deleted_count
            
        except Exception as e:
            logger.error(f"❌ Error deleting time series data: {str(e)}")
            return 0
    
    def get_data_statistics(self, equity: str, frequency: str) -> Dict[str, Any]:
        """
        Get statistics about stored time series data
        
        Args:
            equity: Equity symbol
            frequency: Time frequency
            
        Returns:
            Dictionary with statistics
        """
        if frequency not in self.SUPPORTED_FREQUENCIES:
            return {}
        
        try:
            collection_name = f'timeseries_{frequency}'
            collection = self.db[collection_name]
            
            # Get count
            count = collection.count_documents({'equity': equity})
            
            # Get date range
            oldest = collection.find_one(
                {'equity': equity},
                sort=[('timestamp', ASCENDING)]
            )
            newest = collection.find_one(
                {'equity': equity},
                sort=[('timestamp', DESCENDING)]
            )
            
            stats = {
                'equity': equity,
                'frequency': frequency,
                'total_records': count,
                'oldest_date': oldest['timestamp'] if oldest else None,
                'newest_date': newest['timestamp'] if newest else None
            }
            
            return stats
            
        except Exception as e:
            logger.error(f"❌ Error getting statistics: {str(e)}")
            return {}
    
    # ==================== Utility Methods ====================
    
    def get_connection_status(self) -> Dict[str, Any]:
        """Get database connection status"""
        try:
            self.client.admin.command('ping')
            
            # Get database stats
            stats = self.db.command('dbStats')
            
            return {
                'connected': True,
                'database': self.database_name,
                'collections': self.db.list_collection_names(),
                'size_mb': round(stats.get('dataSize', 0) / (1024 * 1024), 2),
                'supported_frequencies': self.SUPPORTED_FREQUENCIES
            }
        except Exception as e:
            return {
                'connected': False,
                'error': str(e)
            }

    def check_ready(self) -> Dict[str, Any]:
        """Verify DB connectivity and expected indexes.

        Returns a dict with `ok` boolean plus `issues` and lightweight `details`.
        """
        issues: List[str] = []
        details: Dict[str, Any] = {}

        try:
            self.client.admin.command('ping')
        except Exception as e:
            return {'ok': False, 'issues': [f"ping failed: {e}"], 'details': {}}

        # agents_weights indexes
        try:
            weights_info = self.db['agents_weights'].index_information()
            details['agents_weights_indexes'] = list(weights_info.keys())
            has_agent_id_unique = any(
                info.get('unique') is True and info.get('key') == [('agent_id', 1)]
                for info in weights_info.values()
            )
            if not has_agent_id_unique:
                issues.append("agents_weights missing unique index on agent_id")
        except Exception as e:
            issues.append(f"could not inspect agents_weights indexes: {e}")

        # timeseries indexes
        missing_ts_indexes: List[str] = []
        for freq in self.SUPPORTED_FREQUENCIES:
            coll_name = f'timeseries_{freq}'
            try:
                idx_info = self.db[coll_name].index_information()
                has_unique = any(
                    info.get('unique') is True
                    and info.get('key') == [('equity', 1), ('timestamp', 1)]
                    for info in idx_info.values()
                )
                if not has_unique:
                    missing_ts_indexes.append(coll_name)
            except Exception as e:
                issues.append(f"could not inspect {coll_name} indexes: {e}")

        if missing_ts_indexes:
            issues.append(
                "missing unique (equity, timestamp) index on: " + ", ".join(missing_ts_indexes)
            )

        return {'ok': len(issues) == 0, 'issues': issues, 'details': details}
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("🔌 MongoDB connection closed")


# ==================== Example Usage ====================

def test_mongodb_service():
    """Test function for MongoDB service"""
    try:
        # Initialize services
        service = MongoDBService()
        print("✅ MongoDB service initialized")

        # Quick readiness check (connectivity + expected indexes)
        ready = service.check_ready()
        print(f"🧪 Ready check: {ready}")

        # Test connection status
        status = service.get_connection_status()
        print(f"📋 Connection status: {status}")

        # Test saving agent weights
        print("\n📦 Testing agent weights storage...")
        dummy_weights = {
            'layer1': torch.randn(10, 10),
            'layer2': torch.randn(10, 5),
        }

        success = service.save_agent_weights(
            agent_id="test_agent_001",
            agent_name="SignalingModelV1",
            weights=dummy_weights,
            version="v1.0",
            equity="AAPL",
            metadata={"architecture": "ResNet", "layers": 5},
            performance_metrics={"accuracy": 0.85, "loss": 0.23},
        )
        print(f"  Save result: {'✅' if success else '❌'}")

        # Test loading agent weights
        loaded = service.load_agent_weights("test_agent_001")
        if loaded:
            print(f"  ✅ Loaded agent: {loaded['agent_name']}")
            print(f"  Performance: {loaded['performance_metrics']}")

        # Test time series data
        print("\n📈 Testing time series storage...")
        test_data = [
            TimeSeriesData(
                equity="AAPL",
                frequency="1d",
                timestamp=datetime(2026, 2, 24, 9, 30),
                open=150.25,
                high=152.80,
                low=149.50,
                close=151.75,
                volume=1000000,
            ),
            TimeSeriesData(
                equity="AAPL",
                frequency="1d",
                timestamp=datetime(2026, 2, 25, 9, 30),
                open=151.80,
                high=153.20,
                low=150.90,
                close=152.50,
                volume=1200000,
            ),
        ]

        # Test DataFrame ingestion helper
        print("\n🧾 Testing DataFrame ingestion (df_datastore)...")
        df_test = pd.DataFrame(
            {
                'timestamp': [dp.timestamp for dp in test_data],
                'open': [dp.open for dp in test_data],
                'high': [dp.high for dp in test_data],
                'low': [dp.low for dp in test_data],
                'close': [dp.close for dp in test_data],
                'volume': [dp.volume for dp in test_data],
            }
        )
        success = service.df_datastore(df_test, "1d", "AAPL")
        print(f"  df_datastore result: {'✅' if success else '❌'}")

        success = service.save_timeseries_data("AAPL", "1d", test_data)
        print(f"  Save result: {'✅' if success else '❌'}")

        # Retrieve time series data
        retrieved = service.get_timeseries_data("AAPL", "1d", limit=5)
        print(f"  ✅ Retrieved {len(retrieved)} data points")
        if retrieved:
            latest = retrieved[-1]
            print(f"  Latest: {latest['timestamp']} - Close: ${latest['close']}")

        # Get latest price via direct DESCENDING query
        latest_price = service.get_latest_price("AAPL", "1d")
        print(f"  Latest price doc: {latest_price}")

        # Get statistics
        stats = service.get_data_statistics("AAPL", "1d")
        print(f"  📊 Stats: {stats}")

        # List agents and equities
        agents = service.list_agents()
        print(f"\n👥 Total agents: {len(agents)}")

        equities = service.list_equities("1d")
        print(f"📊 Equities with data: {equities}")

        # Clean up
        service.close()

    except Exception as e:
        print(f"❌ Test failed: {str(e)}")


if __name__ == "__main__":
    test_mongodb_service()
