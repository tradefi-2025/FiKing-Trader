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

import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import (
    ConnectionFailure, 
    OperationFailure, 
    DuplicateKeyError,
    PyMongoError
)
from bson import ObjectId
from dotenv import load_dotenv
import pickle
import numpy as np

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AgentWeights:
    """Data class for agent model weights"""
    agent_id: str
    agent_name: str
    version: str
    weights_data: bytes  # Serialized weights
    metadata: Dict[str, Any]
    equity: Optional[str] = None
    training_date: Optional[datetime] = None
    performance_metrics: Optional[Dict[str, float]] = None
    created_at: datetime = None
    updated_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()


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
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with formatted values"""
        data = {
            "equity": self.equity,
            "frequency": self.frequency,
            "timestamp": self.timestamp,
            "open": f"{self.open:.2f}",
            "high": f"{self.high:.2f}",
            "low": f"{self.low:.2f}",
            "close": f"{self.close:.2f}",
            "volume": self.volume
        }
        if self.additional_data:
            data["additional_data"] = self.additional_data
        return data


class MongoDBService:
    """Service for MongoDB database operations"""
    
    # Supported time frequencies
    SUPPORTED_FREQUENCIES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M']
    
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
                # Compound index for efficient queries by equity and timestamp
                ts_collection.create_index([
                    ('equity', ASCENDING),
                    ('timestamp', DESCENDING)
                ])
                ts_collection.create_index([('timestamp', DESCENDING)])
                
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
            weights: Model weights (will be serialized with pickle)
            version: Model version
            equity: Associated equity symbol
            metadata: Additional metadata
            performance_metrics: Training/evaluation metrics
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Serialize weights
            weights_bytes = pickle.dumps(weights)
            
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
                # Deserialize weights
                weights = pickle.loads(result['weights_data'])
                
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
    
    # ==================== Time Series Methods ====================
    
    def save_timeseries_data(self, 
                            equity: str,
                            frequency: str,
                            data_points: List[TimeSeriesData]) -> bool:
        """
        Save time series data for an equity at a specific frequency
        
        Args:
            equity: Equity symbol
            frequency: Time frequency (e.g., '1m', '5m', '1h', '1d')
            data_points: List of TimeSeriesData objects
            
        Returns:
            True if successful, False otherwise
        """
        if frequency not in self.SUPPORTED_FREQUENCIES:
            logger.error(f"❌ Unsupported frequency: {frequency}")
            return False
        
        try:
            collection_name = f'timeseries_{frequency}'
            collection = self.db[collection_name]
            
            # Convert data points to documents
            documents = []
            for data_point in data_points:
                doc = {
                    'equity': equity,
                    'frequency': frequency,
                    'timestamp': data_point.timestamp,
                    'open': data_point.open,
                    'high': data_point.high,
                    'low': data_point.low,
                    'close': data_point.close,
                    'volume': data_point.volume,
                    'additional_data': data_point.additional_data or {}
                }
                documents.append(doc)
            
            # Bulk insert with ordered=False to continue on duplicate key errors
            if documents:
                try:
                    result = collection.insert_many(documents, ordered=False)
                    logger.info(f"✅ Inserted {len(result.inserted_ids)} data points for {equity} at {frequency}")
                except DuplicateKeyError:
                    logger.warning(f"⚠️ Some duplicate entries skipped for {equity} at {frequency}")
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"❌ Error saving time series data: {str(e)}")
            return False
    
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
            
            # Execute query
            cursor = collection.find(query).sort('timestamp', DESCENDING)
            
            if limit:
                cursor = cursor.limit(limit)
            
            # Convert to list and format
            results = []
            for doc in cursor:
                doc['_id'] = str(doc['_id'])  # Convert ObjectId to string
                # Format price values to 2 decimal places
                doc['open'] = f"{doc['open']:.2f}"
                doc['high'] = f"{doc['high']:.2f}"
                doc['low'] = f"{doc['low']:.2f}"
                doc['close'] = f"{doc['close']:.2f}"
                results.append(doc)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error retrieving time series data: {str(e)}")
            return []
    
    def get_latest_price(self, equity: str, frequency: str = '1m') -> Optional[Dict[str, Any]]:
        """
        Get the most recent price data for an equity
        
        Args:
            equity: Equity symbol
            frequency: Time frequency
            
        Returns:
            Latest price data or None
        """
        data = self.get_timeseries_data(equity, frequency, limit=1)
        return data[0] if data else None
    
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
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("🔌 MongoDB connection closed")


# ==================== Example Usage ====================

def test_mongodb_service():
    """Test function for MongoDB service"""
    try:
        # Initialize service
        service = MongoDBService()
        print("✅ MongoDB service initialized")
        
        # Test connection status
        status = service.get_connection_status()
        print(f"📋 Connection status: {status}")
        
        # Test saving agent weights
        print("\n📦 Testing agent weights storage...")
        dummy_weights = {
            'layer1': np.random.randn(10, 10),
            'layer2': np.random.randn(10, 5)
        }
        
        success = service.save_agent_weights(
            agent_id="test_agent_001",
            agent_name="SignalingModelV1",
            weights=dummy_weights,
            version="v1.0",
            equity="AAPL",
            metadata={"architecture": "ResNet", "layers": 5},
            performance_metrics={"accuracy": 0.85, "loss": 0.23}
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
                volume=1000000
            ),
            TimeSeriesData(
                equity="AAPL",
                frequency="1d",
                timestamp=datetime(2026, 2, 25, 9, 30),
                open=151.80,
                high=153.20,
                low=150.90,
                close=152.50,
                volume=1200000
            )
        ]
        
        success = service.save_timeseries_data("AAPL", "1d", test_data)
        print(f"  Save result: {'✅' if success else '❌'}")
        
        # Retrieve time series data
        retrieved = service.get_timeseries_data("AAPL", "1d", limit=5)
        print(f"  ✅ Retrieved {len(retrieved)} data points")
        if retrieved:
            latest = retrieved[0]
            print(f"  Latest: {latest['timestamp']} - Close: ${latest['close']}")
        
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
