"""
MongoDB Database Handler

Collections:
- news_articles: { _id, Date, Stock_symbol, Article, Article_title, Publisher,
                   Category, Date_parsed, embedding, ... }
- timeseries_{freq}: { name (equity), file (pickled DataFrame) }
- agents_weights: { agent_id, weights_data (torch-serialized) }
"""

import os
import io
import pickle
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from pymongo import MongoClient, DESCENDING
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MongoDBService:

    SUPPORTED_FREQUENCIES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w', '1M']

    def __init__(self):
        self.host = os.getenv('MONGO_HOST', 'localhost')
        self.database_name = os.getenv('MONGO_DATABASE', 'admin')
        self.username = os.getenv('MONGO_USERNAME')
        self.password = os.getenv('MONGO_PASSWORD')
        self.client = None
        self.db = None
        self._connect()

    def _connect(self):
        try:
            if self.username and self.password:
                uri = (
                    f"mongodb+srv://{self.username}:{self.password}"
                    f"@{self.host}/{self.database_name}"
                    f"?tls=true&authSource=admin"
                )
            else:
                uri = f"mongodb://{self.host}:27017/"
            self.client = MongoClient(uri, serverSelectionTimeoutMS=5000)
            self.client.admin.command('ping')
            self.db = self.client[self.database_name]
            logger.info(f"✅ Connected to MongoDB: {self.database_name}")
        except ConnectionFailure as e:
            logger.error(f"❌ Failed to connect: {e}")
            raise

    def close(self):
        if self.client:
            self.client.close()

    # ==================== News Articles ====================

    def get_news(self, equity: str, start: datetime = None, 
                 end: datetime = None, limit: int = None) -> List[Dict]:
        """Fetch news articles for equity within date range."""
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
        Return dict with 'timestamps' (list) and 'embeddings' (stacked tensor).
        
        Returns:
            {'timestamps': List[datetime], 'embeddings': Tensor (N, embedding_dim)}
            Empty dict keys if no articles with embeddings found.
        """
        articles = self.get_news(equity, start, end)
        timestamps = []
        embeddings = []
        for doc in articles:
            emb = doc.get('embedding')
            if emb is not None:
                timestamps.append(doc.get('Date_parsed') or doc.get('Date'))
                embeddings.append(torch.tensor(emb) if not isinstance(emb, torch.Tensor) else emb)
        
        if embeddings:
            return {'timestamps': timestamps, 'embeddings': torch.stack(embeddings)}
        return {'timestamps': [], 'embeddings': torch.empty(0)}

    
    # ==================== Time Series ====================

    def get_timeseries(self, equity: str, frequency: str) -> Optional[pd.DataFrame]:
        """Load pickled DataFrame for equity at frequency."""
        doc = self.db[f'timeseries_{frequency}'].find_one({'name': equity})
        if doc and 'file' in doc:
            return pickle.loads(doc['file'])
        return None

    def save_timeseries(self, equity: str, frequency: str, df: pd.DataFrame) -> bool:
        """Store DataFrame for equity at frequency."""
        try:
            self.db[f'timeseries_{frequency}'].update_one(
                {'name': equity},
                {'$set': {'name': equity, 'file': pickle.dumps(df)}},
                upsert=True,
            )
            return True
        except Exception as e:
            logger.error(f"❌ Error saving timeseries: {e}")
            return False

    def list_equities(self, frequency: str = '1d') -> List[str]:
        """List equities with data at frequency."""
        return sorted(self.db['news_articles'].distinct('Stock_symbol'))

    # ==================== Agent Weights ====================

    def get_weights(self, agent_id: str) -> Optional[Any]:
        """Load torch weights for agent."""
        doc = self.db['agents_weights'].find_one({'agent_id': agent_id})
        if doc and 'weights_data' in doc:
            return torch.load(io.BytesIO(doc['weights_data']), weights_only=True)
        return None

    def save_weights(self, agent_id: str, weights: Any) -> bool:
        """Store torch weights for agent."""
        try:
            buf = io.BytesIO()
            torch.save(weights, buf)
            self.db['agents_weights'].update_one(
                {'agent_id': agent_id},
                {'$set': {'agent_id': agent_id, 'weights_data': buf.getvalue()}},
                upsert=True,
            )
            return True
        except Exception as e:
            logger.error(f"❌ Error saving weights: {e}")
            return False

    def list_agents(self) -> List[str]:
        """List all agent IDs."""
        return [doc['agent_id'] for doc in self.db['agents_weights'].find({}, {'agent_id': 1})]

    def delete_weights(self, agent_id: str) -> bool:
        """Delete weights for agent."""
        return self.db['agents_weights'].delete_one({'agent_id': agent_id}).deleted_count > 0

    def get_stock_symbols_with_news(self) -> List[str]:
        """Return list of stock symbols that have news articles."""
        r=sorted(self.db['news_articles'].distinct('Stock_symbol'))
        return {symbol:i for i,symbol in enumerate(r)}
    

def test_mongodb_service():
    service = MongoDBService()
    # print("Equities:", service.list_equities())
    print("Agents:", service.list_agents())
    news = service.get_news("aapl", limit=2)
    print("Sample News:", [n.get('embeddings') for n in news])
    service.close()
# Alias for dl.py compatibility
if __name__ == "__main__":
    service = MongoDBService()
    d=service.get_stock_symbols_with_news()
    import json
    #update the stock symbol in ../../configs/entities.json
    with open('./configs/entities.json', 'w') as f:
        json.dump(d, f, indent=4)


