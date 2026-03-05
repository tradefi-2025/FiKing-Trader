"""
Deep Learning Data Loader for Signaling
Includes utilities for fetching and embedding financial news articles
"""

import os
import logging
from datetime import datetime
from typing import List, Optional, Union
import pandas as pd
import torch
import numpy as np
from pymongo import MongoClient
from dotenv import load_dotenv
from urllib.parse import quote_plus

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)


class NewsEmbeddingFetcher:
    """Utility class for fetching news embeddings from MongoDB"""
    
    def __init__(self):
        """Initialize MongoDB connection"""
        self.mongo_username = os.getenv('MONGO_USERNAME')
        self.mongo_password = os.getenv('MONGO_PASSWORD')
        self.mongo_host = os.getenv('MONGO_HOST')
        self.mongo_database = os.getenv('MONGO_DATABASE', 'admin')
        
        # Build connection string
        encoded_password = quote_plus(self.mongo_password)
        self.mongo_uri = f"mongodb+srv://{self.mongo_username}:{encoded_password}@{self.mongo_host}/{self.mongo_database}?retryWrites=true&w=majority"
        
        self.client = None
        self.db = None
        self.collection = None
        self._connect()
    
    def _connect(self):
        """Establish connection to MongoDB"""
        try:
            self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=5000)
            self.db = self.client[self.mongo_database]
            self.collection = self.db['news_articles']
            logger.info(f"✅ Connected to MongoDB for news embeddings")
        except Exception as e:
            logger.error(f"❌ Failed to connect to MongoDB: {str(e)}")
            raise
    
    def fetch_by_time_window(self, 
                            equity: str, 
                            from_timestamp: Union[datetime, str], 
                            to_timestamp: Union[datetime, str],
                            prompt: Optional[str] = None) -> List[dict]:
        """
        Fetch news articles for an equity within a time window
        
        Args:
            equity: Stock symbol (e.g., "AAPL", "TSLA")
            from_timestamp: Start time
            to_timestamp: End time
            prompt: Optional filter for article content
            
        Returns:
            List of news article documents
        """
        # Convert string timestamps to datetime if needed
        if isinstance(from_timestamp, str):
            from_timestamp = pd.to_datetime(from_timestamp)
        if isinstance(to_timestamp, str):
            to_timestamp = pd.to_datetime(to_timestamp)
        
        # Build query
        query = {
            'Stock_symbol': equity.lower(),
            'Date_parsed': {
                '$gte': from_timestamp,
                '$lte': to_timestamp
            },
            'embedding': {'$exists': True}  # Only get articles with embeddings
        }
        
        # Add prompt-based filtering if provided
        if prompt:
            # Search in article text or title
            query['$or'] = [
                {'Article': {'$regex': prompt, '$options': 'i'}},
                {'Article_title': {'$regex': prompt, '$options': 'i'}}
            ]
        
        # Fetch documents
        try:
            documents = list(self.collection.find(query).sort('Date_parsed', -1))
            logger.info(f"📰 Fetched {len(documents)} news articles for {equity.upper()}")
            return documents
        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}")
            return []
    
    def fetch_all_with_embeddings(self, 
                                 equity: str,
                                 prompt: Optional[str] = None,
                                 resources: Optional[List[str]] = None) -> List[dict]:
        """
        Fetch all news articles for an equity with embeddings
        
        Args:
            equity: Stock symbol
            prompt: Optional content filter
            resources: Optional list of publisher names to filter by
            
        Returns:
            List of news article documents
        """
        query = {
            'Stock_symbol': equity.lower(),
            'embedding': {'$exists': True}
        }
        
        # Add prompt filtering
        if prompt:
            query['$or'] = [
                {'Article': {'$regex': prompt, '$options': 'i'}},
                {'Article_title': {'$regex': prompt, '$options': 'i'}}
            ]
        
        # Add resource filtering
        if resources:
            query['Publisher'] = {'$in': resources}
        
        try:
            documents = list(self.collection.find(query).sort('Date_parsed', -1))
            logger.info(f"📰 Fetched {len(documents)} news articles for {equity.upper()}")
            return documents
        except Exception as e:
            logger.error(f"Error fetching news: {str(e)}")
            return []
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()


# Global fetcher instance (lazy initialization)
_fetcher = None

def _get_fetcher():
    """Get or create global news fetcher instance"""
    global _fetcher
    if _fetcher is None:
        _fetcher = NewsEmbeddingFetcher()
    return _fetcher


def fetch_news_embeddings(equity: str, 
                         prompt: str, 
                         From: Union[datetime, str], 
                         To: Union[datetime, str]) -> torch.Tensor:
    """
    Fetch news articles related to the equity between the given timestamps,
    pass them through a large language model to get embeddings, and return
    a tensor of shape (N, embedding_dim) where N is the number of news articles.
    
    Args:
        equity: Target financial instrument (e.g., "AAPL", "TSLA")
        prompt: Prompt guiding the news retrieval or filtering
        From: Start of the news retrieval window (datetime or string)
        To: End of the news retrieval window (datetime or string)
    
    Returns:
        torch.Tensor: Shape (N, embedding_dim) where N is number of articles
                     Returns shape (0, 384) if no articles found
    
    Example:
        >>> embeddings = fetch_news_embeddings(
        ...     equity="AAPL",
        ...     prompt="earnings",
        ...     From="2025-01-01",
        ...     To="2025-01-31"
        ... )
        >>> print(embeddings.shape)  # (N, 384)
    """
    try:
        fetcher = _get_fetcher()
        documents = fetcher.fetch_by_time_window(equity, From, To, prompt)
        
        if not documents:
            logger.warning(f"No news articles found for {equity} from {From} to {To}")
            # Return empty tensor with correct shape (0, 384)
            return torch.zeros((0, 384), dtype=torch.float32)
        
        # Extract embeddings from documents
        embeddings = []
        for doc in documents:
            if 'embedding' in doc and doc['embedding']:
                embeddings.append(doc['embedding'])
        
        if not embeddings:
            logger.warning(f"No embeddings found in documents for {equity}")
            return torch.zeros((0, 384), dtype=torch.float32)
        
        # Convert to tensor
        embeddings_tensor = torch.tensor(embeddings, dtype=torch.float32)
        logger.info(f"✅ Returning {embeddings_tensor.shape[0]} news embeddings for {equity}")
        
        return embeddings_tensor
        
    except Exception as e:
        logger.error(f"Error in fetch_news_embeddings: {str(e)}")
        return torch.zeros((0, 384), dtype=torch.float32)


def fetch_news_embeddings_dataset(equity: str, 
                                  prompt: str, 
                                  resources: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Fetch news articles related to the equity from the given resources,
    pass them through a large language model guided by `prompt`, and return
    a DataFrame with columns ['timestamp', 'embedding'] where 'embedding'
    is a tensor (embedding_dim,).
    
    Args:
        equity: Ticker / entity name, e.g., "AAPL"
        prompt: Retrieval prompt that guides which news to fetch
        resources: List of news source identifiers (publisher names, etc.)
                  If None, fetches from all sources
    
    Returns:
        pd.DataFrame: Columns ['timestamp', 'embedding', 'title', 'publisher', 'url']
                     'embedding' column contains torch.Tensor objects
                     Returns empty DataFrame if no articles found
    
    Example:
        >>> df = fetch_news_embeddings_dataset(
        ...     equity="AAPL",
        ...     prompt="product launch",
        ...     resources=["Benzinga", "Reuters"]
        ... )
        >>> print(df.shape)
        >>> print(df['embedding'].iloc[0].shape)  # (384,)
    """
    try:
        fetcher = _get_fetcher()
        documents = fetcher.fetch_all_with_embeddings(equity, prompt, resources)
        
        if not documents:
            logger.warning(f"No news articles found for {equity}")
            return pd.DataFrame(columns=['timestamp', 'embedding', 'title', 'publisher', 'url'])
        
        # Build DataFrame
        data = {
            'timestamp': [],
            'embedding': [],
            'title': [],
            'publisher': [],
            'url': []
        }
        
        for doc in documents:
            if 'embedding' in doc and doc['embedding']:
                data['timestamp'].append(doc.get('Date_parsed', None))
                data['embedding'].append(torch.tensor(doc['embedding'], dtype=torch.float32))
                data['title'].append(doc.get('Article_title', ''))
                data['publisher'].append(doc.get('Publisher', ''))
                data['url'].append(doc.get('Url', ''))
        
        df = pd.DataFrame(data)
        
        # Sort by timestamp (most recent first)
        if len(df) > 0:
            df = df.sort_values('timestamp', ascending=False).reset_index(drop=True)
        
        logger.info(f"✅ Returning dataset with {len(df)} news articles for {equity}")
        return df
        
    except Exception as e:
        logger.error(f"Error in fetch_news_embeddings_dataset: {str(e)}")
        return pd.DataFrame(columns=['timestamp', 'embedding', 'title', 'publisher', 'url'])


class SignalingDataLoader:
    """DataLoader for signaling with news integration"""
    
    def __init__(self, config):
        self.config = config

    def fetch_data(self):
        """Fetch raw data"""
        pass

    def fetch_training_dataloader(self):
        """Return training dataloader"""
        pass

    def fetch_test_dataset(self):
        """Return test dataset"""
        pass
