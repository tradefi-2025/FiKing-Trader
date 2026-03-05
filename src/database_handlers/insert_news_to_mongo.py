"""
Script to insert JSONL news data from news_data directory into MongoDB with embeddings
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from pymongo import MongoClient, ASCENDING
from pymongo.errors import ConnectionFailure, DuplicateKeyError, BulkWriteError
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import torch
from concurrent.futures import ThreadPoolExecutor
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NewsDataInserter:
    """Insert news data from JSONL files into MongoDB with embeddings"""
    
    def __init__(self, batch_size=100, embedding_model='all-MiniLM-L6-v2'):
        """
        Initialize MongoDB connection and embedding model
        
        Args:
            batch_size: Number of documents to insert at once (default: 100)
            embedding_model: Name of the sentence-transformers model (default: 'all-MiniLM-L6-v2')
        """
        # Get MongoDB credentials from .env
        self.mongo_username = os.getenv('MONGO_USERNAME')
        self.mongo_password = os.getenv('MONGO_PASSWORD')
        self.mongo_host = os.getenv('MONGO_HOST')
        self.mongo_database = os.getenv('MONGO_DATABASE', 'admin')
        
        # Build connection string for DigitalOcean MongoDB using SRV format
        # URL encode the password to handle special characters
        from urllib.parse import quote_plus
        encoded_password = quote_plus(self.mongo_password)
        
        # Use mongodb+srv:// format for DigitalOcean
        self.mongo_uri = f"mongodb+srv://{self.mongo_username}:{encoded_password}@{self.mongo_host}/{self.mongo_database}?retryWrites=true&w=majority"
        
        self.client = None
        self.db = None
        self.collection = None
        self.batch_size = batch_size
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        self.embedding_model = SentenceTransformer(embedding_model, device=self.device)
        logger.info(f"✅ Embedding model loaded")
        
        self.stats = {
            'files_processed': 0,
            'records_inserted': 0,
            'duplicates_skipped': 0,
            'errors': 0,
            'embeddings_computed': 0
        }
        
    def connect(self):
        """Connect to MongoDB"""
        try:
            logger.info(f"Connecting to MongoDB at {self.mongo_host}...")
            logger.info(f"Database: {self.mongo_database}, User: {self.mongo_username}")
            self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=30000)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.mongo_database]
            self.collection = self.db['news_articles']
            logger.info(f"✅ Connected to MongoDB database: {self.mongo_database}")
            self._setup_indexes()
            return True
        except ConnectionFailure as e:
            logger.error(f"❌ Failed to connect to MongoDB: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error: {str(e)}")
            return False
    
    def _setup_indexes(self):
        """Create indexes for optimized queries"""
        try:
            # Create indexes for efficient querying
            self.collection.create_index([('Stock_symbol', ASCENDING)])
            self.collection.create_index([('Date', ASCENDING)])
            self.collection.create_index([('Publisher', ASCENDING)])
            self.collection.create_index([('Category', ASCENDING)])
            # Unique index on URL to prevent duplicates
            self.collection.create_index([('Url', ASCENDING)], unique=True)
            logger.info("✅ Indexes created successfully")
        except Exception as e:
            logger.warning(f"⚠️ Error creating indexes: {str(e)}")
    
    def parse_date(self, date_str):
        """Parse date string to datetime object"""
        try:
            # Format: "2025-06-28 11:11:03 UTC"
            return datetime.strptime(date_str, "%Y-%m-%d %H:%M:%S UTC")
        except:
            return None
    
    def compute_embeddings(self, texts):
        """
        Compute embeddings for a list of texts
        
        Args:
            texts: List of text strings
            
        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.embedding_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Error computing embeddings: {str(e)}")
            return [None] * len(texts)
    
    def process_jsonl_file(self, file_path):
        """Process a single JSONL file with batch operations"""
        inserted = 0
        duplicates = 0
        errors = 0
        batch = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        # Parse JSON line
                        data = json.loads(line.strip())
                        
                        # Convert date string to datetime if needed
                        if 'Date' in data:
                            data['Date_parsed'] = self.parse_date(data['Date'])
                        
                        # Add metadata
                        data['inserted_at'] = datetime.now()
                        
                        batch.append(data)
                        
                        # Process batch when it reaches batch_size
                        if len(batch) >= self.batch_size:
                            ins, dup, err = self._insert_batch(batch)
                            inserted += ins
                            duplicates += dup
                            errors += err
                            batch = []
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"  ⚠️ JSON decode error at line {line_num}: {str(e)}")
                        errors += 1
                    except Exception as e:
                        logger.warning(f"  ⚠️ Error at line {line_num}: {str(e)}")
                        errors += 1
                
                # Process remaining batch
                if batch:
                    ins, dup, err = self._insert_batch(batch)
                    inserted += ins
                    duplicates += dup
                    errors += err
            
            return inserted, duplicates, errors
            
        except Exception as e:
            logger.error(f"  ❌ Error reading file {file_path}: {str(e)}")
            return 0, 0, 1
    
    def _insert_batch(self, batch):
        """
        Insert a batch of documents with embeddings
        
        Args:
            batch: List of document dictionaries
            
        Returns:
            Tuple of (inserted, duplicates, errors)
        """
        if not batch:
            return 0, 0, 0
        
        try:
            # Extract articles for embedding computation
            articles = [doc.get('Article', '') for doc in batch]
            
            # Compute embeddings for all articles in batch
            embeddings = self.compute_embeddings(articles)
            
            # Add embeddings to documents
            for doc, embedding in zip(batch, embeddings):
                if embedding is not None:
                    doc['embedding'] = embedding
                    self.stats['embeddings_computed'] += 1
            
            # Bulk insert with ordered=False to continue on duplicates
            try:
                result = self.collection.insert_many(batch, ordered=False)
                return len(result.inserted_ids), 0, 0
            except BulkWriteError as bwe:
                # Count successful inserts and duplicates
                inserted = bwe.details.get('nInserted', 0)
                write_errors = bwe.details.get('writeErrors', [])
                duplicates = sum(1 for err in write_errors if err.get('code') == 11000)
                other_errors = len(write_errors) - duplicates
                return inserted, duplicates, other_errors
                
        except Exception as e:
            logger.error(f"  ⚠️ Batch insert error: {str(e)}")
            return 0, 0, len(batch)
    
    def process_news_directory(self, news_dir='C:\\Users\\ASK\\Desktop\\FikingTrader\\news_data'):
        """Process all JSONL files in the news_data directory"""
        news_path = Path(news_dir)
        
        if not news_path.exists():
            logger.error(f"❌ Directory not found: {news_dir}")
            return False
        
        # Get all subdirectories
        subdirs = [d for d in news_path.iterdir() if d.is_dir()]
        logger.info(f"Found {len(subdirs)} subdirectories to process")
        
        total_files = 0
        for subdir in subdirs:
            # Count JSONL files in subdirectory
            jsonl_files = list(subdir.glob('*.jsonl'))
            total_files += len(jsonl_files)
        
        logger.info(f"Total JSONL files to process: {total_files}")
        
        # Process each subdirectory
        for subdir_idx, subdir in enumerate(subdirs, 1):
            logger.info(f"\n📁 Processing directory {subdir_idx}/{len(subdirs)}: {subdir.name}")
            
            # Get all JSONL files in subdirectory
            jsonl_files = list(subdir.glob('*.jsonl'))
            
            for file_idx, jsonl_file in enumerate(jsonl_files, 1):
                logger.info(f"  📄 [{file_idx}/{len(jsonl_files)}] Processing: {jsonl_file.name}")
                
                inserted, duplicates, errors = self.process_jsonl_file(jsonl_file)
                
                # Update statistics
                self.stats['files_processed'] += 1
                self.stats['records_inserted'] += inserted
                self.stats['duplicates_skipped'] += duplicates
                self.stats['errors'] += errors
                
                if inserted > 0:
                    logger.info(f"    ✅ Inserted: {inserted} records")
                if duplicates > 0:
                    logger.info(f"    ⏭️  Skipped: {duplicates} duplicates")
                if errors > 0:
                    logger.info(f"    ⚠️  Errors: {errors}")
        
        return True
    
    def print_summary(self):
        """Print summary statistics"""
        logger.info("\n" + "="*60)
        logger.info("📊 INSERTION SUMMARY")
        logger.info("="*60)
        logger.info(f"Files processed:     {self.stats['files_processed']}")
        logger.info(f"Records inserted:    {self.stats['records_inserted']}")
        logger.info(f"Embeddings computed: {self.stats['embeddings_computed']}")
        logger.info(f"Duplicates skipped:  {self.stats['duplicates_skipped']}")
        logger.info(f"Errors encountered:  {self.stats['errors']}")
        logger.info("="*60)
        
        # Get collection stats
        try:
            total_docs = self.collection.count_documents({})
            logger.info(f"Total documents in collection: {total_docs}")
            
            # Get count of documents with embeddings
            docs_with_embeddings = self.collection.count_documents({'embedding': {'$exists': True}})
            logger.info(f"Documents with embeddings: {docs_with_embeddings}")
            
            # Get unique stock symbols
            unique_symbols = self.collection.distinct('Stock_symbol')
            logger.info(f"Unique stock symbols: {len(unique_symbols)}")
        except Exception as e:
            logger.warning(f"Could not retrieve collection stats: {str(e)}")
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("✅ MongoDB connection closed")


def main():
    """Main execution function"""
    logger.info("🚀 Starting news data insertion to MongoDB with embeddings")
    
    start_time = time.time()
    
    # Initialize with batch processing (adjust batch_size for performance)
    # Larger batch = faster but more memory. 100-500 is recommended
    inserter = NewsDataInserter(batch_size=200)
    
    # Connect to MongoDB
    if not inserter.connect():
        logger.error("Failed to connect to MongoDB. Exiting.")
        return
    
    # Process all news data
    success = inserter.process_news_directory()
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    logger.info(f"\n⏱️ Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    # Print summary
    inserter.print_summary()
    
    # Close connection
    inserter.close()
    
    if success:
        logger.info("\n✅ News data insertion completed successfully!")
    else:
        logger.error("\n❌ News data insertion completed with errors")


if __name__ == "__main__":
    main()
