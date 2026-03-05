"""
Script to add embeddings to existing documents in MongoDB that don't have them
"""

import os
import logging
from datetime import datetime
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import torch
from urllib.parse import quote_plus
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EmbeddingUpdater:
    """Update existing documents with embeddings"""
    
    def __init__(self, batch_size=100, embedding_model='all-MiniLM-L6-v2'):
        """
        Initialize MongoDB connection and embedding model
        
        Args:
            batch_size: Number of documents to process at once
            embedding_model: Sentence transformer model name
        """
        # Get MongoDB credentials from .env
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
        self.batch_size = batch_size
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {embedding_model}...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Using device: {self.device}")
        self.embedding_model = SentenceTransformer(embedding_model, device=self.device)
        logger.info(f"✅ Embedding model loaded")
        
        self.stats = {
            'documents_processed': 0,
            'documents_updated': 0,
            'embeddings_computed': 0,
            'errors': 0
        }
        
    def connect(self):
        """Connect to MongoDB"""
        try:
            logger.info(f"Connecting to MongoDB at {self.mongo_host}...")
            self.client = MongoClient(self.mongo_uri, serverSelectionTimeoutMS=30000)
            # Test connection
            self.client.admin.command('ping')
            self.db = self.client[self.mongo_database]
            self.collection = self.db['news_articles']
            logger.info(f"✅ Connected to MongoDB database: {self.mongo_database}")
            return True
        except ConnectionFailure as e:
            logger.error(f"❌ Failed to connect to MongoDB: {str(e)}")
            return False
        except Exception as e:
            logger.error(f"❌ Unexpected error: {str(e)}")
            return False
    
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
    
    def update_documents_without_embeddings(self):
        """Find and update all documents that don't have embeddings"""
        
        # Count documents without embeddings
        query = {'embedding': {'$exists': False}}
        total_without_embeddings = self.collection.count_documents(query)
        
        logger.info(f"\n📊 Found {total_without_embeddings} documents without embeddings")
        
        if total_without_embeddings == 0:
            logger.info("✅ All documents already have embeddings!")
            return True
        
        logger.info(f"Processing in batches of {self.batch_size}...")
        
        processed = 0
        batch_count = 0
        
        # Process documents in batches
        while True:
            # Fetch a batch of documents without embeddings
            documents = list(self.collection.find(query).limit(self.batch_size))
            
            if not documents:
                break
            
            batch_count += 1
            logger.info(f"\n📦 Processing batch {batch_count} ({len(documents)} documents)")
            
            # Process this batch
            updated = self._process_batch(documents)
            
            processed += len(documents)
            self.stats['documents_processed'] += len(documents)
            self.stats['documents_updated'] += updated
            
            # Progress update
            percent = (processed / total_without_embeddings) * 100
            logger.info(f"   Progress: {processed}/{total_without_embeddings} ({percent:.1f}%)")
            logger.info(f"   Updated: {updated} documents in this batch")
        
        return True
    
    def _process_batch(self, documents):
        """
        Process a batch of documents: compute embeddings and update
        
        Args:
            documents: List of document dictionaries
            
        Returns:
            Number of documents successfully updated
        """
        if not documents:
            return 0
        
        updated_count = 0
        
        try:
            # Extract articles for embedding computation
            articles = [doc.get('Article', '') for doc in documents]
            doc_ids = [doc['_id'] for doc in documents]
            
            # Compute embeddings for all articles in batch
            embeddings = self.compute_embeddings(articles)
            
            # Update documents individually
            for doc_id, embedding in zip(doc_ids, embeddings):
                if embedding is not None:
                    try:
                        result = self.collection.update_one(
                            {'_id': doc_id},
                            {
                                '$set': {
                                    'embedding': embedding,
                                    'embedding_updated_at': datetime.now()
                                }
                            }
                        )
                        if result.modified_count > 0:
                            updated_count += 1
                            self.stats['embeddings_computed'] += 1
                    except Exception as e:
                        logger.warning(f"  ⚠️ Error updating document {doc_id}: {str(e)}")
                        self.stats['errors'] += 1
            
            return updated_count
                
        except Exception as e:
            logger.error(f"  ⚠️ Batch processing error: {str(e)}")
            self.stats['errors'] += len(documents)
            return 0
    
    def print_summary(self):
        """Print summary statistics"""
        logger.info("\n" + "="*60)
        logger.info("📊 UPDATE SUMMARY")
        logger.info("="*60)
        logger.info(f"Documents processed: {self.stats['documents_processed']}")
        logger.info(f"Documents updated:   {self.stats['documents_updated']}")
        logger.info(f"Embeddings computed: {self.stats['embeddings_computed']}")
        logger.info(f"Errors encountered:  {self.stats['errors']}")
        logger.info("="*60)
        
        # Get collection stats
        try:
            total_docs = self.collection.count_documents({})
            docs_with_embeddings = self.collection.count_documents({'embedding': {'$exists': True}})
            docs_without_embeddings = self.collection.count_documents({'embedding': {'$exists': False}})
            
            logger.info(f"Total documents in collection:   {total_docs}")
            logger.info(f"Documents with embeddings:       {docs_with_embeddings}")
            logger.info(f"Documents without embeddings:    {docs_without_embeddings}")
            
            if total_docs > 0:
                percent = (docs_with_embeddings / total_docs) * 100
                logger.info(f"Coverage: {percent:.1f}%")
        except Exception as e:
            logger.warning(f"Could not retrieve collection stats: {str(e)}")
    
    def close(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("✅ MongoDB connection closed")


def main():
    """Main execution function"""
    logger.info("🚀 Starting embedding update for existing documents")
    
    start_time = time.time()
    
    # Initialize updater with batch processing
    updater = EmbeddingUpdater(batch_size=200)
    
    # Connect to MongoDB
    if not updater.connect():
        logger.error("Failed to connect to MongoDB. Exiting.")
        return
    
    # Update documents
    success = updater.update_documents_without_embeddings()
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    logger.info(f"\n⏱️ Total execution time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
    
    # Print summary
    updater.print_summary()
    
    # Close connection
    updater.close()
    
    if success:
        logger.info("\n✅ Embedding update completed successfully!")
    else:
        logger.error("\n❌ Embedding update completed with errors")


if __name__ == "__main__":
    main()
