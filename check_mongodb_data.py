"""
Quick check of available data in MongoDB news_articles collection
"""

import os
from pymongo import MongoClient
from dotenv import load_dotenv
from urllib.parse import quote_plus

load_dotenv()

# Connect to MongoDB
mongo_username = os.getenv('MONGO_USERNAME')
mongo_password = os.getenv('MONGO_PASSWORD')
mongo_host = os.getenv('MONGO_HOST')
mongo_database = os.getenv('MONGO_DATABASE', 'admin')

encoded_password = quote_plus(mongo_password)
mongo_uri = f"mongodb+srv://{mongo_username}:{encoded_password}@{mongo_host}/{mongo_database}?retryWrites=true&w=majority"

try:
    client = MongoClient(mongo_uri, serverSelectionTimeoutMS=10000)
    db = client[mongo_database]
    collection = db['news_articles']
    
    print("="*70)
    print("MongoDB News Articles Collection Summary")
    print("="*70)
    
    # Total count
    total_docs = collection.count_documents({})
    print(f"\nTotal documents: {total_docs}")
    
    # Count with embeddings
    docs_with_embeddings = collection.count_documents({'embedding': {'$exists': True}})
    print(f"Documents with embeddings: {docs_with_embeddings}")
    
    # Sample stock symbols
    print("\n" + "-"*70)
    print("Sample stock symbols (first 20):")
    symbols = collection.distinct('Stock_symbol')[:20]
    print(", ".join(symbols))
    print(f"\nTotal unique symbols: {len(collection.distinct('Stock_symbol'))}")
    
    # Sample publishers
    print("\n" + "-"*70)
    print("Available publishers:")
    publishers = collection.distinct('Publisher')
    print(", ".join(publishers))
    
    # Date range
    print("\n" + "-"*70)
    print("Date range:")
    oldest = collection.find_one(sort=[('Date_parsed', 1)])
    newest = collection.find_one(sort=[('Date_parsed', -1)])
    if oldest and newest:
        print(f"Oldest: {oldest.get('Date_parsed')}")
        print(f"Newest: {newest.get('Date_parsed')}")
    
    # Sample document
    print("\n" + "-"*70)
    print("Sample document:")
    sample = collection.find_one({'embedding': {'$exists': True}})
    if sample:
        print(f"  Symbol: {sample.get('Stock_symbol')}")
        print(f"  Date: {sample.get('Date_parsed')}")
        print(f"  Title: {sample.get('Article_title', 'N/A')[:80]}...")
        print(f"  Publisher: {sample.get('Publisher')}")
        print(f"  Has embedding: {len(sample.get('embedding', [])) if 'embedding' in sample else 0} dimensions")
    
    print("\n" + "="*70)
    
    client.close()
    
except Exception as e:
    print(f"Error: {str(e)}")
