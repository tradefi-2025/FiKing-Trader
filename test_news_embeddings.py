"""
Test script for news embedding retrieval functions
Demonstrates usage of fetch_news_embeddings and fetch_news_embeddings_dataset
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.services.signaling.dl import (
    fetch_news_embeddings,
    fetch_news_embeddings_dataset
)
from datetime import datetime, timedelta
import torch
import pandas as pd


def test_fetch_news_embeddings():
    """Test fetching news embeddings for a time window"""
    print("="*70)
    print("Test 1: fetch_news_embeddings - Time Window Query")
    print("="*70)
    
    # Example 1: Fetch embeddings for AAPL in January 2025
    equity = "AAPL"
    prompt = "earnings"  # Filter for earnings-related news
    from_date = "2025-01-01"
    to_date = "2025-01-31"
    
    print(f"\nFetching news embeddings for {equity}")
    print(f"Time window: {from_date} to {to_date}")
    print(f"Filter prompt: '{prompt}'")
    
    embeddings = fetch_news_embeddings(
        equity=equity,
        prompt=prompt,
        From=from_date,
        To=to_date
    )
    
    print(f"\n✅ Result:")
    print(f"   Shape: {embeddings.shape}")
    print(f"   Type: {type(embeddings)}")
    print(f"   Dtype: {embeddings.dtype}")
    
    if embeddings.shape[0] > 0:
        print(f"   First embedding (first 10 dims): {embeddings[0][:10]}")
        print(f"   Mean: {embeddings.mean():.4f}, Std: {embeddings.std():.4f}")
    
    # Example 2: Without prompt filter
    print("\n" + "-"*70)
    print("Fetching without prompt filter...")
    
    embeddings_all = fetch_news_embeddings(
        equity="AA",  # Alcoa Corporation
        prompt="",
        From="2025-06-01",
        To="2025-06-30"
    )
    
    print(f"\n✅ Result:")
    print(f"   Shape: {embeddings_all.shape}")
    print(f"   Found {embeddings_all.shape[0]} articles")


def test_fetch_news_embeddings_dataset():
    """Test fetching news embeddings as a dataset"""
    print("\n" + "="*70)
    print("Test 2: fetch_news_embeddings_dataset - Dataset Query")
    print("="*70)
    
    # Example 1: Fetch all news for TSLA with specific publishers
    equity = "AAPL"
    prompt = "product"
    resources = ["Benzinga"]  # Only from these publishers
    
    print(f"\nFetching news dataset for {equity}")
    print(f"Filter prompt: '{prompt}'")
    print(f"Publishers: {resources}")
    
    df = fetch_news_embeddings_dataset(
        equity=equity,
        prompt=prompt,
        resources=resources
    )
    
    print(f"\n✅ Result:")
    print(f"   DataFrame shape: {df.shape}")
    print(f"   Columns: {df.columns.tolist()}")
    
    if len(df) > 0:
        print(f"\n   Sample data:")
        print(f"   {'Index':<8} {'Timestamp':<20} {'Publisher':<15} {'Title':<50}")
        print("   " + "-"*93)
        for idx, row in df.head(3).iterrows():
            title = row['title'][:47] + "..." if len(row['title']) > 50 else row['title']
            timestamp = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S') if pd.notna(row['timestamp']) else 'N/A'
            print(f"   {idx:<8} {timestamp:<20} {row['publisher']:<15} {title:<50}")
        
        print(f"\n   Embedding shape: {df['embedding'].iloc[0].shape}")
        print(f"   First embedding (first 10 dims): {df['embedding'].iloc[0][:10]}")
    
    # Example 2: Without resource filter
    print("\n" + "-"*70)
    print("Fetching from all publishers...")
    
    df_all = fetch_news_embeddings_dataset(
        equity="AA",
        prompt="",
        resources=None
    )
    
    print(f"\n✅ Result:")
    print(f"   DataFrame shape: {df_all.shape}")
    
    if len(df_all) > 0:
        print(f"   Publishers found: {df_all['publisher'].unique().tolist()}")
        print(f"   Date range: {df_all['timestamp'].min()} to {df_all['timestamp'].max()}")


def test_integration_example():
    """Example of using embeddings in a downstream model"""
    print("\n" + "="*70)
    print("Test 3: Integration Example - Using Embeddings in a Model")
    print("="*70)
    
    equity = "AAPL"
    
    # Fetch recent news embeddings
    print(f"\nFetching recent news for {equity}...")
    embeddings = fetch_news_embeddings(
        equity=equity,
        prompt="",
        From="2025-01-01",
        To="2025-12-31"
    )
    
    print(f"✅ Got {embeddings.shape[0]} news embeddings")
    
    if embeddings.shape[0] > 0:
        # Example: Aggregate embeddings (mean pooling)
        print("\n📊 Aggregating embeddings...")
        mean_embedding = embeddings.mean(dim=0)
        print(f"   Mean embedding shape: {mean_embedding.shape}")
        
        # Example: Use most recent N embeddings
        N = min(5, embeddings.shape[0])
        recent_embeddings = embeddings[:N]
        print(f"\n📰 Using {N} most recent embeddings")
        print(f"   Combined shape: {recent_embeddings.shape}")
        
        # Example: Concatenate with other features
        print("\n🔗 Combining with other features...")
        market_features = torch.randn(50)  # Simulated market features
        combined = torch.cat([mean_embedding, market_features])
        print(f"   Combined feature vector shape: {combined.shape}")
        print(f"   (384 news dims + 50 market dims = {combined.shape[0]} total dims)")


def main():
    """Run all tests"""
    print("\n" + "🚀 " + "="*66)
    print("Testing News Embedding Retrieval Functions")
    print("="*70)
    
    try:
        # Run tests
        test_fetch_news_embeddings()
        test_fetch_news_embeddings_dataset()
        test_integration_example()
        
        print("\n" + "="*70)
        print("✅ All tests completed successfully!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
