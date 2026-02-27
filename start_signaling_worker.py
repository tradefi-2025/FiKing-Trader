#!/usr/bin/env python3
"""
Script to start the Signaling Worker
Waits on RabbitMQ queue for training requests
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, project_root)

from src.services.signaling.worker import start_worker

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Signaling Service Worker")
    print("Queue: signaling_training_queue")
    print("=" * 60)
    
    try:
        start_worker()
    except KeyboardInterrupt:
        print("\n🛑 Worker stopped by user")
    except Exception as e:
        print(f"\n❌ Worker error: {str(e)}")
        sys.exit(1)
