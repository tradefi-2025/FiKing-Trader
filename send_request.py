import requests
import json

BASE_URL = "http://127.0.0.1:5000"

def send_training_request():
    """Send a training request via HTTP POST"""
    training_request = {
        "model_id": "signaling_model_v1_aapl",
        "service" : "signaling",
        "agent_name": "AAPL Signaling Agent",
        "equity": "AAPL",
        "time_frequency": "1min",
        "observation_horizon": 1000,
        "prediction_horizon": 1000,
        "signal_frequency": 3600,
        "confidence_level": 0.75,
        "news_observation_horizon": 1000,
        "change_percentage_threshold": 0.1,
        "news_retrieval_prompt": "macroeconomic news and earnings reports affecting AAPL",
        "news_resources": ["finnhub"]
    }

    try:
        response = requests.post(f"{BASE_URL}/model/create", json=training_request)
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def send_launch_request():
    """Send a launch request via HTTP POST"""
    launch_request = {
        "model_id": "signaling_model_v1_aapl",
        "equity": "AAPL",
        "time_frequency": "1min",
        "observation_horizon": 1000,
        "prediction_horizon": 24,
        "signal_frequency": 3600,
        "confidence_level": 0.75,
        "news_observation_horizon": 1000,
        "change_percentage_threshold": 0.1,
        "news_retrieval_prompt": "macroeconomic news and earnings reports affecting AAPL",
        "news_resources": ["finnhub"]
    }

    try:
        response = requests.post(f"{BASE_URL}/signaling/launch", json=launch_request)
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

def send_inference_request(model_id):
    """Send an inference request via HTTP POST"""
    inference_request = {
        "model_id": model_id,
        "input_data": [[0.1, 0.05, 0.02, 0.03, 100000]],
        "confidence_level": 0.8
    }

    try:
        response = requests.post(f"{BASE_URL}/signaling/inference", json=inference_request)
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        command = sys.argv[1].lower()

        if command == "train":
            send_training_request()
        elif command == "launch":
            send_launch_request()
        elif command == "inference":
            model_id = sys.argv[2] if len(sys.argv) > 2 else "signaling_model_v1_aapl"
            send_inference_request(model_id)
        else:
            print(f"Unknown command: {command}")
            print("Usage: python send_request.py [train|launch|inference] [model_id]")
    else:
        print("Usage: python send_request.py [train|launch|inference] [model_id]")
        print("\nExample commands:")
        print("  python send_request.py train              # POST /signaling/train")
        print("  python send_request.py launch             # POST /signaling/launch")
        print("  python send_request.py inference model_id # POST /signaling/inference")
