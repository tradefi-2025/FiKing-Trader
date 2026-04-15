import logging
import signal
import signal
import sys
import threading
from time import time
import uuid
from venv import logger
import pika
from src.utils.env import load_env
import redis

from .config import SignalingConfig
from .dl import SignalingDataLoader
from .model import SignalingModelV3
from .verification import SignalingVerification
from .agent import Agent
from ...database_handlers.mongoDB import MongoDBService
from ...external_api.live import RefinitivService
from ...external_api.news import FinnhubNewsCollector
import pandas as pd
import torch.nn.functional as F
import torch
import json
import matplotlib.pyplot as plt
def _request_contextualizer( equity, timestamps, ts, text) -> torch.Tensor:
        params = pika.ConnectionParameters(
            host="localhost",
            heartbeat=60,
            blocked_connection_timeout=30,
            connection_attempts=3,
            retry_delay=2,
            socket_timeout=30,
        )
        
        _ctx_connection = pika.BlockingConnection(params)
        _ctx_channel = _ctx_connection.channel()
        _ctx_channel.queue_declare(queue="contextualization_requests")
        channel = _ctx_channel
        callback_queue = channel.queue_declare(queue="", exclusive=True).method.queue
        correlation_id = str(uuid.uuid4())
        request = {
            "equity": equity,
            "timestamps": [str(t) for t in timestamps],
            "ts": ts.tolist() if ts is not None else None,
            "text": text,
        }

        response = {"body": None}

        def on_response(ch, method, props, body):
            if props.correlation_id == correlation_id:
                response["body"] = body

        consumer_tag = channel.basic_consume(
            queue=callback_queue,
            on_message_callback=on_response,
            auto_ack=True,
        )

        channel.basic_publish(
            exchange="",
            routing_key="contextualization_requests",
            body=json.dumps(request),
            properties=pika.BasicProperties(
                reply_to=callback_queue,
                correlation_id=correlation_id,
            ),
        )

        deadline = time.monotonic() + 1000.0
        while response["body"] is None:
            _ctx_connection.process_data_events(time_limit=1)
            if time.monotonic() >= deadline:
                channel.basic_cancel(consumer_tag)
                raise TimeoutError("Contextualizer request timed out")

        channel.basic_cancel(consumer_tag)
        payload = json.loads(response["body"])
        return torch.tensor(payload, dtype=torch.float32)


config = SignalingConfig()
mongo_service = MongoDBService()
model_data = mongo_service.get_agent_weights("test_1")
mongo_service.close()
model = SignalingModelV3(config)
model.load_state_dict(model_data['weights'])  # Load the trained weights
model.eval()

rd = RefinitivService()
finnhub_collector = FinnhubNewsCollector()

time_now = pd.Timestamp.now()- pd.Timedelta(days=3)  # ensure we get the latest completed candle
start_time = time_now - pd.Timedelta(days=13)
ts= rd.get_ohlc_tensor(
    equity_ric="AAPL",
    start=start_time,
    end=time_now,
    interval="1min",
)
news = finnhub_collector.collect(
    entities=["AAPL"],
    start_date=start_time.strftime("%Y-%m-%d"),
    end_date=time_now.strftime("%Y-%m-%d"),
    fields=["Date", "Article"]
)

list_articles = [r["Article"] for r in news]
if not list_articles:
    list_articles = [""]

timestamps =  [n['Date'] for n in news]
ts_window = ts.unsqueeze(0)
representation = _request_contextualizer(
    "AAPL",
    timestamps,
    None,
    list_articles,
).unsqueeze(0)

ds = torch
def test(self,test_dataset):
        self.eval()
        batch_size = self.config.batch_size
        with torch.no_grad():
            ts,ctx,mask, target = test_dataset.tensors
            ts,ctx,mask,target = ts.chunk(batch_size), ctx.chunk(batch_size), mask.chunk(batch_size), target.chunk(batch_size)
            for i in range(len(ts)):
                pred, _ = self(ts[i], ctx[i], mask[i])
                del _
                loss=F.mse_loss(pred.squeeze(-1), target[i])
                print(f"Test batch {i} loss: {loss.item()}")
                if i==0:
                    all_pred=pred.cpu()
                    all_target=target[i].cpu()
                else:
                    all_pred=torch.cat([all_pred,pred.cpu()],dim=0)
                    all_target=torch.cat([all_target,target[i].cpu()],dim=0)
            
            loss=F.mse_loss(all_pred.squeeze(-1), all_target)
        print(f"Test loss: {loss.item()}")
        plt.plot(all_pred.cpu().numpy(), label="Predicted")
        plt.plot(all_target.cpu().numpy(), label="Actual")
        plt.legend()
        plt.title("Predicted vs Actual")
        plt.show()
    
