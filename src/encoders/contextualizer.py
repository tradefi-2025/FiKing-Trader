from src.encoders.text_handler import FlangService
from src.encoders.ts_handler import KronosService
import torch
import pika
import json
import logging
from concurrent.futures import ThreadPoolExecutor

class NaiveContextualizer:

    def __init__(self):
        with open("src/encoders/config.json", "r") as f:
            config = json.load(f)
        self.config = config
        self.d_model = self.config["ts_encoder"]["output_dim"]
        self.kronos_service = KronosService()
        self.flang_service = FlangService()


    def contextualize(self, equity, timestamps,ts,text):
        ts_embeddings = self.kronos_service.encode_timeseries_batch(ts)
        text_embeddings = torch.stack([self.flang_service.encode(t, pooling="cls").mean(dim=0) for t in text])

        return torch.cat([ts_embeddings, text_embeddings], dim=1)
    


logger = logging.getLogger(__name__)


class launcher:
    
    def __init__(self):
        self.contextualizer = NaiveContextualizer()
        self._contextualize = self.contextualizer.contextualize
        self._executor = ThreadPoolExecutor(max_workers=4)
        params = pika.ConnectionParameters(
            host="localhost",
            heartbeat=60,
            blocked_connection_timeout=30,
            connection_attempts=3,
            retry_delay=2,
            socket_timeout=30,
        )
        self.connection = pika.BlockingConnection(params)
        self.channel = self.connection.channel()
        self.channel.queue_declare(queue="contextualization_requests")
        self.channel.basic_qos(prefetch_count=1)

    def start(self):
        print("Contextualizer service started. Waiting for requests...")
        self.channel.basic_consume(
            queue="contextualization_requests",
            on_message_callback=self.on_request,
            auto_ack=False,
        )
        self.channel.start_consuming()

    def on_request(self, ch, method, properties, body):
        self._executor.submit(
            self._handle_request,
            method.delivery_tag,
            properties.reply_to,
            properties.correlation_id,
            body,
        )

    def _handle_request(self, delivery_tag, reply_to, correlation_id, body):
        try:
            request = json.loads(body)
            equity = request["equity"]
            timestamps = request["timestamps"]
            ts = torch.as_tensor(request["ts"], dtype=torch.float32)
            text = request["text"]

            print(
                f"Received request for {equity} with {len(timestamps)} timestamps and {len(text)} articles."
            )
            with torch.inference_mode():
                representation = self._contextualize(equity, timestamps, ts, text)
            print(f"Generated representation of shape: {representation.shape}")

            payload = json.dumps(representation.tolist(), separators=(",", ":"))
            self.connection.add_callback_threadsafe(
                lambda: self._publish_response(delivery_tag, reply_to, correlation_id, payload)
            )
        except Exception:
            logger.exception("Failed to handle contextualization request")
            self.connection.add_callback_threadsafe(
                lambda: self.channel.basic_nack(delivery_tag=delivery_tag, requeue=True)
            )

    def _publish_response(self, delivery_tag, reply_to, correlation_id, payload):
        if reply_to:
            self.channel.basic_publish(
                exchange="",
                routing_key=reply_to,
                body=payload,
                properties=pika.BasicProperties(correlation_id=correlation_id),
            )
        self.channel.basic_ack(delivery_tag=delivery_tag)


def test():
    service = NaiveContextualizer()
    from src.external_api.live import RefinitivService
    from src.external_api.news import FinnhubNewsCollector
    from datetime import datetime, timedelta
    rd=RefinitivService()
    collector = FinnhubNewsCollector()
    equity = "AAPL"
    strt1= "2026-03-23"
    strt2= "2026-03-24"
    end1= "2026-03-24"
    end2= "2026-03-25"
    news1 = collector.collect(
        entities=["AAPL"],
        start_date="2026-03-23",
        end_date="2026-03-24",
        fields=["Date", "Article"]
    )
    articles = [r["Article"] for r in news1]
    print(f"Collected {len(articles)} articles for {equity} from {strt1} to {end1}")
    ts1 = rd.get_ohlc_tensor(equity, datetime.strptime(strt1, "%Y-%m-%d"), datetime.strptime(end1, "%Y-%m-%d"), interval="1min")

    news2 = collector.collect(
        entities=["AAPL"],
        start_date="2026-03-24",
        end_date="2026-03-25",
        fields=["Date", "Article"]
    )
    articles2 = [r["Article"] for r in news2]
    ts2 = rd.get_ohlc_tensor(equity, datetime.strptime(strt2, "%Y-%m-%d"), datetime.strptime(end2, "%Y-%m-%d"), interval="1min")

    min_len = min(ts1.shape[0], ts2.shape[0])
    ts1 = ts1[:min_len]
    ts2 = ts2[:min_len]
    ts=torch.stack([ts1,ts2])
    news = [articles, articles2]
    rep = service.contextualize(equity, [strt1, strt2], ts, news)
    print("TS Embeddings shape:", rep.shape)  # (2, window_size

if __name__ == "__main__":
    l=launcher()
    l.start()