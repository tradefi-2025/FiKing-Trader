import json
import threading

import pytest
import torch

from src.encoders import contextualizer as contextualizer_module


class DummyContextualizer:
    def contextualize(self, equity, timestamps, ts, text):
        return torch.zeros((ts.shape[0], 2), dtype=torch.float32)


class DummyMethod:
    def __init__(self, delivery_tag):
        self.delivery_tag = delivery_tag


class DummyProperties:
    def __init__(self, reply_to, correlation_id):
        self.reply_to = reply_to
        self.correlation_id = correlation_id


class DummyChannel:
    def __init__(self):
        self.consume_callback = None
        self.published = []
        self.acked = []
        self.nacked = []
        self.consume_ready = threading.Event()
        self.stop_event = threading.Event()
        self.published_event = threading.Event()

    def queue_declare(self, queue):
        return None

    def basic_qos(self, prefetch_count):
        return None

    def basic_consume(self, queue, on_message_callback, auto_ack):
        self.consume_callback = on_message_callback
        self.consume_ready.set()

    def start_consuming(self):
        self.stop_event.wait(timeout=5)

    def basic_publish(self, exchange, routing_key, body, properties):
        self.published.append(
            {
                "exchange": exchange,
                "routing_key": routing_key,
                "body": body,
                "properties": properties,
            }
        )
        self.published_event.set()

    def basic_ack(self, delivery_tag):
        self.acked.append(delivery_tag)

    def basic_nack(self, delivery_tag, requeue):
        self.nacked.append((delivery_tag, requeue))

    def deliver(self, body, reply_to, correlation_id):
        if not self.consume_callback:
            raise RuntimeError("Consumer callback not registered")
        method = DummyMethod(delivery_tag=1)
        properties = DummyProperties(reply_to=reply_to, correlation_id=correlation_id)
        self.consume_callback(self, method, properties, body)


class DummyConnection:
    def __init__(self, channel):
        self._channel = channel

    def channel(self):
        return self._channel

    def add_callback_threadsafe(self, callback):
        callback()


def test_contextualizer_threaded_request(monkeypatch):
    channel = DummyChannel()
    connection = DummyConnection(channel)

    monkeypatch.setattr(contextualizer_module, "NaiveContextualizer", DummyContextualizer)
    monkeypatch.setattr(
        contextualizer_module.pika,
        "BlockingConnection",
        lambda params: connection,
    )

    service = contextualizer_module.launcher()

    def run_service():
        service.start()

    service_thread = threading.Thread(target=run_service, daemon=True)
    service_thread.start()

    assert channel.consume_ready.wait(timeout=1)

    def send_request():
        payload = json.dumps(
            {
                "equity": "AAPL",
                "timestamps": ["t1", "t2"],
                "ts": [[1.0, 2.0, 5.0], [3.0, 4.0, 6.0]],
                "text": ["news-1", "news-2"],
            }
        )
        channel.deliver(payload, reply_to="reply-q", correlation_id="corr-1")

    sender_thread = threading.Thread(target=send_request, daemon=True)
    sender_thread.start()

    assert channel.published_event.wait(timeout=2)
    channel.stop_event.set()

    sender_thread.join(timeout=1)
    service_thread.join(timeout=1)

    assert channel.published
    assert channel.published[0]["routing_key"] == "reply-q"
    assert channel.acked == [1]


def main():
    monkeypatch = pytest.MonkeyPatch()
    try:
        test_contextualizer_threaded_request(monkeypatch)
        print("Test passed!")
    finally:
        monkeypatch.undo()


if __name__ == "__main__":
    main()
