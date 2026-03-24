from src.services.signaling.worker import SignalingWorker


class DummyChannel:
    def __init__(self):
        self.published = []

    def basic_publish(self, exchange, routing_key, body, properties):
        self.published.append(
            {
                "exchange": exchange,
                "routing_key": routing_key,
                "body": body,
                "properties": properties,
            }
        )


class DummyConnection:
    def __init__(self, channel):
        self._channel = channel
        self.closed = False

    def channel(self):
        return self._channel

    def close(self):
        self.closed = True


def test_send_launch_response(monkeypatch):
    worker = SignalingWorker()
    channel = DummyChannel()
    connection = DummyConnection(channel)

    def fake_connection():
        return connection

    monkeypatch.setattr(worker, "_get_rabbitmq_connection", fake_connection)

    worker._send_launch_response(
        reply_to="queue-1",
        correlation_id="corr-1",
        agent_id="agent-1",
        success=True,
        message="ok",
    )

    assert channel.published
    assert channel.published[0]["routing_key"] == "queue-1"
    assert connection.closed is True


def test_process_launch_request_agent_already_running():
    worker = SignalingWorker()
    worker.active_agents["agent-1"] = object()

    result = worker._process_launch_request({"agent_id": "agent-1"})

    assert result is False
