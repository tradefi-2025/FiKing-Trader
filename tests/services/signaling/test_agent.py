import time

from src.services.signaling.agent import Agent


class DummyMongo:
    def __init__(self, *args, **kwargs):
        pass


class DummyPostgres:
    def __init__(self, *args, **kwargs):
        pass


class DummyModel:
    def inference(self, input_data, confidence_level):
        return {
            "estimated action": 0,
        }


class DummyDL:
    def fetch_inference_data(self):
        return "data"


class DummyVerification:
    def verify(self, result):
        return {"is_valid": True}


def test_agent_launch_and_stop(monkeypatch):
    monkeypatch.setattr("src.services.signaling.agent.MongoDBService", DummyMongo)
    monkeypatch.setattr("src.services.signaling.agent.PostgreSQLService", DummyPostgres)

    meta = {"agent_id": "agent-1"}
    agent = Agent(meta_data=meta, model=DummyModel(), dataloader=DummyDL(), verification=DummyVerification())

    flags = {"main": False, "queue": False}

    def fake_main_loop():
        flags["main"] = True

    def fake_consume_queue():
        flags["queue"] = True

    monkeypatch.setattr(agent, "main_loop", fake_main_loop)
    monkeypatch.setattr(agent, "_consume_inference_queue", fake_consume_queue)

    agent.launch()
    time.sleep(0.05)
    print("Flags after launch:", flags)
    assert not agent.stop_event.is_set()
    assert flags["main"] is True
    assert flags["queue"] is True

    agent.stop()
    assert agent.stop_event.is_set()


if __name__ == "__main__":
    test_agent_launch_and_stop()
    print("Test passed!")