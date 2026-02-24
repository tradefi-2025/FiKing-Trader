from .config import SignalingConfig
from .dl import SignalingDataLoader
from .model import SignalingModelV1
from .verification import SignalingVerification


class SignalingWorker:

    def __init__(self):
        self.config = SignalingConfig()
        self.dataloader = SignalingDataLoader(self.config)
        self.model = SignalingModelV1(self.config)
        self.verification = SignalingVerification(self.model)

    def run(self):
        data = self.dataloader.fetch_training_dataloader()
        self.model.fit(data)
