from dataclasses import dataclass


@dataclass
class SignalingConfig:
    name: str = "signaling"
    version: str = "v1"
    batch_size: int = 32
    learning_rate: float = 0.001
