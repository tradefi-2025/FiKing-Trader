from dataclasses import dataclass
import torch
import torch.nn as nn

@dataclass
class SignalingConfig:
    name: str = "signaling"
    version: str = "v1"
    batch_size: int = 32
    learning_rate: float = 0.001
    d_model: int = 256
    nb_layers: int = 6
    activation = nn.GELU()
    lr: float = 0.001  # learning rate alias
    epoch: int = 100  # number of training epochs
    default_frequency: int = 60  # seconds
    default_confidence_level: float = 0.95
