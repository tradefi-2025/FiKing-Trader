from dataclasses import dataclass
import logging
import torch
import torch.nn as nn
import json
contextualizer_config = json.load(open("src/encoders/config.json", "r"))
logger = logging.getLogger(__name__)
@dataclass
class SignalingConfig:
    name: str = "signaling"
    version: str = "v2"
    batch_size: int = 16
    learning_rate: float = 1e-3
    d_model: int=512
    d_ts: int = contextualizer_config["ts_encoder"]["output_dim"]
    d_text: int = contextualizer_config["text_encoder"]["output_dim"]
    lr: float = 1e-4  # learning rate alias
    epoch: int = 5  # number of training epochs
    default_frequency: int = 60  # seconds
    default_confidence_level: float = 0.95
    step_size: int = 10  # epochs between learning rate decay
    activation: nn.Module = nn.LeakyReLU()
    max_news_per_window: int = 64  # maximum number of news items to consider per time window
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    @staticmethod
    def send_signal(message,destination):
        """Send signal to destination (e.g., trading engine)"""
        logger.info(f"Sending signal to {destination}: {message}")
        #TODO: Implement actual communication logic (e.g., via RabbitMQ, REST API, etc.)

    def get(self, key, default=None):
        return getattr(self, key, default)