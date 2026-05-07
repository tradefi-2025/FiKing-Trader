from dataclasses import dataclass
import logging
import torch
import torch.nn as nn
import json
from src.database_handlers.postgres import DatabaseClient
from datetime import datetime, timezone


contextualizer_config = json.load(open("src/encoders/config.json", "r"))
logger = logging.getLogger(__name__)
@dataclass
class SignalingConfig:
    db=DatabaseClient()  # Database client for agent state management
    name: str = "signaling"
    version: str = "v2"
    batch_size: int = 8
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
    def send_signal(generated_signal,agent_id):
        """Send signal to destination (e.g., trading engine)"""
        try:
            SignalingConfig.db.create_signal(
                agent_id=agent_id,
                signal_date=datetime.now(timezone.utc),
                estimated_action=generated_signal.get("estimated_action"),
                signal=generated_signal.get("signal"),
                probability=generated_signal.get("probability"),
                probabilities=generated_signal.get("probabilities", {}),
                volume=generated_signal.get("volume"),
                notional=generated_signal.get("notional"),
                stop_loss_price=generated_signal.get("stop_loss_price"),
                risk_amount=generated_signal.get("risk_amount"),
                sizing_method=generated_signal.get("sizing_method"),
                warnings=generated_signal.get("warnings", []),
                status="NEW",
            )
        except Exception as e:
            logger.error(
                f"Failed to persist generated signal for agent {agent_id}, user {g.user_id}: {e}; signal={generated_signal}"
            )
    def get(self, key, default=None):
        return getattr(self, key, default)