from datetime import datetime
import os
import json
import time
import uuid
import logging
import pandas as pd
import torch
import pika
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from .config import SignalingConfig
from ...database_handlers.mongoDB import MongoDBService
from ...external_api.live import RefinitivService
from ...external_api.news import FinnhubNewsCollector

logger = logging.getLogger(__name__)

def fetch_news_embeddings(equity: str,prompt: str, From,To) -> torch.Tensor:
    '''
    Fetch news articles related to the equity between the given timestamps, pass them through a large language model to get embeddings, and return a tensor of shape (N, embedding_dim) where N is the number of news articles.
    '''
    # Placeholder implementation — replace with actual news fetching and embedding logic
    dummy_embeddings = torch.randn(10, 768)  # 10 news articles, 768-dimensional embeddings
    return dummy_embeddings


class SignalingDataLoader:
    """
    Multimodal dataloader for signaling models.

    - Fetches OHLCV data from MongoDB.
    - Normalizes to scale-free returns.
    - Builds sliding windows (X, Y) with PyTorch.
    - Aligns pre-computed news embeddings from MongoDB to each window.
    - Uses a large encoder model to fuse (OHLCV, news) into representations.
    - Returns a DataLoader for training and a TensorDataset for testing.
    """

    def __init__(self, metadata: dict) -> None:
        self.config = SignalingConfig()
        self.meta_data = metadata

        # Core parameters
        self.equity: str = metadata.get("equity")
        self.frequency: str = metadata.get("time_frequency", "1h")

        self.frequency_to_timedelta = {
            "1min": pd.Timedelta(minutes=1),
            "5min": pd.Timedelta(minutes=5),
            "15min": pd.Timedelta(minutes=15),
            "30min": pd.Timedelta(minutes=30),
            "1h": pd.Timedelta(hours=1),
            "4h": pd.Timedelta(hours=4),
            "1d": pd.Timedelta(days=1),
        }

        self.window_size: int = metadata.get("observation_horizon", 50)
        self.prediction_horizon: int = metadata.get("prediction_horizon", 1)

        # News / multimodal parameters
        base_delta = self.frequency_to_timedelta[self.frequency]
        news_horizon_steps = metadata.get("news_observation_horizon", 50)
        self.news_observation_horizon = base_delta * news_horizon_steps

        self.news_retrieval_prompt: str = metadata.get("news_retrieval_prompt", "")
        self.news_resources: list = metadata.get("news_resources", [])

        # DB client
        self.db = MongoDBService()
        self.rd = RefinitivService()

        self.finnhub_collector = FinnhubNewsCollector()

        self.rabbitmq_host = os.getenv("RABBITMQ_HOST", "localhost")
        self.rabbitmq_port = int(os.getenv("RABBITMQ_PORT", 5672))
        self.rabbitmq_user = os.getenv("RABBITMQ_USER", "guest")
        self.rabbitmq_password = os.getenv("RABBITMQ_PASSWORD", "guest")
        self.contextualizer_queue = os.getenv(
            "CONTEXTUALIZER_QUEUE", "contextualization_requests"
        )
        self._ctx_connection = None
        self._ctx_channel = None



    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_df(self) -> pd.DataFrame:
        """
        Fetch historical OHLCV candles from MongoDB.

        Returns:
            DataFrame indexed by timestamp with columns:
            [open, high, low, close, volume]
        """
        records = self.db.get_timeseries(
            equity=self.equity,
            frequency=self.frequency,
        )
        if not records:
            records=self.rd.get_ohlc_df_for_mongo(self.equity, self.frequency)
        
        if not records:
            raise ValueError(f"No OHLCV records found for {self.equity} at {self.frequency}")

        df = pd.DataFrame(records)[["timestamp", "open", "high", "low", "close", "volume"]]
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df.set_index("timestamp").sort_index()

   
    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert raw OHLCV into scale-free log returns.

        - open, high, low, close: log return vs previous close.
        - volume: log return vs previous volume.
        """
        df = df.copy()
        norm = pd.DataFrame(index=df.index)

        # price log-returns: ln(price_t / close_{t-1})
        prev_close = df["close"].shift(1)
        norm["open"] = np.log(df["open"] / prev_close)
        norm["high"] = np.log(df["high"] / prev_close)
        norm["low"] = np.log(df["low"] / prev_close)
        norm["close"] = np.log(df["close"] / prev_close)

        # volume log-returns (use log1p on pct to be numerically safer)
        vol_ret = df["volume"].pct_change()
        norm["volume"] = np.log1p(vol_ret)  # ≈ log(volume_t / volume_{t-1})[web:46][web:48]

        return norm.dropna().astype("float32")

    def _build_sequences(self, norm_df: pd.DataFrame):
        """
        Build (X, Y) sliding-window pairs using PyTorch.

        X shape: (M, window_size, 5)
        Y shape: (M, 1) — log return from last candle in window
                 to the prediction-horizon candle.
        """
        prediction_horizon = self.prediction_horizon
        step = self.config.step_size

        values = torch.tensor(norm_df.values, dtype=torch.float32)
        
        close_col = torch.tensor(norm_df["close"].values, dtype=torch.float32)

        # Need space to look prediction_horizon steps ahead
        values = values[:-prediction_horizon]
        values = (values[1:]/(values[:-1]+1e-8)).log()  # log returns between consecutive candles
        close_col = close_col[:-prediction_horizon]

        # Build X with unfold over time dimension
        X = values.unfold(0, self.window_size, step)  # (M_raw, 5, window_size)
        
        X = X.permute(0, 2, 1)                        # (M_raw, window_size, 5)
        # Build Y as the future return from last candle in window to prediction horizon
        end_indices = torch.arange(self.window_size - 1, len(values)-prediction_horizon, step)  # indices of last candle in each window
        pred_horizons = end_indices+prediction_horizon  # indices of the candle we want to predict
        Y = (close_col[pred_horizons]/close_col[end_indices]).log().unsqueeze(-1)  # log return from last candle in window to prediction horizon candle
        end_timestamps = norm_df.index[end_indices.tolist()].tolist()
        return X, Y, end_timestamps
    def _ensure_contextualizer_channel(self) -> None:
        if self._ctx_connection and self._ctx_connection.is_open:
            if self._ctx_channel and self._ctx_channel.is_open:
                return
        params = pika.ConnectionParameters(
            host="localhost",
            heartbeat=60,
            blocked_connection_timeout=30,
            connection_attempts=3,
            retry_delay=2,
            socket_timeout=30,
        )
        
        self._ctx_connection = pika.BlockingConnection(params)
        self._ctx_channel = self._ctx_connection.channel()
        self._ctx_channel.queue_declare(queue=self.contextualizer_queue)

    def _request_contextualizer(self, equity, timestamps, ts, text) -> torch.Tensor:
        self._ensure_contextualizer_channel()
        channel = self._ctx_channel
        callback_queue = channel.queue_declare(queue="", exclusive=True).method.queue
        correlation_id = str(uuid.uuid4())
        request = {
            "equity": equity,
            "timestamps": [str(t) for t in timestamps],
            "ts": ts.tolist(),
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
            routing_key=self.contextualizer_queue,
            body=json.dumps(request),
            properties=pika.BasicProperties(
                reply_to=callback_queue,
                correlation_id=correlation_id,
            ),
        )

        deadline = time.monotonic() + 10.0
        while response["body"] is None:
            self._ctx_connection.process_data_events(time_limit=1)
            if time.monotonic() >= deadline:
                channel.basic_cancel(consumer_tag)
                raise TimeoutError("Contextualizer request timed out")

        channel.basic_cancel(consumer_tag)
        payload = json.loads(response["body"])
        return torch.tensor(payload, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Public — called by the worker
    # ------------------------------------------------------------------

        # Timestamps corresponding to each window's end
        end_timestamps = norm_df.index[end_indices.tolist()].tolist()

        return X, Y, end_timestamps

    def fetch_news_embeddings_dataset(
        self,
        equity: str,
        prompt: str = "",
        resources: list | None = None,
    ) -> dict:
        """
        Fetch pre-computed news embeddings for this equity from MongoDB.

        Returns:
            dict with:
                'timestamps': List[datetime]
                'embeddings': torch.Tensor of shape (N, embedding_dim)
        """
        # prompt and resources parameters are placeholders for a future
        # external retrieval / embedding pipeline.
        return self.db.get_news_embeddings(equity=equity, start=None, end=None)

    def _align_news_to_windows(self, end_timestamps: list) -> list[torch.Tensor]:
        """
        Align news embeddings to each OHLCV window.

        For each window end timestamp, collect all news embeddings whose
        timestamp lies in [end_ts - news_observation_horizon, end_ts].

        Returns:
            List[torch.Tensor]; one tensor per window, each with shape
            (N_i, embedding_dim), where N_i is #articles in that window.
        """
        news = self.fetch_news_embeddings_dataset(
            self.equity,
            prompt=self.news_retrieval_prompt,
            resources=self.news_resources,
        )
        timestamps = news["timestamps"]
        embeddings = news["embeddings"]

        if embeddings.numel() == 0:
            # No news at all — return a single zero vector per window
            return [torch.zeros(1, 768) for _ in end_timestamps]

        # Ensure timestamps and embeddings are sorted ascending by time
        if timestamps:
            idx_sorted = sorted(range(len(timestamps)), key=lambda i: timestamps[i])
            timestamps = [timestamps[i] for i in idx_sorted]
            embeddings = embeddings[idx_sorted]

        embedding_dim = embeddings.shape[1]
        horizon = self.news_observation_horizon

        all_window_embeddings: list[torch.Tensor] = []

        for end_ts in end_timestamps:
            start_ts = end_ts - horizon

            # Indices where timestamp is in [start_ts, end_ts]
            indices = [
                i for i, ts in enumerate(timestamps)
                if start_ts <= ts <= end_ts
            ]

            if indices:
                # Sort by time descending: most recent first
                indices = sorted(indices, key=lambda i: timestamps[i], reverse=True)
                window_embs = embeddings[indices]  # (K, embedding_dim)
            else:
                # No news in this window
                window_embs = torch.zeros(1, embedding_dim)

            all_window_embeddings.append(window_embs)

        return all_window_embeddings

    def _pad_news_windows(self, X_news_list: list[torch.Tensor]) -> torch.Tensor:
        """
        Pad a list of (N_i, embedding_dim) tensors to (M, max_N, embedding_dim)
        with zero padding.
        """
        if not X_news_list:
            raise ValueError("X_news_list is empty.")

        max_len = max(t.size(0) for t in X_news_list)
        emb_dim = X_news_list[0].size(1)

        out = torch.zeros(len(X_news_list), max_len, emb_dim, dtype=X_news_list[0].dtype)
        for i, t in enumerate(X_news_list):
            out[i, : t.size(0)] = t
        return out

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_dataloader(self, train_ratio: float = 0.8):
        """
        Full pipeline:

            MongoDB → normalize → OHLCV windows
            + news embeddings aligned to each window
            → large_model.encode(X_ohlcv, X_news)
            → DataLoader(TensorDataset) for training and testing.

        Args:
            large_model: encoder model with .encode(X_ohlcv, X_news)
                         returning a representation tensor (M, d_model).
            train_ratio: fraction of samples used for training.

        Returns:
            train_loader: DataLoader yielding (representation, Y) batches.
            test_dataset: TensorDataset for evaluation.
        """
        # Step 1: OHLCV → normalize → sliding windows
        raw_df = self._fetch_df()
        # norm_df = self._normalize(raw_df)
        X_ohlcv, Y, end_timestamps = self._build_sequences(raw_df)
        
        # X_ohlcv: (M, window_size, 5), Y: (M, 1)

        # Step 2: fetch and align news embeddings
        X_news_list = self._align_news_to_windows(end_timestamps)
        X_news = self._pad_news_windows(X_news_list)
        
        # X_news: (M, max_news_per_window, embedding_dim)

        # Step 3: pass (OHLCV, news) pairs through the large model
        # The large model fuses both modalities and produces representation vectors
        

        # representations : (M, d_model=256)
        ts_representations = self._request_contextualizer(
            self.equity,
            end_timestamps,
            X_ohlcv,
            None,
        )
        

        representations = torch.cat([ts_representations, X_news.mean(dim=1)], dim=1)
        # Step 4: shuffle before splitting
        perm = torch.randperm(len(representations))
        representations = representations[perm]
        Y = Y[perm]

        split = int(len(representations) * train_ratio)
        print(f"Total samples: {len(representations)}, Train: {split}, Test: {len(representations) - split}")
        train_loader = DataLoader(
            TensorDataset(representations[:split], Y[:split]),
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )
        test_dataset = TensorDataset(representations[split:], Y[split:])

        return train_loader, test_dataset

    def fetch_inference_data(self) -> torch.Tensor:
        """
        Fetch latest candles from external APIs and return the most recent
        normalized window as a tensor ready for model.inference().

        Currently:
            - Uses RefinitivService for live OHLCV.
            - (Optional) can be extended to use NewsServices for live news.

        Returns:
            torch.Tensor of shape (1, window_size, 5)
        """
        try:
            time_now = pd.Timestamp.now()-pd.Timedelta(days=3)  # buffer to ensure we get the latest closed candle
            start_time = time_now - self.frequency_to_timedelta[self.frequency] * self.window_size
            ts= self.rd.get_ohlc_tensor(
                equity_ric=self.equity,
                start=start_time,
                end=time_now,
                interval=self.frequency,
            )
            news = self.finnhub_collector.collect(
                entities=[self.equity],
                start_date=start_time.strftime("%Y-%m-%d"),
                end_date=time_now.strftime("%Y-%m-%d"),
                fields=["Date", "Article"]
            )

            list_articles = [r["Article"] for r in news]
            if not list_articles:
                list_articles = [""]
            
            timestamps =  [n['Date'] for n in news]
            ts_window = ts.unsqueeze(0)
            representation = self._request_contextualizer(
                self.equity,
                timestamps,
                ts_window,
                [list_articles],
            )

            return representation
        except Exception:
            logger.exception("Failed to fetch inference data")
            return None  # Return a zero vector on failure


def test_fetch_dataloader():
    metadata = {
        "equity": "AAPL",
        "time_frequency": "1min",
        "observation_horizon": 50,
        "prediction_horizon": 1,
        "news_observation_horizon": 50,
        "news_retrieval_prompt": "Fetch news articles related to {equity} in the last {news_observation_horizon} hours.",
        "news_resources": ["finnhub"],
    }
    loader = SignalingDataLoader(metadata)
    train_loader, test_dataset = loader.fetch_dataloader()
    print(f"Train batches: {len(train_loader)}, Test samples: {len(test_dataset)}")

def test_fetch_inference_data():
    metadata = {
        "equity": "AAPL",
        "time_frequency": "1min",
        "observation_horizon": 50,
        "prediction_horizon": 1,
        "news_observation_horizon": 50,
        "news_retrieval_prompt": "Fetch news articles related to {equity} in the last {news_observation_horizon} hours.",
        "news_resources": ["finnhub"],
    }
    loader = SignalingDataLoader(metadata)
    representation = loader.fetch_inference_data()
    print(f"Inference representation shape: {representation.shape}")
if __name__ == "__main__":
    # test_fetch_dataloader()
    test_fetch_inference_data()
