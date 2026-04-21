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
class ContextWrapper:
    def __init__(self, equity, ctxt, ts):
        self.equity     = equity
        self.timestamps = pd.to_datetime(ctxt["timestamps"], utc=True)  # always UTC-aware
        self.ts         = ts
        self.text_embd  = ctxt["embeddings"]

    def get(self, from_, to_):
        from_utc = pd.Timestamp(from_, tz="UTC")
        to_utc   = pd.Timestamp(to_,   tz="UTC")

        # ts index is tz-naive → strip tz for slicing
        from_naive = from_utc.tz_localize(None)
        to_naive   = to_utc.tz_localize(None)

        mask = (self.timestamps >= from_utc) & (self.timestamps <= to_utc)
        return torch.FloatTensor(self.ts.loc[from_naive:to_naive].values), self.text_embd[mask]
        
def gaussian_blur_1d(x: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    half = kernel_size // 2
    grid = torch.arange(-half, half + 1, dtype=torch.float32)
    kernel = torch.exp(-0.5 * (grid / sigma) ** 2)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1)
    x = x.view(1, 1, -1)
    return torch.nn.functional.conv1d(x, kernel, padding=half).squeeze()
def tan(x):
    return torch.tan(x-torch.pi/2)
def fetch_news_embeddings(equity: str,prompt: str, From,To) -> torch.Tensor:
    '''
    Fetch news articles related to the equity between the given timestamps, pass them through a large language model to get embeddings, and return a tensor of shape (N, embedding_dim) where N is the number of news articles.
    '''
    # Placeholder implementation — replace with actual news fetching and embedding logic
    dummy_embeddings = torch.randn(10, 768)  # 10 news articles, 768-dimensional embeddings
    return dummy_embeddings

def build_label(
    ts: torch.Tensor,
    change_percentage_threshold: float = 0.02,
    kernel_size: int = 5,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    1. Smooth the full window with a 1D Gaussian filter.
    2. Compute percentage change of every step relative to the first step.
    3. Assign ternary labels:
         +1  if pct_change >  threshold  (upward move)
          0  if |pct_change| <= threshold (flat / noise)
         -1  if pct_change < -threshold  (downward move)

    Args:
        ts:                          1D tensor of prices, shape (L,).
        change_percentage_threshold: Symmetric threshold (e.g. 0.02 = 2%).
        kernel_size:                 Gaussian kernel size (must be odd).
        sigma:                       Gaussian standard deviation.

    Returns:
        labels: float tensor of shape (L,) with values in {-1, 0, 1}.
    """
    # ── 1. Build normalised Gaussian kernel ─────────────────────────────
    half   = kernel_size // 2
    grid   = torch.arange(-half, half + 1, dtype=ts.dtype, device=ts.device)
    kernel = torch.exp(-0.5 * (grid / sigma) ** 2)
    kernel = kernel / kernel.sum()

    # ── 2. Smooth the full window ────────────────────────────────────────
    ts_3d = ts.view(1, 1, -1)                             # (1, 1, L)

    if half > 0:
        ts_padded = torch.nn.functional.pad(
            ts_3d, (half, half), mode="replicate"
        )
        smoothed = torch.nn.functional.conv1d(
            ts_padded,
            kernel.view(1, 1, -1),
            padding=0,
        ).view(-1)                                        # avoid squeeze() on 0-dim risk
    else:
        # kernel_size == 1: no-op smoothing
        smoothed = ts_3d.view(-1)

    # ── 3. Percentage change: last step vs. first step ───────────────────
    ref        = smoothed[0]
    pct_change = (smoothed - ref) / (ref.abs() + 1e-8)   # (L,)
    pct_change = pct_change.mean()  # aggregate over window (can experiment with other aggregations)
    # ── 4. Ternary labelling ─────────────────────────────────────────────

    if pct_change > change_percentage_threshold:
        return torch.tensor(2.0)  # upward move
    elif pct_change < -change_percentage_threshold:
        return torch.tensor(0.0)  # downward move
    else:
        return torch.tensor(1.0)  # flat / noise
    
    # for pct in pct_change:
    #     if pct > change_percentage_threshold:
    #         return torch.tensor(2.0)
    #     elif pct < -change_percentage_threshold:

    #         return torch.tensor(0.0)
    # return torch.tensor(1.0)



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
        logger.info(f"Fetching OHLCV data for {self.equity} at {self.frequency} from MongoDB...")
        # records = self.db.get_timeseries(
        #     equity=self.equity,
        #     frequency=self.frequency,
        # )
        records=None
        if not records:
            logger.warning(f"No OHLCV records found for {self.equity} at {self.frequency} fetching from MongoDB, falling back to Refinitiv...")
            records=self.rd.get_ohlc_df_for_mongo(self.equity, interval=self.frequency)
            records=records.reset_index().to_dict(orient="records")
        if not records:
            
            raise ValueError(f"No OHLCV records found for {self.equity} at {self.frequency}")
        
        logger.info(f"Fetched {len(records)} OHLCV records for {self.equity} at {self.frequency}. Sample:\n{records[:5]}")
        df = pd.DataFrame(records)[["timestamp", "open", "high", "low", "close", "volume"]]
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        logger.info(f"Constructed OHLCV DataFrame with shape {df.shape} and columns {df.columns.tolist()}")
        return df.set_index("timestamp").sort_index()

    def _fetch_backtest_df(self) -> pd.DataFrame:
        """
        Fetch historical OHLCV candles for the past week from Refinitiv for backtesting.

        Returns:
            DataFrame indexed by timestamp with columns:
            [open, high, low, close, volume]
        """
        time_now = pd.Timestamp.now(tz="America/New_York")  # ensure we get the latest completed candle
        start_time = time_now - pd.Timedelta(days=7)
        logger.info(f"Fetching backtest OHLCV data for {self.equity} from {start_time} to {time_now}...")
        for suffix in [".O", ".N"]:
            ric = self.equity + suffix
            logger.info(
                "Attempting to fetch %s from %s to %s with interval=%s",
                ric,
                start_time.isoformat(timespec="seconds"),
                time_now.isoformat(timespec="seconds"),
                self.frequency,
            )
            df = self.rd.get_ohlc_df(ric, start_time, time_now, interval=self.frequency)
            if df is not None and not df.empty:
                logger.info(
                    "✅ Successfully fetched %d rows for %s (interval=%s)",
                    len(df),
                    ric,
                    self.frequency,
                )
                break
        if df is None or df.empty:
            logger.error(
                "❌ Failed to fetch OHLCV data for %s with both .O and .N suffixes",
                self.equity,
            )
            raise ValueError(f"No OHLCV records found for {self.equity} at {self.frequency} for backtesting")

        df = pd.DataFrame(df)[["timestamp", "open", "high", "low", "close", "volume"]]
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        logger.info(f"Constructed OHLCV DataFrame with shape {df.shape} and columns {df.columns.tolist()}")
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
        values = (values - values.mean(dim=0, keepdim=True)) / (values.std(dim=0, keepdim=True) + 1e-8)  # standardize each feature

        # Build X with unfold over time dimension
        X = values.unfold(0, self.window_size, step)  # (M_raw, 5, window_size)
        
        # Build Y as the future return from last candle in window to prediction horizon
        end_indices = torch.arange(self.window_size - 1, len(values), step)  # indices of last candle in each window
        pred_horizons = end_indices+prediction_horizon  # indices of the candle we want to predict
        Y=[]
        for e,p in zip(end_indices, pred_horizons):
            Y.append(build_label(close_col[e:p], change_percentage_threshold=self.meta_data.get("change_percentage_threshold", 0.02)))
        Y = torch.stack(Y).unsqueeze(-1)  # (M, 1)
        # Y = ((close_col[pred_horizons]/close_col[end_indices]).log()*1e2).unsqueeze(-1)  # log return from last candle in window to prediction horizon candle
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
            "ts": ts.tolist() if ts is not None else None,
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

        deadline = time.monotonic() + 1000.0
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
        from_: pd.Timestamp | None = None,
        to_: pd.Timestamp | None = None,
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
        # return self.db.get_news_embeddings(equity=equity, start=None, end=None)
        news=self.finnhub_collector.collect(entities=equity, start_date=from_, end_date=to_, fields=["Date", "Article"])
        list_articles = [r["Article"] for r in news]
        if not list_articles:
                list_articles = [""]
        timestamps =  [pd.to_datetime(n['Date']) for n in news]
        #print(f"Fetched {len(list_articles)} news articles for {equity}. Sample:\n{list_articles[:3]}")
        embeddings= self._request_contextualizer(
                equity,
                timestamps,
                None,
                list_articles,
            )
        return {"timestamps": timestamps, "embeddings": embeddings}
    


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
        def _to_utc(ts):
            t = pd.Timestamp(ts)
            if t.tzinfo is None:
                return t.tz_localize("UTC")
            return t.tz_convert("UTC")

        timestamps = [_to_utc(t) for t in news["timestamps"]]
        end_timestamps = [_to_utc(t) for t in end_timestamps]
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
                # print(f"No news for window ending at {end_ts} (lookback to {start_ts})")
                # No news in this window
                window_embs = torch.zeros(1, embedding_dim)

            all_window_embeddings.append(window_embs)

        return all_window_embeddings

    def _pad_news_windows(self, X_news_list: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pad a list of (N_i, embedding_dim) tensors to (M, max_N, embedding_dim)
        with zero padding. Each tensor is truncated to max_news_per_window if needed.
        """
        if not X_news_list:
            raise ValueError("X_news_list is empty.")

        max_len = min(
            max(t.size(0) for t in X_news_list),
            self.config.max_news_per_window,
        )
        emb_dim = X_news_list[0].size(1)

        out  = torch.zeros(len(X_news_list), max_len, emb_dim, dtype=X_news_list[0].dtype)
        mask = torch.zeros(len(X_news_list), max_len, dtype=torch.bool)

        for i, t in enumerate(X_news_list):
            t = t[: max_len]              # ← truncate to max_len before assignment
            n = t.size(0)
            out[i, :n]  = t
            mask[i, :n] = True

        return out, mask

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fetch_dataloader(self, train_ratio: float = 0.95):
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
        logger.info("Starting to fetch dataloader...")
        # Step 1: OHLCV → normalize → sliding windows
        raw_df = self._fetch_df()
        logger.info(f"Fetched raw OHLCV data with {len(raw_df)} records. Sample:\n{raw_df.head()}")
        # norm_df = self._normalize(raw_df)
        X_ohlcv, Y, end_timestamps = self._build_sequences(raw_df)
        all=(Y.shape[0])

        falls=(Y==0).sum().item()
        raises=(Y==2).sum().item()
        flats=(Y==1).sum().item()
        
        falls_perc=(raises+flats)/2/all
        raises_perc=(falls+flats)/2/all
        flats_perc=(raises+falls)/2/all

        
        #print(X_ohlcv.shape, Y.shape, len(end_timestamps))
        logger.info(f"Built {len(X_ohlcv)} OHLCV windows. Sample window shape: {X_ohlcv[0].shape}, Y shape: {Y.shape}")
        # X_ohlcv: (M, window_size, 5), Y: (M, 1)

        # Step 2: fetch and align news embeddings
        X_news_list = self._align_news_to_windows(end_timestamps)
        X_news, X_news_mask = self._pad_news_windows(X_news_list)
        #print(X_news)
        #print(X_news.shape)
        # X_news: (M, embedding_dim)

        # Step 3: pass (OHLCV, news) pairs through the large model
        # The large model fuses both modalities and produces representation vectors
        logger.info("Passing data through the large model...")

        # representations : (M, d_model=256)
        # ctx_batch_size = 100
        # ctx_batches = []
        # for i in range(0, len(end_timestamps), ctx_batch_size):
        #     j = i + ctx_batch_size
        #     batch_timestamps = end_timestamps[i:j]
        #     batch_ts = X_ohlcv[i:j]
        #     batch_reps = self._request_contextualizer(
        #         self.equity,
        #         batch_timestamps,
        #         batch_ts,
        #         None,
        #     )
        #     ctx_batches.append(batch_reps)
        # ts_representations = torch.cat(ctx_batches, dim=0)
        # logger.info(f"Received contextualizer representations with shape: {ts_representations.shape}")
        

        # representations = torch.cat([ts_representations, X_news], dim=1)
        # Step 4: shuffle before splitting
        # perm = torch.randperm(X_ohlcv.shape[0])
        # X_ohlcv, X_news, X_news_mask, Y = X_ohlcv[perm], X_news[perm], X_news_mask[perm], Y[perm]
        # Y = Y[perm]

        split = int(len(X_ohlcv) * train_ratio)
        logger.info(f"Total samples: {len(X_ohlcv)}, Train: {split}, Test: {len(X_ohlcv) - split}")
        train_loader = DataLoader(
            TensorDataset(X_ohlcv[:split], X_news[:split], X_news_mask[:split], Y[:split]),
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )
        test_dataset = TensorDataset(X_ohlcv[split:], X_news[split:], X_news_mask[split:], Y[split:])
        weights= torch.tensor([falls_perc, flats_perc, raises_perc], dtype=torch.float32)
        return train_loader, test_dataset,weights
    

    def fetch_alliged_data(self,from_ , to_):
        news = self.fetch_news_embeddings_dataset(
            self.equity,
            prompt=self.news_retrieval_prompt,
            from_=from_,
            to_=to_,
        )
        from_ = pd.Timestamp(from_)   # ← coerce str → Timestamp
        to_   = pd.Timestamp(to_)

        records = self.rd.get_ohlc_df_interval(
            self.equity, interval=self.frequency, start=from_, end=to_
        )

        records=self.rd.get_ohlc_df_interval(self.equity, interval=self.frequency, start=from_, end=to_)
        df = pd.DataFrame(records)[["timestamp", "open", "high", "low", "close", "volume"]]
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
            df[col] = (df[col] - df[col].mean()) / (df[col].std() + 1e-8)  # standardize each feature
        df=df.set_index("timestamp").sort_index()

        wrapper=ContextWrapper(self.equity, news, df)
        return wrapper
    


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
            time_now = pd.Timestamp.now(tz="America/New_York")- pd.Timedelta(days=7)  # ensure we get the latest completed candle
            start_time = time_now - self.frequency_to_timedelta[self.frequency] * self.window_size
            logger.info(f"Fetching inference data for {self.equity} from {start_time} to {time_now}...")
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
            ts=(ts-ts.mean(dim=0,keepdim=True))/(ts.std(dim=0,keepdim=True)+1e-8)  # standardize
            ts_window = ts.unsqueeze(0)
            representation = self._request_contextualizer(
                self.equity,
                timestamps,
                None,
                list_articles,
            ).unsqueeze(0)

            return ts_window, representation,torch.ones((1,representation.shape[1]), dtype=torch.bool),
        except Exception:
            logger.exception("Failed to fetch inference data")
            return None  # Return a zero vector on failure
    def fetch_backtest_data(self) -> torch.Tensor:
        try:
            raw_df = self._fetch_backtest_df()
            
            # Step 1: OHLCV → normalize → sliding windows
            logger.info(f"Fetched raw OHLCV data with {len(raw_df)} records. Sample:\n{raw_df.head()}")
            # norm_df = self._normalize(raw_df)
            X_ohlcv, Y, end_timestamps = self._build_sequences(raw_df)
            

            
            #print(X_ohlcv.shape, Y.shape, len(end_timestamps))
            logger.info(f"Built {len(X_ohlcv)} OHLCV windows. Sample window shape: {X_ohlcv[0].shape}, Y shape: {Y.shape}")
            # X_ohlcv: (M, window_size, 5), Y: (M, 1)

            # Step 2: fetch and align news embeddings
            X_news_list = self._align_news_to_windows(end_timestamps)
            X_news, X_news_mask = self._pad_news_windows(X_news_list)
            #print(X_news)
            #print(X_news.shape)
            # X_news: (M, embedding_dim)

            # Step 3: pass (OHLCV, news) pairs through the large model
            # The large model fuses both modalities and produces representation vectors
            logger.info("Passing data through the large model...")

            return X_ohlcv, X_news, X_news_mask, Y
        
        except Exception:
            logger.exception("Failed to fetch backtest data")
            return None

def test_fetch_dataloader():
    metadata = {
        "equity": "AAPL",
        "time_frequency": "1h",
        "observation_horizon": 1000,
        "change_percentage_threshold": 0.02,
        "prediction_horizon": 240,
        "news_observation_horizon": 3000,
        "news_retrieval_prompt": "Fetch news articles related to {equity} in the last {news_observation_horizon} hours.",
        "news_resources": ["finnhub"],
    }
    loader = SignalingDataLoader(metadata)
    train_loader, test_dataset,label_perc = loader.fetch_dataloader()
    for batch in train_loader:
        X_batch,X_news_batch, X_news_mask_batch, Y_batch = batch
        logger.info(f"Train batch - X shape: {X_batch.shape}, X_news shape: {X_news_batch.shape}, X_news_mask shape: {X_news_mask_batch.shape}, Y shape: {Y_batch}")
        #print(Y_batch)
        #print(X_batch[:,-364:])
        break
    print(f"Train batches: {len(train_loader)}, Test samples: {len(test_dataset)}")

    
def test_fetch_inference_data():
    metadata = {
        "equity": "AAPL",
        "time_frequency": "1min",
        "observation_horizon": 1000,
        "prediction_horizon": 240,
        "news_observation_horizon": 1240,
        "news_retrieval_prompt": "Fetch news articles related to {equity} in the last {news_observation_horizon} hours.",
        "news_resources": ["finnhub"],
    }
    loader = SignalingDataLoader(metadata)
    representation = loader.fetch_inference_data()
    print(f"Inference representation shape: {representation.shape}")

def test_fetch_allinged_data():
    metadata = {
        "equity": "AAPL",
        "time_frequency": "1min",
        "observation_horizon": 1000,
        "prediction_horizon": 240,
        "news_observation_horizon": 1240,
        "news_retrieval_prompt": "Fetch news articles related to {equity} in the last {news_observation_horizon} hours.",
        "news_resources": ["finnhub"],
    }
    loader = SignalingDataLoader(metadata)
    from_ = "2026-01-01"
    to_ = "2026-01-07"
    wrapper = loader.fetch_alliged_data(from_, to_)
    res=wrapper.get("2026-01-01","2026-01-03")
    print(f"Aligned data from {from_} to {to_}: {res[0]}" )
    print(f"ContextWrapper equity: {wrapper.equity}")
    print(f"ContextWrapper timestamps: {wrapper.timestamps[:5]}")
    print(f"ContextWrapper ts shape: {wrapper.ts.shape}")
    print(f"ContextWrapper text_embd shape: {wrapper.text_embd.shape}")


if __name__ == "__main__":
    dl=SignalingDataLoader({
        "equity": "AAPL",
        "time_frequency": "1min",
        "observation_horizon": 1000,
        "prediction_horizon": 240,
        "news_observation_horizon": 1240,
        "news_retrieval_prompt": "Fetch news articles related to {equity} in the last {news_observation_horizon} hours.",
        "news_resources": ["finnhub"],
    })
    # dl.fetch_backtest_data()
    # test_fetch_dataloader()
    # test_fetch_inference_data()
    test_fetch_allinged_data()