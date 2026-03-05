import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import SignalingConfig
from ...database_handlers.mongoDB import MongoDBHandler

def fetch_news_embeddings(equity: str,prompt: str, From,To) -> torch.Tensor:
    '''
    Fetch news articles related to the equity between the given timestamps, pass them through a large language model to get embeddings, and return a tensor of shape (N, embedding_dim) where N is the number of news articles.
    '''
    # Placeholder implementation — replace with actual news fetching and embedding logic
    dummy_embeddings = torch.randn(10, 768)  # 10 news articles, 768-dimensional embeddings
    return dummy_embeddings

def fetch_news_embeddings_dataset(equity: str, prompt: str = "", resources: list = None) -> pd.DataFrame:
    '''
    Fetch news articles related to the equity from the given resources, pass them
    through a large language model guided by `prompt`, and return a DataFrame with
    columns ['timestamp', 'embedding'] where 'embedding' is a tensor (embedding_dim,).

    Args:
        equity    : ticker / entity name, e.g. "AAPL"
        prompt    : retrieval prompt that guides which news to fetch
        resources : list of news source identifiers (URLs, provider names, etc.)
    '''
    # Placeholder implementation — replace with actual news fetching and embedding logic
    data = {
        'timestamp': pd.date_range(start='2023-01-01', periods=10, freq='H'),
        'embedding': [torch.randn(768) for _ in range(10)]
    }
    return pd.DataFrame(data)

class SignalingDataLoader:
    """
    Fetches OHLCV data from MongoDB, builds sliding windows using PyTorch,
    and returns a DataLoader ready to be passed into model.fit().

    All hyper-parameters (window size, horizons, news config) come from
    `metadata` because they are set by the user on the platform when
    creating the agent — the worker passes them in directly.

    Usage (by the worker):
        dl = SignalingDataLoader(metadata=metadata_dataloader)
        train_loader, test_dataset = dl.fetch_dataloader(large_model)
        model.fit(train_loader)
    """

    def __init__(self, metadata: dict) -> None:
        self.config = SignalingConfig()               # created internally — no need to pass from outside
        self.meta_data = metadata

        # OHLCV / training parameters
        self.equity: str                 = metadata.get("equity")
        self.frequency: str              = metadata.get("time_frequency", "1h")
        self.frequency_to_timedelta = {
            "1m": pd.Timedelta(minutes=1),
            "5m": pd.Timedelta(minutes=5),
            "15m": pd.Timedelta(minutes=15),
            "30m": pd.Timedelta(minutes=30),
            "1h": pd.Timedelta(hours=1),
            "4h": pd.Timedelta(hours=4),
            "1d": pd.Timedelta(days=1),
        }
        self.window_size: int            = metadata.get("observation_horizon", 50)
        self.prediction_horizon: int     = metadata.get("prediction_horizon", 1)

        # News / multimodal parameters
        self.news_observation_horizon    = self.frequency_to_timedelta[self.frequency] * metadata.get("news_observation_horizon", 50)
        self.news_retrieval_prompt: str  = metadata.get("news_retrieval_prompt", "")
        self.news_resources: list        = metadata.get("news_resources", [])

        self.db = MongoDBHandler()

        
    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------
    def test_normalization(self):
        """
        Test that normalization works as expected on a simple example.
        """
        data = {
            "timestamp": [1, 2, 3],
            "open": [100, 110, 120],
            "high": [110, 120, 130],
            "low": [90, 100, 110],
            "close": [105, 115, 125],
            "volume": [1000, 1500, 2000],
        }
        df = pd.DataFrame(data)
        norm_df = self._normalize(df)

        expected_open = [(100/105 - 1), (110/115 - 1)]
        expected_high = [(110/105 - 1), (120/115 - 1)]
        expected_low = [(90/105 - 1), (100/115 - 1)]
        expected_close = [(105/105 - 1), (115/115 - 1)]
        expected_volume = [0.5, (2000/1500 - 1)]

        assert norm_df["open"].tolist() == expected_open
        assert norm_df["high"].tolist() == expected_high
        assert norm_df["low"].tolist() == expected_low
        assert norm_df["close"].tolist() == expected_close
        assert norm_df["volume"].tolist() == expected_volume
        print("Normalization test passed!")

    def _fetch_df(self) -> pd.DataFrame:
        """
        Fetch historical OHLCV candles from MongoDB.
        The data is NOT stored as an object — it is fetched fresh each time.
        """
        records = self.db.get_timeseries_data(
            equity=self.equity,
            frequency=self.frequency,
        )
        df = pd.DataFrame(records)[["timestamp", "open", "high", "low", "close", "volume"]]
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df.set_index("timestamp").sort_index()

    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert raw OHLCV into scale-free % changes.
        A 2% move on a $10 stock looks identical to a 2% move on a $1000 stock.

        open, high, low, close  →  % change relative to previous candle's close
        volume                  →  % change relative to previous candle's volume
        """
        norm = pd.DataFrame(index=df.index)
        norm["open"]   = df["open"]   / df["close"].shift(1) - 1
        norm["high"]   = df["high"]   / df["close"].shift(1) - 1
        norm["low"]    = df["low"]    / df["close"].shift(1) - 1
        norm["close"]  = df["close"]  / df["close"].shift(1) - 1
        norm["volume"] = df["volume"].pct_change()
        return norm.dropna()

    def _build_sequences(self, norm_df: pd.DataFrame):
        """
        Build (X, Y) sliding-window pairs using PyTorch unfold.

        N = total normalised candles
        M = N - window_size  (number of complete sequences)

        X  shape (M, window_size, 5)  — each row is window_size candles × 5 OHLCV values
        Y  shape (M, 1)               — each row is the NEXT candle's close return

        PyTorch unfold(dim, size, step):
            takes a tensor of shape (N, 5)
            slides a window of `size` along dim=0 with step=1
            produces shape (N - window_size + 1, 5, window_size)
            after permute → (N - window_size + 1, window_size, 5)
        """
        prediction_horizon = self.prediction_horizon
        values    = torch.tensor(norm_df.values.astype("float32"))[:-prediction_horizon]  # (N, 5)

        close_col = torch.FloatTensor(norm_df['close'].values.astype("float32") ) # (N,)

        step=self.config.step_size
        # Build X with unfold — no manual loop needed
        X = values.unfold(0, self.window_size, step)  # (N - window_size + 1, 5, window_size)
        
        X = X.permute(0, 2, 1)                     # (N - window_size + 1, window_size, 5)
        X = X[:-1]                                  # drop last (no Y exists for it) → (M, window_size, 5)

        # Y: next candle's close return after each window
        # Window i covers rows [i : i+window_size], so Y[i] = row[i+window_size].close
        end_indices = torch.arange(self.window_size - 1, len(values)-prediction_horizon, step)  # indices of last candle in each window
        pred_horizons = end_indices+prediction_horizon  # indices of the candle we want to predict
        Y = (close_col[pred_horizons]/close_col[end_indices]).log().unsqueeze(-1)  # log return from last candle in window to prediction horizon candle
        end_timestamps = norm_df.index[end_indices.tolist()].tolist()  # timestamps corresponding to each Y
        return X, Y, end_timestamps   # M = N - window_size

    # ------------------------------------------------------------------
    # Public — called by the worker
    # ------------------------------------------------------------------

    def _align_news_to_windows(self, end_timestamps: list) -> torch.Tensor:
        """
        For each OHLCV window, collect ALL news embeddings whose timestamp
        falls within [end_timestamp - news_observation_horizon, end_timestamp].

        This ensures the model never sees future news — no data leakage.
        Windows with fewer than N articles are zero-padded; windows with
        more than N articles keep only the N most recent ones.

        Args:
            end_timestamps : list of timestamps, one per OHLCV window (length M)

        Returns:
            torch.Tensor  shape (M, N, embedding_dim)
            N = news_observation_horizon (max articles per window).
            If fewer articles exist for a window, remaining slots are zero vectors.
        """
        news_df = fetch_news_embeddings_dataset(
            self.equity,
            prompt=self.news_retrieval_prompt,
            resources=self.news_resources,
        )
        news_df = news_df.sort_values("timestamp").reset_index(drop=True)

        embedding_dim = news_df["embedding"].iloc[0].shape[0]
        horizon = self.news_observation_horizon  # max news articles per window


        all_window_embeddings = []  # will hold M tensors of shape (N, embedding_dim)

        for end_ts in end_timestamps:
            start_ts = end_ts - horizon

            # Filter news within [start_ts, end_ts] — no future leakage
            mask = (news_df["timestamp"] >= start_ts) & (news_df["timestamp"] <= end_ts)
            window_news = news_df.loc[mask].sort_values("timestamp", ascending=False)

            # Collect embeddings (most recent first), cap at N
            embs = []
            for _, row in window_news.iterrows():
                emb = row["embedding"]
                if emb is None or isinstance(emb, float):
                    embs.append(torch.zeros(embedding_dim))
                else:
                    embs.append(emb)

            # Pad with zero vectors if fewer than N articles
            if not len(embs):
                embs.append(torch.zeros(embedding_dim))


            all_window_embeddings.append(torch.stack(embs))  # (N, embedding_dim)

        return all_window_embeddings  # (M, N, embedding_dim)

    def fetch_dataloader(self, large_model, train_ratio: float = 0.8):
        """
        Full pipeline:
            MongoDB → normalize → OHLCV windows
            + news embeddings aligned to each window
            → large model encodes (OHLCV, news) pairs → representation vectors
            → DataLoader of representations ready for model.fit()

        The DataLoader is a stack of representation vectors from the large model,
        not raw OHLCV — the large model has already fused both modalities.

        Args:
            large_model  : the encoder model with a .encode(X_ohlcv, X_news) method
                           that returns a representation tensor (M, d_model)
            train_ratio  : fraction used for training (default 0.8)

        Returns:
            train_loader  : DataLoader  — pass directly into model.fit()
                            batches of (representation, Y)
                            representation.shape = (batch_size, d_model)
                            Y.shape              = (batch_size, 1)
            test_dataset  : TensorDataset — evaluate once after training
        """
        # Step 1: OHLCV → normalize → sliding windows
        norm_df = self._normalize(self._fetch_df())
        X_ohlcv, Y, end_timestamps = self._build_sequences(norm_df)
        # X_ohlcv : (M, window_size, 5)
        # Y        : (M, 1)

        # Step 2: fetch news embeddings aligned to each OHLCV window
        X_news = self._align_news_to_windows(end_timestamps)
        # X_news : (M, embedding_dim)

        # Step 3: pass (OHLCV, news) pairs through the large model
        # The large model fuses both modalities and produces representation vectors
        with torch.no_grad():
            representations = large_model.encode(X_ohlcv, X_news)
        # representations : (M, d_model=256)

        # Step 4: shuffle before split
        perm = torch.randperm(len(representations))
        representations = representations[perm]
        Y = Y[perm]

        split = int(len(representations) * train_ratio)

        train_loader = DataLoader(
            TensorDataset(representations[:split], Y[:split]),
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True,
        )
        test_dataset = TensorDataset(representations[split:], Y[split:])

        return train_loader, test_dataset

    def fetch_inference_data(self) -> torch.Tensor:
        '''
        For theinference we need to first get the current time, then compute the timestamp for the input data ( current time - window_size * frequency), then fetch the candles from external api starting from that timestamp, normalize them, and return the most recent window as a tensor ready for model.inference().
        '''
        """
        Fetch latest candles from MongoDB and return the most recent window
        as a tensor ready for model.inference().

        Returns:
            torch.Tensor  shape (1, window_size, 5)
        """
        norm_df = self._normalize(self._fetch_df())

        if len(norm_df) < self.window_size:
            raise ValueError(
                f"Need at least {self.window_size} candles, got {len(norm_df)}."
            )

        last_window = torch.tensor(
            norm_df.values[-self.window_size:].astype("float32")
        )                            # (window_size, 5)
        return last_window.unsqueeze(0)  # (1, window_size, 5)

