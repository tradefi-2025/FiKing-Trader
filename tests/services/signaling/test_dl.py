import pandas as pd
import torch

from src.services.signaling.dl import SignalingDataLoader


class DummyMongo:
    def get_timeseries_data(self, equity, frequency):
        now = pd.Timestamp("2024-01-01")
        rows = []
        for i in range(10):
            rows.append(
                {
                    "timestamp": now + pd.Timedelta(minutes=i),
                    "open": 100 + i,
                    "high": 101 + i,
                    "low": 99 + i,
                    "close": 100 + i,
                    "volume": 1000 + i,
                }
            )
        return rows

    def get_news_embeddings(self, equity, start=None, end=None):
        return {
            "timestamps": [],
            "embeddings": torch.zeros(0, 768),
        }


def _make_loader():
    metadata = {
        "equity": "AAPL",
        "time_frequency": "1h",
        "observation_horizon": 3,
        "prediction_horizon": 1,
        "news_observation_horizon": 2,
        "news_retrieval_prompt": "",
        "news_resources": [],
    }
    loader = SignalingDataLoader(metadata=metadata)
    loader.db = DummyMongo()
    loader.config.step_size = 1
    return loader


def test_normalize_values():
    loader = _make_loader()
    df = pd.DataFrame(
        {
            "timestamp": [1, 2, 3],
            "open": [100, 110, 120],
            "high": [110, 120, 130],
            "low": [90, 100, 110],
            "close": [105, 115, 125],
            "volume": [1000, 1500, 2000],
        }
    )
    df = df.set_index("timestamp")

    norm_df = loader._normalize(df)

    assert norm_df["open"].tolist() == [(100 / 105 - 1), (110 / 115 - 1)]
    assert norm_df["high"].tolist() == [(110 / 105 - 1), (120 / 115 - 1)]
    assert norm_df["low"].tolist() == [(90 / 105 - 1), (100 / 115 - 1)]
    assert norm_df["close"].tolist() == [(105 / 105 - 1), (115 / 115 - 1)]
    assert norm_df["volume"].tolist() == [0.5, (2000 / 1500 - 1)]


def test_build_sequences_shapes():
    loader = _make_loader()
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=6, freq="h"),
            "open": [100, 101, 102, 103, 104, 105],
            "high": [101, 102, 103, 104, 105, 106],
            "low": [99, 100, 101, 102, 103, 104],
            "close": [100, 101, 102, 103, 104, 105],
            "volume": [1000, 1001, 1002, 1003, 1004, 1005],
        }
    ).set_index("timestamp")

    norm_df = loader._normalize(df)
    x, y, end_ts = loader._build_sequences(norm_df)

    assert x.ndim == 3
    assert y.ndim == 2
    assert len(end_ts) == y.shape[0]


def test_align_news_to_windows_empty():
    loader = _make_loader()

    def fake_news_dataset(equity, prompt="", resources=None):
        return {
            "timestamps": [],
            "embeddings": torch.zeros(0, 768),
        }

    loader.fetch_news_embeddings_dataset = fake_news_dataset
    end_ts = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02")]

    out = loader._align_news_to_windows(end_ts)
    assert len(out) == 2
    assert out[0].shape[1] == 768


def test_fetch_inference_data_shape():
    loader = _make_loader()

    def fake_fetch_df():
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
                "open": range(10),
                "high": range(10),
                "low": range(10),
                "close": range(10),
                "volume": range(10),
            }
        ).set_index("timestamp")

    loader._fetch_df = fake_fetch_df
    data = loader.fetch_inference_data()

    assert data.shape == (1, loader.window_size, 5)


def test_fetch_inference_data_raises_on_short_series():
    loader = _make_loader()

    def fake_fetch_df():
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=2, freq="h"),
                "open": [1, 2],
                "high": [1, 2],
                "low": [1, 2],
                "close": [1, 2],
                "volume": [1, 2],
            }
        ).set_index("timestamp")

    loader._fetch_df = fake_fetch_df

    try:
        loader.fetch_inference_data()
        assert False, "Expected ValueError for short series"
    except ValueError:
        assert True
