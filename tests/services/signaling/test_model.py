import torch

from src.services.signaling.config import SignalingConfig
from src.services.signaling.model import SignalingModelV1, t_test


def _small_config():
    config = SignalingConfig()
    config.d_model = 8
    config.epoch = 1
    return config


def test_model_forward_shapes():
    config = _small_config()
    model = SignalingModelV1(config)

    x = torch.randn(2, config.d_model)
    pred, mu, logvar = model(x)

    assert pred.shape == (2, 1)
    assert mu.shape == (2,)
    assert logvar.shape == (2,)


def test_model_inference_output():
    config = _small_config()
    model = SignalingModelV1(config)

    x = torch.randn(1, config.d_model)
    result = model.inference(x, confidence_level=0.95)

    assert "estimated price" in result
    assert "prob" in result
    assert "mu" in result
    assert "logvar" in result
    assert "estimated action" in result


def test_t_test_signs():
    mu_pos = torch.tensor([1.0])
    logvar_small = torch.tensor([-10.0])
    res_pos = t_test(mu_pos, logvar_small, 0.95)

    mu_neg = torch.tensor([-1.0])
    res_neg = t_test(mu_neg, logvar_small, 0.95)

    mu_zero = torch.tensor([0.0])
    logvar_large = torch.tensor([1.0])
    res_zero = t_test(mu_zero, logvar_large, 0.95)

    assert res_pos.item() == 1
    assert res_neg.item() == -1
    assert res_zero.item() == 0
