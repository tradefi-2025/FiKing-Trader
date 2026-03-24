from src.services.signaling.config import SignalingConfig


def test_signaling_config_defaults():
    config = SignalingConfig()

    assert config.name == "signaling"
    assert config.version == "v1"
    assert config.batch_size == 32
    assert config.learning_rate == 0.001
    assert config.d_model == 256
    assert config.nb_layers == 6
    assert config.lr == 0.001
    assert config.epoch == 100
    assert config.default_frequency == 60
    assert config.default_confidence_level == 0.95
