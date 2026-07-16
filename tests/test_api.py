from smarteda import Config, SmartEDA


def test_public_api_exposes_smarteda_and_config():
    assert SmartEDA is not None
    assert Config().random_state == 42


def test_config_allows_custom_sampling():
    config = Config(sample_size=100, correlation_threshold=0.3)

    assert config.sample_size == 100
    assert config.correlation_threshold == 0.3
