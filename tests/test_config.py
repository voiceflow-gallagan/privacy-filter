import os
from app.config import Settings


def test_defaults_match_spec(monkeypatch):
    for key in ("DEVICE", "DEFAULT_MODE", "MAX_TEXT_LENGTH",
                "CHUNK_SIZE_TOKENS", "CHUNK_OVERLAP_TOKENS", "SMART_SPLIT",
                "RATE_LIMIT_ENABLED", "RATE_LIMIT_PER_IP",
                "MAX_BATCH_SIZE", "MAX_BATCH_TOTAL_TOKENS",
                "MAX_CONCURRENT_INFERENCES", "MODEL_NAME", "LOG_LEVEL"):
        monkeypatch.delenv(key, raising=False)

    s = Settings()
    assert s.device == "cpu"
    assert s.default_mode == "balanced"
    assert s.max_text_length == 524_288
    assert s.chunk_size_tokens == 120_000
    assert s.chunk_overlap_tokens == 512
    assert s.smart_split is True
    assert s.rate_limit_enabled is True
    assert s.rate_limit_per_ip == "60/10minutes"
    assert s.max_batch_size == 32
    assert s.max_batch_total_tokens == 200_000
    assert s.max_concurrent_inferences == 2
    assert s.model_name == "openai/privacy-filter"
    assert s.log_level == "info"


def test_overrides_from_env(monkeypatch):
    monkeypatch.setenv("DEVICE", "cuda")
    monkeypatch.setenv("MAX_BATCH_SIZE", "8")
    monkeypatch.setenv("RATE_LIMIT_ENABLED", "false")
    s = Settings()
    assert s.device == "cuda"
    assert s.max_batch_size == 8
    assert s.rate_limit_enabled is False
