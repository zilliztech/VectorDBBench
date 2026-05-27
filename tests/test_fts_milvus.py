from vectordb_bench.backend.clients.milvus.config import MilvusFtsConfig
from vectordb_bench.backend.clients.milvus.milvus import Milvus


def test_milvus_fts_config_uses_analyzer_max_token_length():
    config = MilvusFtsConfig(
        analyzer_tokenizer="standard",
        analyzer_enable_lowercase=True,
        analyzer_max_token_length=12,
        analyzer_stop_words="the,and",
    )

    params = config.index_param()["analyzer_params"]

    assert params["tokenizer"] == "standard"
    assert "lowercase" in params["filter"]
    assert {"type": "length", "max": 12} in params["filter"]
    assert {"type": "stop", "stop_words": ["the", "and"]} in params["filter"]


def test_milvus_declares_full_text_support():
    assert Milvus.supports_full_text_search() is True
