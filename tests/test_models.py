import pytest
from vector_db_bench.models import (
    DB,
    IndexType, MetricType,
    MilvusConfig,
    HNSWConfig, IVFFlatConfig
)


class TestModels:
    def test_db_milvus(self):
        assert DB.Milvus.value == "Milvus"
        assert DB.Milvus.config == MilvusConfig
        assert DB.Milvus.case_config_cls(IndexType.HNSW) == HNSWConfig
        assert DB.Milvus.case_config_cls(IndexType.IVFFlat) == IVFFlatConfig

        milvus_case_config_cls = DB.Milvus.case_config_cls(IndexType.Flat)
        c = milvus_case_config_cls(metric_type=MetricType.COSIN)
        assert c.index_param() == {
            'metric_type': "L2",
            'index_type': "FLAT",
            'params': {},
        }

        assert c.search_param() == {
            "metric_type": "L2",
            "params": {},
        }

        c = milvus_case_config_cls()
        c.metric_type = MetricType.COSIN
        assert c.index_param() == {
            'metric_type': "L2",
            'index_type': "FLAT",
            'params': {},
        }

        assert c.search_param() == {
            "metric_type": "L2",
            "params": {},
        }

        with pytest.raises(AssertionError):
            DB.Milvus.case_config_cls()
