from vectordb_bench.backend.clients import MetricType
from vectordb_bench.backend.clients.elastic_cloud.config import ElasticCloudIndexConfig, ESElementType


def test_elasticsearch_hamming_metric_uses_bit_dense_vector():
    config = ElasticCloudIndexConfig(metric_type=MetricType.HAMMING, element_type=ESElementType.float)

    index_param = config.index_param()

    assert index_param["element_type"] == "bit"
    assert index_param["similarity"] == "l2_norm"
