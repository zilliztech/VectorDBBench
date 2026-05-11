from vectordb_bench.backend.cases import CloudPayloadSearchCase
from vectordb_bench.backend.clients.milvus.milvus import Milvus
from vectordb_bench.backend.dataset import Dataset
from vectordb_bench.backend.filter import FilterOp
from vectordb_bench.backend.payload import PayloadProfile


class _Client:
    def __init__(self):
        self.search_kwargs = None

    def search(self, **kwargs):
        self.search_kwargs = kwargs
        return [[{"pk": 1}]]


def test_laion_100m_declares_scalar_label_assets():
    laion = Dataset.LAION.get(100_000_000)

    assert laion.with_scalar_labels is True
    assert laion.scalar_labels_file == "scalar_labels.parquet"
    assert 0.01 in laion.scalar_label_percentages


def test_scalar_label_payload_profile_estimates_small_string_payload():
    assert PayloadProfile.SCALAR_LABEL.estimated_bytes_per_query(k=100, dim=768) == 3600


def test_cloud_payload_case_can_combine_label_filter_with_scalar_label_payload():
    case = CloudPayloadSearchCase(
        payload_profile="scalar_label",
        label_percentage=0.01,
    )

    assert case.payload_profile == PayloadProfile.SCALAR_LABEL
    assert case.filters.type == FilterOp.StrEqual
    assert case.filters.label_value == "label_1p"


def test_milvus_scalar_label_payload_requests_label_output_field():
    db = Milvus.__new__(Milvus)
    db.client = _Client()
    db.case_config = type("CaseConfig", (), {"search_param": lambda self: {}})()
    db.collection_name = "collection"
    db.expr = "label == 'label_1p'"
    db._primary_field = "pk"
    db._vector_field = "vector"
    db._scalar_label_field = "label"

    assert db.supports_payload_profile(PayloadProfile.SCALAR_LABEL)
    assert db.search_embedding([0.1, 0.2], payload_profile=PayloadProfile.SCALAR_LABEL) == [1]
    assert db.client.search_kwargs["output_fields"] == ["label"]
