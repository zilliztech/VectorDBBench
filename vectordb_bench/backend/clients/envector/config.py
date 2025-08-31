from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType, SQType


class EnVectorConfig(DBConfig):
    uri: SecretStr = SecretStr("http://localhost:50050")
    key_path: str = "keys"
    key_id: str = "default"

    def to_dict(self) -> dict:
        return {
            "uri": self.uri.get_secret_value(),
            "key_path": self.key_path,
            "key_id": self.key_id,
        }


class EnVectorIndexConfig(BaseModel):
    """Base config for envector"""

    index: IndexType
    metric_type: MetricType | None = None
    use_partition_key: bool = True  # for label-filter

    @property
    def is_gpu_index(self) -> bool:
        return self.index in [
            IndexType.GPU_CAGRA,
            IndexType.GPU_IVF_FLAT,
            IndexType.GPU_IVF_PQ,
            IndexType.GPU_BRUTE_FORCE,
        ]

    def parse_metric(self) -> str:
        if not self.metric_type:
            return ""

        if self.is_gpu_index and self.metric_type == MetricType.COSINE:
            return MetricType.L2.value
        return self.metric_type.value


class FlatIndexConfig(EnVectorIndexConfig, DBCaseConfig):
    index: IndexType = IndexType.Flat
    metric_type: MetricType = MetricType.COSINE  # envector는 cosine 유사도만 지원

    def index_param(self) -> dict:
        return {
            "metric_type": "COSINE",  # envector는 내적 기반 cosine만 지원
            "index_type": self.index.value,
            "params": {},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": "COSINE",  # envector는 내적 기반 cosine만 지원
        }


_envector_case_config = {
    IndexType.Flat: FlatIndexConfig,
}
