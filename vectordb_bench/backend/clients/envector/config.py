from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType, SQType


class EnVectorConfig(DBConfig):
    uri: SecretStr = SecretStr("http://localhost:50050")
    key_path: str = "keys"
    key_id: str = "default_key"

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
    metric_type: MetricType = MetricType.COSINE  # envector supports cosine similarity only
    eval_mode: str = "mm"  # default eval_mode

    def index_param(self) -> dict:
        return {
            "metric_type": "COSINE",
            "index_type": self.index.value,
            "eval_model": self.eval_mode,
            "params": {"index_type": "FLAT"},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": "COSINE",
            "search_params": {},
        }


class IVFFlatIndexConfig(EnVectorIndexConfig, DBCaseConfig):
    index: IndexType = IndexType.IVFFlat
    metric_type: MetricType = MetricType.COSINE  # envector supports cosine similarity only
    nlist : int = 0  # default nlist
    nprobe: int = 0  # default nprobe
    eval_mode: str = "mm"  # default eval_mode
    train_centroids: bool = False  # whether to train centroids before inserting data
    centroids: str | None = None  # path to centroids file
    is_vct: bool = False  # whether use VCT index

    def index_param(self) -> dict:
        return {
            "metric_type": "COSINE",
            "index_type": self.index.value,
            "eval_model": self.eval_mode,
            "params": {"index_type": "IVF_FLAT", "nlist": self.nlist, "default_nprobe": self.nprobe},
            "train_centroids": self.train_centroids,
            "centroids": self.centroids,
            "is_vct": self.is_vct,
        }

    def search_param(self) -> dict:
        return {
            "metric_type": "COSINE",
            "search_params": {"nprobe": self.nprobe},
        }


_envector_case_config = {
    IndexType.Flat: FlatIndexConfig,
    IndexType.IVFFlat: IVFFlatIndexConfig,
}
