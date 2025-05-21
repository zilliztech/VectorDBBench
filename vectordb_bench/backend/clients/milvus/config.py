from pydantic import BaseModel, SecretStr, validator

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType, SQType


class MilvusConfig(DBConfig):
    uri: SecretStr = "http://localhost:19530"
    user: str | None = None
    password: SecretStr | None = None

    def to_dict(self) -> dict:
        return {
            "uri": self.uri.get_secret_value(),
            "user": self.user if self.user else None,
            "password": self.password.get_secret_value() if self.password else None,
        }

    @validator("*")
    def not_empty_field(cls, v: any, field: any):
        if (
            field.name in cls.common_short_configs()
            or field.name in cls.common_long_configs()
            or field.name in ["user", "password"]
        ):
            return v
        if isinstance(v, str | SecretStr) and len(v) == 0:
            raise ValueError("Empty string!")
        return v


class MilvusIndexConfig(BaseModel):
    """Base config for milvus"""

    index: IndexType
    metric_type: MetricType | None = None

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


class AutoIndexConfig(MilvusIndexConfig, DBCaseConfig):
    index: IndexType = IndexType.AUTOINDEX

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
        }


class HNSWConfig(MilvusIndexConfig, DBCaseConfig):
    M: int
    efConstruction: int
    ef: int | None = None
    index: IndexType = IndexType.HNSW

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {"M": self.M, "efConstruction": self.efConstruction},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {"ef": self.ef},
        }


class HNSWSQConfig(HNSWConfig, DBCaseConfig):
    index: IndexType = IndexType.HNSW_SQ
    sq_type: SQType = SQType.SQ8
    refine: bool = True
    refine_type: SQType = SQType.FP32
    refine_k: float = 1

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {
                "M": self.M,
                "efConstruction": self.efConstruction,
                "sq_type": self.sq_type.value,
                "refine": self.refine,
                "refine_type": self.refine_type.value,
            },
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {"ef": self.ef, "refine_k": self.refine_k},
        }


class HNSWPQConfig(HNSWConfig):
    index: IndexType = IndexType.HNSW_PQ
    m: int = 32
    nbits: int = 8
    refine: bool = True
    refine_type: SQType = SQType.FP32
    refine_k: float = 1

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {
                "M": self.M,
                "efConstruction": self.efConstruction,
                "m": self.m,
                "nbits": self.nbits,
                "refine": self.refine,
                "refine_type": self.refine_type.value,
            },
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {"ef": self.ef, "refine_k": self.refine_k},
        }


class HNSWPRQConfig(HNSWPQConfig):
    index: IndexType = IndexType.HNSW_PRQ
    nrq: int = 2

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {
                "M": self.M,
                "efConstruction": self.efConstruction,
                "m": self.m,
                "nbits": self.nbits,
                "nrq": self.nrq,
                "refine": self.refine,
                "refine_type": self.refine_type.value,
            },
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {"ef": self.ef, "refine_k": self.refine_k},
        }


class DISKANNConfig(MilvusIndexConfig, DBCaseConfig):
    search_list: int | None = None
    index: IndexType = IndexType.DISKANN

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {"search_list": self.search_list},
        }


class IVFFlatConfig(MilvusIndexConfig, DBCaseConfig):
    nlist: int
    nprobe: int | None = None
    index: IndexType = IndexType.IVFFlat

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {"nlist": self.nlist},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {"nprobe": self.nprobe},
        }


class IVFPQConfig(MilvusIndexConfig, DBCaseConfig):
    nlist: int
    nprobe: int | None = None
    m: int = 32
    nbits: int = 8
    index: IndexType = IndexType.IVFPQ

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {"nlist": self.nlist, "m": self.m, "nbits": self.nbits},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {"nprobe": self.nprobe},
        }


class IVFSQ8Config(MilvusIndexConfig, DBCaseConfig):
    nlist: int
    nprobe: int | None = None
    index: IndexType = IndexType.IVFSQ8

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {"nlist": self.nlist},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {"nprobe": self.nprobe},
        }


class IVFRABITQConfig(IVFSQ8Config):
    index: IndexType = IndexType.IVF_RABITQ
    rbq_bits_query: int = 0  # 0, 1, 2, ..., 8
    refine: bool = True
    refine_type: SQType = SQType.FP32
    refine_k: float = 1

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {
                "nlist": self.nlist,
                "refine": self.refine,
                "refine_type": self.refine_type.value,
            },
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {"nprobe": self.nprobe, "rbq_bits_query": self.rbq_bits_query, "refine_k": self.refine_k},
        }


class FLATConfig(MilvusIndexConfig, DBCaseConfig):
    index: IndexType = IndexType.Flat

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {},
        }


class GPUIVFFlatConfig(MilvusIndexConfig, DBCaseConfig):
    nlist: int = 1024
    nprobe: int = 64
    cache_dataset_on_device: str
    refine_ratio: float | None = None
    index: IndexType = IndexType.GPU_IVF_FLAT

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {
                "nlist": self.nlist,
                "cache_dataset_on_device": self.cache_dataset_on_device,
            },
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {"nprobe": self.nprobe, "refine_ratio": self.refine_ratio},
        }


class GPUBruteForceConfig(MilvusIndexConfig, DBCaseConfig):
    limit: int = 10  # Default top-k for search
    metric_type: str  # Metric type (e.g., 'L2', 'IP', etc.)
    index: IndexType = IndexType.GPU_BRUTE_FORCE  # Index type set to GPU_BRUTE_FORCE

    def index_param(self) -> dict:
        """
        Returns the parameters for creating the GPU_BRUTE_FORCE index.
        No additional parameters required for index building.
        """
        return {
            "metric_type": self.parse_metric(),  # Metric type for distance calculation (L2, IP, etc.)
            "index_type": self.index.value,  # GPU_BRUTE_FORCE index type
            "params": {},  # No additional parameters for GPU_BRUTE_FORCE
        }

    def search_param(self) -> dict:
        """
        Returns the parameters for performing a search on the GPU_BRUTE_FORCE index.
        Only metric_type and top-k (limit) are needed for search.
        """
        return {
            "metric_type": self.parse_metric(),  # Metric type for search
            "params": {
                "nprobe": 1,  # For GPU_BRUTE_FORCE, set nprobe to 1 (brute force search)
                "limit": self.limit,  # Top-k for search
            },
        }


class GPUIVFPQConfig(MilvusIndexConfig, DBCaseConfig):
    nlist: int = 1024
    m: int = 0
    nbits: int = 8
    nprobe: int = 32
    refine_ratio: float | None = None
    cache_dataset_on_device: str
    index: IndexType = IndexType.GPU_IVF_PQ

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {
                "nlist": self.nlist,
                "m": self.m,
                "nbits": self.nbits,
                "cache_dataset_on_device": self.cache_dataset_on_device,
            },
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {"nprobe": self.nprobe, "refine_ratio": self.refine_ratio},
        }


class GPUCAGRAConfig(MilvusIndexConfig, DBCaseConfig):
    intermediate_graph_degree: int = 64
    graph_degree: int = 32
    itopk_size: int = 128
    team_size: int = 0
    search_width: int = 4
    min_iterations: int = 0
    max_iterations: int = 0
    build_algo: str = "IVF_PQ"  # IVF_PQ; NN_DESCENT;
    cache_dataset_on_device: str
    refine_ratio: float | None = None
    index: IndexType = IndexType.GPU_CAGRA

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {
                "intermediate_graph_degree": self.intermediate_graph_degree,
                "graph_degree": self.graph_degree,
                "build_algo": self.build_algo,
                "cache_dataset_on_device": self.cache_dataset_on_device,
            },
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {
                "team_size": self.team_size,
                "search_width": self.search_width,
                "itopk_size": self.itopk_size,
                "min_iterations": self.min_iterations,
                "max_iterations": self.max_iterations,
                "refine_ratio": self.refine_ratio,
            },
        }


_milvus_case_config = {
    IndexType.AUTOINDEX: AutoIndexConfig,
    IndexType.HNSW: HNSWConfig,
    IndexType.HNSW_SQ: HNSWSQConfig,
    IndexType.HNSW_PQ: HNSWPQConfig,
    IndexType.HNSW_PRQ: HNSWPRQConfig,
    IndexType.DISKANN: DISKANNConfig,
    IndexType.IVFFlat: IVFFlatConfig,
    IndexType.IVFPQ: IVFPQConfig,
    IndexType.IVFSQ8: IVFSQ8Config,
    IndexType.IVF_RABITQ: IVFRABITQConfig,
    IndexType.Flat: FLATConfig,
    IndexType.GPU_IVF_FLAT: GPUIVFFlatConfig,
    IndexType.GPU_IVF_PQ: GPUIVFPQConfig,
    IndexType.GPU_CAGRA: GPUCAGRAConfig,
    IndexType.GPU_BRUTE_FORCE: GPUBruteForceConfig,
}
