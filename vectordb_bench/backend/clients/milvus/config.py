from pydantic import BaseModel, SecretStr, validator

from ..api import DBCaseConfig, DBConfig, IndexType, MetricType, SQType


class MilvusConfig(DBConfig):
    uri: SecretStr = "http://localhost:19530"
    user: str | None = None
    password: SecretStr | None = None
    num_shards: int = 1
    replica_number: int = 1

    def to_dict(self) -> dict:
        return {
            "uri": self.uri.get_secret_value(),
            "user": self.user if self.user else None,
            "password": self.password.get_secret_value() if self.password else None,
            "num_shards": self.num_shards,
            "replica_number": self.replica_number,
        }

    @validator("*", allow_reuse=True)
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
    use_partition_key: bool = False  # for label-filter

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


class SCANNConfig(MilvusIndexConfig, DBCaseConfig):
    nlist: int = 1024
    with_raw_data: bool = False
    nprobe: int = 64
    reorder_k: int | None = 100
    index: IndexType = IndexType.SCANN_MILVUS

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": "SCANN",
            "params": {
                "nlist": self.nlist,
                "with_raw_data": self.with_raw_data,
            },
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "params": {
                "nprobe": self.nprobe,
                "reorder_k": self.reorder_k,
            },
        }


class MilvusFtsConfig(BaseModel, DBCaseConfig):
    """
    1. inverted_index_algo: 索引算法选择
       - "DAAT_MAXSCORE" (默认): 适合高k值或包含大量术语的查询,性能均衡
       - "DAAT_WAND": 适合k值较小的查询或较短的查询,更快
       - "TAAT_NAIVE": 能动态适应Collections变化(如avgdl),但速度较慢
    2. bm25_k1: BM25 词频饱和度控制 [1.2, 2.0],默认1.5
       - 较高值:增加词频在文档排名中的重要性
       - 建议范围:1.2-1.8,根据查询特点调整
    3. bm25_b: BM25 文档长度归一化控制 [0.0, 1.0],默认0.75
       - 1.0: 不进行长度归一化,长文档占优势
       - 0.0: 完全归一化,短文档占优势
       - 0.75: 平衡长度归一化,常用默认值
    4. analyzer_tokenizer: 分词器类型,默认"standard"
       - "standard": 标准分词器,适合英文
       - "whitespace": 按空白字符分割
       - "keyword": 不分词,保持原文本
    5. analyzer_enable_lowercase: 启用小写转换,默认True
       - True: 所有文本转为小写,提高匹配率
       - False: 保持原始大小写
    6. analyzer_max_token_length: 单个token最大长度,默认40
       - 限制过长词汇的长度
       - 设为None禁用此限制
    7. analyzer_stop_words: 停用词列表,默认None
       - 用逗号分隔的停用词,如"of,to,the,and,or"
       - 这些词会被过滤掉,不参与索引和搜索
    8. drop_ratio_search: 搜索时忽略最小值的比例 [0.0, 1.0],默认None
       - 0.0: 保留所有值,最高召回率
       - 0.1-0.3: 提升搜索速度10-20%,轻微影响召回率
    """

    index_type: str = "SPARSE_INVERTED_INDEX"
    metric_type: str = "BM25"
    inverted_index_algo: str = "DAAT_MAXSCORE"  # DAAT_MAXSCORE | DAAT_WAND | TAAT_NAIVE
    bm25_k1: float = 1.5  # BM25 k1参数,控制词频饱和度 [1.2, 2.0]
    bm25_b: float = 0.75  # BM25 b参数,控制文档长度归一化 [0.0, 1.0]
    analyzer_tokenizer: str = "standard"  # 分词器类型,standard, whitespace, etc.
    analyzer_enable_lowercase: bool = True
    analyzer_max_token_length: int | None = 40
    analyzer_stop_words: str | None = None  # 停用词列表,用逗号分隔
    drop_ratio_search: float | None = None

    def index_param(self) -> dict:
        params = {
            "inverted_index_algo": self.inverted_index_algo,
        }
        if hasattr(self, "bm25_k1") and self.bm25_k1 is not None:
            params["bm25_k1"] = self.bm25_k1
        if hasattr(self, "bm25_b") and self.bm25_b is not None:
            params["bm25_b"] = self.bm25_b

        # 构建分析器参数
        analyzer_params = {"type": "english"}
        # 设置分词器
        if hasattr(self, "analyzer_tokenizer") and self.analyzer_tokenizer:
            analyzer_params["tokenizer"] = self.analyzer_tokenizer
        # 构建过滤器数组
        filters = []
        if hasattr(self, "analyzer_enable_lowercase") and self.analyzer_enable_lowercase:
            filters.append("lowercase")

        if hasattr(self, "analyzer_max_token_length") and self.analyzer_max_token_length:
            filters.append({"type": "length", "max": self.analyzer_max_token_length})

        if hasattr(self, "analyzer_stop_words") and self.analyzer_stop_words:
            stop_words = [word.strip() for word in self.analyzer_stop_words.split(",") if word.strip()]
            if stop_words:
                filters.append({"type": "stop", "stop_words": stop_words})

        if filters:
            analyzer_params["filter"] = filters

        return {
            "index_type": self.index_type,
            "metric_type": self.metric_type,
            "params": params,
            "analyzer_params": analyzer_params,
        }

    def search_param(self) -> dict:

        params: dict = {}
        if self.drop_ratio_search is not None:
            params["drop_ratio_search"] = self.drop_ratio_search
        return {
            "metric_type": self.metric_type,
            "params": params,
        }


_milvus_case_config = {
    IndexType.AUTOINDEX: AutoIndexConfig,
    IndexType.FTS_AUTOINDEX: MilvusFtsConfig,
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
    IndexType.SCANN_MILVUS: SCANNConfig,
}
