from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType


class PinotConfig(DBConfig):
    controller_host: str = "localhost"
    controller_port: int = 9000
    broker_host: str = "localhost"
    broker_port: int = 8099
    username: str | None = None
    password: SecretStr | None = None
    # Rows buffered before flushing one Pinot segment (one ingestFromFile call).
    # Larger values → fewer segments → better IVF training & query perf.
    # 100_000 rows x 768-dim float32 ~= 300 MB in-memory.
    ingest_batch_size: int = 100_000

    def to_dict(self) -> dict:
        return {
            "controller_host": self.controller_host,
            "controller_port": self.controller_port,
            "broker_host": self.broker_host,
            "broker_port": self.broker_port,
            "username": self.username,
            "password": self.password.get_secret_value() if self.password else None,
            "ingest_batch_size": self.ingest_batch_size,
        }


class PinotHNSWConfig(BaseModel, DBCaseConfig):
    """HNSW vector index config for Apache Pinot (Lucene-based)."""

    metric_type: MetricType | None = None
    m: int = 16  # maxCon: max connections per node
    ef_construction: int = 100  # beamWidth: construction beam width
    ef: int | None = None  # ef_search: HNSW candidate list size at query time (default=k)

    def index_param(self) -> dict:
        return {
            "vectorIndexType": "HNSW",
            "maxCon": str(self.m),
            "beamWidth": str(self.ef_construction),
        }

    def search_param(self) -> dict:
        # ef controls the HNSW candidate list during search via vectorSimilarity(col, q, ef).
        # Larger ef → better recall, slightly higher latency. Defaults to k if not set.
        return {"ef": self.ef} if self.ef is not None else {}


class PinotIVFFlatConfig(BaseModel, DBCaseConfig):
    """IVF_FLAT vector index config for Apache Pinot."""

    metric_type: MetricType | None = None
    nlist: int = 128  # number of Voronoi cells (centroids)
    quantizer: str = "FLAT"  # FLAT, SQ8, or SQ4
    train_sample_size: int | None = None  # defaults to max(nlist*50, 1000) if None
    nprobe: int = 8  # number of cells to probe at query time

    def index_param(self) -> dict:
        params: dict = {
            "vectorIndexType": "IVF_FLAT",
            "nlist": str(self.nlist),
            "quantizer": self.quantizer,
        }
        if self.train_sample_size is not None:
            params["trainSampleSize"] = str(self.train_sample_size)
        return params

    def search_param(self) -> dict:
        return {"nprobe": self.nprobe}


class PinotIVFPQConfig(BaseModel, DBCaseConfig):
    """IVF_PQ vector index config for Apache Pinot (residual product quantization)."""

    metric_type: MetricType | None = None
    nlist: int = 128  # number of Voronoi cells (centroids)
    pq_m: int = 8  # number of sub-quantizers (must divide vectorDimension)
    pq_nbits: int = 8  # bits per sub-quantizer code: 4, 6, or 8
    train_sample_size: int = 6400  # training sample size (must be >= nlist)
    nprobe: int = 8  # number of cells to probe at query time

    def index_param(self) -> dict:
        return {
            "vectorIndexType": "IVF_PQ",
            "nlist": str(self.nlist),
            "pqM": str(self.pq_m),
            "pqNbits": str(self.pq_nbits),
            "trainSampleSize": str(self.train_sample_size),
        }

    def search_param(self) -> dict:
        return {"nprobe": self.nprobe}
