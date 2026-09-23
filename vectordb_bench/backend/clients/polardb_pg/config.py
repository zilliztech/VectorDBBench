from typing import Literal

from pydantic import AliasChoices, Field, PositiveInt, model_validator

from ..api import IndexType
from ..pgvector.config import PgVectorConfig, PgVectorHNSWConfig, PgVectorIndexParam

DEFAULT_GRAPH_CACHE_TIMEOUT_SECONDS = 3600


class PolarDBPgConfig(PgVectorConfig):
    """Connection configuration for PolarDB for PostgreSQL."""


class PolarDBPgHNSWConfig(PgVectorHNSWConfig):
    """PolarDB HNSW-specific benchmark options."""

    post_load_index: bool | None = None
    iterative_scan: str = "off"
    graph_cache: bool = True
    graph_cache_timeout: PositiveInt = DEFAULT_GRAPH_CACHE_TIMEOUT_SECONDS
    quantization: Literal["pq", "sq4", "sq8", "rabitq"] | None = Field(
        default=None,
        validation_alias=AliasChoices("quantization", "hnsw_quantization"),
    )
    pq_m: int | None = None
    train_samples: int | None = None
    quantization_nbits: Literal[1, 4, 8] | None = None

    @model_validator(mode="after")
    def validate_polardb_options(self) -> "PolarDBPgHNSWConfig":
        if self.post_load_index is not None:
            self.create_index_before_load = not self.post_load_index
            self.create_index_after_load = self.post_load_index
        if self.iterative_scan not in {"off", "strict_order", "relaxed_order"}:
            msg = "iterative_scan must be one of: off, strict_order, relaxed_order"
            raise ValueError(msg)
        if self.graph_cache and self.iterative_scan != "off":
            msg = "Graph Cache requires iterative_scan=off"
            raise ValueError(msg)
        if self.pq_m is not None and self.quantization != "pq":
            msg = "pq_m is only valid with quantization=pq"
            raise ValueError(msg)
        if self.quantization_nbits is not None and self.quantization != "rabitq":
            msg = "quantization_nbits is only valid with quantization=rabitq"
            raise ValueError(msg)
        if self.train_samples is not None and self.quantization is None:
            msg = "train_samples requires a quantization method"
            raise ValueError(msg)
        return self

    def index_param(self) -> PgVectorIndexParam:
        index_param = super().index_param()
        polar_options = {
            "quantization": self.quantization,
            "pq_m": self.pq_m,
            "train_samples": self.train_samples,
            "quantization_nbits": self.quantization_nbits,
            "cache": "on" if self.graph_cache else None,
        }
        index_param["index_creation_with_options"] = [
            *index_param["index_creation_with_options"],
            *self._optionally_build_with_options(polar_options),
        ]
        return index_param


_polardb_pg_case_config = {
    IndexType.HNSW: PolarDBPgHNSWConfig,
    IndexType.ES_HNSW: PolarDBPgHNSWConfig,
}
