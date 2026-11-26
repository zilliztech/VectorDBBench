from pydantic import SecretStr

from vectordb_bench.backend.clients.api import DBConfig


class EndeeConfig(DBConfig):
    """Config for the Endee collections-based API (endee SDK >= 2.2.0)."""

    token: SecretStr
    base_url: str | None = None
    space_type: str = "cosine"
    precision: str = "int16"
    m: int | None = 16
    ef_con: int | None = 128
    ef_search: int | None = 128
    collection_name: str
    prefilter_cardinality_threshold: int | None = 10000
    filter_boost_percentage: int | None = 0

    def to_dict(self) -> dict:
        return {
            "token": self.token.get_secret_value(),
            "base_url": self.base_url,
            "space_type": self.space_type,
            "precision": self.precision,
            "m": self.m,
            "ef_con": self.ef_con,
            "ef_search": self.ef_search,
            "collection_name": self.collection_name,
            "prefilter_cardinality_threshold": self.prefilter_cardinality_threshold,
            "filter_boost_percentage": self.filter_boost_percentage,
        }
