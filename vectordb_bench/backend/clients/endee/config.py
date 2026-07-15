from pydantic import SecretStr

from vectordb_bench.backend.clients.api import DBConfig


class EndeeConfig(DBConfig):
    """Config for the Endee collections-based API (v2)."""

    token: SecretStr | None = None
    region: str | None = None
    base_url: str | None = None
    space_type: str = "cosine"
    precision: str = "int16"
    version: str | None = None
    m: int | None = 16
    ef_con: int | None = 128
    ef_search: int | None = 128
    collection_name: str
    prefilter_cardinality_threshold: int | None = 10000
    filter_boost_percentage: int | None = 0

    def to_dict(self) -> dict:
        return {
            "token": self.token.get_secret_value() if self.token else None,
            "region": self.region,
            "base_url": self.base_url,
            "space_type": self.space_type,
            "precision": self.precision,
            "version": self.version,
            "m": self.m,
            "ef_con": self.ef_con,
            "ef_search": self.ef_search,
            "collection_name": self.collection_name,
            "prefilter_cardinality_threshold": self.prefilter_cardinality_threshold,
            "filter_boost_percentage": self.filter_boost_percentage,
        }


class EndeeOSSConfig(DBConfig):
    """Config for Endee OSS (v1, index-based API).

    Python Package: pip install endee==1.0.0
    Docs: https://docs.endee.io/v1/overview
    OSS Repo: https://github.com/endee-io/endee
    """

    token: SecretStr | None = None
    region: str | None = ""
    base_url: str = "http://127.0.0.1:8080/api/v1"
    space_type: str = "cosine"
    precision: str = "int16"
    version: int | None = 1
    m: int | None = 16
    ef_con: int | None = 128
    ef_search: int | None = 128
    index_name: str

    def to_dict(self) -> dict:
        return {
            "token": self.token.get_secret_value() if self.token else None,
            "region": self.region,
            "base_url": self.base_url,
            "space_type": self.space_type,
            "precision": self.precision,
            "version": self.version,
            "m": self.m,
            "ef_con": self.ef_con,
            "ef_search": self.ef_search,
            "index_name": self.index_name,
        }
