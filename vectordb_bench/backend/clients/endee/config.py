from pydantic import SecretStr

from vectordb_bench.backend.clients.api import DBConfig


class EndeeConfig(DBConfig):
    token: SecretStr | None = None
    region: str | None = ""
    base_url: str = "http://127.0.0.1:8080/api/v1"
    space_type: str = "cosine"
    precision: str = "int8d"
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
