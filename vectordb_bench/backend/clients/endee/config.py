from typing import Optional
from pydantic import SecretStr
from vectordb_bench.backend.clients.api import DBConfig


class EndeeConfig(DBConfig):
    token: Optional[SecretStr] = None
    region: Optional[str] = "as1"
    base_url: str = "http://127.0.0.1:8080/api/v1"
    space_type: str = "cosine"
    precision: str = "int8d"
    version: Optional[int] = 1
    m: Optional[int] = 16
    ef_con: Optional[int] = 128
    ef_search: Optional[int] = 128
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