from enum import StrEnum

from pydantic import BaseModel, SecretStr

from ..api import DBConfig
from ..elastic_cloud.config import ElasticCloudIndexConfig


class AliyunESQueryWireFormat(StrEnum):
    base64_f32be = "base64-f32be"
    cbor_f32le = "cbor-f32le"


class AliyunElasticsearchConfig(DBConfig, BaseModel):
    #: Protocol in use to connect to the node
    scheme: str = "http"
    host: str = ""
    port: int = 9200
    user: str = "elastic"
    password: SecretStr

    def to_dict(self) -> dict:
        return {
            "hosts": [{"scheme": self.scheme, "host": self.host, "port": self.port}],
            "basic_auth": (self.user, self.password.get_secret_value()),
        }


class AliyunElasticsearchIndexConfig(ElasticCloudIndexConfig):
    query_wire_format: AliyunESQueryWireFormat = AliyunESQueryWireFormat.base64_f32be

    def __eq__(self, obj: object) -> bool:
        return super().__eq__(obj) and self.query_wire_format == getattr(obj, "query_wire_format", None)

    def __hash__(self) -> int:
        return hash((super().__hash__(), self.query_wire_format))
