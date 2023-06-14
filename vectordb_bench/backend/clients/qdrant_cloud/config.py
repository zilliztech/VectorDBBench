from pydantic import BaseModel, SecretStr

from ..api import DBConfig


class QdrantConfig(DBConfig, BaseModel):
    url: SecretStr | None = None
    api_key: SecretStr | None = None
    prefer_grpc: bool = True

    def to_dict(self) -> dict:
        return {
            "url": self.url.get_secret_value(),
            "api_key": self.api_key.get_secret_value(),
            "prefer_grpc": self.prefer_grpc,
        }
