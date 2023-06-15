from pydantic import SecretStr

from ..api import DBConfig


class QdrantConfig(DBConfig):
    url: SecretStr
    api_key: SecretStr

    def to_dict(self) -> dict:
        return {
            "url": self.url.get_secret_value(),
            "api_key": self.api_key.get_secret_value(),
            "prefer_grpc": True,
        }
