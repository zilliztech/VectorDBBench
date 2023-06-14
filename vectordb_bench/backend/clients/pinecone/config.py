from pydantic import BaseModel, SecretStr
from ..api import DBConfig


class PineconeConfig(DBConfig, BaseModel):
    api_key: SecretStr | None = None
    environment: SecretStr | None = None
    index_name: str

    def to_dict(self) -> dict:
        return {
            "api_key": self.api_key.get_secret_value(),
            "environment": self.environment.get_secret_value(),
            "index_name": self.index_name,
        }
