from pydantic import SecretStr

from ..api import DBConfig


class PineconeConfig(DBConfig):
    api_key: SecretStr
    index_name: str

    def to_dict(self) -> dict:
        return {
            "api_key": self.api_key.get_secret_value(),
            "index_name": self.index_name,
        }
