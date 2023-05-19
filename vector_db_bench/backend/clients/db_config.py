from pydantic import BaseModel, SecretStr
from abc import ABC, abstractmethod
import weaviate

class DBConfig(ABC):
    @abstractmethod
    def to_dict(self) -> dict:
        raise NotImplementedError


class MilvusConfig(DBConfig, BaseModel):
    uri: str = "http://localhost:19530"

    def to_dict(self) -> dict:
        return {"uri": self.uri}


class ZillizCloudConfig(DBConfig, BaseModel):
    uri: str
    user: str
    password: SecretStr | None = None

    def to_dict(self) -> dict:
        return {"uri": self.uri, "user": self.user, "password": self.password.get_secret_value()}


class WeaviateConfig(DBConfig, BaseModel):
    url: str
    api_key: SecretStr | None = None

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "auth_client_secret": weaviate.AuthApiKey(api_key=self.api_key.get_secret_value()),
        }
