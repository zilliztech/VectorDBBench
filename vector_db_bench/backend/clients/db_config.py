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


class QdrantConfig(DBConfig, BaseModel):
    url: str
    api_key: SecretStr | None = None
    prefer_grpc: bool = True

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "api_key": self.api_key.get_secret_value(),
            "prefer_grpc": self.prefer_grpc,
        }


class ElasticsearchConfig(DBConfig, BaseModel):
    cloud_id: str
    password: str

    def to_dict(self) -> dict:
        return {"cloud_id": self.cloud_id, "basic_auth": ("elastic", self.password)}


class PineconeConfig(DBConfig, BaseModel):
    api_key: str
    environment: str
    index_name: str

    def to_dict(self) -> dict:
        return {
            "api_key": self.api_key,
            "environment": self.environment,
            "index_name": self.index_name,
        }
