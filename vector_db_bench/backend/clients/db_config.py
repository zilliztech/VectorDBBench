from pydantic import BaseModel, SecretStr
from abc import ABC, abstractmethod
import weaviate


class DBConfig(ABC, BaseModel):

    db_label: str | None = None

    @abstractmethod
    def to_dict(self) -> dict:
        raise NotImplementedError


class MilvusConfig(DBConfig, BaseModel):
    uri: SecretStr | None = "http://localhost:19530"

    def to_dict(self) -> dict:
        return {"uri": self.uri.get_secret_value()}


class ZillizCloudConfig(DBConfig, BaseModel):
    uri: SecretStr | None = None
    user: str
    password: SecretStr | None = None

    def to_dict(self) -> dict:
        return {
            "uri": self.uri.get_secret_value(),
            "user": self.user,
            "password": self.password.get_secret_value(),
        }


class WeaviateConfig(DBConfig, BaseModel):
    url: SecretStr | None = None
    api_key: SecretStr | None = None

    def to_dict(self) -> dict:
        return {
            "url": self.url.get_secret_value(),
            "auth_client_secret": weaviate.AuthApiKey(api_key=self.api_key.get_secret_value()),
        }


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


class ElasticsearchConfig(DBConfig, BaseModel):
    cloud_id: SecretStr
    password: SecretStr | None = None

    def to_dict(self) -> dict:
        return {
            "cloud_id": self.cloud_id.get_secret_value(),
            "basic_auth": ("elastic", self.password.get_secret_value()),
        }


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
