from pydantic import BaseModel, Field
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
    password: str = Field(exclude=True)

    def to_dict(self) -> dict:
        return {"uri": self.uri, "user": self.user, "password": self.password}


class WeaviateConfig(DBConfig, BaseModel):
    url: str
    api_key: str

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "auth_client_secret": weaviate.AuthApiKey(apikey=self.api_key),
        }
