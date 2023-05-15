from pydantic import BaseModel
from abc import ABC, abstractmethod


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
    password: str

    def to_dict(self) -> dict:
        return {"uri": self.uri, "user": self.user, "password": self.password}
