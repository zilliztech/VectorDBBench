from abc import ABC, abstractmethod
from typing import Any, NamedTuple
from enum import IntEnum
from pydantic import BaseModel


class TaskConfig(NamedTuple):
    db: DB
    db_config: BaseDBConfig
    case_config: BaseCaseConfig


class DB(IntEnum):
    """Database types

    Examples:
        >>> DB.Milvus
        100
        >>> DB.Milvus.name
        "Milvus"
    """
    Milvus: 100
    ZillizCloud: 101

    def config(self) -> Any:
        """Get configs of the DB type
        Examples:
            >>> DB.Milvus.config()

        Returns:
            None, if the database not in the db2config
        """
        return db2config.get(self.name, None)


class BaseDBConfig(ABC):
    """Base interface for database configs"""
    pass


class MilvusConfig(BaseDBConfig, BaseModel):
    host: str
    port: int | str


    def __repr__(self) -> str:
        return f"MilvusConfig<host={self.host}, port={self.port}>"

    def parse_client(self):
        return 



class ZillizCloudConfig(BaseModel):
    uri:        str
    user:       str
    password:   str

    def __repr__(self) -> str:
        return f"ZillizCloudConfig<uri={self.uri}, user={self.user}>"


class TokenConfig(BaseModel):
    uri:    str
    token:  str

    def __repr__(self) -> str:
        return f"TokenConfig<uri={self.uri}>"


db2config = {
    "Milvus": MilvusConfig,
    "ZillizCloud": ZillizCloudConfig,
}
