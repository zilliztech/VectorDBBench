from typing import Any
from enum import IntEnum


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


class MilvusConfig:
    host: str
    port: int | str

    def __init__(self, host: str, port: int | str):
        self.host = host
        self.port = port

    def __repr__(self) -> str:
        return f"MilvusConfig<host={self.host}, port={self.port}>"


class ZillizCloudConfig:
    uri:        str
    user:       str
    password:   str

    def __init__(self, uri: str, user: str, password: str):
        self.uri = uri
        self.user = user
        self.password = password

    def __repr__(self) -> str:
        return f"ZillizCloudConfig<uri={self.uri}, user={self.user}>"


class TokenConfig:
    uri:    str
    token:  str

    def __init__(self, uri: str, token: str):
        self.uri = uri
        self.token = token

    def __repr__(self) -> str:
        return f"TokenConfig<uri={self.uri}>"


db2config = {
    "Milvus": MilvusConfig,
    "ZillizCloud": ZillizCloudConfig,
}
