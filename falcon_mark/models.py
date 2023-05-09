from typing import Any
from enum import IntEnum
from pydantic import BaseModel
from .metric import Metric


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


class MilvusConfig(BaseModel):
    host: str
    port: int | str


    def __repr__(self) -> str:
        return f"MilvusConfig<host={self.host}, port={self.port}>"



class ZillizCloudConfig(BaseModel):
    uri:        str
    user:       str
    password:   str

    def __repr__(self) -> str:
        return f"ZillizCloudConfig<uri={self.uri}, user={self.user}>"


db2config = {
    "Milvus": MilvusConfig,
    "ZillizCloud": ZillizCloudConfig,
}


class TaskConfig(BaseModel):
    db: DB
    db_config: Any
    case_config: Any


class CaseResult(BaseModel):
    result_id: int
    case_id: int
    case_config: Any
    output_path: str

    metrics: list[Metric]

    def append_to_disk(self):
        pass


class TestResult(BaseModel):
    run_id: int
    results: list[CaseResult]
