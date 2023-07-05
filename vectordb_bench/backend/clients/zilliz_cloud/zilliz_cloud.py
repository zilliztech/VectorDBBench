"""Wrapper around the ZillizCloud vector database over VectorDB"""

from typing import Type
from ..milvus.milvus import Milvus
from ..api import DBConfig, DBCaseConfig, IndexType
from .config import ZillizCloudConfig, AutoIndexConfig


class ZillizCloud(Milvus):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        collection_name: str = "ZillizCloudVectorDBBench",
        drop_old: bool = False,
        name: str = "ZillizCloud",
        **kwargs, 
    ):
        super().__init__(
            dim=dim,
            db_config=db_config,
            db_case_config=db_case_config,
            collection_name=collection_name,
            drop_old=drop_old,
            name=name,
            **kwargs,
        )

    @classmethod
    def config_cls(cls) -> Type[DBConfig]:
        return ZillizCloudConfig


    @classmethod
    def case_config_cls(cls, index_type: IndexType | None = None) -> Type[DBCaseConfig]:
        return AutoIndexConfig
