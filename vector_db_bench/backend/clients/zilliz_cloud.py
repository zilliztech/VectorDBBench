"""Wrapper around the ZillizCloud vector database over VectorDB"""

import logging

from .db_case_config import (
    DBCaseConfig,
)

from .milvus import Milvus

log = logging.getLogger(__name__)


class ZillizCloud(Milvus):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        collection_name: str = "ZillizCloudVectorDBBench",
        drop_old: bool = False,
        name: str = "ZillizCloud"
    ):
        super().__init__(
            dim=dim,
            db_config=db_config,
            db_case_config=db_case_config,
            collection_name=collection_name,
            drop_old=drop_old,
            name=name,
        )
