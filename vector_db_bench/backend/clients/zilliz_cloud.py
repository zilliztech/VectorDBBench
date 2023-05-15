"""Wrapper around the Milvus vector database over VectorDB"""

import logging

from ...models import (
    IndexType,
    DBCaseConfig,
)

from .milvus import Milvus, MilvusIndexConfig

log = logging.getLogger(__name__)


class ZillizCloud(Milvus):
    def __init__(
        self,
        db_config: dict,
        db_case_config: DBCaseConfig,
        collection_name: str = "ZillizCloudVectorDBBench",
        drop_old: bool = False,
    ):
        assert isinstance(DBCaseConfig, AutoIndexConfig)
        super().__init__(
            db_config=db_config,
            db_case_config=db_case_config,
            collection_name=collection_name,
            drop_old=drop_old,
        )


