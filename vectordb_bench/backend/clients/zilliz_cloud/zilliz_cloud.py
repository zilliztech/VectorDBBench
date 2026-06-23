"""Wrapper around the ZillizCloud vector database over VectorDB"""

from ..api import DBCaseConfig
from ..milvus.milvus import Milvus


class ZillizCloud(Milvus):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        collection_name: str = "ZillizCloudVDBBench",
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

    def optimize(self, data_size: int | None = None):
        assert self.client, "Please call self.init() before"
        self.client.flush(self.collection_name)
        self._wait_for_index()
        self.client.refresh_load(self.collection_name)
