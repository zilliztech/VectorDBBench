"""Wrapper around the Milvus vector database over VectorDB"""

import logging

from .db_case_config import (
    DBCaseConfig,
)

from .milvus import Milvus

log = logging.getLogger(__name__)


class ZillizCloud(Milvus):
    def __init__(
        self,
        db_config: dict,
        db_case_config: DBCaseConfig,
        collection_name: str = "ZillizCloudVectorDBBench",
        drop_old: bool = False,
    ):
        super().__init__(
            db_config=db_config,
            db_case_config=db_case_config,
            collection_name=collection_name,
            drop_old=drop_old,
        )


    def ready_to_search(self):
        assert self.col, "Please call self.init() before"
        if not self.col.has_index(index_name=self._index_name):
            log.info("ZillizCloud flush, compact, create index and load")
            try:
                # supported on zilliz cloud?
                self.col.flush()
                self.col.compact()
                self.col.wait_for_compaction_completed()

                # is this sync ?
                self.col.create_index(
                    self._vector_field,
                    self.case_config.index_param(),
                    index_name=self._index_name,
                    #  timeout=600,
                )

                # is this sync ?
                self.col.load()
                #  utility.wait_for_loading_complete(self.collection_name)
            except Exception as e:
                log.warning(f"ZillizCloud ready to search error: {e}")
                raise e from None
