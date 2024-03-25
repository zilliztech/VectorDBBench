import logging
from contextlib import contextmanager
from typing import Iterable

import numpy as np
from transwarp_hippo_api.hippo_client import HippoClient, HippoField
from transwarp_hippo_api.hippo_type import HippoType

from ..api import VectorDB
from .config import HippoIndexConfig

log = logging.getLogger(__name__)


class Hippo(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: HippoIndexConfig,
        drop_old: bool = False,
        **kwargs,
    ):
        """Initialize wrapper around the hippo vector database."""
        self.name = "Hippo"
        self.db_config = db_config
        self.index_config = db_case_config

        self.database_name = "default"
        self.table_name = "vdbbench_table"
        self.index_name = "vector_index"

        self.vector_field_name = "vector"
        self.int_field_name = "label"
        self.pk_field_name = "pk"

        self.insert_batch_size = db_config.get("insert_batch_size")
        self.activated = False

        # if `drop_old`, check table and delete table
        hc = HippoClient(
            **{
                k: db_config[k]
                for k in ["host_port", "username", "pwd"]
                if k in db_config
            }
        )
        if drop_old:
            try:
                table_check = hc.check_table_exists(
                    self.table_name, database_name=self.database_name
                )
                log.info(f"check table exsited: {table_check}")
            except ValueError as e:
                log.error("failed to check table exsited; skip", exc_info=e)
                table_check = False

            if table_check:
                log.info(f"delete table: {self.table_name}")
                hc.delete_table(self.table_name, database_name=self.database_name)
                hc.delete_table_in_trash(
                    self.table_name, database_name=self.database_name
                )

            # create table
            fields = [
                HippoField(self.pk_field_name, True, HippoType.INT64),
                HippoField(self.int_field_name, False, HippoType.INT64),
                HippoField(
                    self.vector_field_name,
                    False,
                    HippoType.FLOAT_VECTOR,
                    type_params={"dimension": dim},
                ),
            ]
            log.info(f"create table: {self.table_name}")
            hc.create_table(
                name=self.table_name,
                fields=fields,
                database_name=self.database_name,
                number_of_shards=db_config.get("number_of_shards"),
                number_of_replicas=db_config.get("number_of_replicas"),
            )

            table = hc.get_table(self.table_name, database_name=self.database_name)
            # create index
            log.info("create index")
            table.create_index(
                field_name=self.vector_field_name,
                index_name=self.index_name,
                index_type=self.index_config.index,
                metric_type=self.index_config.parse_metric(),
                **self.index_config.index_param(),
            )

    def need_normalize_cosine(self) -> bool:
        """Wheather this database need to normalize dataset to support COSINE"""
        return False

    @contextmanager
    def init(self):
        """
        generate connection
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """
        from transwarp_hippo_api.hippo_client import HippoClient

        hc = HippoClient(
            **{
                k: self.db_config[k]
                for k in ["host_port", "username", "pwd"]
                if k in self.db_config
            }
        )
        self.client = hc.get_table(self.table_name, database_name=self.database_name)

        yield

    def _activate_index(self):
        if not self.activated:
            try:
                log.info("start activate index, please wait ...")
                self.client.activate_index(
                    self.index_name, wait_for_completion=True, timeout="25h"
                )
                log.info("index is actived.")
            except Exception as e:
                log.error("failed to activate index; skip", exc_info=e)

            self.activated = True

    def insert_embeddings(
        self, embeddings: Iterable[list[float]], metadata: list[int], **kwargs
    ):
        assert self.client is not None
        insert_count = 0
        try:
            for batch_start_offset in range(0, len(embeddings), self.insert_batch_size):
                log.info("batch offset: %d", batch_start_offset)

                data = [
                    list(
                        metadata[
                            batch_start_offset : batch_start_offset
                            + self.insert_batch_size
                        ]
                    ),
                    list(
                        metadata[
                            batch_start_offset : batch_start_offset
                            + self.insert_batch_size
                        ]
                    ),
                    [
                        i.tolist() if isinstance(i, np.ndarray) else i
                        for i in embeddings[
                            batch_start_offset : batch_start_offset
                            + self.insert_batch_size
                        ]
                    ],
                ]

                self.client.insert_rows(data)
                insert_count += len(data[0])
            # if kwargs.get("last_batch"):
            #     self._activate_index()
        except Exception as e:
            log.error("hippp insert error", exc_info=e)
            return (insert_count, e)

        log.info("total insert: %d", insert_count)

        return (insert_count, None)

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
    ) -> list[int]:
        # assert self.col is not None

        dsl = f"{self.int_field_name} >= {filters['id']}" if filters else ""
        output_fields = [self.int_field_name]
        result = self.client.query(
            self.vector_field_name,
            [query],
            output_fields,
            k,
            dsl=dsl,
            **self.index_config.search_param(),
        )

        return result[0][self.int_field_name]

    def optimize(self, **kwargs):
        self._activate_index()

        if kwargs.get("filters"):
            log.info(f"create scalar index on field: {self.int_field_name}")
            self.client.create_scalar_index(
                field_names=[self.int_field_name],
                index_name="idx_" + self.int_field_name,
            )
            log.info("scalar index created")

    def ready_to_load(self):
        return
