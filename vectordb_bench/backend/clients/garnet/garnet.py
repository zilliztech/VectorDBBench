import json
import logging
from collections.abc import Generator
from contextlib import contextmanager

import numpy as np
import redis

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import GarnetDBCaseConfig

log = logging.getLogger(__name__)


class Garnet(VectorDB):
    conn: redis.Redis | None = None
    supported_filter_types: list[FilterOp] = [FilterOp.NonFilter, FilterOp.NumGE, FilterOp.StrEqual]

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: GarnetDBCaseConfig,
        collection_name: str = "vs0",
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "Garnet"

        self.dim = dim
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = collection_name
        self.filter = None

        if drop_old:
            conn = redis.Redis(**self.db_config, protocol=2)
            try:
                conn.delete(self.collection_name)
            except redis.exceptions.ResponseError:
                log.info(f"Garnet client failed to drop old collection: {self.collection_name}")

            conn.close()

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        self.conn = redis.Redis(**self.db_config, protocol=2)
        yield
        self.conn.close()
        self.conn = None

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception | None]:
        batch_size = 1000
        try:
            with self.conn.pipeline(transaction=False) as pipe:  # ty:ignore[possibly-missing-attribute]
                for i, emb in enumerate(embeddings):
                    attrs = {"id": metadata[i]}
                    if labels_data is not None:
                        attrs["label"] = labels_data[i]

                    command = [
                        "VADD",
                        self.collection_name,
                        "FP32",
                        np.array(emb).astype(np.float32).tobytes(),
                        str(metadata[i]),
                        "NOQUANT",
                        "EF",
                        self.case_config.index_param()["params"]["l_build"],
                        "SETATTR",
                        json.dumps(attrs),
                        "M",
                        self.case_config.index_param()["params"]["max_degree"],
                        "XDISTANCE_METRIC",
                        self.case_config.index_param()["metric_type"],
                    ]
                    pipe.pipeline_execute_command(*command)

                    if (i + 1) % batch_size == 0:
                        _ = pipe.execute()
                _ = pipe.execute()
                num_results = i + 1
        except Exception as e:
            return 0, e

        return num_results, None

    def prepare_filter(self, filter: Filter):
        if filter.type == FilterOp.NonFilter:
            self.filter = None
        elif filter.type == FilterOp.NumGE:
            self.filter = f".id >= {filter.int_value}"
        elif filter.type == FilterOp.StrEqual:
            self.filter = f'.label == "{filter.label_value}"'
        else:
            msg = f"Garnet does not support the filter: {filter}"
            raise ValueError(msg)

    def search_embedding(self, query: list[float], k: int = 100, filters: dict | None = None) -> list[int]:
        command = [
            "VSIM",
            self.collection_name,
            "FP32",
            np.array(query).astype(np.float32).tobytes(),
            "COUNT",
            str(k),
            "EF",
            self.case_config.search_param()["params"]["l_search"],
        ]
        if self.filter:
            command.append("FILTER")
            command.append(self.filter)

        result = self.conn.execute_command(  # ty:ignore[possibly-missing-attribute]
            *command
        )
        return [int(x) for x in result]

    def optimize(self, data_size: int | None = None):
        pass
