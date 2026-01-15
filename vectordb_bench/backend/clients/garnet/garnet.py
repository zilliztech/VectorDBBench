import json
import logging
import threading
from collections.abc import Generator
from contextlib import contextmanager

import numpy as np
import redis

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import GarnetDBCaseConfig

log = logging.getLogger(__name__)


class Garnet(VectorDB):
    thread_safe: bool = True
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

        # Each worker thread gets its own connection (see the `conn` property).
        self._local = threading.local()
        self._conns_lock = threading.Lock()
        self._all_conns: list[redis.Redis] = []

        if drop_old:
            conn = redis.Redis(**self.db_config, protocol=2)
            try:
                conn.delete(self.collection_name)
            except redis.exceptions.ResponseError:
                log.info(f"Garnet client failed to drop old collection: {self.collection_name}")

            conn.close()

    def __getstate__(self) -> dict:
        """Exclude unpicklable thread-local state for ProcessPoolExecutor(spawn)."""
        state = self.__dict__.copy()
        state.pop("_local", None)
        state.pop("_conns_lock", None)
        state.pop("_all_conns", None)
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)
        self._local = threading.local()
        self._conns_lock = threading.Lock()
        self._all_conns = []

    @property
    def conn(self) -> redis.Redis:
        """Return this thread's Redis connection, creating it on first use.

        Each worker thread gets its own dedicated connection so concurrent
        inserts do not share a single connection.
        """
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = redis.Redis(**self.db_config, protocol=2)
            self._local.conn = conn
            with self._conns_lock:
                self._all_conns.append(conn)
        return conn

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        _ = self.conn
        try:
            yield
        finally:
            with self._conns_lock:
                conns, self._all_conns = self._all_conns, []
            for c in conns:
                try:
                    c.close()
                except Exception as e:
                    log.warning(f"Garnet failed to close connection: {e}")
            self._local = threading.local()

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception | None]:
        batch_size = 1000
        try:
            with self.conn.pipeline(transaction=False) as pipe:
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
            command.append("FILTER-EF")
            command.append(str(self.case_config.filter_scale))

        result = self.conn.execute_command(*command)
        return [int(x) for x in result]

    def optimize(self, data_size: int | None = None):
        pass
