import logging
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path

import infino
import pyarrow as pa

from ..api import DBCaseConfig, VectorDB
from .config import InfinoConfig, InfinoIndexConfig

log = logging.getLogger(__name__)

_VECTOR_FIELD = "emb"
_ID_FIELD = "id"


class Infino(VectorDB):
    """VectorDBBench client for Infino, an embedded vector/search engine.

    Infino is in-process: each benchmark worker connects to the same on-disk
    catalog. The instance holds only picklable config so it survives the
    ProcessPoolExecutor(spawn) boundary; the connection and table are opened
    lazily in init().
    """

    # Concurrent same-process writes to one table are not yet validated, so
    # serialize the load (runner clamps max_workers to 1 when thread_safe is False).
    thread_safe: bool = False

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: InfinoIndexConfig,
        collection_name: str = "vdbbench_infino",
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "Infino"
        self.dim = dim
        self.data_path = db_config["data_path"]
        self.table_name = collection_name
        index_param = db_case_config.index_param()
        self.metric = index_param["metric"]
        self.n_cent = index_param["n_cent"]
        self.nprobe = db_case_config.search_param()["nprobe"]

        self._conn = None
        self._table = None

        Path(self.data_path).mkdir(parents=True, exist_ok=True)
        conn = infino.connect(self.data_path)
        if drop_old and self.table_name in conn.list_tables():
            conn.drop_table(self.table_name, purge=True)
        if self.table_name not in conn.list_tables():
            conn.create_table(self.table_name, self._schema(), self._index_spec())

    def _schema(self) -> pa.Schema:
        return pa.schema(
            [
                pa.field(_ID_FIELD, pa.int64(), nullable=False),
                pa.field(_VECTOR_FIELD, pa.list_(pa.float32(), self.dim), nullable=False),
            ],
        )

    def _index_spec(self) -> infino.IndexSpec:
        return infino.IndexSpec().vector(_VECTOR_FIELD, self.dim, self.n_cent, self.metric)

    @classmethod
    def config_cls(cls) -> type[InfinoConfig]:
        return InfinoConfig

    @classmethod
    def case_config_cls(cls, index_type: str | None = None) -> type[DBCaseConfig]:
        return InfinoIndexConfig

    @contextmanager
    def init(self):
        self._conn = infino.connect(self.data_path)
        self._table = self._conn.open_table(self.table_name)
        try:
            yield
        finally:
            self._table = None
            self._conn = None

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> tuple[int, Exception | None]:
        try:
            batch = pa.record_batch(
                [
                    pa.array(metadata, type=pa.int64()),
                    pa.array(embeddings, type=pa.list_(pa.float32(), self.dim)),
                ],
                schema=self._schema(),
            )
            self._table.append(batch)
        except Exception as e:
            log.exception("Failed to insert embeddings into Infino")
            return 0, e
        return len(metadata), None

    def search_embedding(self, query: list[float], k: int = 100, **kwargs) -> list[int]:
        hits = self._table.vector_search(
            _VECTOR_FIELD,
            query,
            k,
            nprobe=self.nprobe,
            projection=[_ID_FIELD],
        )
        return hits.column(_ID_FIELD).to_pylist()

    def optimize(self, data_size: int | None = None):
        with self.init():
            self._table.optimize()

    def need_normalize_cosine(self) -> bool:
        return True
