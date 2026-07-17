import logging
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path

import infino
import pyarrow as pa

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import InfinoFTSConfig, InfinoIndexConfig

log = logging.getLogger(__name__)

_VECTOR_FIELD = "emb"
_ID_FIELD = "id"
_LABEL_FIELD = "label"
_DOC_ID_FIELD = "doc_id"
_TEXT_FIELD = "text"
# vector_search's filter args are FTS-token-only; scalar filters go through the
# SQL vector_search TVF, which post-filters the top-k, so over-fetch to refill.
_FILTER_OVERSAMPLE = 10


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

    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: InfinoIndexConfig,
        collection_name: str = "vdbbench_infino",
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        self.name = "Infino"
        self.dim = dim
        self.data_path = db_config["data_path"]
        self.table_name = collection_name
        self.with_scalar_labels = with_scalar_labels
        self._is_fts = isinstance(db_case_config, InfinoFTSConfig)
        if not self._is_fts:
            index_param = db_case_config.index_param()
            self.metric = index_param["metric"]
            self.n_cent = index_param["n_cent"]
            self.nprobe = db_case_config.search_param()["nprobe"]

        self._where = None
        self._conn = None
        self._table = None

        Path(self.data_path).mkdir(parents=True, exist_ok=True)
        conn = infino.connect(self.data_path)
        if drop_old and self.table_name in conn.list_tables():
            conn.drop_table(self.table_name, purge=True)
        if self.table_name not in conn.list_tables():
            conn.create_table(self.table_name, self._schema(), self._index_spec())

    def _schema(self) -> pa.Schema:
        if self._is_fts:
            return pa.schema(
                [
                    pa.field(_DOC_ID_FIELD, pa.large_utf8(), nullable=False),
                    pa.field(_TEXT_FIELD, pa.large_utf8(), nullable=False),
                ],
            )
        fields = [pa.field(_ID_FIELD, pa.int64(), nullable=False)]
        if self.with_scalar_labels:
            fields.append(pa.field(_LABEL_FIELD, pa.large_utf8(), nullable=False))
        fields.append(pa.field(_VECTOR_FIELD, pa.list_(pa.float32(), self.dim), nullable=False))
        return pa.schema(fields)

    def _index_spec(self) -> infino.IndexSpec:
        if self._is_fts:
            return infino.IndexSpec().fts(_TEXT_FIELD)
        return infino.IndexSpec().vector(_VECTOR_FIELD, self.dim, self.n_cent, self.metric)

    @classmethod
    def supports_full_text_search(cls) -> bool:
        return True

    @contextmanager
    def init(self):
        # Reentrant: nested init() reuses the connection; a second one deadlocks optimize's lock on Linux.
        if self._conn is not None:
            yield
            return
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
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception | None]:
        try:
            arrays = [pa.array(metadata, type=pa.int64())]
            if self.with_scalar_labels:
                arrays.append(pa.array(labels_data, type=pa.large_utf8()))
            arrays.append(pa.array(embeddings, type=pa.list_(pa.float32(), self.dim)))
            self._table.append(pa.record_batch(arrays, schema=self._schema()))
        except Exception as e:
            log.exception("Failed to insert embeddings into Infino")
            return 0, e
        return len(metadata), None

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            self._where = None
        elif filters.type == FilterOp.NumGE:
            self._where = f"{_ID_FIELD} >= {filters.int_value}"
        elif filters.type == FilterOp.StrEqual:
            self._where = f"{_LABEL_FIELD} = '{filters.label_value}'"
        else:
            msg = f"Infino does not support filter {filters.type}"
            raise ValueError(msg)

    def search_embedding(self, query: list[float], k: int = 100, **kwargs) -> list[int]:
        if not self._where:
            hits = self._table.vector_search(
                _VECTOR_FIELD,
                query,
                k,
                nprobe=self.nprobe,
                projection=[_ID_FIELD],
            )
            return hits.column(_ID_FIELD).to_pylist()

        # Scalar filter: SQL vector_search TVF post-filters top-k, so over-fetch.
        vector_literal = ",".join(repr(float(x)) for x in query)
        sql = (
            f"SELECT {_ID_FIELD}, score FROM vector_search("
            f"'{self.table_name}', '{_VECTOR_FIELD}', '{vector_literal}', {k * _FILTER_OVERSAMPLE}) "
            f"WHERE {self._where} ORDER BY score ASC LIMIT {k}"
        )
        return self._conn.query_sql(sql).column(_ID_FIELD).to_pylist()

    def insert_documents(
        self,
        texts: list[str],
        doc_ids: list[str],
        **kwargs,
    ) -> tuple[int, Exception | None]:
        try:
            batch = pa.record_batch(
                [
                    pa.array([str(d) for d in doc_ids], type=pa.large_utf8()),
                    pa.array(texts, type=pa.large_utf8()),
                ],
                schema=self._schema(),
            )
            self._table.append(batch)
        except Exception as e:
            log.exception("Failed to insert documents into Infino")
            return 0, e
        return len(doc_ids), None

    def search_documents(self, query: str, k: int = 100, **kwargs) -> list[str]:
        hits = self._table.bm25_search(_TEXT_FIELD, query, k, projection=[_DOC_ID_FIELD])
        return hits.column(_DOC_ID_FIELD).to_pylist()

    def optimize(self, data_size: int | None = None):
        with self.init():
            self._table.optimize()

    def need_normalize_cosine(self) -> bool:
        return True
