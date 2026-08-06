import logging
from collections.abc import Iterable
from contextlib import contextmanager
from pathlib import Path

import infino
import pyarrow as pa

from ..api import VectorDB
from .config import InfinoFTSConfig, InfinoIndexConfig

log = logging.getLogger(__name__)

_VECTOR_FIELD = "emb"
_ID_FIELD = "id"
_LABEL_FIELD = "label"
_DOC_ID_FIELD = "doc_id"
_TEXT_FIELD = "text"


class Infino(VectorDB):
    """VectorDBBench client for Infino, an embedded vector/search engine.

    Infino is in-process: each benchmark worker connects to the same on-disk
    catalog. The instance holds only picklable config so it survives the
    ProcessPoolExecutor(spawn) boundary; the connection and table are opened
    lazily in init().
    """

    # Serialize the load: concurrent writes to a single table are not supported.
    thread_safe: bool = False

    # NonFilter only (base default): Infino's native filtered ANN is an FTS-token
    # pre-filter that can't express the harness's scalar equality / range filters.

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
        # A cache budget without a cache dir is a silent no-op in the
        # engine (no disk cache is created); default the cache next to the
        # catalog so warm queries are actually warm.
        if db_config.get("cache_budget_bytes") and not db_config.get("cache_dir"):
            db_config = {**db_config, "cache_dir": str(Path(self.data_path) / "cache")}
        # Connection tuning (cache budget, cache dir, object-store options); pass only what is set.
        self._connect_opts = {
            k: db_config[k]
            for k in ("cache_budget_bytes", "cache_dir", "storage_options")
            if db_config.get(k) is not None
        }
        self.table_name = collection_name
        self.with_scalar_labels = with_scalar_labels
        self._is_fts = isinstance(db_case_config, InfinoFTSConfig)
        # Vector-only params; left None for FTS runs, which never call search_embedding.
        self.metric = None
        if self._is_fts:
            # Tokenizer chosen to match the GT analyzer (set by
            # apply_fts_manifest); defaults to the ASCII tokenizer.
            self._analyzer = db_case_config.analyzer
        else:
            self.metric = db_case_config.index_param()["metric"]

        self._conn = None
        self._table = None
        # Engine _id -> caller id, built once per process on first search:
        # search returns the engine-native stable _id for free, while
        # projecting the id column costs a per-query scalar resolve.
        self._id_map = None
        # Build the schema once so table creation and every append stay in lockstep.
        self._schema = self._build_schema()

        Path(self.data_path).mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        if drop_old and self.table_name in conn.list_tables():
            conn.drop_table(self.table_name, purge=True)
        if self.table_name not in conn.list_tables():
            conn.create_table(self.table_name, self._schema, self._index_spec())

    def _connect(self):
        return infino.connect(self.data_path, **self._connect_opts)

    def __getstate__(self) -> dict:
        # Drop the non-picklable live connection so the instance can cross a process boundary.
        return {**self.__dict__, "_conn": None, "_table": None, "_id_map": None}

    def _build_schema(self) -> pa.Schema:
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
            return infino.IndexSpec().fts(_TEXT_FIELD, analyzer=self._analyzer)
        return infino.IndexSpec().vector(_VECTOR_FIELD, self.dim, self.metric)

    @classmethod
    def supports_full_text_search(cls) -> bool:
        return True

    @contextmanager
    def init(self):
        # Reuse one connection for the whole process: reopening is costly and can deadlock.
        if self._conn is None:
            conn = self._connect()
            self._table = conn.open_table(self.table_name)
            self._conn = conn  # assign last so a failed open leaves a clean state to retry
        yield

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
            self._table.append(pa.record_batch(arrays, schema=self._schema))
        except Exception as e:
            log.exception("Failed to insert embeddings into Infino")
            return 0, e
        return len(metadata), None

    def _ensure_id_map(self) -> dict:
        if self._id_map is None:
            m = self._conn.query_sql(f"SELECT _id, {_ID_FIELD} FROM {self.table_name}")
            self._id_map = dict(
                zip(m.column("_id").to_pylist(), m.column(_ID_FIELD).to_pylist())
            )
        return self._id_map

    def search_embedding(self, query: list[float], k: int = 100, **kwargs) -> list[int]:
        # Vector serving is engine-decided; the call carries no tuning kwargs.
        id_map = self._ensure_id_map()
        hits = self._table.vector_search(_VECTOR_FIELD, query, k)
        return [id_map[h] for h in hits.column("_id").to_pylist()]

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
                schema=self._schema,
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
