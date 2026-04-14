"""Wrapper around the LanceDB vector database over VectorDB"""

import logging
import os
from contextlib import contextmanager

import lancedb
import pyarrow as pa

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import IndexType, VectorDB
from .config import LanceDBIndexConfig

log = logging.getLogger(__name__)

# Rows per ``table.add`` call. Each call produces a new Lance data file
# (fragment), so enlarging this value directly controls the on-disk fragment
# size. Override via the ``LANCEDB_BATCH_SIZE`` environment variable.
#
# Rough sizing for float32 vectors: bytes_per_fragment ≈ rows * dim * 4.
# Example: dim=768, rows=170000 -> ~498 MB per fragment.
LANCEDB_BATCH_SIZE = int(os.environ.get("LANCEDB_BATCH_SIZE", "5000"))


class LanceDB(VectorDB):
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]
    thread_safe: bool = False

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: LanceDBIndexConfig,
        collection_name: str = "vector_bench_test",
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        self.name = "LanceDB"
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self.dim = dim
        self.uri = db_config["uri"]
        self.storage_options = db_config.get("storage_options") or None
        self.with_scalar_labels = with_scalar_labels
        self.where_clause = None

        self._id_field = "id"
        self._vector_field = "vector"
        self._label_field = "label"

        # cache search params to avoid repeated calls
        self.search_config = db_case_config.search_param()
        log.info(f"LanceDB search config: {self.search_config}")

        connect_kwargs = {}
        if self.storage_options:
            connect_kwargs["storage_options"] = self.storage_options
        db = lancedb.connect(self.uri, **connect_kwargs)

        if drop_old:
            try:
                db.drop_table(self.table_name)
                log.info(f"LanceDB dropped old table: {self.table_name}")
            except Exception as e:
                log.warning(f"Failed to drop table {self.table_name}: {e}")
            # Always create a fresh table with the correct schema after drop.
            # On remote storage (e.g. GooseFS) drop_table may not fully purge
            # metadata immediately, causing open_table to succeed with a stale
            # schema that is missing expected fields like 'id'.
            schema = self._build_schema()
            db.create_table(self.table_name, schema=schema, mode="overwrite")
            log.info(f"LanceDB created table: {self.table_name} (schema: {schema})")
        else:
            try:
                db.open_table(self.table_name)
            except Exception:
                schema = self._build_schema()
                db.create_table(self.table_name, schema=schema, mode="overwrite")
                log.info(f"LanceDB created table: {self.table_name} (schema: {schema})")

    def _build_schema(self) -> pa.Schema:
        fields = [
            pa.field(self._id_field, pa.int64()),
            pa.field(self._vector_field, pa.list_(pa.float32(), list_size=self.dim)),
        ]
        if self.with_scalar_labels:
            fields.append(pa.field(self._label_field, pa.utf8()))
        return pa.schema(fields)

    @contextmanager
    def init(self):
        connect_kwargs = {}
        if self.storage_options:
            connect_kwargs["storage_options"] = self.storage_options
        self.db = lancedb.connect(self.uri, **connect_kwargs)
        self.table = self.db.open_table(self.table_name)
        yield
        self.db = None
        self.table = None

    def __deepcopy__(self, memo: dict) -> "LanceDB":
        """Custom deepcopy: skip live connection/table handles.

        The LanceDB ``Connection`` / ``Table`` objects wrap Rust bindings that
        are not picklable. ``ConcurrentInsertRunner`` deep-copies the client
        per thread for non-thread-safe DBs; the caller will then invoke
        ``init()`` on the copy, which re-opens a fresh connection.
        """
        cls = self.__class__
        new_obj = cls.__new__(cls)
        memo[id(self)] = new_obj
        from copy import deepcopy as _dc

        for k, v in self.__dict__.items():
            if k in ("db", "table"):
                new_obj.__dict__[k] = None
            else:
                new_obj.__dict__[k] = _dc(v, memo)
        return new_obj

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception | None]:
        assert self.table is not None, "Please call self.init() before"
        try:
            log.info(
                f"LanceDB insert_embeddings called with {len(embeddings)} rows, "
                f"LANCEDB_BATCH_SIZE={LANCEDB_BATCH_SIZE} -> "
                f"{(len(embeddings) + LANCEDB_BATCH_SIZE - 1) // LANCEDB_BATCH_SIZE} fragment(s)"
            )
            for offset in range(0, len(embeddings), LANCEDB_BATCH_SIZE):
                batch_emb = embeddings[offset : offset + LANCEDB_BATCH_SIZE]
                batch_ids = metadata[offset : offset + LANCEDB_BATCH_SIZE]

                id_arr = pa.array(batch_ids, type=pa.int64())
                vec_arr = pa.FixedSizeListArray.from_arrays(
                    pa.array([v for emb in batch_emb for v in emb], type=pa.float32()),
                    list_size=self.dim,
                )

                if self.with_scalar_labels and labels_data is not None:
                    batch_labels = labels_data[offset : offset + LANCEDB_BATCH_SIZE]
                    label_arr = pa.array(batch_labels, type=pa.utf8())
                    batch_table = pa.table(
                        {
                            self._id_field: id_arr,
                            self._vector_field: vec_arr,
                            self._label_field: label_arr,
                        }
                    )
                else:
                    batch_table = pa.table(
                        {
                            self._id_field: id_arr,
                            self._vector_field: vec_arr,
                        }
                    )
                self.table.add(batch_table)

            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into LanceDB table ({self.table_name}), error: {e}")
            return 0, e

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            self.where_clause = None
        elif filters.type == FilterOp.NumGE:
            self.where_clause = f"{self._id_field} >= {filters.int_value}"
        elif filters.type == FilterOp.StrEqual:
            self.where_clause = f"{self._label_field} = '{filters.label_value}'"
        else:
            msg = f"Unsupported filter for LanceDB: {filters}"
            raise ValueError(msg)

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        **kwargs,
    ) -> list[int]:
        assert self.table is not None, "Please call self.init() before"

        # Include ``_distance`` in the projection to opt in to LanceDB's
        # upcoming default behaviour. Without it, lance logs a per-query
        # deprecation warning ("This search specified output columns but did
        # not include `_distance`"). We only consume ``id`` downstream, so the
        # extra column adds negligible overhead.
        q = self.table.search(query).select([self._id_field, "_distance"]).limit(k)

        # apply filter
        if self.where_clause:
            q = q.where(self.where_clause, prefilter=True)

        # apply search parameters based on config
        search_cfg = self.search_config
        if "nprobes" in search_cfg:
            q = q.nprobes(search_cfg["nprobes"])
        if "ef" in search_cfg:
            q = q.ef(search_cfg["ef"])
        if "refine_factor" in search_cfg:
            q = q.refine_factor(search_cfg["refine_factor"])

        results = q.to_list()
        return [int(r[self._id_field]) for r in results]

    def optimize(self, data_size: int | None = None):
        assert self.table is not None, "Please call self.init() before"

        # Build index if configured
        if self.case_config.index != IndexType.NONE:
            index_params = self.case_config.index_param()
            log.info(f"LanceDB creating index on table ({self.table_name}), params: {index_params}")
            self.table.create_index(**index_params)

        # Compact fragments and clean up old versions for better performance.
        # Prefer the unified ``table.optimize()`` API (lancedb >= 0.10), which
        # internally handles both compaction and version cleanup without
        # requiring the optional ``pylance`` package. Fall back to the legacy
        # split APIs only if ``optimize`` is unavailable.
        try:
            if hasattr(self.table, "optimize"):
                self.table.optimize()
                log.info(f"LanceDB optimize completed for table ({self.table_name})")
            else:
                self.table.compact_files()
                self.table.cleanup_old_versions()
                log.info(f"LanceDB compact_files + cleanup_old_versions completed for table ({self.table_name})")
        except Exception as e:
            log.warning(f"LanceDB optimize failed (non-fatal): {e}")
