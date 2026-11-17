import logging
import time
from typing import Any

from ...filter import Filter, FilterOp
from ...payload import PayloadProfile
from .config import SPFreshBuildMode, TiDBSPFreshIndexConfig
from .tidb import TiDB

log = logging.getLogger(__name__)


class TiDBSPFresh(TiDB):
    supported_filter_types = [FilterOp.NonFilter, FilterOp.NumGE, FilterOp.StrEqual]

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: TiDBSPFreshIndexConfig,
        collection_name: str = "vector_bench_test",
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        self.with_scalar_labels = with_scalar_labels
        self.where_clause = ""
        self.where_params: tuple[Any, ...] = ()
        super().__init__(dim, db_config, db_case_config, collection_name, drop_old, **kwargs)
        self.name = "TiDBSPFresh"
        if not drop_old:
            params = self._index_params(search_only=True)
            # Check the reused index and apply settings once in the parent.
            # The optimizer remains free to choose a scalar or vector access path.
            with self.init():
                self._read_ready_status()
                if params:
                    self.cursor.execute(
                        f"ALTER TABLE {self.table_name} ALTER INDEX {self._index_name()} SET VECTOR_INDEX_PARAM %s",
                        params,
                    )
                    self.conn.commit()

    @staticmethod
    def supports_payload_profile(payload_profile: PayloadProfile) -> bool:
        return payload_profile in {PayloadProfile.IDS_ONLY, PayloadProfile.SCALAR_LABEL}

    def _index_name(self) -> str:
        return f"idx_embedding_spfresh_{self.case_config.metric_type.value.lower()}"

    def _index_definition(self) -> str:
        sql = f"VECTOR INDEX {self._index_name()} (({self.search_fn}(embedding))) USING SPFRESH"
        if self.with_scalar_labels:
            sql += " STORING (labels)"
        if self._index_params():
            sql += " VECTOR_INDEX_PARAM %s"
        return sql

    def _index_params(self, *, search_only: bool = False) -> tuple[str] | None:
        params = {} if search_only else self.case_config.index_param()
        params.update(self.case_config.search_param())
        params.pop("metric_fn")
        return (",".join(f"{key}={value}" for key, value in params.items()),) if params else None

    def _create_table(self) -> None:
        columns = ["id BIGINT PRIMARY KEY", f"embedding VECTOR({self.dim}) NOT NULL"]
        if self.with_scalar_labels:
            columns.append("labels VARCHAR(64) NOT NULL")
        inline = self.case_config.spfresh_build_mode == SPFreshBuildMode.INLINE
        if inline:
            columns.append(self._index_definition())
        with self._get_connection() as (conn, cursor):
            cursor.execute(
                f"CREATE TABLE {self.table_name} ({', '.join(columns)})",
                self._index_params() if inline else None,
            )
            conn.commit()

    def optimize(self, data_size: int | None = None) -> None:
        if self.cursor is None or self.conn is None:
            raise RuntimeError("TiDBSPFresh.optimize() requires an active init() context")

        # The standard runner has joined every insert worker before optimize().
        # A fresh TSO therefore bounds all committed inserts, even though loading
        # ran in another process. No per-insert TSO queries or runner state are needed.
        barrier_ts = self._current_tso_marker()
        if self.case_config.spfresh_build_mode == SPFreshBuildMode.NON_INLINE:
            self.cursor.execute(
                f"ALTER TABLE {self.table_name} ADD {self._index_definition()}",
                self._index_params(),
            )
            self.conn.commit()
        self._wait_for_checkpoint(barrier_ts)

    def _current_tso_marker(self) -> int:
        self.cursor.execute("BEGIN")
        self.cursor.execute("SELECT @@tidb_current_ts")
        row = self.cursor.fetchone()
        self.conn.commit()
        if row is None or row[0] is None or int(row[0]) <= 0:
            raise RuntimeError("Failed to read a positive TSO from @@tidb_current_ts")
        return int(row[0])

    def _read_ready_status(self) -> tuple:
        context = f"table={self.table_name} index={self._index_name()}"
        try:
            self.cursor.execute(
                "SELECT applied_base_ts, index_state, is_ready, lag_seconds, owner_lease_expire_ts, observed_at "
                "FROM information_schema.TIDB_SPFRESH_INDEX_STATUS "
                "WHERE table_schema = DATABASE() AND table_name = %s AND index_name = %s",
                (self.table_name, self._index_name()),
            )
            status = self.cursor.fetchone()
        except Exception as e:
            msg = f"SPFresh status query failed: {context}; error={e}"
            raise RuntimeError(msg) from e
        if status is None or len(status) != 6:
            msg = f"Missing or invalid SPFresh status: {context}; status={status}"
            raise RuntimeError(msg)
        applied_ts, index_state, is_ready, _, _, observed_at = status
        if (
            str(index_state).upper() != "READY"
            or is_ready != 1
            or applied_ts is None
            or int(applied_ts) <= 0
            or observed_at is None
        ):
            msg = f"SPFresh index is not ready or has an invalid checkpoint: {context}; status={status}"
            raise RuntimeError(msg)
        return status

    def _wait_for_checkpoint(self, barrier_ts: int) -> None:
        start = time.monotonic()
        deadline = start + self.case_config.spfresh_wait_timeout
        status = None
        context = f"table={self.table_name} index={self._index_name()} barrier_ts={barrier_ts}"
        while time.monotonic() < deadline:
            status = self._read_ready_status()
            now = time.monotonic()
            if now >= deadline:
                break
            if int(status[0]) >= barrier_ts:
                log.info("SPFresh checkpoint reached: %s; status=%s", context, status)
                return
            log.info("Waiting for SPFresh checkpoint: %s; status=%s", context, status)
            time.sleep(min(1.0, deadline - now))
        msg = (
            f"Timed out waiting for SPFresh checkpoint: {context}; "
            f"waited_seconds={time.monotonic() - start:.3f}; last_status={status}"
        )
        raise RuntimeError(msg)

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[int, Exception | None]:
        if self.with_scalar_labels and (labels_data is None or len(labels_data) != len(embeddings)):
            raise ValueError("labels_data must contain one label per embedding")
        return super().insert_embeddings(embeddings, metadata, labels_data=labels_data, **kwargs)

    def _insert_embeddings_serial(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        offset: int,
        size: int,
        labels_data: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        if not self.with_scalar_labels:
            super()._insert_embeddings_serial(embeddings, metadata, offset, size)
            return
        params = []
        for i in range(offset, offset + size):
            params.extend((metadata[i], str(embeddings[i]), labels_data[i]))
        values = ",".join(["(%s, %s, %s)"] * size)
        with self._get_connection() as (conn, cursor):
            cursor.execute(
                f"INSERT INTO {self.table_name} (id, embedding, labels) VALUES {values}",
                tuple(params),
            )
            conn.commit()

    def prepare_filter(self, filters: Filter) -> None:
        if filters.type == FilterOp.NonFilter:
            self.where_clause, self.where_params = "", ()
        elif filters.type == FilterOp.NumGE and filters.int_field == "id":
            self.where_clause, self.where_params = "WHERE id >= %s", (int(filters.int_value),)
        elif filters.type == FilterOp.StrEqual and filters.label_field == "labels" and self.with_scalar_labels:
            self.where_clause, self.where_params = "WHERE labels = %s", (filters.label_value,)
        else:
            msg = f"Unsupported TiDB SPFresh filter or missing scalar labels: {filters}"
            raise ValueError(msg)

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
        payload_profile: PayloadProfile = PayloadProfile.IDS_ONLY,
        **kwargs: Any,
    ) -> list[int]:
        if not self.supports_payload_profile(payload_profile):
            msg = f"Unsupported TiDB SPFresh payload profile: {payload_profile}"
            raise ValueError(msg)
        if payload_profile == PayloadProfile.SCALAR_LABEL and not self.with_scalar_labels:
            raise ValueError("scalar_label payload requires a table with scalar labels")
        columns = "id, labels" if payload_profile == PayloadProfile.SCALAR_LABEL else "id"
        self.cursor.execute(
            f"SELECT {columns} FROM {self.table_name} {self.where_clause} "
            f"ORDER BY {self.search_fn}(embedding, %s) LIMIT %s",
            (*self.where_params, str(query), k),
        )
        return [int(row[0]) for row in self.cursor.fetchall()]
