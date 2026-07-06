#!/usr/bin/env python3
import json
import logging
import os
import struct
import tempfile
from contextlib import contextmanager
from pathlib import Path

import mysql.connector as mysql
import numpy as np

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import VolcMySQLConfigDict, VolcMySQLIndexConfig

log = logging.getLogger(__name__)


def _encode_batch_to_tsv(
    metadata: list[int],
    embeddings: list[list[float]],
    dim: int,
    tsv_path: str,
    *,
    binary: bool = True,
) -> None:
    """Sort (id, vector) pairs by id ascending and write them to ``tsv_path``
    for bulk ``LOAD DATA`` ingestion.

    ``binary=True`` (default) encodes each vector as hex of little-endian
    float32 bytes -> ``<id>\\t<hex>\\n``, consumed by
    ``... (id, @h) SET v = UNHEX(@h)``.

    ``binary=False`` (to_vector fallback) writes the vector as a JSON-style
    ``[f1,f2,...]`` literal -> ``<id>\\t[..]\\n``, consumed by
    ``... (id, @v) SET v = to_vector(@v)``. The literal contains no tab or
    newline, so it stays delimiter-safe in the TSV transport.
    """
    order = np.argsort(metadata)
    with Path(tsv_path).open("w", buffering=1 << 20) as f:
        if binary:
            pack_fmt = f"<{dim}f"
            for i in order:
                hex_str = struct.pack(pack_fmt, *embeddings[i]).hex()
                f.write(f"{metadata[i]}\t{hex_str}\n")
        else:
            for i in order:
                vec_str = "[" + ",".join(repr(float(x)) for x in embeddings[i]) + "]"
                f.write(f"{metadata[i]}\t{vec_str}\n")


def _build_index_attrs_json(index_param: dict) -> str:
    """Build the SECONDARY_ENGINE_ATTRIBUTE JSON payload for CREATE VECTOR INDEX
    from a case-config index_param dict. None values are dropped so the server
    sees only the attributes the user explicitly set.
    """
    attrs = {
        "algorithm": "hnsw",
        "distance": index_param.get("metric_type"),
        "m": index_param.get("M"),
        "ef_construction": index_param.get("ef_construction"),
        "quant_algorithm": index_param.get("quant_algorithm"),
        "quant_type": index_param.get("quant_type"),
    }
    attrs = {k: v for k, v in attrs.items() if v is not None}
    return json.dumps(attrs, ensure_ascii=False, separators=(",", ":"))


class VolcMySQL(VectorDB):
    # mysql.connector is not thread-safe; ConcurrentInsertRunner uses
    # max_workers=1 when False. rate_runner branches on db.name to give
    # each worker thread its own connection.
    thread_safe: bool = False

    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
    ]

    def __init__(
        self,
        dim: int,
        db_config: VolcMySQLConfigDict,
        db_case_config: VolcMySQLIndexConfig,
        collection_name: str = "vec_collection",
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "VolcMySQL"
        self.db_config = db_config
        self.case_config = db_case_config
        self.db_name = "vectordbbench"
        self.table_name = collection_name
        self.dim = dim

        self.conn = None
        self.cursor = None
        self.admin_cursor = None

        # Active filter predicate selected by prepare_filter(). Runners call
        # prepare_filter() (never search_embedding(filters=...)), so the
        # filtered WHERE clause is chosen here, not per query.
        self._filtered = False

        if drop_old:
            self.conn, self.cursor, self.admin_cursor = self._create_connection()
            try:
                self._drop_db()
                self._create_db_table(dim)
            finally:
                self.cursor.close()
                self.admin_cursor.close()
                self.conn.close()
                self.conn = None
                self.cursor = None
                self.admin_cursor = None

    def _create_connection(self):
        conn = mysql.connect(
            host=self.db_config["host"],
            user=self.db_config["user"],
            port=self.db_config["port"],
            password=self.db_config["password"],
            allow_local_infile=True,
        )
        cursor = conn.cursor()
        admin_cursor = conn.cursor()

        assert conn is not None, "Connection is not initialized"
        assert cursor is not None, "Cursor is not initialized"
        assert admin_cursor is not None, "Admin cursor is not initialized"

        return conn, cursor, admin_cursor

    def _drop_db(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.admin_cursor is not None, "Cursor is not initialized"
        log.info(f"{self.name} client drop db : {self.db_name}")

        # flush tables before dropping database to avoid some locking issue
        self.admin_cursor.execute("FLUSH TABLES")
        self.admin_cursor.execute(f"DROP DATABASE IF EXISTS {self.db_name}")
        self.admin_cursor.execute("COMMIT")
        self.admin_cursor.execute("FLUSH TABLES")

    def _create_db_table(self, dim: int):
        assert self.conn is not None, "Connection is not initialized"
        assert self.admin_cursor is not None, "Cursor is not initialized"

        try:
            log.info(f"{self.name} client create database : {self.db_name}")
            self.admin_cursor.execute(f"CREATE DATABASE {self.db_name}")

            log.info(f"{self.name} client create table : {self.table_name}")
            self.admin_cursor.execute(f"USE {self.db_name}")

            create_table_sql = f"""
              CREATE TABLE {self.table_name} (
                id INT PRIMARY KEY,
                v VECTOR({self.dim}) NOT NULL
              ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
            self.admin_cursor.execute(create_table_sql)
            self.admin_cursor.execute("COMMIT")

        except Exception as e:
            log.warning(f"Failed to create table: {self.table_name} error: {e}")
            raise e from None

    def _probe_binary_support(self) -> bool:
        """Probe once whether the server accepts the raw little-endian float32
        binary VECTOR path: ``UNHEX()`` insert and the ``_binary`` query
        introducer. Returns ``True`` only if both succeed; otherwise ``False``
        so :meth:`init` selects the ``to_vector()`` text path for insert and
        search.

        Runs inside a session-local ``TEMPORARY TABLE`` with a throwaway
        4-dim vector, so it never touches the benchmark table and is
        independent of the real vector dimension. A probe failure is expected
        (not an error) on builds without binary VECTOR support and only flips
        the path; it never raises.
        """
        probe = [1.0, 2.0, 3.0, 4.0]
        blob = struct.pack(f"<{len(probe)}f", *probe)
        # Qualify with the (already-existing) benchmark schema: the client never
        # issues USE, so the connection has no default database to host the temp table.
        tmp = f"`{self.db_name}`._vdbb_binprobe"
        cur = self.conn.cursor(buffered=True)
        try:
            cur.execute(f"CREATE TEMPORARY TABLE {tmp} (id INT PRIMARY KEY, v VECTOR(4))")
            cur.execute(f"INSERT INTO {tmp} (id, v) VALUES (1, UNHEX(%s))", (blob.hex(),))
            cur.execute(f"SELECT id FROM {tmp} ORDER BY L2_DISTANCE(v, _binary %s) LIMIT 1", (blob,))
            cur.fetchall()
        except mysql.Error as e:
            log.warning(f"{self.name}: binary VECTOR path unsupported, falling back to to_vector() text path: {e}")
            return False
        else:
            log.info(f"{self.name}: binary VECTOR path supported; using raw-binary insert + query")
            return True
        finally:
            try:
                cur.execute(f"DROP TEMPORARY TABLE IF EXISTS {tmp}")
            except mysql.Error:
                log.debug("Failed to drop binary-probe temp table", exc_info=True)
            cur.close()

    @contextmanager
    def init(self):
        """create and destory connections to database.

        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
        """
        self.conn, self.cursor, self.admin_cursor = self._create_connection()
        try:
            # The binary probe creates a TEMPORARY TABLE qualified with self.db_name.
            # When drop_old=False on a fresh server, the schema may not exist yet --
            # ensure it does so the probe lands on a real database and doesn't
            # spuriously fall back to the text path.
            self.admin_cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.db_name}")

            # Load-phase session tuning. SESSION-scoped; resets when the
            # connection closes. No GLOBAL or instance-level changes.
            try:
                self.admin_cursor.execute("SET SESSION unique_checks = 0")
                self.admin_cursor.execute("SET SESSION foreign_key_checks = 0")
            except mysql.Error as e:
                log.warning(f"Could not apply load-phase session tuning: {e}")

            # Per-batch TSV file numbering for LOAD DATA bulk load.
            self._batch_counter = 0

            index_param = self.case_config.index_param()
            search_param = self.case_config.search_param()

            if search_param.get("ef_search") is not None:
                try:
                    self.admin_cursor.execute(f"SET loose_hnsw_ef_search = {int(search_param['ef_search'])}")
                    self.conn.commit()
                except mysql.Error:
                    log.warning(
                        f"Could not set loose_hnsw_ef_search = {int(search_param['ef_search'])}, "
                        "using server defaults"
                    )

            # prebuild SQL strings
            dist_func = "L2_DISTANCE" if index_param["metric_type"] == "l2" else "COSINE_DISTANCE"
            # Raw-binary VECTOR path: send float32 vectors as little-endian bytes and let the
            # server consume them directly -- UNHEX(@h) on insert, the `_binary` introducer on
            # query -- with no to_vector() text parse and no Python str() formatting (+71% c80
            # QPS, recall identical). `_binary <literal>` stays constant-foldable so the HNSW
            # index scan is preserved (UNHEX() is NOT, hence hex is used only on the load path).
            #
            # Not every MySQL-compatible build accepts the binary path, so we AUTO-PROBE it once
            # per connection (see _probe_binary_support) and fall back to the to_vector() text
            # path -- for BOTH insert and query -- when it is unsupported. VDB_BINARY_VEC overrides
            # the probe: "1" forces binary (skip probe), "0" forces the to_vector() text path.
            force = os.environ.get("VDB_BINARY_VEC")
            if force == "1":
                self._binary_vec = True
            elif force == "0":
                self._binary_vec = False
            else:
                self._binary_vec = self._probe_binary_support()
            vec_expr = "_binary %s" if self._binary_vec else "to_vector(%s)"
            # No FORCE INDEX hint: optimize() creates idx_v, but read_write streaming
            # cases run search before optimize, so hinting a not-yet-existing index
            # would error every pre-optimize search. The optimizer picks idx_v once
            # it exists; before that, the seq scan is the only valid plan anyway.
            self.select_sql = (
                f"SELECT id FROM {self.db_name}.{self.table_name} ORDER BY {dist_func}(v, {vec_expr}) LIMIT %s"
            )
            self.select_sql_with_filter = (
                f"SELECT id FROM {self.db_name}.{self.table_name} WHERE id >= %s ORDER BY "
                f"{dist_func}(v, {vec_expr}) LIMIT %s"
            )

            yield
        finally:
            try:
                if self.cursor is not None:
                    self.cursor.close()
                if self.admin_cursor is not None:
                    self.admin_cursor.close()
                if self.conn is not None:
                    self.conn.close()
            finally:
                self.cursor = None
                self.admin_cursor = None
                self.conn = None

    def optimize(self, data_size: int | None = None) -> None:
        assert self.conn is not None, "Connection is not initialized"
        assert self.admin_cursor is not None, "Admin cursor is not initialized"

        try:
            log.info(f"{self.name} client create index : {self.table_name}")
            self.admin_cursor.execute(f"USE {self.db_name}")

            # Build vector index attributes
            index_param = self.case_config.index_param()

            attrs_json = _build_index_attrs_json(index_param)

            sql = f"CREATE VECTOR INDEX idx_v ON {self.table_name}(v) SECONDARY_ENGINE_ATTRIBUTE='{attrs_json}'"
            log.info(f"{self.name} client execute create index: {sql}")
            self.admin_cursor.execute(sql)
            self.admin_cursor.execute("COMMIT")
        except Exception as e:
            log.warning(f"Failed to create index on {self.table_name}, error: {e}")
            raise e from None

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs,
    ) -> tuple[int, Exception | None]:
        """Insert a batch via LOAD DATA LOCAL INFILE with sorted PK. Uses the
        hex-binary ``UNHEX`` path when the server supports it (probed in
        :meth:`init`), otherwise the ``to_vector()`` text path. Requires
        self.init() context.
        """
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        n = len(metadata)
        if n == 0:
            return 0, None

        tsv_path = Path(tempfile.gettempdir()) / (
            f"vdbb_volc_{self.db_name}_{self.table_name}_{os.getpid()}_{self._batch_counter}.tsv"
        )
        self._batch_counter += 1

        try:
            _encode_batch_to_tsv(metadata, embeddings, self.dim, str(tsv_path), binary=self._binary_vec)

            tsv_literal = str(tsv_path).replace("'", "''")
            set_clause = "(id, @h) SET v = UNHEX(@h)" if self._binary_vec else "(id, @v) SET v = to_vector(@v)"
            load_sql = (
                f"LOAD DATA LOCAL INFILE '{tsv_literal}' "
                f"INTO TABLE `{self.db_name}`.`{self.table_name}` "
                "FIELDS TERMINATED BY '\\t' LINES TERMINATED BY '\\n' "
                f"{set_clause}"
            )
            self.cursor.execute(load_sql)
            self.conn.commit()
        except Exception as e:
            log.warning(f"Failed to LOAD DATA into Vector table ({self.table_name}), error: {e}")
            return 0, e
        else:
            actual = self.cursor.rowcount
            if actual != n:
                # Return the real count so the runner's already_insert_count tracks
                # what's actually on disk; on retry it will slice past the
                # already-loaded prefix instead of duplicate-PK looping on the
                # same batch. self.conn.commit() ran above, so those rows are
                # durable.
                msg = f"LOAD DATA wrote {actual} rows, expected {n}"
                return actual, RuntimeError(msg)
            return n, None
        finally:
            if tsv_path.exists():
                try:
                    tsv_path.unlink()
                except OSError as e:
                    log.warning(f"Failed to unlink staging TSV {tsv_path}: {e}")

    def prepare_filter(self, filters: Filter):
        """Select the filtered vs unfiltered search template for the case.

        Runners apply filters via this hook (called once per case inside the
        init() context), not by passing ``filters`` to search_embedding. Store
        the predicate value so search_embedding can bind it.
        """
        if filters.type == FilterOp.NonFilter:
            self._filtered = False
        elif filters.type == FilterOp.NumGE:
            self._filtered = True
            self._filter_value = filters.int_value
        else:
            msg = f"Unsupported filter for VolcMySQL: {filters}"
            raise ValueError(msg)

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        **kwargs,
    ) -> list[int]:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        try:
            # Binary path: raw LE float32 bytes (C-level struct.pack, ~us) consumed
            # by the server as a binary vector via `_binary %s`; avoids the per-query
            # str() formatting of 1536 floats and the server-side strtof text parse.
            query_param = struct.pack(f"<{len(query)}f", *query) if self._binary_vec else str(query)
            if self._filtered:
                self.cursor.execute(self.select_sql_with_filter, (self._filter_value, query_param, k))
            else:
                self.cursor.execute(self.select_sql, (query_param, k))
            return [row[0] for row in self.cursor.fetchall()]

        except mysql.Error:
            log.exception("Failed to execute search query")
            raise
