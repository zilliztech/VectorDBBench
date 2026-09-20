import array
import hashlib
import logging
import re
from collections.abc import Generator
from contextlib import contextmanager
from typing import Any

import oracledb

from vectordb_bench.backend.filter import Filter, FilterOp
from vectordb_bench.backend.payload import PayloadProfile

from ..api import IndexType, VectorDB

log = logging.getLogger(__name__)


def validate_sql_identifier(identifier: str) -> str:
    if not identifier or len(identifier.encode("utf-8")) > 128:
        msg = f"Invalid SQL identifier length ({len(identifier)}): {identifier}"
        raise ValueError(msg)
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", identifier):
        msg = f"Invalid SQL identifier characters: {identifier}"
        raise ValueError(msg)
    return identifier


def get_unique_index_name(table_name: str, index_type_str: str) -> str:
    suffix = f"_{index_type_str.lower()}_idx"
    max_prefix_len = 128 - len(suffix) - 9
    short_table = table_name[:max_prefix_len] if max_prefix_len > 0 else "vdb"
    h = hashlib.md5(table_name.encode("utf-8"), usedforsecurity=False).hexdigest()[:8]
    index_name = f"{short_table}_{h}{suffix}"
    return validate_sql_identifier(index_name)


class Oracle(VectorDB):
    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]
    thread_safe: bool = False

    conn: oracledb.Connection | None = None
    cursor: oracledb.Cursor | None = None

    def __init__(
        self,
        dim: int,
        db_config: dict[str, Any],
        db_case_config: Any | None,
        collection_name: str = "vdbbench_oracle_test",
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs: Any,
    ):
        self.name = "Oracle"
        self.dim = int(dim)
        self.db_config = db_config
        self.case_config = db_case_config
        raw_table = db_config.get("collection_name") or db_config.get("table_name") or collection_name
        self.table_name = validate_sql_identifier(raw_table)
        self.with_scalar_labels = with_scalar_labels

        self.where_clause = ""
        self.filter_bind_key = None
        self.filter_bind_val = None

        with self.init():
            if drop_old:
                self._drop_table()
                self._create_table()
            else:
                self._create_table_if_not_exists()

    def _create_connection(self) -> tuple[oracledb.Connection, oracledb.Cursor]:
        user = self.db_config["user"]
        password = self.db_config["password"]
        host = self.db_config["host"]
        port = self.db_config["port"]
        service_name = self.db_config["service_name"]
        sysdba = self.db_config.get("sysdba", False)

        dsn = f"{host}:{port}/{service_name}"
        mode = (
            oracledb.AUTH_MODE_SYSDBA
            if sysdba or user.lower() == "sys"
            else oracledb.DEFAULT_AUTH
        )

        conn = oracledb.connect(
            user=user,
            password=password,
            dsn=dsn,
            mode=mode,
        )
        cursor = conn.cursor()
        return conn, cursor

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        conn, cursor = self._create_connection()
        try:
            self.conn = conn
            self.cursor = cursor
            yield
        finally:
            if self.cursor:
                self.cursor.close()
            if self.conn:
                self.conn.close()
            self.cursor = None
            self.conn = None

    def _drop_table(self) -> None:
        assert self.cursor is not None, "Cursor is not initialized"
        sql = f"""
        BEGIN
            EXECUTE IMMEDIATE 'DROP TABLE {self.table_name} PURGE';
        EXCEPTION
            WHEN OTHERS THEN
                IF SQLCODE != -942 THEN RAISE; END IF;
        END;
        """
        self.cursor.execute(sql)

    def _create_table(self) -> None:
        assert self.cursor is not None, "Cursor is not initialized"
        if self.with_scalar_labels:
            sql = f"""
            CREATE TABLE {self.table_name} (
                id NUMBER(19) NOT NULL PRIMARY KEY,
                embedding VECTOR({self.dim}, FLOAT32),
                label VARCHAR2(64)
            )
            """
        else:
            sql = f"""
            CREATE TABLE {self.table_name} (
                id NUMBER(19) NOT NULL PRIMARY KEY,
                embedding VECTOR({self.dim}, FLOAT32)
            )
            """
        self.cursor.execute(sql)

    def _create_table_if_not_exists(self) -> None:
        assert self.cursor is not None, "Cursor is not initialized"
        try:
            self._create_table()
        except oracledb.DatabaseError as e:
            if e.args and hasattr(e.args[0], "code") and e.args[0].code == 955:
                log.info(f"Table {self.table_name} already exists. Reusing existing table.")
            else:
                raise

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        tenant_labels_data: list[str] | None = None,
        **kwargs: Any,
    ) -> tuple[int, Exception | None]:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        if self.with_scalar_labels:
            assert labels_data is not None, "labels_data must be provided when with_scalar_labels=True"

        try:
            batch_data = []
            for i, doc_id in enumerate(metadata):
                vec = embeddings[i]
                if len(vec) != self.dim:
                    msg = f"Vector dimension mismatch: expected {self.dim}, got {len(vec)}"
                    raise ValueError(msg)  # noqa: TRY301
                vec_arr = array.array("f", vec)
                if self.with_scalar_labels:
                    batch_data.append((int(doc_id), vec_arr, str(labels_data[i])))
                else:
                    batch_data.append((int(doc_id), vec_arr))

            if self.with_scalar_labels:
                insert_sql = f"INSERT INTO {self.table_name} (id, embedding, label) VALUES (:1, :2, :3)"
            else:
                insert_sql = f"INSERT INTO {self.table_name} (id, embedding) VALUES (:1, :2)"

            self.cursor.executemany(insert_sql, batch_data)
            self.conn.commit()
            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into Oracle table ({self.table_name}), error: {e}")
            return 0, e

    def prepare_filter(self, filters: Filter) -> None:
        if filters.type == FilterOp.NonFilter:
            self.where_clause = ""
            self.filter_bind_key = None
            self.filter_bind_val = None
        elif filters.type == FilterOp.NumGE:
            self.where_clause = "WHERE id >= :filter_val"
            self.filter_bind_key = "filter_val"
            self.filter_bind_val = filters.int_value
        elif filters.type == FilterOp.StrEqual:
            self.where_clause = "WHERE label = :filter_val"
            self.filter_bind_key = "filter_val"
            self.filter_bind_val = filters.label_value
        else:
            msg = f"Unsupported filter type for Oracle: {filters.type}"
            raise ValueError(msg)

    def optimize(self, data_size: int | None = None) -> None:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        if not self.case_config or not getattr(self.case_config, "create_index_after_load", True):
            log.info("Index creation after load is disabled for this case.")
            return

        index_params = self.case_config.index_param()
        index_type = index_params.get("index_type")

        if index_type == IndexType.Flat.value:
            log.info("Flat exact search case; no vector index created.")
            return

        try:
            self.cursor.execute("SELECT value FROM v$parameter WHERE name = 'vector_memory_size'")
            row = self.cursor.fetchone()
            if row and row[0] == "0":
                log.warning("Oracle vector_memory_size is set to 0. In-memory vector index creation may fail.")
        except Exception as e:
            log.debug(f"Could not check vector_memory_size (insufficient privileges?): {e}")

        metric = index_params.get("metric", "COSINE")
        index_name = get_unique_index_name(self.table_name, index_type)

        try:
            if index_type == IndexType.HNSW.value:
                neighbors = index_params.get("neighbors", 32)
                ef_construction = index_params.get("ef_construction", 200)
                index_acc = index_params.get("index_target_accuracy", 95)
                ddl = f"""
                CREATE VECTOR INDEX {index_name} ON {self.table_name} (embedding)
                ORGANIZATION INMEMORY NEIGHBOR GRAPH
                DISTANCE {metric}
                WITH TARGET ACCURACY {index_acc}
                PARAMETERS (
                    TYPE HNSW,
                    NEIGHBORS {neighbors},
                    EFCONSTRUCTION {ef_construction}
                )
                """
            elif index_type == IndexType.IVFFlat.value:
                partitions = index_params.get("neighbor_partitions", 1024)
                samples = index_params.get("samples_per_partition", 10)
                min_vecs = index_params.get("min_vectors_per_partition", 5)
                index_acc = index_params.get("index_target_accuracy", 90)
                ddl = f"""
                CREATE VECTOR INDEX {index_name} ON {self.table_name} (embedding)
                ORGANIZATION NEIGHBOR PARTITIONS
                DISTANCE {metric}
                WITH TARGET ACCURACY {index_acc}
                PARAMETERS (
                    TYPE IVF,
                    NEIGHBOR PARTITIONS {partitions},
                    SAMPLES_PER_PARTITION {samples},
                    MIN_VECTORS_PER_PARTITION {min_vecs}
                )
                """
            else:
                msg = f"Unsupported index type for Oracle: {index_type}"
                raise ValueError(msg)

            log.info(f"Creating Oracle vector index {index_name} with DDL:\n{ddl}")
            self.cursor.execute(ddl)
            self.conn.commit()
            log.info(f"Oracle vector index {index_name} created successfully.")
        except oracledb.DatabaseError as e:
            log.exception(f"Failed to create Oracle vector index ({index_name})")
            msg = f"Oracle vector index creation failed: {e}"
            raise RuntimeError(msg) from e

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        payload_profile: PayloadProfile = PayloadProfile.IDS_ONLY,
        tenant: str | None = None,
        **kwargs: Any,
    ) -> list[int]:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        if not self.supports_payload_profile(payload_profile):
            msg = f"Unsupported payload profile: {payload_profile}"
            raise ValueError(msg)

        search_params = self.case_config.search_param() if self.case_config else {}
        metric = search_params.get("metric", "COSINE")

        q_vec = array.array("f", query)
        if len(q_vec) != self.dim:
            msg = f"Query vector dimension mismatch: expected {self.dim}, got {len(q_vec)}"
            raise ValueError(msg)

        binds: dict[str, Any] = {
            "q_vec": q_vec,
            "k_val": k,
        }
        if self.filter_bind_key and self.filter_bind_val is not None:
            binds[self.filter_bind_key] = self.filter_bind_val

        index_type = (
            getattr(self.case_config, "index_param", dict)().get("index_type")
            if self.case_config
            else None
        )

        if (
            index_type == IndexType.Flat.value
            or not self.case_config
            or not getattr(self.case_config, "create_index_after_load", True)
        ):
            search_sql = f"""
            SELECT id
            FROM {self.table_name}
            {self.where_clause}
            ORDER BY VECTOR_DISTANCE(embedding, :q_vec, {metric}) ASC
            FETCH EXACT FIRST :k_val ROWS ONLY
            """
        else:
            target_acc = search_params.get("search_target_accuracy", 95)
            binds["target_acc"] = target_acc
            search_sql = f"""
            SELECT id
            FROM {self.table_name}
            {self.where_clause}
            ORDER BY VECTOR_DISTANCE(embedding, :q_vec, {metric}) ASC
            FETCH APPROXIMATE FIRST :k_val ROWS ONLY WITH TARGET ACCURACY :target_acc
            """

        self.cursor.execute(search_sql, binds)
        res = self.cursor.fetchall()
        return [int(row[0]) for row in res]
