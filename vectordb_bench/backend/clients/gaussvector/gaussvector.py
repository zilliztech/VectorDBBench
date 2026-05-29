"""Wrapper around the GaussVector vector database for VectorDBBench"""

import logging
import pprint
from contextlib import contextmanager
from typing import Any, Generator, Optional, Tuple, Sequence

import numpy as np
import psycopg2
from psycopg2 import OperationalError
from psycopg2.extensions import connection, cursor
from psycopg2 import sql
from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import GaussVectorConfigDict, GaussVectorIndexConfig


# Initialize logger
log = logging.getLogger(__name__)

class GaussVector(VectorDB):
    """Use psycopg2 to interact with GaussVector database"""
    con: None  # Connection object for psycopg2
    cur: None

    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    _unfiltered_search: sql.Composed

    def __init__(
        self,
        dim: int,
        db_config: GaussVectorConfigDict,
        db_case_config: GaussVectorIndexConfig,
        collection_name: str ='gaussvector_collection',
        drop_old: bool = False,  # Drop existing table
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        self.name ='GaussVector'
        self.db_config =db_config
        self.case_config =db_case_config
        self.table_name =collection_name
        self.dim =dim

        self._index_name ='gaussvector_index'
        self._primary_field ='id'
        self._scalar_id_field = "id"  # Integer scalar field for filtering by id
        self._scalar_label_field = "label"
        self._vector_field ='embedding'
        self.with_scalar_labels = with_scalar_labels
        self.with_scalar_id=False
        if kwargs.get("filters") and kwargs.get("filters").type == FilterOp.NumGE:
            self.with_scalar_id=True

        # Create connection
        self.con, self.cur = self._create_connection(**self.db_config)
        self.con.commit()

        log.info(f"{self.name} db_config values: {self.db_config}")
        log.info(f"{self.name} case_config values: {self.case_config}")
        if not any(
            (
                self.case_config.create_index_before_load,
                self.case_config.create_index_after_load,
            )
        ):
            err = f"{self.name} config must create an index using create_index_before_load or create_index_after_load"
            log.error(err)
            raise RuntimeError(
                f"{err}\n{pprint.pformat(self.db_config)}\n{pprint.pformat(self.case_config)}"
            )
        log.info(f"drop_old:{drop_old}")
        if drop_old:
            self._drop_index()
            self._drop_table()
            self._create_table(dim)
            if self.case_config.create_index_before_load:
                self._create_index()

        self.cur.close()
        self.con.close()
        self.cur = None
        self.con = None

    # Create connection
    @staticmethod
    def _create_connection(**kwargs):
        con = psycopg2.connect(**kwargs, connect_timeout=100000, client_encoding='utf-8')
        con.autocommit = True  # Must be set to True
        cur = con.cursor()

        assert con is not None, "Connection is not initialized"
        assert cur is not None, "Cursor is not initialized"
        return con, cur

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """

        self.con, self.cur = self._create_connection(**self.db_config)

        # index configuration may have commands defined that we should set during each client session
        session_options: Sequence[dict[str, Any]] = self.case_config.session_param()["session_options"]

        if len(session_options) > 0:
            for setting in session_options:
                if setting['parameter']['setting_name']!='modify_vector_index_mode':
                    command = sql.SQL("SET {setting_name} " + "= {val};").format(
                        setting_name=sql.Identifier(setting['parameter']['setting_name']),
                        val=sql.Identifier(str(setting['parameter']['val'])),
                    )
                    log.debug(command.as_string(self.cur))
                    self.cur.execute(command)
            self.con.commit()

        # Initialize default filter and search query
        self.expr = ""
        self._search = self._generate_search_query()
        
        try:
            yield
        finally:
            self.cur.close()
            self.con.close()
            self.cur = None
            self.con = None

    def _drop_table(self):
        assert self.con is not None, "Connection is not initialized"
        assert self.cur is not None, "Cursor is not initialized"
        log.info(f"{self.name} client drop table : {self.table_name}")

        self.cur.execute(
            sql.SQL("DROP TABLE IF EXISTS public.{table_name}").format(
                table_name=sql.Identifier(self.table_name)
            )
        )
        self.con.commit()

    def ready_to_load(self):
        pass

    def optimize(self, data_size: int | None = None):
        try:
            self._post_insert()
            self._set_index_mode()
            return True, None  # Return success status and empty result
        except Exception as e:
            log.error(f"Optimize failed: {e}")
            return False, str(e)

    def _post_insert(self):
        log.info(f"{self.name} post insert before optimize")
        if self.case_config.create_index_after_load:
            self._drop_index()
            self._create_index()

    def _drop_index(self):
        assert self.con is not None, "Connection is not initialized"
        assert self.cur is not None, "Cursor is not initialized"
        log.info(f"{self.name} client drop index : {self._index_name}")

        drop_index_sql = sql.SQL("DROP INDEX IF EXISTS {index_name}").format(
            index_name=sql.Identifier(self._index_name)
        )
        log.debug(drop_index_sql.as_string(self.cur))
        self.cur.execute(drop_index_sql)
        self.con.commit()

    def _set_parallel_index_build_param(self):
        assert self.con is not None, "Connection is not initialized"
        assert self.cur is not None, "Cursor is not initialized"

        session_param = self.case_config.session_param()
        if "session_options" in session_param.keys():
            session_options = session_param["session_options"]
            log.info(f"session_options: {session_options}")

            for i in range(len(session_options)):
                if session_options[i]['parameter']['setting_name']!='modify_vector_index_mode':
                    self.cur.execute(
                        sql.SQL(
                            f""" SET {session_options[i]["parameter"]["setting_name"]} = "{session_options[i]["parameter"]["val"]}";"""
                        ))
        self.con.commit()

    def _create_index(self):
        assert self.con is not None, "Connection is not initialized"
        assert self.cur is not None, "Cursor is not initialized"
        log.info(f"{self.name} client create index : {self._index_name}")

        index_param = self.case_config.index_param()
        self._set_parallel_index_build_param()
        options = []

        self.cur.execute(sql.SQL("SHOW maintenance_work_mem;"))
        results = self.cur.fetchall()
        log.info(f"{self.name} session maintenance_work_mem parameters: {results}")

        self.cur.execute(sql.SQL("SHOW diskann_probe_ncandidates;"))
        results = self.cur.fetchall()
        log.info(f"{self.name} session diskann_probe_ncandidates parameters: {results}")

        for option in index_param["index_creation_with_options"]:
            if option['val'] is not None:
                options.append(
                    sql.SQL("{option_name} = {val}").format(
                        option_name=sql.SQL(option['option_name']),
                        val=sql.SQL(str(option['val'])),
                    )
                )    
        index_type = index_param["index_type"]
        if any(options):
            with_clause = sql.SQL(" WITH ({});").format(sql.SQL(", ").join(options))
        else:
            with_clause = sql.Composed("")

        if self.with_scalar_labels:
            index_create_sql = sql.SQL(
                "CREATE INDEX IF NOT EXISTS {index_name} ON public.{table_name} USING {index_type}(embedding {embedding_metric}, label)"
            ).format(
                index_name=sql.SQL(self._index_name),
                table_name=sql.SQL(self.table_name),
                index_type=sql.SQL(index_type),
                embedding_metric=sql.SQL(index_param["metric"]),
            )
        elif self.with_scalar_id :
            assert 'diskann' in index_type, "scalar id only supported diskann"
            index_create_sql = sql.SQL(
                "CREATE INDEX IF NOT EXISTS {index_name} ON public.{table_name} USING {index_type}(embedding {embedding_metric}, id)"
            ).format(
                index_name=sql.SQL(self._index_name),
                table_name=sql.SQL(self.table_name),
                index_type=sql.SQL(index_type),
                embedding_metric=sql.SQL(index_param["metric"]),
            )
        else:
            index_create_sql = sql.SQL(
                "CREATE INDEX IF NOT EXISTS {index_name} ON public.{table_name} USING {index_type}(embedding {embedding_metric})"
            ).format(
                index_name=sql.SQL(self._index_name),
                table_name=sql.SQL(self.table_name),
                index_type=sql.SQL(index_type),
                embedding_metric=sql.SQL(index_param["metric"]),
            )
        index_create_sql_with_with_clause = (
            index_create_sql + with_clause
        )
        log.info(index_create_sql_with_with_clause.as_string(self.cur))
        self.cur.execute(index_create_sql_with_with_clause)
        self.con.commit()

    def _create_table(self, dim: int):
        assert self.con is not None, "Connection is not initialized"
        assert self.cur is not None, "Cursor is not initialized"
        try:
            log.info(f"{self.name} client create table : {self.table_name}")

            if self.with_scalar_labels:
                self.cur.execute(
                    sql.SQL(
                        "CREATE TABLE IF NOT EXISTS public.{table_name} (id BIGINT PRIMARY KEY, label varchar(256), embedding floatvector({dim}));"
                    ).format(table_name=sql.Identifier(self.table_name), dim=sql.Literal(dim))
                )
            else:
                self.cur.execute(
                    sql.SQL(
                        "CREATE TABLE IF NOT EXISTS public.{table_name} (id BIGINT PRIMARY KEY, embedding floatvector({dim}));"
                    ).format(table_name=sql.Identifier(self.table_name), dim=sql.Literal(dim))
                )
            self.con.commit()
        except Exception as e:
            log.warning(
                f"Failed to create gaussvector table: {self.table_name} error: {e}"
            )
            raise e from None

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs: Any,
    ) -> Tuple[int, Optional[Exception]]:
        assert self.con is not None, "Connection is not initialized"
        assert self.cur is not None, "Cursor is not initialized"
        try:
            values_list = []
            if self.with_scalar_labels:
                for i, (meta, label, emb) in enumerate(zip(metadata, labels_data, embeddings)):
                    emb_str = str(emb).replace("[", "\'[").replace("]", "]\'")
                    values_list.append(f"({meta}, '{label}', {emb_str})")
                values_str = ", ".join(values_list)
                insert_sql = f"INSERT INTO public.{self.table_name} (id, label, embedding) VALUES {values_str}"
            else:
                for i, (meta, emb) in enumerate(zip(metadata, embeddings)):
                    emb_str = str(emb).replace("[", "\'[").replace("]", "]\'")
                    values_list.append(f"({meta}, {emb_str})")
                values_str = ", ".join(values_list)
                insert_sql = f"INSERT INTO public.{self.table_name} (id, embedding) VALUES {values_str};"

            self.cur.execute(insert_sql)
            self.con.commit()
            if kwargs.get("last_batch"):
                self._post_insert()

            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into gaussvector table ({self.table_name}), error: {e}")
            return 0, e
    def _set_index_mode(self):
        assert self.con is not None, "Connection is not initialized"
        assert self.cur is not None, "Cursor is not initialized"
        log.info(f"{self.name} client alter index mode : {self._index_name}")
        session_options: Sequence[dict[str, Any]] = self.case_config.session_param()["session_options"]
        if len(session_options) > 0:
            for setting in session_options:
                if setting['parameter']['setting_name']=='modify_vector_index_mode' and setting['parameter']['val']!='3':
                    command = sql.SQL("SELECT gs_modify_vector_index_mode('{index_name}',{val});").format(
                        index_name=sql.SQL(self._index_name),
                        val=sql.SQL(setting['parameter']['val']),
                    )
                    log.debug(command.as_string(self.cur))
                    self.cur.execute(command)
            self.con.commit()

    def prepare_filter(self, filters: Filter):
        """
        Prepare SQL WHERE clause based on filter type.
        Only supports FilterOp.NumGE and FilterOp.StrEqual.
        """
        if filters.type == FilterOp.NonFilter:
            self.expr = ""
        elif filters.type == FilterOp.NumGE:
            # Add condition: id >= value
            self.expr = f"WHERE {self._scalar_id_field} >= {filters.int_value}"
        elif filters.type == FilterOp.StrEqual:
            # Add condition: label = 'value'
            self.expr = f"WHERE {self._scalar_label_field} = '{filters.label_value}'"
        else:
            msg = f"Not support Filter for GaussVector - {filters}"
            raise ValueError(msg)

        # Regenerate search query with new filter (following pgvector pattern)
        self._search = self._generate_search_query()

    def _generate_search_query(self) -> sql.Composed:
        """Generate parameterized search query."""
        metric_fun_op = self.case_config.search_param()["metric_fun_op"]
        where_clause = sql.SQL(self.expr) if self.expr else sql.SQL("1=1")
        
        return sql.Composed([
            sql.SQL("SELECT id FROM public."),
            sql.Identifier(self.table_name),
            sql.SQL(" WHERE "),
            where_clause,
            sql.SQL(" ORDER BY embedding "),
            sql.SQL(metric_fun_op),
            sql.SQL(" "),
            sql.Placeholder(),  # Vector parameter
            sql.SQL(" LIMIT "),
            sql.Placeholder(),  # k parameter
        ])

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        timeout: int | None = None,
    ) -> list[int]:
        assert self.con is not None, "Connection is not initialized"
        assert self.cur is not None, "Cursor is not initialized"

        query_str = str(list(query))
        self.cur.execute(self._search, (query_str, k))
        result = self.cur.fetchall()
        return [int(i[0]) for i in result]