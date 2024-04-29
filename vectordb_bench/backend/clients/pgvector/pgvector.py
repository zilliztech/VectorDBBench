"""Wrapper around the Pgvector vector database over VectorDB"""

import io
import logging
from contextlib import contextmanager
from typing import Any
import pandas as pd
import psycopg2
import psycopg2.extras

from ..api import IndexType, VectorDB, DBCaseConfig

log = logging.getLogger(__name__)

class PgVector(VectorDB):
    """ Use SQLAlchemy instructions"""
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        collection_name: str = "PgVectorCollection",
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "PgVector"
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self.dim = dim

        self._index_name = "pqvector_index"
        self._primary_field = "id"
        self._vector_field = "embedding"

        # construct basic units
        self.conn = psycopg2.connect(**self.db_config)
        self.conn.autocommit = False
        self.cursor = self.conn.cursor()

        # create vector extension
        self.cursor.execute('CREATE EXTENSION IF NOT EXISTS vector')
        self.conn.commit()

        if drop_old :
            log.info(f"Pgvector client drop table : {self.table_name}")
            # self.pg_table.drop(pg_engine, checkfirst=True)
            self._drop_index()
            self._drop_table()
            self._create_table(dim)
            self._create_index()

        self.cursor.close()
        self.conn.close()
        self.cursor = None
        self.conn = None

    @contextmanager
    def init(self) -> None:
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """
        self.conn = psycopg2.connect(**self.db_config)
        self.conn.autocommit = False
        self.cursor = self.conn.cursor()

        try:
            yield
        finally:
            self.cursor.close()
            self.conn.close()
            self.cursor = None
            self.conn = None

    def _drop_table(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        self.cursor.execute(f'DROP TABLE IF EXISTS public."{self.table_name}"')
        self.conn.commit()

    def ready_to_load(self):
        pass

    def optimize(self):
        log.info(f"{self.name} optimizing")
        self._drop_index()
        self._create_index()

    def ready_to_search(self):
        pass

    def _drop_index(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        self.cursor.execute(f'DROP INDEX IF EXISTS "{self._index_name}"')
        self.conn.commit()

    def _create_index(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        index_param = self.case_config.index_param()
        if self.case_config.index == IndexType.HNSW:
            log.debug(f'Creating HNSW index. m={index_param["m"]}, ef_construction={index_param["ef_construction"]}')
            self.cursor.execute(f'CREATE INDEX IF NOT EXISTS {self._index_name} ON public."{self.table_name}" USING hnsw (embedding {index_param["metric"]}) WITH (m={index_param["m"]}, ef_construction={index_param["ef_construction"]});')
        elif self.case_config.index == IndexType.IVFFlat:
            log.debug(f'Creating IVFFLAT index. list={index_param["lists"]}')
            self.cursor.execute(f'CREATE INDEX IF NOT EXISTS {self._index_name} ON public."{self.table_name}" USING ivfflat (embedding {index_param["metric"]}) WITH (lists={index_param["lists"]});')
        else:
            assert "Invalid index type {self.case_config.index}"
        self.conn.commit()

    def _create_table(self, dim : int):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        try:
            # create table
            self.cursor.execute(f'CREATE TABLE IF NOT EXISTS public."{self.table_name}" (id BIGINT PRIMARY KEY, embedding vector({dim}));')
            self.cursor.execute(f'ALTER TABLE public."{self.table_name}" ALTER COLUMN embedding SET STORAGE PLAIN;')
            self.conn.commit()
        except Exception as e:
            log.warning(f"Failed to create pgvector table: {self.table_name} error: {e}")
            raise e from None

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> (int, Exception):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        try:
            items = {
                "id": metadata,
                "embedding": embeddings
            }
            df = pd.DataFrame(items)
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False, header=False)
            csv_buffer.seek(0)
            self.cursor.copy_expert(f"COPY public.\"{self.table_name}\" FROM STDIN WITH (FORMAT CSV)", csv_buffer)
            self.conn.commit()

            return len(metadata), None
        except Exception as e:
            log.warning(f"Failed to insert data into pgvector table ({self.table_name}), error: {e}")
            return 0, e

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
    ) -> list[int]:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        search_param =self.case_config.search_param()

        if self.case_config.index == IndexType.HNSW:
            self.cursor.execute(f'SET hnsw.ef_search = {search_param["ef"]}')
            self.cursor.execute(f"SELECT id FROM public.\"{self.table_name}\" ORDER BY embedding {search_param['metric_fun_op']} '{query}' LIMIT {k};")
        elif self.case_config.index == IndexType.IVFFlat:
            self.cursor.execute(f'SET ivfflat.probes = {search_param["probes"]}')
            self.cursor.execute(f"SELECT id FROM public.\"{self.table_name}\" ORDER BY embedding {search_param['metric_fun_op']} '{query}' LIMIT {k};")
        else:
            assert "Invalid index type {self.case_config.index}"
        self.conn.commit()
        result = self.cursor.fetchall()

        return [int(i[0]) for i in result]
