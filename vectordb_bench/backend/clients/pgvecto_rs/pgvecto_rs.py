"""Wrapper around the Pgvector vector database over VectorDB"""

import io
import logging
from contextlib import contextmanager
from typing import Any
import pandas as pd

import psycopg2

from ..api import VectorDB, DBCaseConfig

log = logging.getLogger(__name__)


class PgVectoRS(VectorDB):
    """Use SQLAlchemy instructions"""

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        collection_name: str = "PgVectorCollection",
        drop_old: bool = False,
        **kwargs,
    ):
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
        self.cursor.execute("CREATE EXTENSION IF NOT EXISTS vectors")
        self.conn.commit()

        if drop_old:
            log.info(f"Pgvecto.rs client drop table : {self.table_name}")
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
        pass

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

        try:
            # create table
            self.cursor.execute(
                f'CREATE INDEX IF NOT EXISTS {self._index_name} ON public."{self.table_name}" \
                    USING vectors (embedding {index_param["metric"]}) WITH (options = $${index_param["options"]}$$);'
            )
            self.conn.commit()
        except Exception as e:
            log.warning(
                f"Failed to create pgvector table: {self.table_name} error: {e}"
            )
            raise e from None

    def _create_table(self, dim: int):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        try:
            # create table
            self.cursor.execute(
                f'CREATE TABLE IF NOT EXISTS public."{self.table_name}" \
                    (id Integer PRIMARY KEY, embedding vector({dim}));'
            )
            self.cursor.execute(
                f'ALTER TABLE public."{self.table_name}" ALTER COLUMN embedding SET STORAGE PLAIN;'
            )
            self.conn.commit()
        except Exception as e:
            log.warning(
                f"Failed to create pgvector table: {self.table_name} error: {e}"
            )
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
            items = {"id": metadata, "embedding": embeddings}
            df = pd.DataFrame(items)
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False, header=False)
            csv_buffer.seek(0)
            self.cursor.copy_expert(
                f'COPY public."{self.table_name}" FROM STDIN WITH (FORMAT CSV)',
                csv_buffer,
            )
            self.conn.commit()
            return len(metadata), None
        except Exception as e:
            log.warning(
                f"Failed to insert data into pgvector table ({self.table_name}), error: {e}"
            )

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
    ) -> list[int]:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        search_param = self.case_config.search_param()

        if filters:
            gt = filters.get("id")
            self.cursor.execute(
                f"SELECT id FROM (SELECT * FROM public.\"{self.table_name}\" ORDER BY embedding \
                    {search_param['metrics_op']} '{query}' LIMIT {k}) AS X WHERE id > {gt} ;"
            )
        else:
            self.cursor.execute(
                f"SELECT id FROM public.\"{self.table_name}\" ORDER BY embedding \
                    {search_param['metrics_op']} '{query}' LIMIT {k};"
            )
        self.conn.commit()
        result = self.cursor.fetchall()

        return [int(i[0]) for i in result]
