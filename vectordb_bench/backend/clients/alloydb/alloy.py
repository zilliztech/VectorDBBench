
import numpy as np
from pgvector.psycopg2 import register_vector
from psycopg2.extras import execute_values
import logging
import pprint
from contextlib import contextmanager
from typing import Any, Generator, Optional, Tuple, Sequence
import psycopg2
import numpy as np

from ..api import VectorDB
from .config import PgVectorConfigDict, PgVectorIndexConfig

log = logging.getLogger(__name__)



class alloyDB(VectorDB):

    def __init__(
        self,
        dim: int,
        db_config: PgVectorConfigDict,
        db_case_config: PgVectorIndexConfig,
        collection_name: str = "pg_vector_collection",
        drop_old: bool = False,
        **kwargs,
    ):
        self.name = "AlloyDB"
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name
        self.dim = dim

        self._index_name = "hnsw"
        self._primary_field = "id"
        self._vector_field = "embedding"

        # construct basic units
        self.conn, self.cursor = self._create_connection(**self.db_config)

        # create vector extension
        self.conn.commit()
        print(self.conn)

        if drop_old:
            # self.pg_table.drop(pg_engine, checkfirst=True)
            self._drop_index()
            self._drop_table()
            self._create_table(dim)
            if self.case_config.create_index_before_load:
                self._create_index()

        self.cursor.close()
        self.conn.close()
        self.cursor = None
        self.conn = None



    @staticmethod
    def _create_connection(**kwargs):
        '''No problem'''
        conn = psycopg2.connect(
            host = kwargs['host'],
            port = kwargs['port'],
            user = kwargs['user'],
            password = kwargs['password']
        )
        conn.autocommit = False
        cursor = conn.cursor()
        cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        register_vector(conn)

        #cursor.execute(';')
        assert conn is not None, "Connection is not initialized"
        assert cursor is not None, "Cursor is not initialized"
        return conn, cursor




    def _drop_table(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        self.cursor.execute(
            f'''
                DROP TABLE IF EXISTS public.{self.table_name}
            '''
        )
        self.conn.commit()
    

    def ready_to_load(self):
        pass

    def optimize(self):
        self._post_insert()

    def _post_insert(self):
        log.info(f"{self.name} post insert before optimize")
        if self.case_config.create_index_after_load:
            self._drop_index()
            self._create_index()



    def _drop_index(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"
        log.info(f"{self.name} client drop index : {self._index_name}")

        drop_index_sql = f'''
            DROP INDEX IF EXISTS {self._index_name}
        '''

        self.cursor.execute(drop_index_sql)
        self.conn.commit()


    @contextmanager
    def init(self) -> Generator[None, None, None]:
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """

        self.conn, self.cursor = self._create_connection(**self.db_config)

        # index configuration may have commands defined that we should set during each client session
        
        session_options: Sequence[dict[str, Any]] = self.case_config.session_param()["session_options"]


        if len(session_options) > 0:
            for setting in session_options:
                command = f'''SET {setting['parameter']['setting_name']} = {setting['parameter']['val']}'''
                
                self.cursor.execute(command)
            self.conn.commit()


        try:
            yield
        finally:
            self.cursor.close()
            self.conn.close()
            self.cursor = None
            self.conn = None


    def _set_parallel_index_build_param(self):
        pass



    def _create_index(self):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        index_param = self.case_config.index_param()
        index_create_sql = f'''CREATE INDEX IF NOT EXISTS {self._index_name} ON public.{self.table_name} USING {index_param["index_type"]} (embedding {index_param["metric"]})'''

        self.cursor.execute(index_create_sql)
        self.conn.commit()





    def _create_table(self, dim: int):
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        try:
            # create table
            self.cursor.execute(
                f'''
                CREATE TABLE IF NOT EXISTS public.{self.table_name} (id BIGINT PRIMARY KEY, embedding vector({self.dim}));
                '''
            )
            self.cursor.execute(
                f'''
                    ALTER TABLE public.{self.table_name} ALTER COLUMN embedding SET STORAGE PLAIN;
                '''
            )
            self.conn.commit()
        except Exception as e:
            raise e from None



    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> Tuple[int, Optional[Exception]]:
        assert self.conn is not None, "Connection is not initialized"
        assert self.cursor is not None, "Cursor is not initialized"

        try:
            metadata_arr = np.array(metadata)
            embeddings_arr = np.array(embeddings)

            for i in range(len(metadata_arr)):
                meta = metadata[i]
                arr = np.array(embeddings_arr[i])
                self.cursor.execute(
                    f'insert into {self.table_name} (id, embedding) values (%s, %s);', (meta, arr)
                )
            self.conn.commit()

            return len(metadata), None
        except Exception as e:
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

        arr = np.array(query)
        try:
            self.cursor.execute(f'''
            SELECT id FROM public.{self.table_name} ORDER BY embedding <=> %s LIMIT {k};
            ''', (arr,))
        except Exception as e:
            raise e from None

        result = self.cursor.fetchall()
        return [int(i[0]) for i in result]


