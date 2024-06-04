import psycopg
import psycopg2.extras
import vectordb_bench.backend.clients.pgvector.config as pgconfig
import vectordb_bench.backend.clients.pgvector.pgvector as pgvector
import vectordb_bench.backend.clients.alloydb.alloy as alloy

import random
import vectordb_bench.backend.clients.alloydb.config as c
import psycopg2
import numpy as np
from pgvector.psycopg2 import register_vector
from psycopg import sql
from typing import Any, Generator, Optional, Tuple, Sequence
from io import StringIO
from psycopg2.extras import execute_values

tmp: pgconfig.PgVectorConfigDict = {
    "user": 'postgres',
    "password": "postgres",
    "host": "35.245.99.238",
    "port": 5432,
    "dbname": "my_database"
}


host = "35.245.99.238"
port = 5432
user = 'postgres'
password = 'postgres'
table_name = 'test_table'
index_name = 'hnsw'

dim = 10
case_config = pgconfig.PgVectorIndexConfig
db_config = pgconfig.PgVectorConfigDict
index_param = {
    'metric': 'vector_l2_ops',
    'index_type': 'hnsw',
    'index_creation_with_options': None,
    'maintenance_work_mem': None,
    'max_parallel_workers': None
}


def test_connection(**kwargs):
    '''No problem'''
    conn = psycopg2.connect(
        host = kwargs['host'],
        port = kwargs['port'],
        user = kwargs['user'],
        password = kwargs['password']
    )
    conn.autocommit = False
    register_vector(conn)
    cursor = conn.cursor()
    return conn, cursor


def test_drop_table(conn, cursor):
    assert conn is not None, "Connection is not initialized"
    assert cursor is not None, "Cursor is not initialized"

    cursor.execute(
        f'''
            DROP TABLE IF EXISTS public.{table_name}
        '''
    )
    conn.commit()


def test_drop_index(conn, cursor):
    assert conn is not None, "Connection is not initialized"
    assert cursor is not None, "Cursor is not initialized"

    drop_index_sql = f'''
        DROP INDEX IF EXISTS {index_name}
    '''

    cursor.execute(drop_index_sql)
    conn.commit()


def test_create_index(conn, cursor, index_param):
    assert conn is not None, "Connection is not initialized"
    assert cursor is not None, "Cursor is not initialized"


    index_create_sql = f'''CREATE INDEX IF NOT EXISTS {index_name} 
                ON public.{table_name} USING {index_param["index_type"]} 
                (embedding {index_param["metric"]})'''

    cursor.execute(index_create_sql)
    conn.commit()



def test_create_table(conn, cursor, dim: int):
    assert conn is not None, "Connection is not initialized"
    assert cursor is not None, "Cursor is not initialized"

    try:
        # create table
        cursor.execute(
            f'''
            CREATE TABLE IF NOT EXISTS public.{table_name} 
            (id BIGINT PRIMARY KEY, embedding vector({dim}));
            '''
        )
        cursor.execute(
            f'''
                ALTER TABLE public.{table_name} 
                ALTER COLUMN embedding SET STORAGE PLAIN;
            '''
        )
        conn.commit()
    except Exception as e:
        raise e from None



def test_insert_embeddings(
    conn,
    cursor,
    embeddings: list[list[float]],
    metadata: list[int],
    **kwargs: Any,
) -> Tuple[int, Optional[Exception]]:
    
    assert conn is not None, "Connection is not initialized"
    assert cursor is not None, "Cursor is not initialized"

    try:
        metadata_arr = np.array(metadata)
        embeddings_arr = np.array(embeddings)

        for i in range(len(metadata_arr)):
            meta = int(i)
            arr = np.array(embeddings_arr[i])
            cursor.execute(
                f'insert into {table_name} (id, embedding) values (%s, %s);', (meta, arr)
            )

        conn.commit()

        return len(metadata), None
    except Exception as e:
        return 0, e



def test_search_embedding(
    conn,
    cursor,
    query: list[float],
    k: int = 100,
    filters: dict | None = None,
    timeout: int | None = None,
) -> list[int]:
    assert conn is not None, "Connection is not initialized"
    assert cursor is not None, "Cursor is not initialized"

    arr = np.array(query)
    try:
        cursor.execute(f'''
        SELECT embedding FROM public.{table_name}
        ORDER BY embedding <=> %s LIMIT 10;
        ''', (arr,))
    except Exception as e:
        raise e from None

    result = cursor.fetchall()
    return result






list = np.random.rand(1000, dim).tolist()
meta = np.random.randint(0, 10, size = 1000).tolist()
query = [random.uniform(0, 1) for _ in range(10)]

conn, cursor = test_connection(**tmp)
test_drop_table(conn, cursor)
test_drop_index(conn, cursor)
test_create_table(conn, cursor, dim)
test_create_index(conn, cursor, index_param)
len = test_insert_embeddings(conn, cursor, list, meta)
print(len)
res = test_search_embedding(conn, cursor, query, 10)
for i in res:
    print(i)

