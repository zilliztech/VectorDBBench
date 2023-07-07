"""Wrapper around the Pgvector vector database over VectorDB"""

import logging
import time
from contextlib import contextmanager
from typing import Any, Type

from ..api import VectorDB, DBConfig, DBCaseConfig, IndexType
from .config import PgVectorConfig, PgVectorIndexConfig
from sqlalchemy.orm import (
    declarative_base, 
    mapped_column, 
    Session
)

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
    ):
        self.db_config = db_config
        self.case_config = db_case_config
        self.table_name = collection_name

        self._index_name = "pqvector_index"
        self._primary_field = "id"
        self._vector_field = "embedding"

        # construct basic units
        pq_metadata = MetaData()
        self.pg_engine = create_engine(**self.db_config)
        # create vector extension
        with self.pg_engine as conn: 
            conn.execute(text('CREATE EXTENSION IF NOT EXISTS vector'))
            conn.commit()

        self.pg_table = Table(
            self.table_name,
            pq_metadata,
            Column(self._primary_field, Integer, primary_key=True),
            Column(self._vector_field, Vector(dim))
        )
        if drop_old and self.table_name in pq_metadata.tables:
            log.info(f"Pgvector client drop table : {self.table_name}")
            self.self.pq_table.drop(bind = engine)
        
        self._create_table(dim)

    
    @classmethod
    def config_cls(cls) -> Type[DBConfig]:
        return PgVectorConfig

    @classmethod
    def case_config_cls(cls, index_type: IndexType | None = None) -> Type[DBCaseConfig]:
        return PgVectorIndexConfig

    @contextmanager
    def init(self) -> None:
        """
        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
            >>>     self.search_embedding()
        """
        self.pq_session = Session(self.pg_engine)
        yield 
        self.pq_session = None
        del (self.pq_session)
    
    def ready_to_load(self):
        pass

    def ready_to_search(self):
        pass
    
    def _create_index(self):
        index = Index(self._index_name, self.pq_table.embedding, **self.case_config.index_param())
        index.create(self.pg_engine)

    def _create_table(self, dim : int):
        try:
            self.pg_table.create(bind = self.pg_engine, checkfirst = True)
            self._create_index()
        except Exception as e:
            log.warning(f"Failed to create pgvector table: {self.table_name} error: {e}")
            raise e from None

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> (int, Exception):
        try:
            items = [dict(id = metadata[i], embedding=embeddings[i]) for i in range(metadata)]
            self.pq_session.execute(insert(table), items)
            self.pq_session.commit()
            return len(items), None
        except Exception as e:
            log.warning(f"Failed to insert data into pgvector table ({self.table_name}), error: {e}")   
            return 0, e

    def search_embedding(        
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> list[int]:
        assert self.pq_table is not None
        with self.pg_engine as conn: 
            conn.execute(text(f'SET ivfflat.probes = {kwargs["probes"]}'))
            conn.commit()
        op_fun = getattr(table.c.embedding, kwargs["metric_fun"])
        if filters:
            res = self.pq_session.scalars(select(self.pq_table.order_by(op_fun(query)).filter(self.pq_table.c.id > filters.get('id')).limit(k)))
        else: 
            res = self.pq_session.scalars(select(self.pq_table.order_by(op_fun(query)).limit(k)))
        return list(res)
        