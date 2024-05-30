import logging
from contextlib import contextmanager
from typing import Any, Generator, Optional, Tuple

from ..api import DBCaseConfig, VectorDB

log = logging.getLogger(__name__)


class Test(VectorDB):
    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: DBCaseConfig,
        drop_old: bool = False,
        **kwargs,
    ):
        self.db_config = db_config
        self.case_config = db_case_config

        log.info("Starting Test DB")

    @contextmanager
    def init(self) -> Generator[None, None, None]:
        """create and destroy connections to database.

        Examples:
            >>> with self.init():
            >>>     self.insert_embeddings()
        """

        yield

    def ready_to_load(self) -> bool:
        return True

    def optimize(self) -> None:
        pass

    def insert_embeddings(
        self,
        embeddings: list[list[float]],
        metadata: list[int],
        **kwargs: Any,
    ) -> Tuple[int, Optional[Exception]]:
        """Insert embeddings into the database.
        Should call self.init() first.
        """
        raise RuntimeError("Not implemented")
        return len(metadata), None

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        filters: dict | None = None,
        timeout: int | None = None,
        **kwargs: Any,
    ) -> list[int]:
        raise NotImplementedError
        return [i for i in range(k)]
