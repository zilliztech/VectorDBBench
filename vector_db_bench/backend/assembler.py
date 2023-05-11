"""Assembler assembles cases with datasets and runners"""

from typing import Any
from .cases import Case
from .client import Client

class Assembler:
    """
    Examples:
    """

    def assemble() -> list[Any]:
        pass


    def _get_db_clients(self, dbs: list[Any]) -> list[Client]:
        pass

    def _get_cases(cases_config: list[Any]) -> list[Case]:
        pass
