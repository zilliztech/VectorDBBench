from vectordb_bench.backend.cases import (
    PerformanceCase,
    CaseType,
)

import vectordb_bench.backend.dataset as ds
from pydantic.dataclasses import dataclass

@dataclass
class Cohere_S(ds.Cohere):
    label: str = "SMALL"
    size: int  = 100_000

@dataclass
class Glove_S(ds.Glove):
    label: str = "SMALL"
    size : int = 100_000


class Performance100K99p(PerformanceCase):
    case_id: CaseType = CaseType.PerformanceSHigh
    filter_rate: float | int | None = 0.99
    dataset: ds.DataSet = ds.get(ds.Name.Cohere, ds.Label.SMALL)
    name: str = "Filtering Search Performance Test (100K Dataset, 768 Dim, Filter 99%)"
    description: str = """This case tests the search performance of a vector database with a small dataset (<b>Cohere 100K vectors</b>, 768 dimensions) under a high filtering rate (<b>99% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""

class Performance100K1p(PerformanceCase):
    case_id: CaseType = CaseType.PerformanceSLow
    filter_rate: float | int | None = 0.01
    dataset: ds.DataSet = ds.get(ds.Name.Cohere, ds.Label.SMALL)
    name: str = "Filtering Search Performance Test (100K Dataset, 768 Dim, Filter 1%)"
    description: str = (
        """This case tests the search performance of a vector database with a small dataset (<b>Cohere 100K vectors</b>, 768 dimensions) under a low filtering rate (<b>1% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS.""",
    )


class Performance100K(PerformanceCase):
    case_id: CaseType = CaseType.PerformanceSZero
    dataset: ds.DataSet = ds.get(ds.Name.Cohere, ds.Label.SMALL)
    name: str = "Search Performance Test (100K Dataset, 768 Dim)"
    description: str = """This case tests the search performance of a vector database with a small dataset (<b>Cohere 100K vectors</b>, 768 dimensions) at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""


