from vectordb_bench.backend.cases import (
    PerformanceCase,
    CaseType,
)

from vectordb_bench.backend.datase import Dataset, DatasetManager


class Performance100K99p(PerformanceCase):
    case_id: CaseType = 100
    filter_rate: float | int | None = 0.99
    dataset: DatasetManager = Dataset.COHERE.manager(100_000)
    name: str = "Filtering Search Performance Test (100K Dataset, 768 Dim, Filter 99%)"
    description: str = """This case tests the search performance of a vector database with a small dataset (<b>Cohere 100K vectors</b>, 768 dimensions) under a high filtering rate (<b>99% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""

class Performance100K1p(PerformanceCase):
    case_id: CaseType = 100
    filter_rate: float | int | None = 0.01
    dataset: DatasetManager = Dataset.COHERE.manager(100_000)
    name: str = "Filtering Search Performance Test (100K Dataset, 768 Dim, Filter 1%)"
    description: str = (
        """This case tests the search performance of a vector database with a small dataset (<b>Cohere 100K vectors</b>, 768 dimensions) under a low filtering rate (<b>1% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS.""",
    )


class Performance100K(PerformanceCase):
    case_id: CaseType = 100
    dataset: DatasetManager = Dataset.COHERE.manager(100_000)
    name: str = "Search Performance Test (100K Dataset, 768 Dim)"
    description: str = """This case tests the search performance of a vector database with a small dataset (<b>Cohere 100K vectors</b>, 768 dimensions) at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
