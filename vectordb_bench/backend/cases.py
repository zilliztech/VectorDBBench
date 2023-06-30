import typing
import logging
from enum import Enum, auto

from vectordb_bench import config
from vectordb_bench.base import BaseModel

from .dataset import Dataset, DatasetManager


log = logging.getLogger(__name__)

Case = typing.TypeVar("Case")


class CaseType(Enum):
    """
    Example:
        >>> case_cls = CaseType.CapacityDim128.case_cls
        >>> assert c is not None
        >>> CaseType.CapacityDim128.case_name
        "Capacity Test (128 Dim Repeated)"
    """

    CapacityDim128 = 1
    CapacityDim960 = 2

    Performance100M = 3
    Performance10M = 4
    Performance1M = 5

    Performance10M1P = 6
    Performance1M1P = 7
    Performance10M99P = 8
    Performance1M99P = 9

    Custom = 100

    @property
    def case_cls(self, custom_configs: dict | None = None) -> Case:
        return type2case.get(self)

    @property
    def case_name(self) -> str:
        c = self.case_cls
        if c is not None:
            return c().name
        raise ValueError("Case unsupported")

    @property
    def case_description(self) -> str:
        c = self.case_cls
        if c is not None:
            return c().description
        raise ValueError("Case unsupported")


class CaseLabel(Enum):
    Load = auto()
    Performance = auto()


class Case(BaseModel):
    """Undifined case

    Fields:
        case_id(CaseType): default 9 case type plus one custom cases.
        label(CaseLabel): performance or load.
        dataset(DataSet): dataset for this case runner.
        filter_rate(float | None): one of 99% | 1% | None
        filters(dict | None): filters for search
    """

    case_id: CaseType
    label: CaseLabel
    name: str
    description: str
    dataset: DatasetManager

    load_timeout: float | int
    optimize_timeout: float | int | None

    filter_rate: float | None

    @property
    def filters(self) -> dict | None:
        if self.filter_rate is not None:
            ID = round(self.filter_rate * self.dataset.data.size)
            return {
                "metadata": f">={ID}",
                "id": ID,
            }

        return None


class CapacityCase(Case, BaseModel):
    label: CaseLabel = CaseLabel.Load
    filter_rate: float | None = None
    load_timeout: float | int = config.CAPACITY_TIMEOUT_IN_SECONDS
    optimize_timeout: float | int | None = None


class PerformanceCase(Case, BaseModel):
    label: CaseLabel = CaseLabel.Performance
    filter_rate: float | None = None


class CapacityDim960(CapacityCase):
    case_id: CaseType = CaseType.CapacityDim960
    dataset: DatasetManager = Dataset.GIST.manager(100_000)
    name: str = "Capacity Test (960 Dim Repeated)"
    description: str = """This case tests the vector database's loading capacity by repeatedly inserting large-dimension vectors (GIST 100K vectors, <b>960 dimensions</b>) until it is fully loaded.
Number of inserted vectors will be reported."""


class CapacityDim128(CapacityCase):
    case_id: CaseType = CaseType.CapacityDim128
    dataset: DatasetManager = Dataset.SIFT.manager(500_000)
    name: str = "Capacity Test (128 Dim Repeated)"
    description: str = """This case tests the vector database's loading capacity by repeatedly inserting small-dimension vectors (SIFT 100K vectors, <b>128 dimensions</b>) until it is fully loaded.
Number of inserted vectors will be reported."""


class Performance10M(PerformanceCase):
    case_id: CaseType = CaseType.Performance10M
    dataset: DatasetManager = Dataset.COHERE.manager(10_000_000)
    name: str = "Search Performance Test (10M Dataset, 768 Dim)"
    description: str = """This case tests the search performance of a vector database with a large dataset (<b>Cohere 10M vectors</b>, 768 dimensions) at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_10M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_10M


class Performance1M(PerformanceCase):
    case_id: CaseType = CaseType.Performance1M
    dataset: DatasetManager = Dataset.COHERE.manager(1_000_000)
    name: str = "Search Performance Test (1M Dataset, 768 Dim)"
    description: str = """This case tests the search performance of a vector database with a medium dataset (<b>Cohere 1M vectors</b>, 768 dimensions) at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_1M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_1M


class Performance10M1P(PerformanceCase):
    case_id: CaseType = CaseType.Performance10M1P
    filter_rate: float | int | None = 0.01
    dataset: DatasetManager = Dataset.COHERE.manager(10_000_000)
    name: str = "Filtering Search Performance Test (10M Dataset, 768 Dim, Filter 1%)"
    description: str = """This case tests the search performance of a vector database with a large dataset (<b>Cohere 10M vectors</b>, 768 dimensions) under a low filtering rate (<b>1% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_10M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_10M


class Performance1M1P(PerformanceCase):
    case_id: CaseType = CaseType.Performance1M1P
    filter_rate: float | int | None = 0.01
    dataset: DatasetManager = Dataset.COHERE.manager(1_000_000)
    name: str = "Filtering Search Performance Test (1M Dataset, 768 Dim, Filter 1%)"
    description: str = """This case tests the search performance of a vector database with a medium dataset (<b>Cohere 1M vectors</b>, 768 dimensions) under a low filtering rate (<b>1% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_1M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_1M


class Performance10M99P(PerformanceCase):
    case_id: CaseType = CaseType.Performance10M99P
    filter_rate: float | int | None = 0.99
    dataset: DatasetManager = Dataset.COHERE.manager(10_000_000)
    name: str = "Filtering Search Performance Test (10M Dataset, 768 Dim, Filter 99%)"
    description: str = """This case tests the search performance of a vector database with a large dataset (<b>Cohere 10M vectors</b>, 768 dimensions) under a high filtering rate (<b>99% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_10M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_10M


class Performance1M99P(PerformanceCase):
    case_id: CaseType = CaseType.Performance1M99P
    filter_rate: float | int | None = 0.99
    dataset: DatasetManager = Dataset.COHERE.manager(1_000_000)
    name: str = "Filtering Search Performance Test (1M Dataset, 768 Dim, Filter 99%)"
    description: str = """This case tests the search performance of a vector database with a medium dataset (<b>Cohere 1M vectors</b>, 768 dimensions) under a high filtering rate (<b>99% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_1M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_1M



class Performance100M(PerformanceCase):
    case_id: CaseType = CaseType.Performance100M
    filter_rate: float | int | None = None
    dataset: DatasetManager = Dataset.LAION.manager(100_000_000)
    name: str = "Search Performance Test (100M Dataset, 768 Dim)"
    description: str = """This case tests the search performance of a vector database with a large 100M dataset (<b>LAION 100M vectors</b>, 768 dimensions), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""
    load_timeout: float | int = config.LOAD_TIMEOUT_100M
    optimize_timeout: float | int | None = config.OPTIMIZE_TIMEOUT_100M


type2case = {
    CaseType.CapacityDim960: CapacityDim960,
    CaseType.CapacityDim128: CapacityDim128,

    CaseType.Performance100M: Performance100M,
    CaseType.Performance10M: Performance10M,
    CaseType.Performance1M: Performance1M,

    CaseType.Performance10M1P: Performance10M1P,
    CaseType.Performance1M1P: Performance1M1P,
    CaseType.Performance10M99P: Performance10M99P,
    CaseType.Performance1M99P: Performance1M99P,
}
