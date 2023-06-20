import typing
import logging
from enum import Enum, auto

from . import dataset as ds
from ..base import BaseModel


log = logging.getLogger(__name__)

Case = typing.TypeVar("Case")


class CaseType(Enum):
    """
    Value will be displayed in UI

    Example:
        >>> c = CaseType.CapacitySDim.get()()
        >>> assert c is not None
        >>> c.name
        "Capacity Test (128 Dim Repeated)"
        >>> c.description
        ""
    """

    CapacitySDim = "Capacity Test (128 Dim Repeated)"
    CapacityLDim = "Capacity Test (960 Dim Repeated)"

    Performance100M = "Search Performance Test (100M Dataset, 768 Dim)"
    PerformanceLZero = "Search Performance Test (10M Dataset, 768 Dim)"
    PerformanceMZero = "Search Performance Test (1M Dataset, 768 Dim)"
    PerformanceSZero = "Search Performance Test (100K Dataset, 768 Dim)"

    PerformanceLLow = (
        "Filtering Search Performance Test (10M Dataset, 768 Dim, Filter 1%)"
    )
    PerformanceMLow = (
        "Filtering Search Performance Test (1M Dataset, 768 Dim, Filter 1%)"
    )
    PerformanceSLow = (
        "Filtering Search Performance Test (100K Dataset, 768 Dim, Filter 1%)"
    )
    PerformanceLHigh = (
        "Filtering Search Performance Test (10M Dataset, 768 Dim, Filter 99%)"
    )
    PerformanceMHigh = (
        "Filtering Search Performance Test (1M Dataset, 768 Dim, Filter 99%)"
    )
    PerformanceSHigh = (
        "Filtering Search Performance Test (100K Dataset, 768 Dim, Filter 99%)"
    )

    def get(self) -> Case:
        return type2case.get(self)


class CaseLabel(Enum):
    Load = auto()
    Performance = auto()


class Case(BaseModel):
    """Undifined case

    Fields:
        case_id(CaseType): default 11 case type plus one custom cases.
        label(CaseLabel): performance or load.
        dataset(DataSet): dataset for this case runner.
        filter_rate(float | None): one of 99% | 1% | None
        filters(dict | None): filters for search
    """

    case_id: CaseType
    label: CaseLabel
    name: str
    description: str
    dataset: ds.DataSet

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
    filter_rate: float | int | None = None


class PerformanceCase(Case, BaseModel):
    label: CaseLabel = CaseLabel.Performance
    filter_rate: float | int | None = None


class CapacityLDimCase(CapacityCase):
    case_id: CaseType = CaseType.CapacityLDim
    dataset: ds.DataSet = ds.get(ds.Name.GIST, ds.Label.SMALL)
    name: str = "Capacity Test (960 Dim Repeated)"
    description: str = """This case tests the vector database's loading capacity by repeatedly inserting large-dimension vectors (GIST 100K vectors, <b>960 dimensions</b>) until it is fully loaded.  
Number of inserted vectors will be reported."""


class CapacitySDimCase(CapacityCase):
    case_id: CaseType = CaseType.CapacitySDim
    dataset: ds.DataSet = ds.get(ds.Name.SIFT, ds.Label.SMALL)
    name: str = "Capacity Test (128 Dim Repeated)"
    description: str = """This case tests the vector database's loading capacity by repeatedly inserting small-dimension vectors (SIFT 100K vectors, <b>128 dimensions</b>) until it is fully loaded.  
Number of inserted vectors will be reported."""


class PerformanceLZero(PerformanceCase):
    case_id: CaseType = CaseType.PerformanceLZero
    dataset: ds.DataSet = ds.get(ds.Name.Cohere, ds.Label.LARGE)
    name: str = "Search Performance Test (10M Dataset, 768 Dim)"
    description: str = """This case tests the search performance of a vector database with a large dataset (<b>Cohere 10M vectors</b>, 768 dimensions) at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""


class PerformanceMZero(PerformanceCase):
    case_id: CaseType = CaseType.PerformanceMZero
    dataset: ds.DataSet = ds.get(ds.Name.Cohere, ds.Label.MEDIUM)
    name: str = "Search Performance Test (1M Dataset, 768 Dim)"
    description: str = """This case tests the search performance of a vector database with a medium dataset (<b>Cohere 1M vectors</b>, 768 dimensions) at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""


class PerformanceSZero(PerformanceCase):
    case_id: CaseType = CaseType.PerformanceSZero
    dataset: ds.DataSet = ds.get(ds.Name.Cohere, ds.Label.SMALL)
    name: str = "Search Performance Test (100K Dataset, 768 Dim)"
    description: str = """This case tests the search performance of a vector database with a small dataset (<b>Cohere 100K vectors</b>, 768 dimensions) at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""


class PerformanceLLow(PerformanceCase):
    case_id: CaseType = CaseType.PerformanceLLow
    filter_rate: float | int | None = 0.01
    dataset: ds.DataSet = ds.get(ds.Name.Cohere, ds.Label.LARGE)
    name: str = "Filtering Search Performance Test (10M Dataset, 768 Dim, Filter 1%)"
    description: str = """This case tests the search performance of a vector database with a large dataset (<b>Cohere 10M vectors</b>, 768 dimensions) under a low filtering rate (<b>1% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""


class PerformanceMLow(PerformanceCase):
    case_id: CaseType = CaseType.PerformanceMLow
    filter_rate: float | int | None = 0.01
    dataset: ds.DataSet = ds.get(ds.Name.Cohere, ds.Label.MEDIUM)
    name: str = "Filtering Search Performance Test (1M Dataset, 768 Dim, Filter 1%)"
    description: str = """This case tests the search performance of a vector database with a medium dataset (<b>Cohere 1M vectors</b>, 768 dimensions) under a low filtering rate (<b>1% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""


class PerformanceSLow(PerformanceCase):
    case_id: CaseType = CaseType.PerformanceSLow
    filter_rate: float | int | None = 0.01
    dataset: ds.DataSet = ds.get(ds.Name.Cohere, ds.Label.SMALL)
    name: str = "Filtering Search Performance Test (100K Dataset, 768 Dim, Filter 1%)"
    description: str = (
        """This case tests the search performance of a vector database with a small dataset (<b>Cohere 100K vectors</b>, 768 dimensions) under a low filtering rate (<b>1% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS.""",
    )


class PerformanceLHigh(PerformanceCase):
    case_id: CaseType = CaseType.PerformanceLHigh
    filter_rate: float | int | None = 0.99
    dataset: ds.DataSet = ds.get(ds.Name.Cohere, ds.Label.LARGE)
    name: str = "Filtering Search Performance Test (10M Dataset, 768 Dim, Filter 99%)"
    description: str = """This case tests the search performance of a vector database with a large dataset (<b>Cohere 10M vectors</b>, 768 dimensions) under a high filtering rate (<b>99% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""


class PerformanceMHigh(PerformanceCase):
    case_id: CaseType = CaseType.PerformanceMHigh
    filter_rate: float | int | None = 0.99
    dataset: ds.DataSet = ds.get(ds.Name.Cohere, ds.Label.MEDIUM)
    name: str = "Filtering Search Performance Test (1M Dataset, 768 Dim, Filter 99%)"
    description: str = """This case tests the search performance of a vector database with a medium dataset (<b>Cohere 1M vectors</b>, 768 dimensions) under a high filtering rate (<b>99% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""


class PerformanceSHigh(PerformanceCase):
    case_id: CaseType = CaseType.PerformanceSHigh
    filter_rate: float | int | None = 0.99
    dataset: ds.DataSet = ds.get(ds.Name.Cohere, ds.Label.SMALL)
    name: str = "Filtering Search Performance Test (100K Dataset, 768 Dim, Filter 99%)"
    description: str = """This case tests the search performance of a vector database with a small dataset (<b>Cohere 100K vectors</b>, 768 dimensions) under a high filtering rate (<b>99% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""


class Performance100M(PerformanceCase):
    case_id: CaseType = CaseType.Performance100M
    filter_rate: float | int | None = None
    dataset: ds.DataSet = ds.get(ds.Name.LAION, ds.Label.LARGE)
    name: str = "Search Performance Test (100M Dataset, 768 Dim)"
    description: str = """This case tests the search performance of a vector database with a large 100M dataset (<b>LAION 100M vectors</b>, 768 dimensions), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""


type2case = {
    CaseType.CapacityLDim: CapacityLDimCase,
    CaseType.CapacitySDim: CapacitySDimCase,

    CaseType.Performance100M: Performance100M,
    CaseType.PerformanceLZero: PerformanceLZero,
    CaseType.PerformanceMZero: PerformanceMZero,
    CaseType.PerformanceSZero: PerformanceSZero,

    CaseType.PerformanceLLow: PerformanceLLow,
    CaseType.PerformanceMLow: PerformanceMLow,
    CaseType.PerformanceSLow: PerformanceSLow,
    CaseType.PerformanceLHigh: PerformanceLHigh,
    CaseType.PerformanceMHigh: PerformanceMHigh,
    CaseType.PerformanceSHigh: PerformanceSHigh,
}
