import typing
import logging
from enum import Enum, auto

from . import dataset as ds
from ..base import BaseModel


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
    filter_rate: float | None = None


class PerformanceCase(Case, BaseModel):
    label: CaseLabel = CaseLabel.Performance
    filter_rate: float | None = None


class CapacityDim960(CapacityCase):
    case_id: CaseType = CaseType.CapacityDim960
    dataset: ds.DataSet = ds.get(ds.Name.GIST, ds.Label.SMALL)
    name: str = "Capacity Test (960 Dim Repeated)"
    description: str = """This case tests the vector database's loading capacity by repeatedly inserting large-dimension vectors (GIST 100K vectors, <b>960 dimensions</b>) until it is fully loaded.
Number of inserted vectors will be reported."""


class CapacityDim128(CapacityCase):
    case_id: CaseType = CaseType.CapacityDim128
    dataset: ds.DataSet = ds.get(ds.Name.SIFT, ds.Label.SMALL)
    name: str = "Capacity Test (128 Dim Repeated)"
    description: str = """This case tests the vector database's loading capacity by repeatedly inserting small-dimension vectors (SIFT 100K vectors, <b>128 dimensions</b>) until it is fully loaded.
Number of inserted vectors will be reported."""


class Performance10M(PerformanceCase):
    case_id: CaseType = CaseType.Performance10M
    dataset: ds.DataSet = ds.get(ds.Name.Cohere, ds.Label.LARGE)
    name: str = "Search Performance Test (10M Dataset, 768 Dim)"
    description: str = """This case tests the search performance of a vector database with a large dataset (<b>Cohere 10M vectors</b>, 768 dimensions) at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""


class Performance1M(PerformanceCase):
    case_id: CaseType = CaseType.Performance1M
    dataset: ds.DataSet = ds.get(ds.Name.Cohere, ds.Label.MEDIUM)
    name: str = "Search Performance Test (1M Dataset, 768 Dim)"
    description: str = """This case tests the search performance of a vector database with a medium dataset (<b>Cohere 1M vectors</b>, 768 dimensions) at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""


class Performance10M1P(PerformanceCase):
    case_id: CaseType = CaseType.Performance10M1P
    filter_rate: float | int | None = 0.01
    dataset: ds.DataSet = ds.get(ds.Name.Cohere, ds.Label.LARGE)
    name: str = "Filtering Search Performance Test (10M Dataset, 768 Dim, Filter 1%)"
    description: str = """This case tests the search performance of a vector database with a large dataset (<b>Cohere 10M vectors</b>, 768 dimensions) under a low filtering rate (<b>1% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""


class Performance1M1P(PerformanceCase):
    case_id: CaseType = CaseType.Performance1M1P
    filter_rate: float | int | None = 0.01
    dataset: ds.DataSet = ds.get(ds.Name.Cohere, ds.Label.MEDIUM)
    name: str = "Filtering Search Performance Test (1M Dataset, 768 Dim, Filter 1%)"
    description: str = """This case tests the search performance of a vector database with a medium dataset (<b>Cohere 1M vectors</b>, 768 dimensions) under a low filtering rate (<b>1% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""


class Performance10M99P(PerformanceCase):
    case_id: CaseType = CaseType.Performance10M99P
    filter_rate: float | int | None = 0.99
    dataset: ds.DataSet = ds.get(ds.Name.Cohere, ds.Label.LARGE)
    name: str = "Filtering Search Performance Test (10M Dataset, 768 Dim, Filter 99%)"
    description: str = """This case tests the search performance of a vector database with a large dataset (<b>Cohere 10M vectors</b>, 768 dimensions) under a high filtering rate (<b>99% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""


class Performance1M99P(PerformanceCase):
    case_id: CaseType = CaseType.Performance1M99P
    filter_rate: float | int | None = 0.99
    dataset: ds.DataSet = ds.get(ds.Name.Cohere, ds.Label.MEDIUM)
    name: str = "Filtering Search Performance Test (1M Dataset, 768 Dim, Filter 99%)"
    description: str = """This case tests the search performance of a vector database with a medium dataset (<b>Cohere 1M vectors</b>, 768 dimensions) under a high filtering rate (<b>99% vectors</b>), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""



class Performance100M(PerformanceCase):
    case_id: CaseType = CaseType.Performance100M
    filter_rate: float | int | None = None
    dataset: ds.DataSet = ds.get(ds.Name.LAION, ds.Label.LARGE)
    name: str = "Search Performance Test (100M Dataset, 768 Dim)"
    description: str = """This case tests the search performance of a vector database with a large 100M dataset (<b>LAION 100M vectors</b>, 768 dimensions), at varying parallel levels.
Results will show index building time, recall, and maximum QPS."""


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
