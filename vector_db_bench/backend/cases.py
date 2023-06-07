import logging
from enum import Enum, auto

from . import dataset as ds
from ..base import BaseModel
from ..models import CaseType


log = logging.getLogger(__name__)


class CaseLabel(Enum):
    Load = auto()
    Performance = auto()


class Case(BaseModel):
    """ Undifined case

    Fields:
        case_id(CaseType): default 11 case type plus one custom cases.
        label(CaseLabel): performance or load.
        dataset(DataSet): dataset for this case runner.
        filter_rate(float | int | None): one of 100 | 90% | 99% | 1% | None
        filters(dict | None): filters for search
    """

    case_id: CaseType
    label: CaseLabel
    dataset: ds.DataSet

    filter_rate: float | int | None

    @property
    def filters(self) -> dict | None:
        if self.filter_rate is not None:
            if isinstance(self.filter_rate, int) and self.filter_rate == 100:
                return {
                    "metadata": f">={self.filter_rate}",
                    "id": self.filter_rate,
                }
            else:
                ID = round(self.filter_rate * self.dataset.data.size)
                return {
                    "metadata": f">={ID}",
                    "id": ID,
                }

        return None


class LoadCase(Case, BaseModel):
    label: CaseLabel = CaseLabel.Load
    filter_rate: float | int | None = None

class PerformanceCase(Case, BaseModel):
    label: CaseLabel = CaseLabel.Performance
    filter_rate: float | int | None = None

class LoadLDimCase(LoadCase):
    case_id: CaseType = CaseType.LoadLDim
    dataset: ds.DataSet = ds.get(ds.Name.GIST, ds.Label.SMALL)

class LoadSDimCase(LoadCase):
    case_id: CaseType = CaseType.LoadSDim
    dataset: ds.DataSet = ds.get(ds.Name.SIFT, ds.Label.SMALL)

class PerformanceLZero(PerformanceCase):
    case_id: CaseType = CaseType.PerformanceLZero
    dataset: ds.DataSet = ds.get(ds.Name.Cohere, ds.Label.LARGE)

class PerformanceMZero(PerformanceCase):
    case_id: CaseType = CaseType.PerformanceMZero
    dataset: ds.DataSet = ds.get(ds.Name.Cohere, ds.Label.MEDIUM)

class PerformanceSZero(PerformanceCase):
    case_id: CaseType = CaseType.PerformanceSZero
    dataset: ds.DataSet = ds.get(ds.Name.Cohere, ds.Label.SMALL)

class PerformanceLLow(PerformanceCase):
    case_id: CaseType = CaseType.PerformanceLLow
    filter_rate: float | int | None = 100
    dataset: ds.DataSet = ds.get(ds.Name.Cohere, ds.Label.LARGE)

class PerformanceMLow(PerformanceCase):
    case_id: CaseType = CaseType.PerformanceMLow
    filter_rate: float | int | None = 100
    dataset: ds.DataSet = ds.get(ds.Name.Cohere, ds.Label.MEDIUM)

class PerformanceSLow(PerformanceCase):
    case_id: CaseType = CaseType.PerformanceSLow
    filter_rate: float | int | None = 100
    dataset: ds.DataSet = ds.get(ds.Name.Cohere, ds.Label.SMALL)

class PerformanceLHigh(PerformanceCase):
    case_id: CaseType = CaseType.PerformanceLHigh
    filter_rate: float | int | None = 0.9
    dataset: ds.DataSet = ds.get(ds.Name.Cohere, ds.Label.LARGE)

class PerformanceMHigh(PerformanceCase):
    case_id: CaseType = CaseType.PerformanceMHigh
    filter_rate: float | int | None = 0.9
    dataset: ds.DataSet = ds.get(ds.Name.Cohere, ds.Label.MEDIUM)

class PerformanceSHigh(PerformanceCase):
    case_id: CaseType = CaseType.PerformanceSLow
    filter_rate: float | int | None = 0.9
    dataset: ds.DataSet = ds.get(ds.Name.Cohere, ds.Label.SMALL)

type2case = {
    CaseType.LoadLDim: LoadLDimCase,
    CaseType.LoadSDim: LoadSDimCase,

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
