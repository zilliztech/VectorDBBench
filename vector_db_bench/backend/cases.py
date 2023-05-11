import logging
from typing import Any, Type
from pydantic import BaseModel, ConfigDict
from .clients import api
from . import dataset as ds
from ..models import CaseResult, CaseType
from ..metric import Metric
from .runner import MultiProcessingInsertRunner


log = logging.getLogger(__name__)


class Case(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    case_id: CaseType
    run_id: int
    dataset: ds.DataSet

    metric: Metric
    filter_rate: float
    filter_size: int
    runner: Any = None

    db_class: Type[api.VectorDB]

    def prepare(self):
        """Prepare runner, dataset, and db"""
        pass

    def run(self, run_id: int) -> CaseResult:
        pass

    def stop(self):
        pass


class LoadCase(Case, BaseModel):
    #  metric: Metric = LoadMetric()
    metric: Metric = None
    filter_rate: float = 0
    filter_size: int = 0


class PerformanceCase(Case, BaseModel):
    #  metric: Metric = PerformanceMetric()
    metric: Metric = None # TODO
    filter_rate: float = 0
    filter_size: int = 0

    def run(self) -> CaseResult:
        log.debug("start run")

        log.debug("stop run")

    def _insert_train_data(self):
        # TODO reduce the iterated data columns in dataset
        results = []
        for data in self.dataset:
            runner = MultiProcessingInsertRunner(self.db_class, data)
            res = runner.run()
            results.append(res)
            #  res = runner.run_sequentially()
        return results

    def prepare_train_data_in_db(self):
        self.dataset.prepare()

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
    dataset: ds.DataSet = ds.get(ds.Name.Cohere, ds.Label.LARGE)

class PerformanceSZero(PerformanceCase):
    case_id: CaseType = CaseType.PerformanceSZero
    dataset: ds.DataSet = ds.get(ds.Name.Cohere, ds.Label.SMALL)

class PerformanceLLow(PerformanceCase):
    case_id: CaseType = CaseType.PerformanceLLow
    filter_size: int = 100
    dataset: ds.DataSet = ds.get(ds.Name.Cohere, ds.Label.LARGE)

class PerformanceMLow(PerformanceCase):
    case_id: CaseType = CaseType.PerformanceMLow
    filter_size: int = 100
    dataset: ds.DataSet = ds.get(ds.Name.Cohere, ds.Label.MEDIUM)

class PerformanceSLow(PerformanceCase):
    case_id: CaseType = CaseType.PerformanceSLow
    filter_size: int = 100
    dataset: ds.DataSet = ds.get(ds.Name.Cohere, ds.Label.SMALL)

class PerformanceLHigh(PerformanceCase):
    case_id: CaseType = CaseType.PerformanceLHigh
    filter_rate: float = 0.9
    dataset: ds.DataSet = ds.get(ds.Name.Cohere, ds.Label.LARGE)

class PerformanceMHigh(PerformanceCase):
    case_id: CaseType = CaseType.PerformanceMHigh
    filter_rate: float = 0.9
    dataset: ds.DataSet = ds.get(ds.Name.Cohere, ds.Label.MEDIUM)

class PerformanceSHigh(PerformanceCase):
    case_id: CaseType = CaseType.PerformanceSLow
    filter_rate: float = 0.9
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
