import logging
from typing import Any, Type
from pydantic import BaseModel, ConfigDict
from .clients import api
from . import dataset as ds
from ..models import CaseResult, CaseType
from ..metric import Metric
from .runner import MultiProcessingInsertRunner, MultiProcessingSearchRunner
from . import utils


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
    """ DataSet, filter_rate/filter_size, db_class with db config

    Static params:
        k = 100
        concurrency = [1, 5, 10, 15, 20, 25, 30, 35]
        run_dur(k, concurrency) = 30s

    Dynamic params:
        dataset = GIST | Glove | Cohere | SIFT
        filter_rate/filter_size = 0 | 100 | 90%

        db_class = Type[api.VectorDB]
        case_config = CaseConfig

    Result metrics:
        QPS
        Recall
        serial_latency
        ready_elapse # TODO rename
    """
    #  metric: Metric = PerformanceMetric()
    metric: Metric = None # TODO
    filter_rate: float = 0 # TODO
    filter_size: int = 0 # TODO

    def run(self) -> CaseResult:
        log.debug("start run")
        self.dataset.prepare()
        self.load_dataset_into_db()
        self.search()
        log.debug("stop run")

    @utils.Timer(name="insert_train", logger=log.info)
    def _insert_train_data(self):
        """Insert train data and get the insert_duration"""
        # TODO reduce the iterated data columns in dataset
        results = []
        for data in self.dataset:
            runner = MultiProcessingInsertRunner(self.db_class, data)
            res = runner.run()
            results.append(res)
            #  res = runner.run_sequentially()
        return results

    @utils.Timer(name="ready_elapse", logger=log.info)
    def load_dataset_into_db(self):
        self._insert_train_data()
        self.db_class().ready_to_search()

    @utils.Timer(name="search", logger=log.info)
    def search(self):
        runner = MultiProcessingSearchRunner(
            db_class=self.db_class,
            test_df=self.dataset.test_data,
            ground_truth=self.dataset.ground_truth,
        )

        runner.run()


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
