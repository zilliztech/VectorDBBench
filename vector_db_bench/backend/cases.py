import logging
from typing import Any
from pydantic import BaseModel, ConfigDict, computed_field
from .clients import api
from . import dataset as ds
from ..models import CaseResult, CaseType
from ..metric import Metric
from .runner import (
    MultiProcessingInsertRunner,
    MultiProcessingSearchRunner,
)
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

    db: api.VectorDB

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

    def run(self) -> CaseResult:
        log.debug("start run")
        self.prep()
        self.load()
        log.debug("stop run")

    def prep(self):
        self.dataset.prepare()
        self.db.init()

    @utils.Timer(name="insert_train", logger=log.info)
    def load(self):
        """Insert train data and get the insert_duration"""
        # datasets for load tests are quite small, can fit into memory
        data_dfs = [data_df for data_df in self.dataset]
        assert len(data_dfs) == 1

        runner = MultiProcessingInsertRunner(self.db, data_dfs[0])
        dur, count = runner.run_sequentially_endlessness()
        runner.clean()

        log.info("load reach limit: dur={dur}, insertion counts={count}")

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
    filter_rate: float = 0
    filter_size: int = 0

    @computed_field
    @property
    def filters(self) -> dict | None:
        if abs(self.filter_rate - 0) > 1e-6:
            ID = round(self.filter_rate * self.dataset.data.size)
            return {
                "metadata": f">={ID}",
                "id": ID,
            }

        if self.filter_size > 0:
            return {
                "metadata": f">={self.filter_size}",
                "id": self.filter_size,
            }

    def run(self) -> CaseResult:
        log.debug("start run")
        self.dataset.prepare()
        self._insert_train_data()
        self.search()
        log.debug("stop run")

    @utils.Timer(name="insert_train", logger=log.info)
    def _insert_train_data(self):
        """Insert train data and get the insert_duration"""
        results = []
        for data in self.dataset:
            runner = MultiProcessingInsertRunner(self.db, data)
            res = runner.run()
            results.append(res)
            runner.clean()
        return results

    #  @utils.Timer(name="ready_elapse", logger=log.info)
    #  def load_dataset_into_db(self):
    #      self._insert_train_data()

    @utils.Timer(name="search", logger=log.info)
    def search(self):
        runner = MultiProcessingSearchRunner(
            db=self.db,
            test_df=self.dataset.test_data,
            ground_truth=self.dataset.ground_truth,
            filters=self.filters,
        )

        runner.run()
        runner.clean()


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
