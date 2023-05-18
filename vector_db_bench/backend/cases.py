import logging
from pydantic import BaseModel, ConfigDict, computed_field
from .clients import api
from . import dataset as ds
from ..models import CaseType
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
    dataset: ds.DataSet

    metric: Metric
    filter_rate: float
    filter_size: int

    db: api.VectorDB | None = None

    def run(self):
        pass

    def stop(self):
        pass


class LoadCase(Case, BaseModel):
    metric: Metric = None
    filter_rate: float = 0.
    filter_size: int = 0

    def run(self) -> int:
        """
        Returns
            int: the max load count
        """
        log.info("start to run load case")
        self._prep()
        self._load()
        log.info("end run load case")

    def _prep(self):
        self.dataset.prepare()
        self.db.init()

    def _load(self):
        """Insert train data and get the insert_duration"""
        # datasets for load tests are quite small, can fit into memory
        data_dfs = [data_df for data_df in self.dataset]
        assert len(data_dfs) == 1

        runner = MultiProcessingInsertRunner(self.db, data_dfs[0])
        try:
            count = runner.run_sequentially_endlessness()
            log.info(f"load reach limit: insertion counts={count}")
            return count
        except Exception as e:
            log.warning(f"run load case error: {e}")
        finally:
            runner.stop()


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
    metric: Metric = None # TODO
    filter_rate: float = 0
    filter_size: int = 0
    search_runner: MultiProcessingSearchRunner | None = None

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
        return None

    def run(self):
        # TODO try catch
        self.dataset.prepare()
        result, insert_dur = self._insert_train_data()
        m = self.search()
        log.info(f"got results: {m}")
        m.load_duration = insert_dur
        return m

    @utils.time_it
    def _insert_train_data(self):
        """Insert train data and get the insert_duration"""
        results = []
        for data in self.dataset:
            runner = MultiProcessingInsertRunner(self.db, data)
            try:
                res = runner.run()
                results.append(res)
            except Exception as e:
                log.warning(f"insert train data error: {e}")
            finally:
                runner.stop()
        return results

    def search(self):
        """ performance tests """
        gt_df = self.dataset.ground_truth if self.filters is None or self.filters['id'] == self.filter_size \
                else self.dataset.ground_truth_90p

        self.search_runner =  MultiProcessingSearchRunner(
            db=self.db,
            test_df=self.dataset.test_data,
            ground_truth=gt_df,
            filters=self.filters,
        )
        try:
            return self.search_runner.run()
        except Exception as e:
            log.warning(f"search error: {str(e)}, {e}")
            raise e from None
        finally:
            self.search_runner.stop()

    def stop(self):
        if self.search_runner:
            self.search_runner.stop()


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
