import traceback
import logging
import concurrent
from typing import Any
#  from pydantic import computed_field

from . import dataset as ds
from .clients import api
from ..base import BaseModel
from ..models import CaseType, DBCaseConfig
from ..metric import Metric
from .runner import (
    MultiProcessingInsertRunner,
    MultiProcessingSearchRunner,
    SerialSearchRunner,
)
from . import utils


log = logging.getLogger(__name__)


class Case(BaseModel):
    case_id: CaseType
    dataset: ds.DataSet

    metric: Metric
    filter_rate: float
    filter_size: int

    db: api.VectorDB | None = None
    # tuple of db init cls, db_config and db_case_config
    db_configs: tuple[Any, dict, DBCaseConfig] | None = None

    def init_db(self) -> None:
        self.db = self.db_configs[0](
            db_config=self.db_configs[1],
            db_case_config=self.db_configs[2],
            drop_old=True,
        )

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
        try:
            log.info("start to run load case")
            self.init_db()
            self.dataset.prepare()
            self._load()
            log.info("end run load case")
        except Exception as e:
            log.warning(f"run load case error: {e}")

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
            raise e from None
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
        Metric: metrics except max_load_count,
            including load_duration, build_duration, qps, serial_latency, p99, recall
    """
    metric: Metric = None
    filter_rate: float = 0
    filter_size: int = 0
    search_runner: MultiProcessingSearchRunner | None = None
    serial_search_runner: SerialSearchRunner | None = None

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

    def run(self) -> Metric:
        try:
            self.dataset.prepare()
            self.init_db()
            _, insert_dur = self._insert_train_data()
            build_dur = self._ready_to_search()
            recall, serial_latency, p99 = self.serial_search()
            qps = self.conc_search()
            m = Metric(
                load_duration=round(insert_dur, 4),
                build_duration=round(build_dur, 4),
                recall=recall,
                serial_latency=serial_latency,
                p99=p99,
                qps=qps,
            )
            log.info(f"got results: {m}")
            return m
        except Exception as e:
            log.warning(f"performance case run error: {e}")
            traceback.print_exc()
            raise e


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
                raise e from None
            finally:
                runner.stop()
        return results

    @utils.time_it
    def _task(self) -> None:
        """"""
        with self.db.init():
            self.db.ready_to_search()

    def _ready_to_search(self) -> float:
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self._task)
            try:
                return future.result()[1]
            except Exception as e:
                log.warning(f"VectorDB ready_to_search error: {e}")
                raise e from None


    def serial_search(self) -> tuple[float, float, float]:
        """Performance serial tests, search the entire test data once,
        calculate the recall, serial_latency, and, p99

        Returns:
            tuple[float, float, float]: recall, serial_latency, p99
        """
        gt_df = self.dataset.ground_truth if self.filters is None or self.filters['id'] == self.filter_size \
                else self.dataset.ground_truth_90p

        self.serial_search_runner = SerialSearchRunner(
            db=self.db,
            test_df=self.dataset.test_data,
            ground_truth=gt_df,
            filters=self.filters,
        )

        try:
            return self.serial_search_runner.run()
        except Exception as e:
            log.warning(f"search error: {str(e)}, {e}")
            raise e from None
        finally:
            self.serial_search_runner.stop()

    def conc_search(self):
        """Performance concurrency tests, search the test data endlessness
        for 30s in several concurrencies

        Returns:
            float: the largest qps in all concurrencies
        """

        self.search_runner =  MultiProcessingSearchRunner(
            db=self.db,
            test_df=self.dataset.test_data,
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
        if self.serial_search_runner:
            self.serial_search_runner.stop()


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
