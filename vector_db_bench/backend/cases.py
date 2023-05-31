import traceback
import logging
import concurrent
from typing import Any
import numpy as np
from enum import Enum, auto

from . import dataset as ds
from .clients import api
from .clients.db_case_config import MetricType
from .clients.milvus import Milvus
from .clients.zilliz_cloud import ZillizCloud
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


class CaseLabel(Enum):
    Load = auto()
    Performance = auto()


class Case(BaseModel):
    case_id: CaseType
    label: CaseLabel
    dataset: ds.DataSet

    metric: Metric
    filter_rate: float
    filter_size: int

    db: api.VectorDB | None = None
    # tuple of db init cls, db_config and db_case_config
    db_configs: tuple[Any, dict, DBCaseConfig] | None = None

    def init_db(self, drop_old: bool = True) -> None:
        self.db = self.db_configs[0](
            dim=self.dataset.data.dim,
            db_config=self.db_configs[1],
            db_case_config=self.db_configs[2],
            drop_old=drop_old,
        )

    @property
    def db_cls(self) -> Any:
        return self.db_configs[0]

    def run(self):
        pass

    def stop(self):
        pass


class LoadCase(Case, BaseModel):
    label: CaseLabel = CaseLabel.Load
    metric: Metric = None
    filter_rate: float = 0.
    filter_size: int = 0

    def run(self, drop_old: bool=False) -> int:
        """
        Args:
            drop_old(bool): no effects, always drops old

        Returns
            int: the max load count
        """
        try:
            log.info("start to run load case")
            self.init_db()
            self.dataset.prepare()
            return self._load()
            log.info("end run load case")
        except Exception as e:
            log.warning(f"run load case error: {e}")

    def _load(self):
        """Insert train data and get the insert_duration"""
        # datasets for load tests are quite small, can fit into memory
        # only 1 file
        data_df = [data_df for data_df in self.dataset][0]

        all_embeddings, all_metadata = np.stack(data_df["emb"]).tolist(), data_df['id'].tolist()
        runner = MultiProcessingInsertRunner(self.db, all_embeddings, all_metadata)
        try:
            count = runner.run_sequentially_endlessness()
            log.info(f"load reach limit: insertion counts={count}")
            return Metric(max_load_count=count)
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
    label: CaseLabel = CaseLabel.Performance
    metric: Metric = None
    filter_rate: float = 0
    filter_size: int = 0
    search_runner: MultiProcessingSearchRunner | None = None
    serial_search_runner: SerialSearchRunner | None = None
    test_emb: np.ndarray | None = None


    @property
    def normalize(self) -> bool:
        assert self.db
        return isinstance(self.db, (Milvus, ZillizCloud)) and \
            self.dataset.data.metric_type == MetricType.COSINE

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

    def run(self, drop_old: True) -> Metric:
        try:
            self.dataset.prepare()
            self.init_db(drop_old)
            self.calc_test_emb()

            m = Metric()
            if drop_old:
                _, insert_dur = self._insert_train_data()
                build_dur = self._ready_to_search()

                m.load_duration = round(insert_dur, 4)
                m.build_duration = round(build_dur, 4)

            m.recall, m.serial_latency, m.p99 = self.serial_search()
            m.qps = self.conc_search()

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

        for data_df in self.dataset:
            try:
                all_metadata = data_df['id'].tolist()
                emb_np = np.stack(data_df['emb'])

                if self.normalize:
                    log.debug("normalize the 100k train data")
                    all_embeddings = emb_np / np.linalg.norm(emb_np, axis=1)[:, np.newaxis].tolist()
                else:
                    all_embeddings = emb_np.tolist()

                del(emb_np)

                log.debug(f"normalized size: {len(all_embeddings)}, {len(all_metadata)}")

                runner = MultiProcessingInsertRunner(self.db, all_embeddings, all_metadata)
                results.append(runner.run())
            except Exception as e:
                raise e from None
            finally:
                if runner:
                    runner.stop()
                    runner = None
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

    def calc_test_emb(self):
        test_emb = np.stack(self.dataset.test_data["emb"])
        if self.normalize:
            test_emb = test_emb / np.linalg.norm(test_emb, axis=1)[:, np.newaxis]
        self.test_emb = test_emb


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
            test_data=self.test_emb.tolist(),
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
            test_data=self.test_emb,
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
