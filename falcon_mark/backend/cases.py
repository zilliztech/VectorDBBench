from pydantic import BaseModel
from .clients import api
from . import dataset as ds
from ..models import CaseResult, CaseType
from ..metric import Metric, LoadMetric, PerformanceMetric


class Case(BaseModel):
    case_id: CaseType
    run_id: int
    metric: Metric = None # TODO
    filter_rate: float 
    filter_size: int

    db_client: api.Client = None

    class Config:
        """configs for pydantic"""
        arbitrary_types_allowed = True

    def run(self, run_id: int) -> CaseResult:
        pass

    def stop(self):
        pass

class LoadCase(Case):
    #  metric: Metric = LoadMetric()
    metric: Metric = None
    filter_rate: float = 0
    filter_size: int = 0


class PerformanceCase(Case):
    #  metric: Metric = PerformanceMetric()
    metric: Metric = None

class LoadLDimCase(LoadCase, ds.GIST_S):

    def __init__(self):
        self.case_id: CaseType = CaseType.LoadLDim

class LoadSDimCase(LoadCase, ds.SIFT_S):
    case_id = CaseType.LoadSDim

class PerformanceLZero(PerformanceCase, ds.Cohere_L):
    case_id = CaseType.PerformanceLZero

class PerformanceMZero(PerformanceCase, ds.Cohere_M):
    case_id = CaseType.PerformanceMZero

class PerformanceSZero(PerformanceCase, ds.Cohere_S):
    case_id = CaseType.PerformanceSZero


class PerformanceLLow(PerformanceCase, ds.Cohere_L):

    case_id = CaseType.PerformanceLLow
    filter_size: int = 100

class PerformanceMLow(PerformanceCase, ds.Cohere_M):

    case_id = CaseType.PerformanceMLow
    filter_size: int = 100

class PerformanceSLow(PerformanceCase, ds.Cohere_S):
    case_id: CaseType = CaseType.PerformanceSLow
    filter_size: int = 100


class PerformanceLHigh(PerformanceCase, ds.Cohere_L):
    case_id: CaseType = CaseType.PerformanceLHigh
    filter_rate: float = 0.9

class PerformanceMHigh(PerformanceCase, ds.Cohere_M):
    case_id: CaseType = CaseType.PerformanceMHigh
    filter_rate: float = 0.9


class PerformanceSHigh(PerformanceCase, ds.Cohere_S):
    case_id: CaseType = CaseType.PerformanceSLow
    filter_rate: float = 0.9

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
