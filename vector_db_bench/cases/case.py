from .clients import api
from .dataset import DataSet
from .results import CaseResult

class Case:
    case_id: int # TODO enum
    run_id: int
    data_set: DataSet

    filter_rate: Any = None # TODO

    def run(self, db_client: api.Client, run_id: int) -> CaseResult:
        pass

    def stop(self):
        pass
