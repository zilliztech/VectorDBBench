from .clients import api
from .dataset import DataSet
from .results import CaseResult

class Case:
    case_id: int # TODO enum
    run_id: int
    data_set: DataSet
    db_client: api.Client

    filter_rate: Any = None # TODO

    def run(self, run_id: int) -> CaseResult:
        pass

    def stop(self):
        pass
