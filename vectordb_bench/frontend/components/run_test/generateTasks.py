from vectordb_bench.backend.clients import DB
from vectordb_bench.models import CaseConfig, CaseConfigParamType, TaskConfig
import logging

log = logging.getLogger(__name__)

def generate_tasks(activedDbList: list[DB], dbConfigs, activedCaseList: list[CaseConfig], allCaseConfigs):
    tasks = []
    for db in activedDbList:
        for case in activedCaseList:
            # 记录参数传递过程
            case_params = allCaseConfigs[db][case]
            log.info(f"Generating task for DB: {db}, case: {case}")
            log.info(f"Case parameters: {case_params}")
            
            # 特别记录 ef_search 参数
            if CaseConfigParamType.ef_search in case_params:
                log.info(f"ef_search parameter: {case_params[CaseConfigParamType.ef_search]}")
            
            task = TaskConfig(
                db=db.value,
                db_config=dbConfigs[db],
                case_config=case,
                db_case_config=db.case_config_cls(allCaseConfigs[db][case].get(CaseConfigParamType.IndexType, None))(
                    **{key.value: value for key, value in allCaseConfigs[db][case].items()}
                ),
            )
            tasks.append(task)

    return tasks
