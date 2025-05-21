from vectordb_bench.backend.clients import DB
from vectordb_bench.models import CaseConfig, CaseConfigParamType, TaskConfig


def generate_tasks(activedDbList: list[DB], dbConfigs, activedCaseList: list[CaseConfig], allCaseConfigs):
    tasks = []
    for db in activedDbList:
        for case in activedCaseList:
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
