from vectordb_bench.backend.clients import DB
from vectordb_bench.models import CaseConfig, CaseConfigParamType, TaskConfig


def generate_tasks(activedDbList: list[DB], dbConfigs, activedCaseList: list[CaseConfig], allCaseConfigs):
    tasks = []
    for db in activedDbList:
        for case in activedCaseList:
            cfg = {key.value: value for key, value in allCaseConfigs[db][case].items()}
            # Many DBCaseConfig models require an `index` field, while the UI stores the selection under `IndexType`.
            # Passing both keeps backwards-compatibility (extra fields are ignored) and enables strict models (e.g. OceanBase).
            if CaseConfigParamType.IndexType in allCaseConfigs[db][case] and "index" not in cfg:
                cfg["index"] = allCaseConfigs[db][case][CaseConfigParamType.IndexType]
            task = TaskConfig(
                db=db.value,
                db_config=dbConfigs[db],
                case_config=case,
                db_case_config=db.case_config_cls(allCaseConfigs[db][case].get(CaseConfigParamType.IndexType, None))(
                    **cfg
                ),
            )
            tasks.append(task)

    return tasks
