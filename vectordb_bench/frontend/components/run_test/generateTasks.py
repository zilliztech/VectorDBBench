from vectordb_bench.models import CaseConfig, CaseConfigParamType, TaskConfig


def generate_tasks(activedDbList, dbConfigs, activedCaseList, allCaseConfigs):
    tasks = [
        TaskConfig(
            db=db.value,
            db_config=dbConfigs[db],
            case_config=CaseConfig(
                case_id=case.value,
                custom_case={},
            ),
            db_case_config=db.init_cls.case_config_cls(
                allCaseConfigs[db][case].get(CaseConfigParamType.IndexType, None)
            )(**{key.value: value for key, value in allCaseConfigs[db][case].items()}),
        )
        for case in activedCaseList
        for db in activedDbList
    ]
    
    return tasks
