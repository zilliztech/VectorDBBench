from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.cases import CaseLabel
from vectordb_bench.models import CaseConfig, CaseConfigParamType, TaskConfig


def generate_tasks(activedDbList: list[DB], dbConfigs, activedCaseList: list[CaseConfig], allCaseConfigs):
    tasks = []
    for db in activedDbList:
        for case in activedCaseList:
            # Get the index type for this case
            index_type = allCaseConfigs[db][case].get(CaseConfigParamType.IndexType, None)

            # Special handling for FTS cases
            if case.case.label == CaseLabel.FullTextSearchPerformance:
                from vectordb_bench.backend.clients.api import IndexType

                index_type = IndexType.FTS_AUTOINDEX
            elif index_type is None:
                # Default to AUTOINDEX for cases without specific index type
                from vectordb_bench.backend.clients.api import IndexType

                index_type = IndexType.AUTOINDEX

            # Create the database case config
            cfg = {key.value: value for key, value in allCaseConfigs[db][case].items()}
            # Many DBCaseConfig models require an `index` field, while the UI stores the selection under `IndexType`.
            # Passing both keeps backwards-compatibility (extra fields are ignored) and enables strict models (e.g. OceanBase).
            if CaseConfigParamType.IndexType in allCaseConfigs[db][case] and "index" not in cfg:
                cfg["index"] = allCaseConfigs[db][case][CaseConfigParamType.IndexType]

            db_case_config = db.case_config_cls(index_type)(**cfg)

            task = TaskConfig(
                db=db.value,
                db_config=dbConfigs[db],
                case_config=case,
                db_case_config=db_case_config,
            )
            tasks.append(task)

    return tasks
