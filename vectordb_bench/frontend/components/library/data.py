from dataclasses import asdict
from vectordb_bench.backend.cases import CaseType
from vectordb_bench.backend.clients.api import TestType
from vectordb_bench.interface import benchMarkRunner
from vectordb_bench.models import CaseResult


def getChartsData():
    allResults = benchMarkRunner.get_results()
    libraryCaseResults: list[CaseResult] = []
    exampleCaseResults: list[CaseResult] = []

    for res in allResults:
        results = res.results
        resIsExample = res.task_label == "example"
        for result in results:
            if result.task_config.case_config.case_id not in [CaseType.Custom]:
                test_type = result.task_config.db_config.test_type
                if test_type == TestType.LIBRARY:
                    if resIsExample:
                        exampleCaseResults.append(result)
                    else:
                        libraryCaseResults.append(result)
    isExample = len(libraryCaseResults) == 0
    data = exampleCaseResults if isExample else libraryCaseResults

    chartsData, labels = formatData(data)
    return chartsData, labels, isExample


def formatData(caseResults: list[CaseResult]):
    data = []
    labels = set(["db", "load_duration", "dbLabel"])
    for caseResult in caseResults:
        db = caseResult.task_config.db.value
        dbLabel = caseResult.task_config.db_config.db_label
        build_configs = caseResult.task_config.db_config.config_json
        search_configs = caseResult.task_config.db_case_config.search_param()
        labels = labels | build_configs.keys() | search_configs.keys()
        dbName = caseResult.task_config.db_name
        case_config = caseResult.task_config.case_config
        case = case_config.case_id.case_cls()
        caseName = case.name
        metrics = caseResult.metrics
        data.append(
            {
                "db": db,
                "dbLabel": dbLabel,
                "dbName": dbName,
                "case": caseName,
                **build_configs,
                **search_configs,
                **asdict(metrics),
            }
        )
    return data, list(labels)
