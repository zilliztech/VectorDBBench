from dataclasses import asdict
from vectordb_bench.backend.cases import CaseType
from vectordb_bench.interface import benchMarkRunner
from vectordb_bench.models import CaseResult, ResultLabel
import pandas as pd


def getNewResults():
    allResults = benchMarkRunner.get_results()
    newResults: list[CaseResult] = []

    for res in allResults:
        # if res.task_label not in ['standard', 'example']:
        if 'g5' in res.task_label:
            results = res.results
            for result in results:
                if result.label == ResultLabel.NORMAL:
                    newResults.append(result)
                    

    df = pd.DataFrame(formatData(newResults))
    return df


def formatData(caseResults: list[CaseResult]):
    data = []
    for caseResult in caseResults:
        db = caseResult.task_config.db.value
        db_label = caseResult.task_config.db_config.db_label
        case_config = caseResult.task_config.case_config
        db_case_config = caseResult.task_config.db_case_config
        itopk_size = db_case_config.itopk_size
        search_width = db_case_config.search_width
        max_iterations = db_case_config.max_iterations
        case = case_config.case_id.case_cls()
        filter_rate = case.filter_rate
        dataset = case.dataset.data.name
        metrics = asdict(caseResult.metrics)
        data.append(
            {
                "db": db,
                "db_label": db_label,
                "case_name": case.name,
                "dataset": dataset,
                "filter_rate": filter_rate,
                "itopk_size": itopk_size,
                "search_width": search_width,
                "max_iterations": max_iterations,
                **metrics,
            }
        )
    return data