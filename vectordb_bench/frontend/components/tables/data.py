from dataclasses import asdict

import pandas as pd

from vectordb_bench.frontend.components.check_results.data import getCaseResultName
from vectordb_bench.interface import benchmark_runner
from vectordb_bench.models import CaseResult, ResultLabel


def getNewResults():
    allResults = benchmark_runner.get_results()
    newResults: list[CaseResult] = []

    for res in allResults:
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
        case = case_config.case
        filter_rate = case.filter_rate
        dataset = case.dataset.data.name
        metrics = asdict(caseResult.metrics)
        data.append(
            {
                "db": db,
                "db_label": db_label,
                "case_name": getCaseResultName(caseResult),
                "k": case_config.k,
                "dataset": dataset,
                "filter_rate": filter_rate,
                **metrics,
            }
        )
    return data
