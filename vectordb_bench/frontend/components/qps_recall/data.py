from collections import defaultdict
from dataclasses import asdict
from vectordb_bench.backend.filter import FilterOp
from vectordb_bench.frontend.components.check_results.data import getFilterTasks
from vectordb_bench.frontend.components.check_results.filters import getShowDbsAndCases, getshownResults
from vectordb_bench.models import CaseResult, ResultLabel, TestResult


def getshownData(st, results: list[TestResult], filter_type: FilterOp = FilterOp.NonFilter, **kwargs):
    # hide the nav
    st.markdown(
        "<style> div[data-testid='stSidebarNav'] {display: none;} </style>",
        unsafe_allow_html=True,
    )
    st.header("Filters")
    shownResults = getshownResults(st, results, **kwargs)
    showDBNames, showCaseNames = getShowDbsAndCases(st, shownResults, filter_type)
    shownData, failedTasks = getChartData(shownResults, showDBNames, showCaseNames)
    return shownData, failedTasks, showCaseNames


def getChartData(
    tasks: list[CaseResult],
    dbNames: list[str],
    caseNames: list[str],
):
    filterTasks = getFilterTasks(tasks, dbNames, caseNames)
    failedTasks = defaultdict(lambda: defaultdict(str))
    nonemergedTasks = []
    for task in filterTasks:
        db_name = task.task_config.db_name
        db = task.task_config.db.value
        db_label = task.task_config.db_config.db_label or ""
        version = task.task_config.db_config.version or ""
        case = task.task_config.case_config.case
        case_name = case.name
        dataset_name = case.dataset.data.full_name
        filter_rate = case.filter_rate
        metrics = asdict(task.metrics)
        label = task.label
        if label == ResultLabel.NORMAL:
            nonemergedTasks.append(
                {
                    "db_name": db_name,
                    "db": db,
                    "db_label": db_label,
                    "dataset_name": dataset_name,
                    "filter_rate": filter_rate,
                    "version": version,
                    "case_name": case_name,
                    "metricsSet": set(metrics.keys()),
                    **metrics,
                }
            )
        else:
            failedTasks[case_name][db_name] = label

    return nonemergedTasks, failedTasks
