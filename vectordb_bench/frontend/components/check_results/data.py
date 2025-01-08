from collections import defaultdict
from dataclasses import asdict
from vectordb_bench.metric import isLowerIsBetterMetric
from vectordb_bench.models import CaseResult, ResultLabel


def getChartData(
    tasks: list[CaseResult],
    dbNames: list[str],
    caseNames: list[str],
):
    filterTasks = getFilterTasks(tasks, dbNames, caseNames)
    mergedTasks, failedTasks = mergeTasks(filterTasks)
    return mergedTasks, failedTasks


def getFilterTasks(
    tasks: list[CaseResult],
    dbNames: list[str],
    caseNames: list[str],
) -> list[CaseResult]:
    filterTasks = [
        task
        for task in tasks
        if task.task_config.db_name in dbNames
        and task.task_config.case_config.case_id.case_cls(task.task_config.case_config.custom_case).name in caseNames
    ]
    return filterTasks


def mergeTasks(tasks: list[CaseResult]):
    dbCaseMetricsMap = defaultdict(lambda: defaultdict(dict))
    for task in tasks:
        db_name = task.task_config.db_name
        db = task.task_config.db.value
        db_label = task.task_config.db_config.db_label or ""
        version = task.task_config.db_config.version or ""
        case = task.task_config.case_config.case_id.case_cls(task.task_config.case_config.custom_case)
        dbCaseMetricsMap[db_name][case.name] = {
            "db": db,
            "db_label": db_label,
            "version": version,
            "metrics": mergeMetrics(
                dbCaseMetricsMap[db_name][case.name].get("metrics", {}),
                asdict(task.metrics),
            ),
            "label": getBetterLabel(
                dbCaseMetricsMap[db_name][case.name].get("label", ResultLabel.FAILED),
                task.label,
            ),
        }

    mergedTasks = []
    failedTasks = defaultdict(lambda: defaultdict(str))
    for db_name, caseMetricsMap in dbCaseMetricsMap.items():
        for case_name, metricInfo in caseMetricsMap.items():
            metrics = metricInfo["metrics"]
            db = metricInfo["db"]
            db_label = metricInfo["db_label"]
            version = metricInfo["version"]
            label = metricInfo["label"]
            if label == ResultLabel.NORMAL:
                mergedTasks.append(
                    {
                        "db_name": db_name,
                        "db": db,
                        "db_label": db_label,
                        "version": version,
                        "case_name": case_name,
                        "metricsSet": set(metrics.keys()),
                        **metrics,
                    }
                )
            else:
                failedTasks[case_name][db_name] = label

    return mergedTasks, failedTasks


def mergeMetrics(metrics_1: dict, metrics_2: dict) -> dict:
    metrics = {**metrics_1}
    for key, value in metrics_2.items():
        metrics[key] = getBetterMetric(key, value, metrics[key]) if key in metrics else value

    return metrics


def getBetterMetric(metric, value_1, value_2):
    try:
        if value_1 < 1e-7:
            return value_2
        if value_2 < 1e-7:
            return value_1
        return min(value_1, value_2) if isLowerIsBetterMetric(metric) else max(value_1, value_2)
    except Exception:
        return value_1


def getBetterLabel(label_1: ResultLabel, label_2: ResultLabel):
    return label_2 if label_1 != ResultLabel.NORMAL else label_1
