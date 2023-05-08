from collections import defaultdict
from dataclasses import asdict
from vector_db_bench.metric import isLowerIsBetterMetric
from vector_db_bench.models import ResultLabel


def getChartData(tasks, dbNames, cases):
    filterTasks = getFilterTasks(tasks, dbNames, cases)
    mergedTasks, failedTasks = mergeTasks(filterTasks)
    return mergedTasks, failedTasks


def getFilterTasks(tasks, dbNames, cases):
    filterTasks = [
        task
        for task in tasks
        if task.task_config.db_name in dbNames
        and task.task_config.case_config.case_id.value in cases
    ]
    return filterTasks


def mergeTasks(tasks):
    dbCaseMetricsMap = defaultdict(lambda: defaultdict(dict))
    for task in tasks:
        db_name = task.task_config.db_name
        db = task.task_config.db.value
        db_label = task.task_config.db_config.db_label or ""
        case = task.task_config.case_config.case_id.value
        dbCaseMetricsMap[db_name][case] = {
            "db": db,
            "db_label": db_label,
            "metrics": mergeMetrics(
                dbCaseMetricsMap[db_name][case].get("metrics", {}), asdict(task.metrics)
            ),
            "label": getBetterLabel(dbCaseMetricsMap[db_name][case].get("label", ResultLabel.FAILED), task.label)
        }

    mergedTasks = []
    failedTasks = defaultdict(lambda: defaultdict(str))
    for db_name, caseMetricsMap in dbCaseMetricsMap.items():
        for case, metricInfo in caseMetricsMap.items():
            metrics = metricInfo["metrics"]
            db = metricInfo["db"]
            db_label = metricInfo["db_label"]
            label = metricInfo["label"]
            if label == ResultLabel.NORMAL:
                mergedTasks.append(
                    {
                        "db_name": db_name,
                        "db": db,
                        "db_label": db_label,
                        "case": case,
                        "metricsSet": set(metrics.keys()),
                        **metrics,
                    }
                )
            else: 
                failedTasks[case][db_name] = label
                
    return mergedTasks, failedTasks


def mergeMetrics(metrics_1: dict, metrics_2: dict) -> dict:
    metrics = {**metrics_1}
    for key, value in metrics_2.items():
        metrics[key] = (
            getBetterMetric(key, value, metrics[key]) if key in metrics else value
        )

    return metrics


def getBetterMetric(metric, value_1, value_2):
    if value_1 < 1e-7:
        return value_2
    if value_2 < 1e-7:
        return value_1
    return (
        min(value_1, value_2)
        if isLowerIsBetterMetric(metric)
        else max(value_1, value_2)
    )
    
def getBetterLabel(label_1: ResultLabel, label_2: ResultLabel):
    return label_2 if label_1 != ResultLabel.NORMAL else label_1
