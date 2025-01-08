from vectordb_bench.frontend.components.check_results.expanderStyle import (
    initMainExpanderStyle,
)
from vectordb_bench.metric import metric_order, isLowerIsBetterMetric, metric_unit_map
from vectordb_bench.frontend.config.styles import *
from vectordb_bench.models import ResultLabel
import plotly.express as px


def drawCharts(st, allData, failedTasks, caseNames: list[str]):
    initMainExpanderStyle(st)
    for caseName in caseNames:
        chartContainer = st.expander(caseName, True)
        data = [data for data in allData if data["case_name"] == caseName]
        drawChart(data, chartContainer, key_prefix=caseName)

        errorDBs = failedTasks[caseName]
        showFailedDBs(chartContainer, errorDBs)


def showFailedDBs(st, errorDBs):
    failedDBs = [db for db, label in errorDBs.items() if label == ResultLabel.FAILED]
    timeoutDBs = [db for db, label in errorDBs.items() if label == ResultLabel.OUTOFRANGE]

    showFailedText(st, "Failed", failedDBs)
    showFailedText(st, "Timeout", timeoutDBs)


def showFailedText(st, text, dbs):
    if len(dbs) > 0:
        st.markdown(
            f"<div style='margin: -16px 0 12px 8px; font-size: 16px; font-weight: 600;'>{text}: &nbsp;&nbsp;{', '.join(dbs)}</div>",
            unsafe_allow_html=True,
        )


def drawChart(data, st, key_prefix: str):
    metricsSet = set()
    for d in data:
        metricsSet = metricsSet.union(d["metricsSet"])
    showMetrics = [metric for metric in metric_order if metric in metricsSet]

    for i, metric in enumerate(showMetrics):
        container = st.container()
        key = f"{key_prefix}-{metric}"
        drawMetricChart(data, metric, container, key=key)


def getLabelToShapeMap(data):
    labelIndexMap = {}

    dbSet = {d["db"] for d in data}
    for db in dbSet:
        labelSet = {d["db_label"] for d in data if d["db"] == db}
        labelList = list(labelSet)

        usedShapes = set()
        i = 0
        for label in labelList:
            if label not in labelIndexMap:
                loopCount = 0
                while i % len(PATTERN_SHAPES) in usedShapes:
                    i += 1
                    loopCount += 1
                    if loopCount > len(PATTERN_SHAPES):
                        break
                labelIndexMap[label] = i
                i += 1
            else:
                usedShapes.add(labelIndexMap[label] % len(PATTERN_SHAPES))

    labelToShapeMap = {label: getPatternShape(index) for label, index in labelIndexMap.items()}
    return labelToShapeMap


def drawMetricChart(data, metric, st, key: str):
    dataWithMetric = [d for d in data if d.get(metric, 0) > 1e-7]
    # dataWithMetric = data
    if len(dataWithMetric) == 0:
        return

    # title = st.container()
    # title.markdown(
    #     f"**{metric}** ({'less' if isLowerIsBetterMetric(metric) else 'more'} is better)"
    # )
    chart = st.container()

    height = len(dataWithMetric) * 24 + 48
    xmin = 0
    xmax = max([d.get(metric, 0) for d in dataWithMetric])
    xpadding = (xmax - xmin) / 16
    xpadding_multiplier = 1.8
    xrange = [xmin, xmax + xpadding * xpadding_multiplier]
    unit = metric_unit_map.get(metric, "")
    labelToShapeMap = getLabelToShapeMap(dataWithMetric)
    categoryorder = "total descending" if isLowerIsBetterMetric(metric) else "total ascending"
    fig = px.bar(
        dataWithMetric,
        x=metric,
        y="db_name",
        color="db",
        height=height,
        # pattern_shape="db_label",
        # pattern_shape_sequence=SHAPES,
        pattern_shape_map=labelToShapeMap,
        orientation="h",
        hover_data={
            "db": False,
            "db_label": False,
            "db_name": True,
        },
        color_discrete_map=COLOR_MAP,
        text_auto=True,
        title=f"{metric.capitalize()} ({'less' if isLowerIsBetterMetric(metric) else 'more'} is better)",
    )
    fig.update_xaxes(showticklabels=False, visible=False, range=xrange)
    fig.update_yaxes(
        # showticklabels=False,
        # visible=False,
        title=dict(
            font=dict(
                size=1,
            ),
            text="",
        )
    )
    fig.update_traces(
        textposition="outside",
        textfont=dict(
            color="#333",
            size=12,
        ),
        marker=dict(pattern=dict(fillmode="overlay", fgcolor="#fff", fgopacity=1, size=7)),
        texttemplate="%{x:,.4~r}" + unit,
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=48, b=12, pad=8),
        bargap=0.25,
        showlegend=False,
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1, title=""),
        # legend=dict(orientation="v", title=""),
        yaxis={"categoryorder": categoryorder},
        title=dict(
            font=dict(
                size=16,
                color="#666",
                # family="Arial, sans-serif",
            ),
            pad=dict(l=16),
            # y=0.95,
            # yanchor="top",
            # yref="container",
        ),
    )

    chart.plotly_chart(fig, use_container_width=True, key=key)
