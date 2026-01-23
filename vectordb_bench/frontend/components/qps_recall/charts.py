from vectordb_bench.frontend.components.check_results.expanderStyle import (
    initMainExpanderStyle,
)
from vectordb_bench.metric import metric_order, isLowerIsBetterMetric, metric_unit_map
from vectordb_bench.frontend.config.styles import *
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt


def drawCharts(st, allData, caseNames: list[str]):
    initMainExpanderStyle(st)
    for caseName in caseNames:
        chartContainer = st.expander(caseName, True)
        data = [data for data in allData if data["case_name"] == caseName]
        drawChart(data, chartContainer, key_prefix=caseName)


def drawChart(data, st, key_prefix: str):
    metricsSet = set()
    for d in data:
        metricsSet = metricsSet.union(d["metricsSet"])
    showlineMetrics = [metric for metric in metric_order[:2] if metric in metricsSet]

    if showlineMetrics:
        metric = showlineMetrics[0]
        key = f"{key_prefix}-{metric}"
        drawlinechart(st, data, metric, key=key)


def drawBestperformance(data, y, group):
    all_filter_points = []
    data = pd.DataFrame(data)
    grouped = data.groupby(group)
    for name, group_df in grouped:
        filter_points = []
        current_start = 0
        for _ in range(len(group_df)):
            if current_start >= len(group_df):
                break
            max_index = group_df[y].iloc[current_start:].idxmax()
            filter_points.append(group_df.loc[max_index])

            current_start = group_df.index.get_loc(max_index) + 1
        all_filter_points.extend(filter_points)

    all_filter_df = pd.DataFrame(all_filter_points)
    remaining_df = data[~data.isin(all_filter_df).any(axis=1)]
    new_data = all_filter_df.to_dict(orient="records")
    remain_data = remaining_df.to_dict(orient="records")
    return new_data, remain_data


def drawlinechart(st, data: list[object], metric, key: str):
    minV = min([d.get(metric, 0) for d in data])
    maxV = max([d.get(metric, 0) for d in data])
    padding = maxV - minV
    rangeV = [
        minV - padding * 0.1,
        maxV + padding * 0.1,
    ]
    x = "recall"
    xrange = [0.8, 1.01]
    y = "qps"
    yrange = rangeV
    data.sort(key=lambda a: a[x])
    group = "db_name"
    new_data, new_remain_data = drawBestperformance(data, y, group)
    unique_db_names = list(set(item["db_name"] for item in new_data + new_remain_data))

    colors = plt.cm.get_cmap("tab10", len(unique_db_names))

    color_map = {
        db: f"rgb({int(colors(i)[0] * 255)}, {int(colors(i)[1] * 255)}, {int(colors(i)[2] * 255)})"
        for i, db in enumerate(unique_db_names)
    }

    fig = go.Figure()

    new_data_df = pd.DataFrame(new_data)

    for db in unique_db_names:
        db_data = new_data_df[new_data_df["db_name"] == db]
        fig.add_trace(
            go.Scatter(
                x=db_data["recall"],
                y=db_data["qps"],
                mode="lines+markers+text",
                name=db,
                line=dict(color=color_map[db]),
                marker=dict(color=color_map[db]),
                showlegend=True,
                hovertemplate="QPS=%{y:.4g}, Recall=%{x:.2f}",
                text=[f"{qps:.4g}@{recall:.2f}" for recall, qps in zip(db_data["recall"], db_data["qps"])],
                textposition="top right",
            )
        )

    for item in new_remain_data:
        fig.add_trace(
            go.Scatter(
                x=[item["recall"]],
                y=[item["qps"]],
                mode="markers",
                name=item["db_name"],
                marker=dict(color=color_map[item["db_name"]]),
                showlegend=False,
            )
        )

    fig.update_xaxes(range=xrange, title_text="Recall")
    fig.update_yaxes(range=yrange, title_text="QPS")
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0, pad=8),
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1, title=""),
    )
    st.plotly_chart(fig, use_container_width=True, key=key)
