import plotly.express as px
from vectordb_bench.metric import metric_unit_map


def drawCharts(st, allData, **kwargs):
    dataset_names = list(set([data["dataset_name"] for data in allData]))
    dataset_names.sort()
    for dataset_name in dataset_names:
        container = st.container()
        container.subheader(dataset_name)
        data = [d for d in allData if d["dataset_name"] == dataset_name]
        drawChartByMetric(container, data, **kwargs)


def drawChartByMetric(st, data, metrics=("qps", "recall"), **kwargs):
    columns = st.columns(len(metrics))
    for i, metric in enumerate(metrics):
        container = columns[i]
        container.markdown(f"#### {metric}")
        drawChart(container, data, metric)


def getRange(metric, data, padding_multipliers):
    minV = min([d.get(metric, 0) for d in data])
    maxV = max([d.get(metric, 0) for d in data])
    padding = maxV - minV
    rangeV = [
        minV - padding * padding_multipliers[0],
        maxV + padding * padding_multipliers[1],
    ]
    return rangeV


def drawChart(st, data: list[object], metric):
    unit = metric_unit_map.get(metric, "")
    x = "filter_rate"
    xrange = getRange(x, data, [0.05, 0.1])

    y = metric
    yrange = getRange(y, data, [0.2, 0.1])

    data.sort(key=lambda a: a[x])

    fig = px.line(
        data,
        x=x,
        y=y,
        color="db_name",
        line_group="db_name",
        text=metric,
        markers=True,
    )
    fig.update_xaxes(range=xrange)
    fig.update_yaxes(range=yrange)
    fig.update_traces(textposition="bottom right", texttemplate="%{y:,.4~r}" + unit)
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0, pad=8),
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1, title=""),
    )
    st.plotly_chart(fig, use_container_width=True)
