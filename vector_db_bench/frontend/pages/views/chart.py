from vector_db_bench.metric import metricOrder, isLowerIsBetterMetric
from vector_db_bench.frontend.const import *
import plotly.express as px


def drawChart(data, st):
    metricsSet = set()
    for d in data:
        metricsSet = metricsSet.union(d["metricsSet"])
    showMetrics = [metric for metric in metricOrder if metric in metricsSet]

    for i, metric in enumerate(showMetrics):
        container = st.container()
        drawMetricChart(data, metric, container)


def drawMetricChart(data, metric, st):
    dataWithMetric = [d for d in data if d.get(metric, 0) > 1e-7]
    # dataWithMetric = data
    if len(dataWithMetric) == 0:
        return

    title = st.container()
    title.markdown(
        f"**{metric}** ({'less' if isLowerIsBetterMetric(metric) else 'more'} is better)"
    )
    chart = st.container()

    height = len(dataWithMetric) * 28
    fig = px.bar(
        dataWithMetric,
        x=metric,
        y="db_name",
        color="db",
        height=height,
        pattern_shape="db_label",
        pattern_shape_sequence=["", "+", "\\", ".", "|", "/", "-"],
        orientation="h",
        hover_data={
            "db": False,
            "db_label": False,
            "db_name": True,
        },
        color_discrete_map=COLOR_MAP,
        text_auto=True,
    )
    fig.update_xaxes(showticklabels=False, visible=False)
    fig.update_yaxes(showticklabels=False, visible=False)
    fig.update_traces(
        textposition="outside",
        marker=dict(
            pattern=dict(fillmode="overlay", fgcolor="#fff", fgopacity=1, size=7)
        ),
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0, pad=8),
        showlegend=False,
        # legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="right", x=1, title=""),
        # legend=dict(orientation="v", title=""),
    )

    chart.plotly_chart(fig, use_container_width=True)
