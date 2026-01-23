import plotly.graph_objects as go

from vectordb_bench.frontend.components.streaming.data import (
    DisplayedMetric,
    StreamingData,
    get_streaming_data,
)
from vectordb_bench.frontend.config.styles import (
    COLORS_10,
    COLORS_2,
    SCATTER_LINE_WIDTH,
    SCATTER_MAKER_SIZE,
    STREAMING_CHART_COLUMNS,
)


def drawChartsByCase(
    st,
    allData,
    showCaseNames: list[str],
    **kwargs,
):
    allData = [d for d in allData if len(d["st_search_stage_list"]) > 0]
    for case_name in showCaseNames:
        data = [d for d in allData if d["case_name"] == case_name]
        if len(data) == 0:
            continue
        container = st.container()
        container.write("")  # blank line
        container.subheader(case_name)
        drawChartByMetric(container, data, case_name=case_name, **kwargs)
        container.write("")  # blank line


def drawChartByMetric(
    st,
    case_data,
    case_name: str,
    line_chart_displayed_y_metrics: list[tuple[DisplayedMetric, str]],
    **kwargs,
):
    columns = st.columns(STREAMING_CHART_COLUMNS)
    streaming_data = get_streaming_data(case_data)

    # line chart
    for i, metric_info in enumerate(line_chart_displayed_y_metrics):
        metric, note = metric_info
        container = columns[i % STREAMING_CHART_COLUMNS]
        container.markdown(f"#### {metric.value.capitalize()}")
        container.markdown(f"{note}")
        key = f"{case_name}-{metric.value}"
        drawLineChart(container, streaming_data, metric=metric, key=key, **kwargs)

    # bar chart
    container = columns[len(line_chart_displayed_y_metrics) % STREAMING_CHART_COLUMNS]
    container.markdown("#### Duration")
    container.markdown(
        "insert more than ideal-insert-duration (dash-line) means exceeding the maximum processing capacity.",
        help="vectordb need more time to process accumulated insert requests.",
    )
    key = f"{case_name}-duration"
    drawBarChart(container, case_data, key=key, **kwargs)
    # drawLineChart(container, data, line_x_displayed_label, label)
    # drawTestChart(container)


def drawLineChart(
    st,
    streaming_data: list[StreamingData],
    metric: DisplayedMetric,
    key: str,
    with_last_optimized_data=True,
    **kwargs,
):
    db_names = list({d.db_name for d in streaming_data})
    db_names.sort()
    x_metric = kwargs.get("line_chart_displayed_x_metric", DisplayedMetric.search_stage)
    fig = go.Figure()
    if x_metric == DisplayedMetric.search_time:
        ideal_insert_duration = streaming_data[0].ideal_insert_duration
        fig.add_shape(
            type="line",
            y0=min([getattr(d, metric.value) for d in streaming_data]),
            y1=max([getattr(d, metric.value) for d in streaming_data]),
            x0=ideal_insert_duration,
            x1=ideal_insert_duration,
            line=dict(color="#999", width=SCATTER_LINE_WIDTH, dash="dot"),
            showlegend=True,
            name="insert 100% standard time",
        )
    for i, db_name in enumerate(db_names):
        data = [d for d in streaming_data if d.db_name == db_name]
        color = COLORS_10[i]
        if with_last_optimized_data:
            fig.add_trace(
                get_optimized_scatter(
                    data,
                    db_name=db_name,
                    metric=metric,
                    color=color,
                    **kwargs,
                )
            )
        fig.add_trace(
            get_normal_scatter(
                data,
                db_name=db_name,
                metric=metric,
                color=color,
                **kwargs,
            )
        )
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0, pad=8),
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="left", x=0, title=""),
    )

    x_title = "Search Stages (%)"
    if x_metric == DisplayedMetric.search_time:
        x_title = "Actual Time (s)"
    fig.update_layout(xaxis_title=x_title)
    st.plotly_chart(fig, use_container_width=True, key=key)


def get_normal_scatter(
    data: list[StreamingData],
    db_name: str,
    metric: DisplayedMetric,
    color: str,
    line_chart_displayed_x_metric: DisplayedMetric,
    **kwargs,
):
    unit = ""
    if "latency" in metric.value:
        unit = "ms"
    data.sort(key=lambda x: getattr(x, line_chart_displayed_x_metric.value))
    data = [d for d in data if not d.optimized]
    hovertemplate = f"%{{text}}% data inserted.<br>{metric.value}=%{{y:.4g}}{unit}"
    if line_chart_displayed_x_metric == DisplayedMetric.search_time:
        hovertemplate = f"%{{text}}% data inserted.<br>actual_time=%{{x:.4g}}s<br>{metric.value}=%{{y:.4g}}{unit}"
    return go.Scatter(
        x=[getattr(d, line_chart_displayed_x_metric.value) for d in data],
        y=[getattr(d, metric.value) for d in data],
        text=[d.search_stage for d in data],
        mode="markers+lines",
        name=db_name,
        marker=dict(color=color, size=SCATTER_MAKER_SIZE),
        line=dict(dash="solid", width=SCATTER_LINE_WIDTH, color=color),
        legendgroup=db_name,
        hovertemplate=hovertemplate,
    )


def get_optimized_scatter(
    data: list[StreamingData],
    db_name: str,
    metric: DisplayedMetric,
    color: str,
    line_chart_displayed_x_metric: DisplayedMetric,
    **kwargs,
):
    unit = ""
    if "latency" in metric.value:
        unit = "ms"
    data.sort(key=lambda x: x.search_stage)
    if not data[-1].optimized or len(data) < 2:
        return go.Scatter()
    data = data[-2:]
    hovertemplate = f"all data inserted and <b style='color: #333;'>optimized</b>.<br>{metric.value}=%{{y:.4g}}{unit}"
    if line_chart_displayed_x_metric == DisplayedMetric.search_time:
        hovertemplate = f"all data inserted and <b style='color: #333;'>optimized</b>.<br>actual_time=%{{x:.4g}}s<br>{metric.value}=%{{y:.4g}}{unit}"
    return go.Scatter(
        x=[getattr(d, line_chart_displayed_x_metric.value) for d in data],
        y=[getattr(d, metric.value) for d in data],
        text=[d.search_stage for d in data],
        mode="markers+lines",
        name=db_name,
        legendgroup=db_name,
        marker=dict(color=color, size=[0, SCATTER_MAKER_SIZE]),
        line=dict(dash="dash", width=SCATTER_LINE_WIDTH, color=color),
        hovertemplate=hovertemplate,
        showlegend=False,
    )


def drawBarChart(
    st,
    data,
    key: str,
    with_last_optimized_data=True,
    **kwargs,
):
    if len(data) < 1:
        return
    fig = go.Figure()

    # ideal insert duration
    ideal_insert_duration = data[0]["st_ideal_insert_duration"]
    fig.add_shape(
        type="line",
        y0=-0.5,
        y1=len(data) - 0.5,
        x0=ideal_insert_duration,
        x1=ideal_insert_duration,
        line=dict(color="#999", width=SCATTER_LINE_WIDTH, dash="dot"),
        showlegend=True,
        name="insert 100% standard time",
    )

    # insert duration
    fig.add_trace(
        get_bar(
            data,
            metric=DisplayedMetric.insert_duration,
            color=COLORS_2[0],
            **kwargs,
        )
    )

    # optimized duration
    if with_last_optimized_data:
        fig.add_trace(
            get_bar(
                data,
                metric=DisplayedMetric.optimize_duration,
                color=COLORS_2[1],
                **kwargs,
            )
        )
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0, pad=8),
        legend=dict(orientation="h", yanchor="bottom", y=1, xanchor="left", x=0, title=""),
    )
    fig.update_layout(xaxis_title="time (s)")
    fig.update_layout(barmode="stack")
    fig.update_traces(width=0.15)
    st.plotly_chart(fig, use_container_width=True, key=key)


def get_bar(
    data: list[StreamingData],
    metric: DisplayedMetric,
    color: str,
    **kwargs,
):
    return go.Bar(
        x=[d[metric.value] for d in data],
        y=[d["db_name"] for d in data],
        name=metric,
        marker_color=color,
        orientation="h",
        hovertemplate="%{y} %{x:.2f}s",
    )
