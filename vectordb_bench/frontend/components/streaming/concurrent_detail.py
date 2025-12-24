import plotly.graph_objects as go
import streamlit as st


def drawConcurrentPerformanceSection(container, case_data, case_name: str):
    """Main section for concurrent performance detail in streaming tests"""

    # Check if data exists
    if not case_data.get("st_conc_qps_list_list") or len(case_data["st_conc_qps_list_list"]) == 0:
        container.info(
            "Concurrent latency detail not available for this test. "
            "Re-run the test with updated code to collect this data."
        )
        return

    container.markdown("---")
    container.subheader("Concurrent Search Performance")
    container.markdown(
        "Detailed Latency → QPS relationship at each stage. " "Displays how latency and QPS vary with concurrency."
    )

    # View mode selector
    view_mode = container.radio(
        "View Mode", options=["Single Stage", "Compare Stages"], horizontal=True, key=f"{case_name}-view-mode"
    )

    if view_mode == "Single Stage":
        drawSingleStageView(container, case_data, case_name)
    else:
        drawCompareStagesView(container, case_data, case_name)


def drawSingleStageView(container, case_data, case_name: str):
    """Show detailed Latency→QPS for one selected stage"""

    stages = case_data["st_search_stage_list"]

    # Find the last stage with data for default selection
    default_stage_idx = len(stages) - 1
    for i in range(len(stages) - 1, -1, -1):
        if case_data["st_conc_qps_list_list"][i]:
            default_stage_idx = i
            break

    # Stage selector (show all stages)
    stage = container.selectbox(
        "Select Stage",
        options=stages,
        index=default_stage_idx,
        format_func=lambda x: f"{x}% data loaded",
        key=f"{case_name}-stage-selector",
    )

    stage_idx = stages.index(stage)

    # Check if this stage has concurrent data
    if not case_data["st_conc_qps_list_list"][stage_idx]:
        container.warning(
            f"No concurrent search data for {stage}% stage.\n\n"
            f"**Reason:** Concurrent tests were skipped because there wasn't enough time "
            f"between stages (< 10s per concurrency level).\n\n"
            f"**Tip:** Use a larger dataset or slower insert rate to get data for all stages."
        )
        return

    # Latency metric selector
    latency_metric = container.radio(
        "Latency Metric", options=["P99", "P95", "Average"], horizontal=True, key=f"{case_name}-latency-metric"
    )

    # Get data for selected stage
    qps_values = case_data["st_conc_qps_list_list"][stage_idx]
    conc_nums = case_data["st_conc_num_list_list"][stage_idx]

    # Get latency based on selection
    if latency_metric == "P99":
        latencies_sec = case_data["st_conc_latency_p99_list_list"][stage_idx]
    elif latency_metric == "P95":
        latencies_sec = case_data["st_conc_latency_p95_list_list"][stage_idx]
    else:
        latencies_sec = case_data["st_conc_latency_avg_list_list"][stage_idx]

    latencies_ms = [l * 1000 for l in latencies_sec]  # Convert to ms

    # Draw chart
    drawQPSLatencyChart(container, qps_values, latencies_ms, stage, latency_metric, case_name)

    # Draw table
    drawMetricsTable(
        container,
        qps_values,
        conc_nums,
        case_data["st_conc_latency_p99_list_list"][stage_idx],
        case_data["st_conc_latency_p95_list_list"][stage_idx],
        case_data["st_conc_latency_avg_list_list"][stage_idx],
        stage,
    )


def drawQPSLatencyChart(container, qps_values, latencies_ms, stage, metric_name, case_name):
    """Draw Latency vs QPS scatter plot"""

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=latencies_ms,
            y=qps_values,
            mode="lines+markers+text",
            text=[f"{qps:.1f}" for qps in qps_values],
            textposition="top center",
            marker=dict(size=12, color="#1f77b4"),
            line=dict(width=2, color="#1f77b4"),
            hovertemplate=("QPS: %{y:.1f}<br>" f"{metric_name} Latency: %{{x:.1f}}ms<br>" "<extra></extra>"),
        )
    )

    fig.update_layout(
        title=f"Latency vs QPS at Stage {stage}%",
        xaxis_title=f"Latency {metric_name} (ms)",
        yaxis_title="Queries Per Second (QPS)",
        height=500,
        hovermode="closest",
        showlegend=False,
    )

    container.plotly_chart(fig, use_container_width=True, key=f"{case_name}-chart-{stage}")


def drawMetricsTable(container, qps_values, conc_nums, p99_list, p95_list, avg_list, stage):
    """Draw detailed metrics table"""

    container.markdown(f"**Detailed Metrics at Stage {stage}%**")

    # Build table data
    table_data = []
    for i in range(len(qps_values)):
        table_data.append(
            {
                "Concurrency": conc_nums[i],
                "QPS": f"{qps_values[i]:.2f}",
                "P99 (ms)": f"{p99_list[i] * 1000:.1f}",
                "P95 (ms)": f"{p95_list[i] * 1000:.1f}",
                "Avg (ms)": f"{avg_list[i] * 1000:.1f}",
            }
        )

    container.table(table_data)


def drawCompareStagesView(container, case_data, case_name: str):
    """Show QPS→Latency curves for multiple stages"""

    stages = case_data["st_search_stage_list"]

    # Find stages with data for default selection
    stages_with_data = [stage for i, stage in enumerate(stages) if case_data["st_conc_qps_list_list"][i]]

    # Stage multi-selector (show all stages, but default to ones with data)
    default_stages = []
    if stages_with_data:
        default_stages = [stages_with_data[0], stages_with_data[-1]] if len(stages_with_data) >= 2 else stages_with_data

    selected_stages = container.multiselect(
        "Select stages to compare",
        options=stages,
        default=default_stages,
        format_func=lambda x: f"{x}%",
        key=f"{case_name}-compare-stages",
        help="Note: Some stages may have no concurrent data if test duration was too short",
    )

    if not selected_stages:
        container.warning("Please select at least one stage to display.")
        return

    # Latency metric selector
    latency_metric = container.radio(
        "Latency Metric", options=["P99", "P95", "Average"], horizontal=True, key=f"{case_name}-compare-metric"
    )

    # Draw comparison chart
    drawComparisonChart(container, case_data, selected_stages, latency_metric, case_name)


def drawComparisonChart(container, case_data, selected_stages, metric_name, case_name):
    """Draw multi-line comparison chart"""

    fig = go.Figure()

    # Color palette
    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]

    stages_plotted = 0
    stages_skipped = []

    for idx, stage in enumerate(selected_stages):
        stage_idx = case_data["st_search_stage_list"].index(stage)

        qps_values = case_data["st_conc_qps_list_list"][stage_idx]

        # Skip stages with no data
        if not qps_values:
            stages_skipped.append(stage)
            continue

        # Get latency based on selection
        if metric_name == "P99":
            latencies_sec = case_data["st_conc_latency_p99_list_list"][stage_idx]
        elif metric_name == "P95":
            latencies_sec = case_data["st_conc_latency_p95_list_list"][stage_idx]
        else:
            latencies_sec = case_data["st_conc_latency_avg_list_list"][stage_idx]

        latencies_ms = [l * 1000 for l in latencies_sec]
        stages_plotted += 1

        fig.add_trace(
            go.Scatter(
                x=latencies_ms,
                y=qps_values,
                mode="lines+markers",
                name=f"{stage}% loaded",
                marker=dict(size=10),
                line=dict(width=2, color=colors[idx % len(colors)]),
                hovertemplate=(
                    f"Stage {stage}%<br>"
                    "QPS: %{y:.1f}<br>"
                    f"{metric_name} Latency: %{{x:.1f}}ms<br>"
                    "<extra></extra>"
                ),
            )
        )

    # Check if any data was plotted
    if stages_plotted == 0:
        container.warning(
            f"None of the selected stages have concurrent search data.\n\n"
            f"**Skipped stages:** {', '.join([f'{s}%' for s in stages_skipped])}\n\n"
            f"**Reason:** Concurrent tests were skipped because there wasn't enough time "
            f"between stages (< 10s per concurrency level).\n\n"
            f"**Tip:** Use a larger dataset or slower insert rate to get data for all stages."
        )
        return

    # Show warning for skipped stages
    if stages_skipped:
        container.info(f"Stages without data: {', '.join([f'{s}%' for s in stages_skipped])}")

    fig.update_layout(
        title=f"Latency vs QPS Evolution Across Stages",
        xaxis_title=f"Latency {metric_name} (ms)",
        yaxis_title="Queries Per Second (QPS)",
        height=600,
        hovermode="closest",
        legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99),
    )

    container.plotly_chart(fig, use_container_width=True, key=f"{case_name}-compare-chart")

    # Add insight
    container.info("**Insight:** Compare curves across stages to understand how performance scales with data growth.")
