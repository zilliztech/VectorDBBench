import logging
import streamlit as st
from vectordb_bench.backend.cases import CaseLabel
from vectordb_bench.frontend.components.check_results.footer import footer
from vectordb_bench.frontend.components.check_results.headerIcon import drawHeaderIcon
from vectordb_bench.frontend.components.check_results.nav import (
    NavToResults,
    NavToRunTest,
    NavToPages,
)
from vectordb_bench.frontend.components.check_results.filters import getshownData
from vectordb_bench.frontend.components.streaming.charts import drawChartsByCase
from vectordb_bench.frontend.components.get_results.saveAsImage import getResults
from vectordb_bench.frontend.components.streaming.data import DisplayedMetric
from vectordb_bench.frontend.config.styles import FAVICON
from vectordb_bench.interface import benchmark_runner
from vectordb_bench.models import CaseResult, TestResult

log = logging.getLogger("vectordb_bench")


def main():
    # set page config
    st.set_page_config(
        page_title="VDBB Streaming Perf",
        page_icon=FAVICON,
        layout="wide",
        # initial_sidebar_state="collapsed",
    )

    # header
    drawHeaderIcon(st)

    # navigate
    NavToPages(st)

    allResults = benchmark_runner.get_results()

    def check_streaming_data(res: TestResult):
        case_results = res.results
        flag = False
        for case_result in case_results:
            if case_result.task_config.case_config.case.label == CaseLabel.Streaming:
                flag = True

        return flag

    checkedResults = [res for res in allResults if check_streaming_data(res)]

    st.title("VDBBench - Streaming Performance")

    # results selector
    resultSelectorContainer = st.sidebar.container()

    def case_results_filter(case_result: CaseResult) -> bool:
        return len(case_result.metrics.st_search_stage_list) > 0

    shownData, _, showCaseNames = getshownData(
        resultSelectorContainer, checkedResults, case_results_filter=case_results_filter
    )

    resultSelectorContainer.divider()

    # nav
    navContainer = st.sidebar.container()
    NavToRunTest(navContainer)
    NavToResults(navContainer)

    # save or share
    resultesContainer = st.sidebar.container()
    getResults(resultesContainer, "vectordb_bench_streaming")

    # # main
    st.markdown("Tests search performance with a **stable** and **fixed** insertion rate.")
    control_panel = st.columns(3)
    compared_with_optimized = control_panel[0].toggle(
        "Compare with **optimezed** performance.",
        value=True,
        help="VectorDB is allowed to do **optimizations** after all insertions done and then test search performance.",
    )
    x_use_actual_time = control_panel[0].toggle(
        "Use **actual time** as X-axis instead of search stage.",
        value=False,
        help="Since vdbbench inserts may be faster than vetordb can process them, the time it actually reaches search_stage may have different delays.",
    )
    
    # Latency type selection
    latency_type = control_panel[2].radio(
        "Latency Type", 
        options=["latency_p99", "latency_p95"], 
        index=0,
        help="Choose between P99 (slowest 1%) or P95 (slowest 5%) latency metrics."
    )
    
    accuracy_metric = DisplayedMetric.recall
    show_ndcg = control_panel[1].toggle(
        "Show **NDCG** instead of Recall.",
        value=False,
        help="A more appropriate indicator to measure ANN search accuracy than Recall.",
    )
    need_adjust = control_panel[1].toggle(
        "Adjust the NDCG/Recall value based on the search stage.",
        value=True,
        help="NDCG/Recall is calculated using the ground truth file of the **entire** database, **divided by the search stage** to simulate the actual value.",
    )
    if show_ndcg:
        if need_adjust:
            accuracy_metric = DisplayedMetric.adjusted_ndcg
        else:
            accuracy_metric = DisplayedMetric.ndcg
    else:
        if need_adjust:
            accuracy_metric = DisplayedMetric.adjusted_recall
            
    # Determine which latency metric to display
    latency_metric = DisplayedMetric.latency_p99 if latency_type == "latency_p99" else DisplayedMetric.latency_p95
    latency_desc = "serial lantency (p99)" if latency_type == "latency_p99" else "serial lantency (p95)"
    
    line_chart_displayed_y_metrics: list[tuple[DisplayedMetric, str]] = [
        (
            DisplayedMetric.qps,
            "max-qps of increasing **concurrency search** tests in each search stage.",
        ),
        (accuracy_metric, "calculated in each search_stage."),
        (
            latency_metric,
            f"{latency_desc} of **serial search** tests in each search stage.",
        ),
    ]
    line_chart_displayed_x_metric = DisplayedMetric.search_stage
    if x_use_actual_time:
        line_chart_displayed_x_metric = DisplayedMetric.search_time

    drawChartsByCase(
        st.container(),
        shownData,
        showCaseNames,
        with_last_optimized_data=compared_with_optimized,
        line_chart_displayed_x_metric=line_chart_displayed_x_metric,
        line_chart_displayed_y_metrics=line_chart_displayed_y_metrics,
    )

    # footer
    footer(st.container())


if __name__ == "__main__":
    main()
