import streamlit as st
from vectordb_bench.backend.cases import CaseLabel
from vectordb_bench.backend.filter import FilterOp
from vectordb_bench.frontend.components.check_results.footer import footer
from vectordb_bench.frontend.components.check_results.headerIcon import drawHeaderIcon
from vectordb_bench.frontend.components.check_results.nav import (
    NavToQuriesPerDollar,
    NavToRunTest,
    NavToPages,
)
from vectordb_bench.frontend.components.qps_recall.charts import drawCharts
from vectordb_bench.frontend.components.qps_recall.data import getshownData
from vectordb_bench.frontend.components.get_results.saveAsImage import getResults

from vectordb_bench.frontend.config.styles import FAVICON
from vectordb_bench.interface import benchmark_runner
from vectordb_bench.models import CaseResult


def main():
    # set page config
    st.set_page_config(
        page_title="Label Filter",
        page_icon=FAVICON,
        layout="wide",
        # initial_sidebar_state="collapsed",
    )

    # header
    drawHeaderIcon(st)

    # navigate
    NavToPages(st)

    allResults = benchmark_runner.get_results()

    st.title("Vector Database Benchmark (Qps & Recall)")

    # results selector and filter
    resultSelectorContainer = st.sidebar.container()

    def case_results_filter(case_result: CaseResult) -> bool:
        case = case_result.task_config.case_config.case
        return case.label == CaseLabel.Performance and case.filters.type == FilterOp.NonFilter

    default_selected_task_labels = ["standard_2025"]
    shownData, failedTasks, showCaseNames = getshownData(
        resultSelectorContainer,
        allResults,
        case_results_filter=case_results_filter,
        default_selected_task_labels=default_selected_task_labels,
    )

    resultSelectorContainer.divider()

    # nav
    navContainer = st.sidebar.container()
    NavToRunTest(navContainer)
    NavToQuriesPerDollar(navContainer)

    # save or share
    resultesContainer = st.sidebar.container()
    getResults(resultesContainer, "vectordb_bench")

    # charts
    drawCharts(st, shownData, showCaseNames)

    # footer
    footer(st.container())


if __name__ == "__main__":
    main()
