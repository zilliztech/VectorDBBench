import streamlit as st
from vectordb_bench.frontend.components.check_results.footer import footer
from vectordb_bench.frontend.components.check_results.stPageConfig import (
    initResultsPageConfig,
)
from vectordb_bench.frontend.components.check_results.headerIcon import drawHeaderIcon
from vectordb_bench.frontend.components.check_results.nav import (
    NavToQuriesPerDollar,
    NavToRunTest,
    NavToPages,
)
from vectordb_bench.frontend.components.check_results.charts import drawCharts
from vectordb_bench.frontend.components.check_results.filters import getshownData
from vectordb_bench.frontend.components.get_results.saveAsImage import getResults

from vectordb_bench.interface import benchmark_runner


def main():
    # set page config
    initResultsPageConfig(st)

    # header
    drawHeaderIcon(st)

    # navigate
    NavToPages(st)

    allResults = benchmark_runner.get_results()

    st.title("Vector Database Benchmark")
    st.caption(
        "Choose your desired test results to display from the sidebar. "
        "For your reference, we've included two standard benchmarks tested by our team. "
        "Note that `standard_2025` was tested in 2025; the others in 2023. "
        "Unless explicitly labeled as distributed multi-node, test with single-node mode by default."
    )
    st.caption("We welcome community contributions for better results, parameter configurations, and optimizations.")
    # results selector and filter
    resultSelectorContainer = st.sidebar.container()
    shownData, failedTasks, showCaseNames = getshownData(resultSelectorContainer, allResults)

    resultSelectorContainer.divider()

    # nav
    navContainer = st.sidebar.container()
    NavToRunTest(navContainer)
    NavToQuriesPerDollar(navContainer)

    # save or share
    resultesContainer = st.sidebar.container()
    getResults(resultesContainer, "vectordb_bench")

    # charts
    drawCharts(st, shownData, failedTasks, showCaseNames)

    # footer
    footer(st.container())


if __name__ == "__main__":
    main()
