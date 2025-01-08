import streamlit as st
from vectordb_bench.frontend.components.check_results.footer import footer
from vectordb_bench.frontend.components.check_results.stPageConfig import (
    initResultsPageConfig,
)
from vectordb_bench.frontend.components.check_results.headerIcon import drawHeaderIcon
from vectordb_bench.frontend.components.check_results.nav import (
    NavToQuriesPerDollar,
    NavToRunTest,
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

    allResults = benchmark_runner.get_results()

    st.title("Vector Database Benchmark")
    st.caption(
        "Except for zillizcloud-v2024.1, which was tested in _January 2024_, all other tests were completed before _August 2023_."
    )
    st.caption("All tested milvus are in _standalone_ mode.")

    # results selector and filter
    resultSelectorContainer = st.sidebar.container()
    shownData, failedTasks, showCaseNames = getshownData(allResults, resultSelectorContainer)

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
