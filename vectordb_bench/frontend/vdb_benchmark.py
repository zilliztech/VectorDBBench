import streamlit as st
from vectordb_bench.frontend.components.check_results.footer import footer
from vectordb_bench.frontend.components.check_results.stPageConfig import initResultsPageConfig
from vectordb_bench.frontend.components.check_results.headerIcon import drawHeaderIcon
from vectordb_bench.frontend.components.check_results.nav import NavToQuriesPerDollar, NavToRunTest
from vectordb_bench.frontend.components.check_results.charts import drawCharts
from vectordb_bench.frontend.components.check_results.filters import getshownData
from vectordb_bench.frontend.components.get_results.saveAsImage import getResults
from vectordb_bench.frontend.const.styles import *
from vectordb_bench.interface import benchMarkRunner


def main():
    # set page config
    initResultsPageConfig(st)

    # header
    drawHeaderIcon(st)

    allResults = benchMarkRunner.get_results()

    st.title("Vector Database Benchmark")

    # results selector and filter
    resultSelectorContainer = st.sidebar.container()
    shownData, failedTasks, showCases = getshownData(
        allResults, resultSelectorContainer
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
    drawCharts(st, shownData, failedTasks, showCases)

    # footer
    footer(st.container())


if __name__ == "__main__":
    main()
