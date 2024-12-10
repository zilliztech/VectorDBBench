import streamlit as st
from vectordb_bench.backend.filter import FilterType
from vectordb_bench.frontend.components.check_results.footer import footer
from vectordb_bench.frontend.components.check_results.headerIcon import drawHeaderIcon
from vectordb_bench.frontend.components.check_results.nav import (
    NavToQuriesPerDollar,
    NavToRunTest,
)
from vectordb_bench.frontend.components.filter.charts import drawCharts
from vectordb_bench.frontend.components.check_results.filters import getshownData
from vectordb_bench.frontend.config.styles import FAVICON

# from vectordb_bench.frontend.config.styles import *
from vectordb_bench.interface import benchMarkRunner


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

    allResults = benchMarkRunner.get_results()

    st.title("Vector Database Benchmark (Label Filter)")

    # results selector and filter
    resultSelectorContainer = st.sidebar.container()
    shownData, failedTasks, showCaseNames = getshownData(
        resultSelectorContainer, allResults, filter_type=FilterType.Label
    )

    resultSelectorContainer.divider()

    # nav
    navContainer = st.sidebar.container()
    NavToRunTest(navContainer)
    NavToQuriesPerDollar(navContainer)

    # charts
    drawCharts(st, shownData, failedTasks, showCaseNames)

    # footer
    footer(st.container())


if __name__ == "__main__":
    main()
