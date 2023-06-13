import streamlit as st
from vector_db_bench.frontend.const import *
from vector_db_bench.frontend.components.check_results.headerIcon import drawHeaderIcon
from vector_db_bench.frontend.components.check_results.nav import NavToRunTest
from vector_db_bench.frontend.components.check_results.charts import drawCharts
from vector_db_bench.frontend.components.check_results.filters import getshownData
from vector_db_bench.interface import benchMarkRunner


def main():
    st.set_page_config(
        page_title="VectorDB Benchmark",
        page_icon="https://assets.zilliz.com/favicon_f7f922fe27.png",
        # layout="wide",
        # initial_sidebar_state="collapsed",
    )

    # header
    drawHeaderIcon(st)

    allResults = benchMarkRunner.get_results()

    st.title("Vector Database Benchmark")
    st.write("description [todo]")

    # results selector
    resultSelectorContainer = st.sidebar.container()
    shownData, showCases = getshownData(allResults, resultSelectorContainer)

    resultSelectorContainer.divider()

    # nav
    navContainer = st.sidebar.container()
    NavToRunTest(navContainer)

    # charts
    drawCharts(st, shownData, showCases)


if __name__ == "__main__":
    main()
