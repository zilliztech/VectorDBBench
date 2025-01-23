import streamlit as st
from vectordb_bench.frontend.components.check_results.footer import footer
from vectordb_bench.frontend.components.check_results.headerIcon import drawHeaderIcon
from vectordb_bench.frontend.components.check_results.nav import (
    NavToResults,
    NavToRunTest,
)
from vectordb_bench.frontend.components.check_results.filters import getshownData
from vectordb_bench.frontend.components.concurrent.charts import drawChartsByCase
from vectordb_bench.frontend.components.get_results.saveAsImage import getResults
from vectordb_bench.frontend.config.styles import FAVICON
from vectordb_bench.interface import benchmark_runner
from vectordb_bench.models import TestResult


def main():
    # set page config
    st.set_page_config(
        page_title="VDBBench Conc Perf",
        page_icon=FAVICON,
        layout="wide",
        # initial_sidebar_state="collapsed",
    )

    # header
    drawHeaderIcon(st)

    allResults = benchmark_runner.get_results()

    def check_conc_data(res: TestResult):
        case_results = res.results
        count = 0
        for case_result in case_results:
            if len(case_result.metrics.conc_num_list) > 0:
                count += 1

        return count > 0

    checkedResults = [res for res in allResults if check_conc_data(res)]

    st.title("VectorDB Benchmark (Concurrent Performance)")

    # results selector
    resultSelectorContainer = st.sidebar.container()
    shownData, _, showCaseNames = getshownData(checkedResults, resultSelectorContainer)

    resultSelectorContainer.divider()

    # nav
    navContainer = st.sidebar.container()
    NavToRunTest(navContainer)
    NavToResults(navContainer)

    # save or share
    resultesContainer = st.sidebar.container()
    getResults(resultesContainer, "vectordb_bench_concurrent")

    # main
    latency_type = st.radio("Latency Type", options=["latency_p99", "latency_avg"])
    drawChartsByCase(shownData, showCaseNames, st.container(), latency_type=latency_type)

    # footer
    footer(st.container())


if __name__ == "__main__":
    main()
