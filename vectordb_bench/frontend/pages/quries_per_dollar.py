import streamlit as st
from vectordb_bench.frontend.components.check_results.footer import footer
from vectordb_bench.frontend.components.check_results.expanderStyle import (
    initMainExpanderStyle,
)
from vectordb_bench.frontend.components.check_results.priceTable import priceTable
from vectordb_bench.frontend.components.check_results.stPageConfig import (
    initResultsPageConfig,
)
from vectordb_bench.frontend.components.check_results.headerIcon import drawHeaderIcon
from vectordb_bench.frontend.components.check_results.nav import (
    NavToResults,
    NavToRunTest,
)
from vectordb_bench.frontend.components.check_results.charts import drawMetricChart
from vectordb_bench.frontend.components.check_results.filters import getshownData
from vectordb_bench.frontend.components.get_results.saveAsImage import getResults

from vectordb_bench.interface import benchmark_runner
from vectordb_bench.metric import QURIES_PER_DOLLAR_METRIC


def main():
    # set page config
    initResultsPageConfig(st)

    # header
    drawHeaderIcon(st)

    allResults = benchmark_runner.get_results()

    st.title("Vector DB Benchmark (QP$)")

    # results selector
    resultSelectorContainer = st.sidebar.container()
    shownData, _, showCaseNames = getshownData(allResults, resultSelectorContainer)

    resultSelectorContainer.divider()

    # nav
    navContainer = st.sidebar.container()
    NavToRunTest(navContainer)
    NavToResults(navContainer)

    # save or share
    resultesContainer = st.sidebar.container()
    getResults(resultesContainer, "vectordb_bench_qp$")

    # price table
    initMainExpanderStyle(st)
    priceTableContainer = st.container()
    priceMap = priceTable(priceTableContainer, shownData)

    # charts
    for caseName in showCaseNames:
        data = [data for data in shownData if data["case_name"] == caseName]
        dataWithMetric = []
        metric = QURIES_PER_DOLLAR_METRIC
        for d in data:
            qps = d.get("qps", 0)
            price = priceMap.get(d["db"], {}).get(d["db_label"], 0)
            if qps > 0 and price > 0:
                d[metric] = d["qps"] / price * 3.6
                dataWithMetric.append(d)
        if len(dataWithMetric) > 0:
            chartContainer = st.expander(caseName, True)
            key = f"{caseName}-{metric}"
            drawMetricChart(data, metric, chartContainer, key=key)

    # footer
    footer(st.container())


if __name__ == "__main__":
    main()
