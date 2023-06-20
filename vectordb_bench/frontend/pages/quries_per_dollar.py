import streamlit as st
from vectordb_bench.frontend.components.check_results.footer import footer
from vectordb_bench.frontend.components.check_results.priceTable import priceTable
from vectordb_bench.frontend.const import *
from vectordb_bench.frontend.components.check_results.headerIcon import drawHeaderIcon
from vectordb_bench.frontend.components.check_results.nav import (
    NavToResults,
    NavToRunTest,
)
from vectordb_bench.frontend.components.check_results.charts import drawMetricChart
from vectordb_bench.frontend.components.check_results.filters import getshownData
from vectordb_bench.interface import benchMarkRunner
from vectordb_bench.metric import QURIES_PER_DOLLAR_METRIC


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

    st.title("Vector DB Benchmark (QP$)")
    st.subheader("Price List")

    # results selector
    resultSelectorContainer = st.sidebar.container()
    shownData, _, showCases = getshownData(allResults, resultSelectorContainer)

    resultSelectorContainer.divider()

    # nav
    navContainer = st.sidebar.container()
    NavToRunTest(navContainer)
    NavToResults(navContainer)

    # price table
    priceTableContainer = st.container()
    priceMap = priceTable(priceTableContainer, shownData)

    # charts
    for case in showCases:
        chartContainer = st.container()
        data = [data for data in shownData if data["case_name"] == case.name]
        dataWithMetric = []
        metric = QURIES_PER_DOLLAR_METRIC
        for d in data:
            qps = d.get("qps", 0)
            price = priceMap.get(d["db"], {}).get(d["db_label"], 0)
            if qps > 0 and price > 0:
                d[metric] = d["qps"] / price * 3.6
                dataWithMetric.append(d)
        if len(dataWithMetric) > 0:
            chartContainer.subheader(case.name)
            drawMetricChart(data, metric, chartContainer)

    # footer
    footer(st.container())


if __name__ == "__main__":
    main()
