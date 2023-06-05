import streamlit as st
from vector_db_bench.frontend.const import *
from vector_db_bench.interface import benchMarkRunner
from dataclasses import asdict
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
from collections import defaultdict
from vector_db_bench.frontend.pages.views.result_selector import getShowResults
from vector_db_bench.frontend.pages.views.chart import drawChartQpsPerHour
from vector_db_bench.frontend.pages.views.data import getChartData
from vector_db_bench.frontend.pages.views.db_case_filter import getShowDbsAndCases
from vector_db_bench.frontend.utils import displayCaseText
from vector_db_bench.backend.clients import DB


def main():
    st.set_page_config(
        page_title="Falcon Mark - Open VectorDB Bench",
        page_icon="🧊",
        # layout="wide",
        initial_sidebar_state="collapsed",
    )

    st.title("Check Results")

    allResults = benchMarkRunner.get_results()

    # results selector
    resultSelectorContainer = st.container()
    selectedResult = getShowResults(allResults, resultSelectorContainer)

    # filters: db_name, case_name
    filterContainer = st.container()
    showDBNames, showCases = getShowDbsAndCases(selectedResult, filterContainer)

    # data
    allData = getChartData(selectedResult, showDBNames, showCases)

    # charts
    for case in showCases:
        chartContainer = st.container()
        data = [data for data in allData if data["case"] == case]
        chartContainer.subheader(displayCaseText(case))
        drawChartQpsPerHour(data, chartContainer)


if __name__ == "__main__":
    main()
