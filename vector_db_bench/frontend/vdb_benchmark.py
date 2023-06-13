import streamlit as st
from vector_db_bench.frontend.const import *
from vector_db_bench.frontend.pages.components.charts import drawCharts
from vector_db_bench.frontend.pages.components.headerIcon import drawHeaderIcon
from vector_db_bench.frontend.pages.components.nav import NavToRunTest
from vector_db_bench.interface import benchMarkRunner
from dataclasses import asdict
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit.components.v1 as components
from collections import defaultdict
from vector_db_bench.frontend.pages.components.filters import getshownData
from vector_db_bench.frontend.utils import displayCaseText


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

