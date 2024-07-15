import streamlit as st
from vectordb_bench.frontend.components.check_results.headerIcon import drawHeaderIcon
from vectordb_bench.frontend.components.tables.data import getNewResults
from vectordb_bench.frontend.config.styles import FAVICON


def main():
    # set page config
    st.set_page_config(
        page_title="Table",
        page_icon=FAVICON,
        layout="wide",
        # initial_sidebar_state="collapsed",
    )

    # header
    drawHeaderIcon(st)

    df = getNewResults()
    st.dataframe(df, height=800)


if __name__ == "__main__":
    main()
