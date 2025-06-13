import streamlit as st
from vectordb_bench.frontend.components.check_results.headerIcon import drawHeaderIcon
from vectordb_bench.frontend.components.custom.initStyle import initStyle
from vectordb_bench.frontend.components.welcome.explainPrams import explainPrams
from vectordb_bench.frontend.components.welcome.welcomePrams import welcomePrams
from vectordb_bench.frontend.config.styles import FAVICON, PAGE_TITLE


def main():
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=FAVICON,
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    # header
    drawHeaderIcon(st)

    # init style
    initStyle(st)

    # page
    welcomePrams(st)

    # description
    explainPrams(st)


if __name__ == "__main__":
    main()
