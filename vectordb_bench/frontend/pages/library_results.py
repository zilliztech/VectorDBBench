import streamlit as st
from streamlit_extras.switch_page_button import switch_page
from vectordb_bench.frontend.components.check_results.headerIcon import drawHeaderIcon
from vectordb_bench.frontend.components.check_results.nav import NavToRunTest
from vectordb_bench.frontend.components.library.attr_selector import attr_selector
from vectordb_bench.frontend.components.library.chart import drawChartByCase
from vectordb_bench.frontend.components.library.data import getChartsData
from vectordb_bench.frontend.const.styles import FAVICON, PAGE_TITLE


def main():
    # set page config
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=FAVICON,
        # layout="wide",
        initial_sidebar_state="collapsed",
    )

    # header
    drawHeaderIcon(st)

    libraryData, labels, isExample = getChartsData()

    if isExample:
        st.markdown(
            "*Sample data is shown now for the sake of user experience. After running your own tests, the sample data will not be shown."
        )

    navClick = st.button("Run Your Test &nbsp;&nbsp;>")
    if navClick:
        switch_page("run test")

    st.markdown("")

    cols = st.columns(2)
    # color
    colorAttr = attr_selector(cols[0], labels, "Color Attrs", libraryData, "index_type")
    # shape
    shapeAttr = attr_selector(cols[1], labels, "Shape Attrs", libraryData, "index_type")
    # text
    textAttr = attr_selector(st, labels, "Text Attrs", libraryData)

    st.markdown("")

    drawChartByCase(
        st,
        libraryData,
        colorAttr=colorAttr,
        shapeAttr=shapeAttr,
        textAttr=textAttr,
        labels=labels,
    )


if __name__ == "__main__":
    main()
