from vectordb_bench.frontend.config.styles import *


def initResultsPageConfig(st):
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=FAVICON,
        # layout="wide",
        # initial_sidebar_state="collapsed",
    )


def initRunTestPageConfig(st):
    st.set_page_config(
        page_title=PAGE_TITLE,
        page_icon=FAVICON,
        # layout="wide",
        initial_sidebar_state="collapsed",
    )
