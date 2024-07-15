from streamlit_autorefresh import st_autorefresh
from vectordb_bench.frontend.config.styles import *


def autoRefresh():
    auto_refresh_count = st_autorefresh(
        interval=MAX_AUTO_REFRESH_INTERVAL,
        limit=MAX_AUTO_REFRESH_COUNT,
        key="streamlit-auto-refresh",
    )
