from streamlit_autorefresh import st_autorefresh
from vector_db_bench.frontend.const import *


def autoRefresh():
    auto_refresh_count = st_autorefresh(
        interval=MAX_AUTO_REFRESH_INTERVAL,
        limit=MAX_AUTO_REFRESH_COUNT,
        key="streamlit-auto-refresh",
    )
