from vectordb_bench.frontend.config.styles import HEADER_ICON
from vectordb_bench.frontend.utils import get_page_url_prefix


def drawHeaderIcon(st):
    url_prefix = get_page_url_prefix(st.get_option("server.baseUrlPath"))
    st.markdown(
        f"""
    <a href="{url_prefix}vdbbench" target="_self">
        <div class="headerIconContainer"></div>
    </a>

    <style>
    .headerIconContainer {{
        position: relative;
        top: 0px;
        height: 50px;
        width: 100%;
        border-bottom: 2px solid #E8EAEE;
        background-image: url({HEADER_ICON});
        background-size: contain;
        background-position: left top;
        background-repeat: no-repeat;
        cursor: pointer;
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )
