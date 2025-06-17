from vectordb_bench.frontend.config.styles import HEADER_ICON


def drawHeaderIcon(st):
    st.markdown(
        f"""
    <a href="/vdbbench" target="_self">
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
