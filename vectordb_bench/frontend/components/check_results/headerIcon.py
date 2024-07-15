from vectordb_bench.frontend.config.styles import HEADER_ICON


def drawHeaderIcon(st):
    st.markdown(
        f"""
<div class="headerIconContainer"></div>

<style>
.headerIconContainer {{
    position: absolute;
    top: -50px;
    height: 50px;
    width: 100%;
    border-bottom: 2px solid #E8EAEE;
    background-image: url({HEADER_ICON});
    background-repeat: no-repeat;
}}
</style
""",
        unsafe_allow_html=True,
    )
