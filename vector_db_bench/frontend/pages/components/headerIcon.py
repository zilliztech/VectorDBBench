def drawHeaderIcon(st):
    st.markdown("""
<div class="headerIconContainer"></div>

<style>
.headerIconContainer {
    position: absolute;
    top: -50px;
    height: 50px;
    width: 100%;
    border-bottom: 2px solid #E8EAEE;
    background-image: url(https://assets.zilliz.com/vdb_benchmark_db790b5387.png);
    background-repeat: no-repeat;
}
</style
""",
        unsafe_allow_html=True,
    )
