from vectordb_bench.frontend.utils import get_page_url_prefix


def NavToRunTest(st):
    st.subheader("Run your test")
    st.write("You can set the configs and run your own test.")
    navClick = st.button("Run Your Test &nbsp;&nbsp;>")
    if navClick:
        st.switch_page("pages/run_test.py")


def NavToQuriesPerDollar(st):
    st.subheader("Compare qps with price.")
    navClick = st.button("QP$ (Quries per Dollar) &nbsp;&nbsp;>")
    if navClick:
        st.switch_page("pages/quries_per_dollar.py")


ZILLIZ_CLOUD_SIGNUP_URL = (
    "https://cloud.zilliz.com/signup"
    "?utm_source=github&utm_medium=referral&utm_campaign=vdbbench&utm_content=app_results"
)
PUBLIC_LEADERBOARD_URL = "https://zilliz.com/vdbbench-leaderboard"


def NavToExternalLinks(st):
    st.markdown(
        f"[View the public leaderboard]({PUBLIC_LEADERBOARD_URL})  \n"
        f"[Run these cases on Zilliz Cloud]({ZILLIZ_CLOUD_SIGNUP_URL})"
    )


def NavToResults(st, key="nav-to-results"):
    navClick = st.button("< &nbsp;&nbsp;Back to Results", key=key)
    if navClick:
        st.switch_page("pages/results.py")


def NavToPages(st):
    options = [
        {"name": "Run Test", "link": "run_test"},
        {"name": "Results", "link": "results"},
        {"name": "Qps & Recall", "link": "qps_recall"},
        {"name": "Full Text Search", "link": "full_text_search"},
        {"name": "Quries Per Dollar", "link": "quries_per_dollar"},
        {"name": "Concurrent", "link": "concurrent"},
        {"name": "Label Filter", "link": "label_filter"},
        {"name": "Int Filter", "link": "int_filter"},
        {"name": "Streaming", "link": "streaming"},
        {"name": "Tables", "link": "tables"},
        {"name": "Custom Dataset", "link": "custom"},
    ]

    url_prefix = get_page_url_prefix(st.get_option("server.baseUrlPath"))
    html = ""
    for i, option in enumerate(options):
        html += f'<a href="{url_prefix}{option["link"]}" target="_self" style="text-decoration: none; padding: 0.1px 0.2px;">{option["name"]}</a>'
        if i < len(options) - 1:
            html += '<span style="color: #888; margin: 0 5px;">|</span>'
    st.markdown(html, unsafe_allow_html=True)
