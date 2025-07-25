from streamlit_extras.switch_page_button import switch_page


def NavToRunTest(st):
    st.subheader("Run your test")
    st.write("You can set the configs and run your own test.")
    navClick = st.button("Run Your Test &nbsp;&nbsp;>")
    if navClick:
        switch_page("run test")


def NavToQuriesPerDollar(st):
    st.subheader("Compare qps with price.")
    navClick = st.button("QP$ (Quries per Dollar) &nbsp;&nbsp;>")
    if navClick:
        switch_page("quries_per_dollar")


def NavToResults(st, key="nav-to-results"):
    navClick = st.button("< &nbsp;&nbsp;Back to Results", key=key)
    if navClick:
        switch_page("results")


def NavToPages(st):
    options = [
        {"name": "Run Test", "link": "run_test"},
        {"name": "Results", "link": "results"},
        {"name": "Quries Per Dollar", "link": "quries_per_dollar"},
        {"name": "Concurrent", "link": "concurrent"},
        {"name": "Label Filter", "link": "label_filter"},
        {"name": "Int Filter", "link": "int_filter"},
        {"name": "Streaming", "link": "streaming"},
        {"name": "Tables", "link": "tables"},
        {"name": "Custom Dataset", "link": "custom"},
    ]

    html = ""
    for i, option in enumerate(options):
        html += f'<a href="/{option["link"]}" target="_self" style="text-decoration: none; padding: 0.1px 0.2px;">{option["name"]}</a>'
        if i < len(options) - 1:
            html += '<span style="color: #888; margin: 0 5px;">|</span>'
    st.markdown(html, unsafe_allow_html=True)
