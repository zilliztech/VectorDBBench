from streamlit_extras.switch_page_button import switch_page


def NavToRunTest(st):
    st.header("Run your test")
    st.write("You can set the configs and run your own test.")
    navClick = st.button("Run Your Test &nbsp;&nbsp;>")
    if navClick:
        switch_page("run test")
        
        
def NavToQuriesPerDollar(st):
    st.write("Compare qps with price.")
    navClick = st.button("QP$ (Quries per Dollar) &nbsp;&nbsp;>")
    if navClick:
        switch_page("quries_per_dollar")
        
        
def NavToResults(st, key="nav-to-results"):
    navClick = st.button("< &nbsp;&nbsp;Back to Results", key=key)
    if navClick:
        switch_page("vdb benchmark")
