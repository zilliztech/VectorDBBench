from streamlit_extras.switch_page_button import switch_page


def NavToRunTest(st):
    st.header("Run your test")
    st.write("You can set the configs and run your own test.")
    navClick = st.button("Run Your Test &nbsp;&nbsp;>")
    if navClick:
        switch_page("run test")
        
        
def NavToQPSWithPrice(st):
    navClick = st.button("QPS with Price &nbsp;&nbsp;>")
    if navClick:
        switch_page("qps with price")
        
        
def NavToResults(st):
    navClick = st.button("< &nbsp;&nbsp;Back to Results")
    if navClick:
        switch_page("vdb benchmark")
