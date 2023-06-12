from streamlit_extras.switch_page_button import switch_page


def NavToRunTest(st):
    st.header("Run your test")
    st.write("You can set the configs and run your own test.")
    navClick = st.button("[Run Your Test &nbsp;&nbsp;>](http://github.com)")
    if navClick:
        switch_page("run test")
