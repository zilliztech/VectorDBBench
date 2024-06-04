import streamlit as st

try:
    from streamlit import rerun
except ImportError:
    from streamlit import experimental_rerun as rerun

from .. import extra


@extra
def button(*args, key=None, **kwargs):
    """
    Works just like a normal streamlit button, but it remembers its state, so that
    it works as a toggle button. If you click it, it will be pressed, and if you click
    it again, it will be unpressed. Args and output are the same as for
    [st.button](https://docs.streamlit.io/library/api-reference/widgets/st.button)
    """

    if key is None:
        raise ValueError("Must pass key")

    if key not in st.session_state:
        st.session_state[key] = False

    if "type" not in kwargs:
        kwargs["type"] = "primary" if st.session_state[key] else "secondary"

    if st.button(*args, **kwargs):
        st.session_state[key] = not st.session_state[key]
        rerun()

    return st.session_state[key]


def example():
    if button("Button 1", key="button1"):
        if button("Button 2", key="button2"):
            if button("Button 3", key="button3"):
                st.write("All 3 buttons are pressed")


__title__ = "Stateful Button"
__desc__ = "Button that keeps track of its state, so that it works as a toggle button"
__icon__ = "🔛"
__examples__ = [example]
__author__ = "Zachary Blackwood"
__forum_url__ = "https://discuss.streamlit.io/t/how-to-use-multiple-buttons-in-st-columns/35088/4?u=blackary"
__experimental_playground__ = False
