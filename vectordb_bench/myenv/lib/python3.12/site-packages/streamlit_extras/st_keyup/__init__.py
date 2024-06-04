import streamlit as st
from st_keyup import st_keyup

from .. import extra

st_keyup = extra(st_keyup)


def example():
    st.write("## Notice how the output doesn't update until you hit enter")
    out = st.text_input("Normal text input")
    st.write(out)
    st.write("## Notice how the output updates with every key you press")
    out2 = st_keyup("Keyup input")
    st.write(out2)


def example_with_debounce():
    st.write("## Notice how the output doesn't update until 500ms has passed")
    out = st_keyup("Keyup with debounce", debounce=500)
    st.write(out)


__title__ = "Keyup text input"
__desc__ = "A text input that updates with every key press"
__icon__ = "🔑"
__author__ = "Zachary Blackwood"
__examples__ = [example, example_with_debounce]
__github_repo__ = "blackary/streamlit-keyup"
__pypi_name__ = "streamlit-keyup"
__package_name__ = "st_keyup"
__experimental_playground__ = False
__stlite__ = False
