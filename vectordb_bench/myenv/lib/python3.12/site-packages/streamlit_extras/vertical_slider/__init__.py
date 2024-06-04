import streamlit as st
from streamlit_vertical_slider import vertical_slider

from .. import extra

vertical_slider = extra(vertical_slider)


def example():
    st.write("## Vertical Slider")
    vertical_slider(
        key="slider",
        default_value=25,
        step=1,
        min_value=0,
        max_value=100,
        track_color="gray",  # optional
        thumb_color="blue",  # optional
        slider_color="red",  # optional
    )


__title__ = "Vertical Slider"  # title of your extra!
__desc__ = (
    "Continuous Vertical Slider with color customizations"  # description of your extra!
)
__icon__ = "🎚"  # give your extra an icon!
__examples__ = [example]  # create some examples to show how cool your extra is!
__author__ = "Carlos D. Serrano"
__pypi_name__ = "streamlit-vertical-slider"
__package_name__ = "streamlit_vertical_slider"
__github_repo__ = "sqlinsights/streamlit-vertical-slider"  # Optional
__forum_url__ = "https://discuss.streamlit.io/t/vertical-slider-component/32229"
__experimental_playground__ = False  # Optional
__stlite__ = False
