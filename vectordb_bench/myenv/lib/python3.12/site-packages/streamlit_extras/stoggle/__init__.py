import streamlit as st
from htbuilder import details, div, p, styles
from htbuilder import summary as smry

from .. import extra


@extra
def stoggle(summary: str, content: str):
    """
    Displays a toggle widget in Streamlit

    Args:
        summary (str): Summary of the toggle (always shown)
        content (str): Content shown after toggling
    """

    st.write(
        str(
            div(
                style=styles(
                    line_height=1.8,
                )
            )(details(smry(summary), p(content)))
        ),
        unsafe_allow_html=True,
    )


def example():
    stoggle(
        "Click me!",
        """🥷 Surprise! Here's some additional content""",
    )


__title__ = "Toggle button"
__desc__ = "Toggle button just like in Notion!"
__icon__ = "➡️"
__examples__ = [example]
__author__ = "Arnaud Miribel"
__github_repo__ = "arnaudmiribel/stoggle"
__streamlit_cloud_url__ = "http://stoggle.streamlitapp.com"
__experimental_playground__ = True
