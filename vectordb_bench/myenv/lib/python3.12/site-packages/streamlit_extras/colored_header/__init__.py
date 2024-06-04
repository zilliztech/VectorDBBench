"""Add colorful headers to your Streamlit app."""
import itertools
from typing import Literal

import streamlit as st

from .. import extra


def color(name):
    """Returns a color from the streamlit color palette, e.g. red-100, as hex."""
    try:
        hue, intensity = name.rsplit("-", 1)
    except (ValueError, KeyError):
        st.error(
            "Input color_name must contain a name (red, orange, ...) and"
            " intensity (10, 20, ... 100) e.g. 'red-70'"
        )
        st.stop()
    return ST_COLOR_PALETTE[hue][intensity]


ST_COLOR_PALETTE = {
    "red": {
        "100": "#7d353b",
        "90": "#bd4043",
        "80": "#ff2b2b",
        "70": "#ff4b4b",
        "60": "#ff6c6c",
        "50": "#ff8c8c",
        "40": "#ffabab",
        "30": "#ffc7c7",
        "20": "#ffdede",
        "10": "#fff0f0",
    },
    "orange": {
        "100": "#d95a00",
        "90": "#ed6f13",
        "80": "#ff8700",
        "70": "#ffa421",
        "60": "#ffbd45",
        "50": "#ffd16a",
        "40": "#ffe08e",
        "30": "#ffecb0",
        "20": "#fff6d0",
        "10": "#fffae8",
    },
    "yellow": {
        "100": "#dea816",
        "90": "#edbb16",
        "80": "#faca2b",
        "70": "#ffe312",
        "60": "#fff835",
        "50": "#ffff59",
        "40": "#ffff7d",
        "30": "#ffffa0",
        "20": "#ffffc2",
        "10": "#ffffe1",
    },
    "green": {
        "100": "#177233",
        "90": "#158237",
        "80": "#09ab3b",
        "70": "#21c354",
        "60": "#3dd56d",
        "50": "#5ce488",
        "40": "#7defa1",
        "30": "#9ef6bb",
        "20": "#c0fcd3",
        "10": "#dffde9",
    },
    "blue-green": {
        "100": "#246e69",
        "90": "#2c867c",
        "80": "#29b09d",
        "70": "#00d4b1",
        "60": "#20e7c5",
        "50": "#45f4d5",
        "40": "#6bfde3",
        "30": "#93ffee",
        "20": "#bafff7",
        "10": "#dcfffb",
    },
    "light-blue": {
        "100": "#15799e",
        "90": "#0d8cb5",
        "80": "#00a4d4",
        "70": "#00c0f2",
        "60": "#24d4ff",
        "50": "#4be4ff",
        "40": "#73efff",
        "30": "#9af8ff",
        "20": "#bffdff",
        "10": "#e0feff",
    },
    "blue": {
        "100": "#004280",
        "90": "#0054a3",
        "80": "#0068c9",
        "70": "#1c83e1",
        "60": "#3d9df3",
        "50": "#60b4ff",
        "40": "#83c9ff",
        "30": "#a6dcff",
        "20": "#c7ebff",
        "10": "#e4f5ff",
    },
    "violet": {
        "100": "#3f3163",
        "90": "#583f84",
        "80": "#6d3fc0",
        "70": "#803df5",
        "60": "#9a5dff",
        "50": "#b27eff",
        "40": "#c89dff",
        "30": "#dbbbff",
        "20": "#ebd6ff",
        "10": "#f5ebff",
    },
    "gray": {
        "100": "#0e1117",
        "90": "#262730",
        "80": "#555867",
        "70": "#808495",
        "60": "#a3a8b8",
        "50": "#bfc5d3",
        "40": "#d5dae5",
        "30": "#e6eaf1",
        "20": "#f0f2f6",
        "10": "#fafafa",
    },
}


HEADER_COLOR_CYCLE = itertools.cycle(
    [
        "light-blue-70",
        "orange-70",
        "blue-green-70",
        "blue-70",
        "violet-70",
        "red-70",
        "green-70",
        "yellow-80",
    ]
)

_SUPPORTED_COLORS = Literal[
    "light-blue-70",
    "orange-70",
    "blue-green-70",
    "blue-70",
    "violet-70",
    "red-70",
    "green-70",
    "yellow-80",
]


@extra
def colored_header(
    label: str = "Nice title",
    description: str = "Cool description",
    color_name: _SUPPORTED_COLORS = "red-70",
):
    """
    Shows a header with a colored underline and an optional description.

    Args:
        label (str, optional): Header label. Defaults to "Nice title".
        description (str, optional): Description shown under the header. Defaults to "Cool description".
        color_name (_SUPPORTED_COLORS, optional): Color of the underline. Defaults to "red-70".
            Supported colors are "light-blue-70", "orange-70", "blue-green-70", "blue-70", "violet-70",
            "red-70", "green-70", "yellow-80".
    """
    if color_name is None:
        color_name = next(HEADER_COLOR_CYCLE)
    st.subheader(label)
    st.write(
        f'<hr style="background-color: {color(color_name)}; margin-top: 0;'
        ' margin-bottom: 0; height: 3px; border: none; border-radius: 3px;">',
        unsafe_allow_html=True,
    )
    if description:
        st.caption(description)


def example():
    colored_header(
        label="My New Pretty Colored Header",
        description="This is a description",
        color_name="violet-70",
    )


__title__ = "Color ya Headers"
__desc__ = """This function makes headers much prettier in Streamlit.
           **Note that this now accessible in native Streamlit in st.header
           with parameter `divider`!**"""
__icon__ = "🖌️"
__examples__ = [example]
__author__ = "Johannes Rieke / Tyler Richards"
__experimental_playground__ = True
