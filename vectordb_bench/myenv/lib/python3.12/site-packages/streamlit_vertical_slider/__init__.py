import os
from idna import valid_contextj
import streamlit.components.v1 as components
import streamlit as st
import altair as alt
import streamlit_toggle as sts
from typing import Literal


_RELEASE = True

if not _RELEASE:
    _component_func = components.declare_component(
        "vertical_slider",
        url="http://localhost:3001",
    )
else:
    parent_dir = os.path.dirname(os.path.abspath(__file__))
    build_dir = os.path.join(parent_dir, "frontend/build")
    _component_func = components.declare_component(
        "streamlit_vertical_slider", path=build_dir
    )

shape_types = Literal["circle", "square", "pill"]


def vertical_slider(
    label: str = None,
    key: str = None,
    height: int = 200,
    thumb_shape: shape_types = "circle",
    step: [int, float] = 1,
    default_value: [int, float] = 0,
    min_value: [int, float] = 0,
    max_value: [int, float] = 10,
    track_color: str = "#E5E9F1",
    slider_color: [str, tuple] = "#FF4B4B",
    thumb_color: str = "#FF4B4B",
    value_always_visible: bool = False,
):
    assert thumb_shape in ["circle", "square", "pill"]
    if default_value < min_value:
        default_value = min_value

    if thumb_shape == "circle":
        thumb_height = ("0.75rem",)
        thumb_style = ("inherit",)
        thumb_color = thumb_color
    if thumb_shape == "square":
        thumb_height = ("0.75rem",)
        thumb_style = ("1px",)
        thumb_color = thumb_color
    if thumb_shape == "pill":
        thumb_height = ("0.35rem",)
        thumb_style = ("0.25rem",)
        thumb_color = thumb_color
    if type(slider_color) == tuple:
        gradient_colors = ",".join(slider_color)
        slider_color = f"linear-gradient({gradient_colors})"
        track_color = slider_color
        opacity = 0
    else:
        opacity = 100

    label_display = "auto" if not value_always_visible else True

    vertical_slider_value = _component_func(
        label=label,
        key=key,
        height=height,
        default_value=default_value,
        thumb_shape=thumb_shape,
        step=step,
        min_value=min_value,
        max_value=max_value,
        track_color=track_color,
        thumb_color=thumb_color,
        thumb_height=thumb_height,
        thumb_style=thumb_style,
        slider_color=slider_color,
        opacity=opacity,
        value_always_visible=label_display,
    )
    return vertical_slider_value if vertical_slider_value else default_value


if not _RELEASE:
    import pandas as pd

    st.set_page_config(layout="wide")
    st.subheader("Vertical Slider")

    bottom_cols = st.columns(15)
    with bottom_cols[0]:
        tst = vertical_slider(
            label="Default Style",
            height=200,
            key="test_0",
            default_value=550,
            step=1,
            min_value=0,
            max_value=1500,
            value_always_visible=False,
        )

    with bottom_cols[1]:
        tst = vertical_slider(
            label="Default Style + Always Visible",
            height=200,
            key="test_1",
            default_value=550,
            thumb_shape="circle",
            step=1,
            min_value=0,
            max_value=1500,
            value_always_visible=True,
        )

    with bottom_cols[2]:
        tst = vertical_slider(
            label="Pill Shaped",
            height=200,
            key="test_2",
            default_value=550,
            thumb_shape="pill",
            step=1,
            min_value=0,
            max_value=1500,
            value_always_visible=False,
        )

    with bottom_cols[3]:
        tst = vertical_slider(
            label="Square Shaped",
            height=200,
            key="test_3",
            default_value=550,
            thumb_shape="square",
            step=1,
            min_value=0,
            max_value=1500,
            value_always_visible=False,
        )

    with bottom_cols[4]:
        tst = vertical_slider(
            label="Custom Colors",
            thumb_color="Red",
            track_color="gray",
            slider_color="orange",
            height=200,
            key="test_4",
            default_value=550,
            step=1,
            min_value=0,
            max_value=1500,
            value_always_visible=False,
        )

    with bottom_cols[5]:
        tst = vertical_slider(
            label="Height Control",
            height=400,
            key="test_5",
            default_value=550,
            step=1,
            min_value=0,
            max_value=1500,
            value_always_visible=False,
        )

    with bottom_cols[6]:
        tst = vertical_slider(
            height=400,
            key="test_6",
            default_value=550,
            step=1,
            min_value=0,
            max_value=1500,
            value_always_visible=False,
        )
    with bottom_cols[7]:
        flt_tst = vertical_slider(
            label="Gradient",
            thumb_color="Red",
            track_color="gray",
            slider_color=("blue", "red"),
            height=200,
            key="test_gradient",
            default_value=150.098,
            step=0.09,
            min_value=0,
            max_value=25.500,
            value_always_visible=False,
        )
