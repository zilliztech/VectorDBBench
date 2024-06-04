from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

from .. import extra

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator


@extra
def bottom() -> DeltaGenerator:
    """
    Insert a multi-element container that sticks to the bottom of the app.
    """
    if hasattr(st, "_bottom"):
        return st._bottom
    else:
        raise Exception(
            "The bottom container is not supported in this Streamlit version."
        )


def example():
    st.write("This is the main container")

    with bottom():
        st.write("This is the bottom container")
        st.text_input("This is a text input in the bottom container")


__title__ = "Bottom Container"
__desc__ = "A multi-element container that sticks to the bottom of the app."
__icon__ = "⬇️"
__examples__ = [example]
__author__ = "Lukas Masuch"
__experimental_playground__ = False
