from __future__ import annotations

from typing import TYPE_CHECKING, List

import streamlit as st

from .. import extra

if TYPE_CHECKING:
    from streamlit.delta_generator import DeltaGenerator


@extra
def stylable_container(key: str, css_styles: str | List[str]) -> "DeltaGenerator":
    """
    Insert a container into your app which you can style using CSS.
    This is useful to style specific elements in your app.

    Args:
        key (str): The key associated with this container. This needs to be unique since all styles will be
            applied to the container with this key.
        css_styles (str | List[str]): The CSS styles to apply to the container elements.
            This can be a single CSS block or a list of CSS blocks.

    Returns:
        DeltaGenerator: A container object. Elements can be added to this container using either the 'with'
            notation or by calling methods directly on the returned object.
    """
    if isinstance(css_styles, str):
        css_styles = [css_styles]

    # Remove unneeded spacing that is added by the style markdown:
    css_styles.append(
        """
> div:first-child {
    margin-bottom: -1rem;
}
"""
    )

    style_text = """
<style>
"""

    for style in css_styles:
        style_text += f"""

div[data-testid="stVerticalBlock"]:has(> div.element-container > div.stMarkdown > div[data-testid="stMarkdownContainer"] > p > span.{key}) {style}

"""

    style_text += f"""
    </style>

<span class="{key}"></span>
"""

    container = st.container()
    container.markdown(style_text, unsafe_allow_html=True)
    return container


def example():
    with stylable_container(
        key="green_button",
        css_styles="""
            button {
                background-color: green;
                color: white;
                border-radius: 20px;
            }
            """,
    ):
        st.button("Green button")

    st.button("Normal button")

    with stylable_container(
        key="container_with_border",
        css_styles="""
            {
                border: 1px solid rgba(49, 51, 63, 0.2);
                border-radius: 0.5rem;
                padding: calc(1em - 1px)
            }
            """,
    ):
        st.markdown("This is a container with a border.")


__title__ = "Styleable Container"
__desc__ = "A container that allows to style its child elements using CSS."
__icon__ = "🎨"
__examples__ = [example]
__author__ = "Lukas Masuch"
__experimental_playground__ = False
