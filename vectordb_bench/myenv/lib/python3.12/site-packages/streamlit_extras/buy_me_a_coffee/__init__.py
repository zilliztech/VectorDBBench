from typing import Literal

import streamlit as st
from streamlit.components.v1 import html

from .. import extra

Font = Literal[
    "Cookie",
    "Lato",
    "Arial",
    "Comic",
    "Inter",
    "Bree",
    "Poppins",
]


@extra
def button(
    username: str,
    floating: bool = True,
    text: str = "Buy me a coffee",
    emoji: str = "",
    bg_color: str = "#FFDD00",
    font: Font = "Cookie",
    font_color: str = "#000000",
    coffee_color: str = "#000000",
    width: int = 220,
):
    """
    Display a button which links to your Buy Me a Coffee page.

    Args:
        username (str): Buy Me a Coffee username
        floating (bool, optional): Whether the button should be floating. Defaults to True.
        text (str, optional): Text to show on the button. Defaults to "Buy me a coffee".
        emoji (str, optional): Emoji to show on the button. Defaults to "".
        bg_color (str, optional): Background of the button. Defaults to "#FFDD00".
        font (Font, optional): Font of the button. Defaults to "Cookie".
        font_color (str, optional): Font color. Defaults to "#000000".
        coffee_color (str, optional): Coffee icon color. Defaults to "#000000".
        width (int, optional): Width of the button. Defaults to 220.
    """ """"""
    button = f"""
        <script type="text/javascript"
            src="https://cdnjs.buymeacoffee.com/1.0.0/button.prod.min.js"
            data-name="bmc-button"
            data-slug="{username}"
            data-color="{bg_color}"
            data-emoji="{emoji}"
            data-font="{font}"
            data-text="{text}"
            data-outline-color="#000000"
            data-font-color="{font_color}"
            data-coffee-color="{coffee_color}" >
        </script>
    """

    html(button, height=70, width=width)

    if floating:
        st.markdown(
            f"""
            <style>
                iframe[width="{width}"] {{
                    position: fixed;
                    bottom: 60px;
                    right: 40px;
                }}
            </style>
            """,
            unsafe_allow_html=True,
        )


def example():
    button(username="fake-username", floating=False, width=221)


__title__ = "Buy Me a Coffee Button"  # title of your extra!
__desc__ = "Adds a floating button which links to your Buy Me a Coffee page"  # description of your extra!
__icon__ = "☕"  # give your extra an icon!
__examples__ = [example]  # create some examples to show how cool your extra is!
__author__ = "Zachary Blackwood"
__forum_url__ = "https://discuss.streamlit.io/t/how-to-add-a-floating-widget-in-streamlit-app/33165/2?u=blackary"
__experimental_playground__ = True  # Optional
