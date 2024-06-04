from __future__ import annotations

from typing import Literal, get_args

import streamlit as st
from htbuilder import a, img

from .. import extra

_SUPPORTED_TYPES = Literal["pypi", "streamlit", "github", "twitter", "buymeacoffee"]


@extra
def badge(type: _SUPPORTED_TYPES, name: str | None = None, url: str | None = None):
    """Easily create a visual badge for PyPI, GitHub, Streamlit Cloud or other social platforms.

    Args:
        type (str): Badge type. Can be "pypi", "github", "streamlit", "twitter" or "buymeacoffee"
        name (str): Name of the PyPI package, GitHub repository, Twitter's username or BuyMeaCoffee Creator's page name.
                    Mandatory when using type="pypi", type="twitter" & type="buymeacoffee"
        url (str): URL of the Streamlit Cloud app. Mandatory when using type="streamlit"
    """

    assert type, "Type must be given!"

    assert type in get_args(_SUPPORTED_TYPES), (
        f"Input type '{type}' is not supported! Supported types are"
        f" {get_args(_SUPPORTED_TYPES)}"
    )

    badge_html = None

    if type == "pypi":
        assert name, "You must give a valid PyPI package name!"
        badge_html = str(
            a(href=f"https://pypi.org/project/{name}")(
                img(src=f"https://badge.fury.io/py/{name}.svg")
            )
        )

    if type == "streamlit":
        assert url, "You must provide a valid URL for the Streamlit app"
        badge_html = str(
            a(href=url)(
                img(
                    src="https://static.streamlit.io/badges/streamlit_badge_black_white.svg"
                )
            )
        )

    if type == "github":
        assert name, (
            "You must give a valid GitHub repository name! Something like"
            " 'author/name'"
        )
        badge_html = str(
            a(href=f"https://github.com/{name}")(
                img(
                    src=f"https://img.shields.io/github/stars/{name}.svg?style=social&label=Star&maxAge=2592000"
                )
            )
        )

    if type == "twitter":
        assert name, "You must provide a valid twitter username"
        badge_html = str(
            a(href=f"https://twitter.com/intent/follow?screen_name={name}")(
                img(
                    src=f"https://img.shields.io/twitter/follow/{name}?style=social&logo=twitter"
                )
            )
        )

    if type == "buymeacoffee":
        assert name, "You must provide a valid Buy-Me-a-Coffee page username"
        badge_html = str(
            a(href=f"https://www.buymeacoffee.com/{name}")(
                img(
                    src="https://img.shields.io/badge/Buy%20me%20a%20coffee--yellow.svg?logo=buy-me-a-coffee&logoColor=orange&style=social"
                )
            )
        )

    if badge_html is not None:
        st.write(badge_html, unsafe_allow_html=True)


def example_pypi():
    badge(type="pypi", name="plost")
    badge(type="pypi", name="streamlit")


def example_streamlit():
    badge(type="streamlit", url="https://plost.streamlitapp.com")


def example_github():
    badge(type="github", name="streamlit/streamlit")


def example_twitter():
    badge(type="twitter", name="streamlit")


def example_buymeacoffee():
    badge(type="buymeacoffee", name="andfanilo")


__title__ = "Badges"
__desc__ = "Create custom badges (e.g. PyPI, Streamlit Cloud, GitHub, Twitter, Buy Me a Coffee)"
__icon__ = "🏷️"
__examples__ = [
    example_pypi,
    example_streamlit,
    example_github,
    example_twitter,
    example_buymeacoffee,
]
__author__ = "Arnaud Miribel, ShruAgarwal"
__experimental_playground__ = True
