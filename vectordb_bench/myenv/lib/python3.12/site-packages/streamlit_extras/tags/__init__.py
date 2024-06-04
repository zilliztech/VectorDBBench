from __future__ import annotations

from textwrap import dedent

import streamlit as st

from .. import extra

# mypy: ignore-errors
TAGGER_COLOR_PALETTE = {
    "lightblue": "#00c0f2",
    "orange": "#ffa421",
    "bluegreen": "#00d4b1",
    "blue": "#1c83e1",
    "violet": "#803df5",
    "red": "#ff4b4b",
    "green": "#21c354",
    "yellow": "#faca2b",
}

_DEFAULT_COLOR = "#808495"


def _get_html(
    content: str,
    tags: list[str],
    color_name: list[str] | str | None = None,
) -> str:
    tags_html = content + " "
    for i in range(len(tags)):
        if color_name is None:
            color = _DEFAULT_COLOR
        elif isinstance(color_name, list):
            color = TAGGER_COLOR_PALETTE.get(color_name[i], color_name[i])
        elif isinstance(color_name, str):
            color = TAGGER_COLOR_PALETTE.get(color_name, color_name)
        else:
            raise ValueError(
                f"color_name must be a list or a string or None. "
                f"color_name = {color_name}, type(color_name) = {type(color_name)}"
            )

        tags_html += dedent(
            f"""
            <span style="display:inline-block;
            background-color: {color};
            padding: 0.1rem 0.5rem;
            font-size: 14px;
            font-weight: 400;
            color:white;
            margin: 5px;
            border-radius: 1rem;">{tags[i]}</span>
            """
        ).strip()

    return tags_html


@extra
def tagger_component(
    content: str,
    tags: list[str],
    color_name: list[str] | str | None = None,
):
    """
    Displays tags next to your text.

    Args:
        content (str): Content to be tagged
        tags (list): A list of tags to be displayed next to the content
        color_name: A list or a string that indicates the color of tags.
            Choose from lightblue, orange, bluegreen, blue, violet, red, green, yellow
    """
    if isinstance(color_name, list) and len(color_name) != len(tags):
        raise ValueError(
            f"color_name must be the same length as tags. "
            f"len(color_name) = {len(color_name)}, len(tags) = {len(tags)}"
        )

    tags_html = _get_html(content, tags, color_name)

    st.write(tags_html, unsafe_allow_html=True)


def example():
    tagger_component("Here is a feature request", ["p2", "🚩triaged", "backlog"])
    tagger_component(
        "Here are colored tags",
        ["turtle", "rabbit", "lion"],
        color_name=["blue", "orange", "lightblue"],
    )
    tagger_component(
        "Annotate the feature",
        ["hallucination"],
        color_name=["blue"],
    )


def test_invalid_color_length():
    import pytest

    with pytest.raises(ValueError):
        tagger_component(
            "Here is a feature request",
            ["p2", "🚩triaged", "backlog"],
            color_name=["blue"],
        )


def test_color_html_list_in_palette():
    output = _get_html("foo", ["bar"], color_name=["blue"])
    assert (
        output
        == dedent(
            f"""
        foo <span style="display:inline-block;
        background-color: {TAGGER_COLOR_PALETTE['blue']};
        padding: 0.1rem 0.5rem;
        font-size: 14px;
        font-weight: 400;
        color:white;
        margin: 5px;
        border-radius: 1rem;">bar</span>
        """
        ).strip()
    )


def test_color_html_list_not_in_palette():
    output = _get_html("foo", ["bar"], color_name=["pink"])
    assert (
        output
        == dedent(
            """
        foo <span style="display:inline-block;
        background-color: pink;
        padding: 0.1rem 0.5rem;
        font-size: 14px;
        font-weight: 400;
        color:white;
        margin: 5px;
        border-radius: 1rem;">bar</span>
        """
        ).strip()
    )


def test_color_html_str():
    output = _get_html("foo", ["bar"], color_name="blue")

    assert (
        output
        == dedent(
            f"""
        foo <span style="display:inline-block;
        background-color: {TAGGER_COLOR_PALETTE['blue']};
        padding: 0.1rem 0.5rem;
        font-size: 14px;
        font-weight: 400;
        color:white;
        margin: 5px;
        border-radius: 1rem;">bar</span>
        """
        ).strip()
    )


def test_color_html_str_multiple_tags():
    output = _get_html("foo", ["bar", "foo"], color_name="blue")
    print(output)
    assert (
        output
        == dedent(
            f"""
        foo <span style="display:inline-block;
        background-color: {TAGGER_COLOR_PALETTE['blue']};
        padding: 0.1rem 0.5rem;
        font-size: 14px;
        font-weight: 400;
        color:white;
        margin: 5px;
        border-radius: 1rem;">bar</span><span style="display:inline-block;
        background-color: {TAGGER_COLOR_PALETTE['blue']};
        padding: 0.1rem 0.5rem;
        font-size: 14px;
        font-weight: 400;
        color:white;
        margin: 5px;
        border-radius: 1rem;">foo</span>
        """
        ).strip()
    )


def test_no_color_html():

    output = _get_html("foo", ["bar"])

    assert (
        output
        == dedent(
            f"""
        foo <span style="display:inline-block;
        background-color: {_DEFAULT_COLOR};
        padding: 0.1rem 0.5rem;
        font-size: 14px;
        font-weight: 400;
        color:white;
        margin: 5px;
        border-radius: 1rem;">bar</span>
        """
        ).strip()
    )


__title__ = "Tags"
__desc__ = "Display tags like github issues!"
__icon__ = "🔖"
__examples__ = [example]
__author__ = "Maggie Liu"
__experimental_playground__ = False
__tests__ = [
    test_invalid_color_length,
    test_color_html_list_in_palette,
    test_color_html_list_not_in_palette,
    test_color_html_str,
    test_color_html_str_multiple_tags,
    test_no_color_html,
]
