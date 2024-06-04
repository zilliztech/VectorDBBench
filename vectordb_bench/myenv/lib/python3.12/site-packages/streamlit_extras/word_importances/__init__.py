from typing import List

import streamlit as st

from .. import extra


@extra
def format_word_importances(words: List[str], importances: List[float]) -> str:
    """Adds a background color to each word based on its importance (float from -1 to 1)

    Args:
        words (list): List of words
        importances (list): List of importances (scores from -1 to 1)

    Returns:
        html (str): HTML string with formatted word


    """
    if importances is None or len(importances) == 0:
        return "<td></td>"
    assert len(words) == len(importances), "Words and importances but be of same length"

    tags = ["<td>"]
    for word, importance in zip(words, importances[: len(words)]):
        color = _get_color(importance)
        unwrapped_tag = (
            '<mark style="background-color: {color}; opacity:1.0;             '
            '        line-height:1.75"><font color="black"> {word}            '
            "        </font></mark>".format(color=color, word=word)
        )
        tags.append(unwrapped_tag)
    tags.append("</td>")
    html = "".join(tags)

    return html


def _get_color(importance: float) -> str:
    # clip values to prevent CSS errors (Values should be from [-1,1])
    importance = max(-1, min(1, importance))
    if importance > 0:
        hue = 120
        sat = 75
        lig = 100 - int(50 * importance)
    else:
        hue = 0
        sat = 75
        lig = 100 - int(-40 * importance)
    return "hsl({}, {}%, {}%)".format(hue, sat, lig)


def example():
    text = (
        "Streamlit Extras is a library to help you discover, learn, share and"
        " use Streamlit bits of code!"
    )
    html = format_word_importances(
        words=text.split(),
        importances=(0.1, 0.2, 0, -1, 0.1, 0, 0, 0.2, 0.3, 0.8, 0.9, 0.6, 0.3, 0.1, 0, 0, 0),  # fmt: skip
    )
    st.write(html, unsafe_allow_html=True)


__title__ = "Word importances"
__desc__ = "Highlight words based on their importances. Inspired from captum library."
__icon__ = "❗"
__examples__ = [example]
__author__ = "Arnaud Miribel"
