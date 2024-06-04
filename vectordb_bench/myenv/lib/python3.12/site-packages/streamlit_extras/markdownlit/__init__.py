import textwrap

import streamlit as st
from markdownlit import mdlit

from .. import extra

mdlit = extra(mdlit)


def mdlit_example(code: str) -> None:
    left, right = st.columns((2, 25))
    with left:
        mdlit("[gray]<small style='float:right'>In:</small>[/gray]")
    right.code(
        f"""mdlit("{code}")""",
        "python",
    )
    left, right = st.columns((2, 25))
    with left:
        mdlit("[gray]<small style='float:right'>Out:</small>[/gray]")
    with right:
        mdlit(code)


def example_link_and_colors():
    mdlit(
        """Tired from [default links](https://extras.streamlit.app)?
    Me too! Discover Markdownlit's `@()` operator. Just insert a link and it
    will figure a nice icon and label for you!
    Example: @(https://extras.streamlit.app)... better, right? You can
    also @(🍐)(manually set the label if you want)(https://extras.streamlit.app)
    btw, and play with a [red]beautiful[/red] [blue]set[/blue] [orange]of[/orange]
    [violet]colors[/violet]. Another perk is those beautiful arrows -> <-
    """
    )


def example_collapsible_content():
    mdlit(
        textwrap.dedent(
            """
    ??? Bonus
        @(🎁)(A very insightful tutorial)(https://www.youtube.com/watch?v=dQw4w9WgXcQ)
    """
        )
    )


__func__ = mdlit
__title__ = "Markdownlit"
__desc__ = """markdownlit adds a set of lit Markdown commands for your Streamlit apps!
    **Note you can now use colored text in markdown with native Streamlit, check
    out [the docs](https://docs.streamlit.io/library/api-reference/text/st.markdown)!**"""
__icon__ = "〽️"
__examples__ = [
    example_link_and_colors,
    example_collapsible_content,
]
__author__ = "Arnaud Miribel"
__github_repo__ = "arnaudmiribel/markdownlit"
__pypi_name__ = "markdownlit"
__package_name__ = "markdownlit"
__experimental_playground__ = True
