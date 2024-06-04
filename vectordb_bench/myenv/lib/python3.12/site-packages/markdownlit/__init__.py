import markdown
import streamlit as st
from markdown.extensions import Extension

# from .extensions.align import AlignProcessor
from .extensions.arrow import ARROW_RE, ArrowProcessor
from .extensions.at_sign import AT_SIGN_RE, AtSignProcessor
from .extensions.color import COLOR_RE, ColorProcessor
from .extensions.double_dash import DOUBLE_DASH_RE, DoubleDashProcessor


def css(body: str) -> None:
    """Apply custom CSS in a Streamlit app.
    Warning: this will inevitably show some blank vertical space wherever this function is being used.

    Args:
        body (str): CSS string to apply
    """
    st.write("<style>" + body + "</style>", unsafe_allow_html=True)


STYLE_HTML = """
<style>
a:hover {
    background-color: rgba(.7, .7, .7, .05);
}
</style>
"""


def md(body: str, extensions: list, extension_configs: dict = dict()) -> None:
    """Display Markdown using the Python markdown library.
    Under the hoods, markdown library converts Markdown to HTML and we use
    st.write(..., unsafe_allow_html=True) to show that HTML.

    Args:
        body (str): Original markdown string
        extensions (list): List of Markdown extensions to support
        extension_configs (dict, optional): Configs for the extensions. Defaults to dict().
    """
    st.write(
        markdown.markdown(
            body,
            extensions=extensions,
            extension_configs=extension_configs,
        )
        + STYLE_HTML,
        unsafe_allow_html=True,
    )


class MarkdownLitExtension(Extension):
    def extendMarkdown(self, md):
        """This is a method to register a bunch of processors into a single Markdown extension."""

        md.inlinePatterns.register(
            item=ArrowProcessor(ARROW_RE, md),
            name="arrow",
            priority=1_000,
        )

        md.inlinePatterns.register(
            item=DoubleDashProcessor(DOUBLE_DASH_RE, md),
            name="double_dash",
            priority=1_000,
        )

        md.inlinePatterns.register(
            item=AtSignProcessor(AT_SIGN_RE, md),
            name="at_sign",
            priority=1_000,
        )

        md.inlinePatterns.register(
            item=ColorProcessor(COLOR_RE, md),
            name="color",
            priority=1_000,
        )

        # md.inlinePatterns.register(
        #     item=AlignProcessor(ALIGN_RE, md),
        #     name="align",
        #     priority=1_000,
        # )


def mdlit(body: str) -> None:
    md(
        body=body,
        extensions=[
            MarkdownLitExtension(),  # Includes at_sign, color, arrow and double_dash
            "pymdownx.details",  # Includes collapsible content
            "pymdownx.tasklist",  # Added for consistency with st.markdown
            "fenced_code",  # Added for consistency with st.markdown
        ],
    )
