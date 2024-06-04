import streamlit as st
from htbuilder import a, img, span
from validators import url as validate_url

from .. import extra

GITHUB_ICON = "https://cdn-icons-png.flaticon.com/512/25/25231.png"
NOTION_ICON = "https://upload.wikimedia.org/wikipedia/commons/4/45/Notion_app_logo.png"
TWITTER_ICON = "https://seeklogo.com/images/T/twitter-icon-circle-blue-logo-0902F48837-seeklogo.com.png"
STREAMLIT_ICON = "https://streamlit.io/images/brand/streamlit-mark-color.png"

STYLE_HTML = """
<style>
a:hover {
    background-color: rgba(.7, .7, .7, .05);
}
</style>
"""


@extra
def mention(label: str, url: str, icon: str = "🔗", write: bool = True):
    """Mention a link with a label and icon.

    Args:
        label (str): Label to use in the mention
        icon (str): Icon to use. Can be an emoji or a URL. Default '🔗'
        url (str): Target URL of the mention
        write (bool): Writes the mention directly. If False, returns the raw HTML.
                      Useful if mention is used inline.
    """

    if icon.lower() == "github":
        icon = GITHUB_ICON
    elif icon.lower() == "notion":
        icon = NOTION_ICON
    elif icon.lower() == "twitter":
        icon = TWITTER_ICON
    elif icon.lower() == "streamlit":
        icon = STREAMLIT_ICON

    if validate_url(icon):
        icon_html = img(
            src=icon,
            style="width:1em;height:1em;vertical-align:-0.15em;border-radius:3px;margin-right:0.3em",
        )
    else:
        icon_html = icon + "  "

    mention_html = a(
        contenteditable=False,
        href=url,
        rel="noopener noreferrer",
        style="color:inherit;text-decoration:inherit; height:auto!important",
        target="_blank",
    )(
        span(),
        icon_html,
        span(
            style=(
                "border-bottom:0.05em solid"
                " rgba(55,53,47,0.25);font-weight:500;flex-shrink:0"
            )
        )(label),
        span(),
    )

    html = STYLE_HTML + str(mention_html)
    if write:
        st.write(html, unsafe_allow_html=True)
    else:
        return html


def example_1():
    mention(
        label="An awesome Streamlit App",
        icon="streamlit",  # Some icons are available... like Streamlit!
        url="https://extras.streamlitapp.com",
    )


def example_2():
    mention(
        label="streamlit-extras",
        icon="🪢",  # You can also just use an emoji
        url="https://github.com/arnaudmiribel/streamlit-extras",
    )


def example_3():
    mention(
        label="example-app-cv-model",
        icon="github",  # GitHub is also featured!
        url="https://github.com/streamlit/example-app-cv-model",
    )


def example_4():
    mention(
        label="That page somewhere in Notion",
        icon="notion",  # Notion is also featured!
        url="https://notion.so",
    )


def example_5():
    inline_mention = mention(
        label="streamlit",
        icon="twitter",  # Twitter is also featured!
        url="https://www.twitter.com/streamlit",
        write=False,
    )

    st.write(
        f"Here's how to use the mention inline:  {inline_mention}. Cool" " right?",
        unsafe_allow_html=True,
    )


__title__ = "Mentions"
__desc__ = "Create nice links with icons, like Notion mentions!"
__icon__ = "🫵"
__examples__ = [
    example_1,
    example_2,
    example_3,
    example_4,
    example_5,
]
__author__ = "Arnaud Miribel"
__experimental_playground__ = True
