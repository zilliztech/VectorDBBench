from streamlit_embedcode import (
    codepen_snippet,
    github_gist,
    gitlab_snippet,
    ideone_snippet,
    pastebin_snippet,
    tagmycode_snippet,
)

from .. import extra

codepen_snippet = extra(codepen_snippet)
github_gist = extra(github_gist)
gitlab_snippet = extra(gitlab_snippet)
ideone_snippet = extra(ideone_snippet)
pastebin_snippet = extra(pastebin_snippet)
tagmycode_snippet = extra(tagmycode_snippet)


def example_github():
    github_gist(
        "https://gist.github.com/randyzwitch/be8c5e9fb5b8e7b046afebcac12e5087/",
        width=700,
        height=400,
    )


def example_gitlab():
    gitlab_snippet(
        "https://gitlab.com/snippets/1995463",
        width=700,
        height=200,
    )


def example_codepen(codepen_snippet):
    codepen_snippet(
        "https://codepen.io/randyzwitch/pen/GRrYrBw",
        width=700,
        height=400,
    )


def example_ideone(ideone_snippet):
    ideone_snippet(
        "https://ideone.com/5V7XZ6",
        width=700,
        height=400,
    )


def example_pastebin(pastebin_snippet):
    pastebin_snippet(
        "https://pastebin.com/8QZ7YjYD",
        width=700,
        height=400,
    )


def example_tagmycode(tagmycode_snippet):
    tagmycode_snippet(
        "https://tagmycode.com/snippet/1038",
        width=700,
        height=400,
    )


__title__ = "Embed code"
__desc__ = "Embed code from various platforms (Gists, snippets...)"
__icon__ = "📋"
__examples__ = {
    example_github: [github_gist],
    example_gitlab: [gitlab_snippet],
}
__author__ = "randyzwitch"
__github_repo__ = "randyzwitch/streamlit-embedcode"
__pypi_name__ = "streamlit-embedcode"
__package_name__ = "streamlit_embedcode"
__experimental_playground__ = False
