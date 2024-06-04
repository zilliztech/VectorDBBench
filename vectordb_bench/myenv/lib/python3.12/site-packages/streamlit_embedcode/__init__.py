import streamlit as st
import streamlit.components.v1 as components


def _clean_link(link):
    """Strip trailing slash if present on link.

    Parameters
    ----------
    link : str
        URL from code sharing website

    Returns
    -------
    str
        Returns value of `link` without trailing slash.

    Example
    -------
    >>> _clean_link("https://gist.github.com/randyzwitch/be8c5e9fb5b8e7b046afebcac12e5087/")
    'https://gist.github.com/randyzwitch/be8c5e9fb5b8e7b046afebcac12e5087'

    >>> _clean_link("https://gist.github.com/randyzwitch/be8c5e9fb5b8e7b046afebcac12e5087")
    'https://gist.github.com/randyzwitch/be8c5e9fb5b8e7b046afebcac12e5087'
    """

    return link[:-1] if link[-1] == "/" else link


def github_gist(link, height=600, width=950, scrolling=True):
    """Embed a GitHub gist.

    Parameters
    ----------
    link : str
        URL from https://gist.github.com/
    height: int
        Height of the resulting iframe
    width: int
        Width of the resulting iframe
    scrolling: bool
        If content is larger than iframe size, provide scrollbars?
    
    Example
    -------
    >>> github_gist("https://gist.github.com/randyzwitch/934d502e53f2adcb48eea2423fe4a47e")
    """

    gistcreator, gistid = _clean_link(link).split("/")[-2:]
    return components.html(
        f"""<script src="https://gist.github.com/{gistcreator}/{gistid}.js"></script>""",
        height=height,
        width=width,
        scrolling=scrolling,
    )


def gitlab_snippet(link, height=600, width=950, scrolling=True):
    """Embed a Gitlab snippet.

    Parameters
    ----------
    link : str
        URL from https://gitlab.com/explore/snippets
    height: int
        Height of the resulting iframe
    width: int
        Width of the resulting iframe
    scrolling: bool
        If content is larger than iframe size, provide scrollbars?
    
    Example
    -------
    >>> gitlab_snippet("https://gitlab.com/snippets/1995463", height = 400)
    """

    snippetnumber = _clean_link(link).split("/")[-1]
    return components.html(
        f"""<script src='https://gitlab.com/snippets/{snippetnumber}.js'></script>""",
        height=height,
        width=width,
        scrolling=scrolling,
    )


def pastebin_snippet(link, height=600, width=950, scrolling=True):
    """Embed a Pastebin snippet.

    Parameters
    ----------
    link : str
        URL from https://pastebin.com/
    height: int
        Height of the resulting iframe
    width: int
        Width of the resulting iframe
    scrolling: bool
        If content is larger than iframe size, provide scrollbars?
    
    Example
    -------
    >>> pastebin_snippet("https://pastebin.com/AWYbziQF", width = 600, scrolling = False)
    """

    snippetnumber = _clean_link(link).split("/")[-1]
    return components.html(
        f"""<script src="https://pastebin.com/embed_js/{snippetnumber}"></script>""",
        height=height,
        width=width,
        scrolling=scrolling,
    )


def codepen_snippet(
    link, height=600, width=950, scrolling=True, theme="light", preview=True
):
    """Embed a CodePen snippet.

    Parameters
    ----------
    link : str
        URL from https://codepen.io/
    height: int
        Height of the resulting iframe
    width: int
        Width of the resulting iframe
    scrolling: bool
        If content is larger than iframe size, provide scrollbars?
    theme: str
        Color theme of snippet (i.e. "light", "dark")
    preview: bool
        Require snippet to be clicked to load. Setting `preview=True` can improve load times.

    
    Example
    -------
    >>> codepen_snippet("https://codepen.io/ste-vg/pen/GRooLza", width = 600, scrolling = False)
    """

    user, _, slughash = _clean_link(link).split("/")[-3:]
    return components.html(
        f"""
        <p class="codepen" 
        data-height="{height}" 
        data-theme-id="{theme}" 
        data-default-tab="html,result" 
        data-user="{user}" 
        data-slug-hash="{slughash}" 
        data-preview="{str(preview).lower()}" 
        style="height: {height}px; box-sizing: border-box; display: flex; align-items: center; justify-content: center; border: 2px solid; margin: 1em 0; padding: 1em;"">
    </p><script async src="https://static.codepen.io/assets/embed/ei.js"></script>
    """,
        height=height,
        width=width,
        scrolling=scrolling,
    )


def ideone_snippet(link, height=600, width=950, scrolling=True):
    """Embed a Ideone snippet.

    Parameters
    ----------
    link : str
        URL from https://ideone.com/
    height: int
        Height of the resulting iframe
    width: int
        Width of the resulting iframe
    scrolling: bool
        If content is larger than iframe size, provide scrollbars?
    
    Example
    -------
    >>> ideone_snippet("https://ideone.com/vQ54cr")
    """

    snippetnumber = _clean_link(link).split("/")[-1]
    return components.html(
        f"""<script src="https://ideone.com/e.js/{snippetnumber}" type="text/javascript" ></script>""",
        height=height,
        width=width,
        scrolling=scrolling,
    )


def tagmycode_snippet(link, height=600, width=950, scrolling=True):
    """Embed a TagMyCode snippet.

    Parameters
    ----------
    link : str
        URL from https://tagmycode.com/
    height: int
        Height of the resulting iframe
    width: int
        Width of the resulting iframe
    scrolling: bool
        If content is larger than iframe size, provide scrollbars?
    
    Example
    -------
    >>> tagmycode_snippet("https://tagmycode.com/snippet/5965/recursive-list-files-in-a-dir#.Xwyc43VKglU")
    """

    snippetnumber = _clean_link(link).split("/")[-2]
    return components.html(
        f"""<script src="https://tagmycode.com/embed/js/{snippetnumber}"></script>""",
        height=height,
        width=width,
        scrolling=scrolling,
    )
