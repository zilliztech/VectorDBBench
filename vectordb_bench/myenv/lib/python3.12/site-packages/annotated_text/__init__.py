import streamlit as st
from . import util

annotation = util.annotation

def annotated_text(*args):
    """Writes text with annotations into your Streamlit app.

    Parameters
    ----------
    *args : str, tuple or htbuilder.HtmlElement
        Arguments can be:
        - strings, to draw the string as-is on the screen.
        - tuples of the form (main_text, annotation_text, background, color) where
          background and foreground colors are optional and should be an CSS-valid string such as
          "#aabbcc" or "rgb(10, 20, 30)"
        - HtmlElement objects in case you want to customize the annotations further. In particular,
          you can import the `annotation()` function from this module to easily produce annotations
          whose CSS you can customize via keyword arguments.
        - lists of any of the above -- or of lists!

    Examples
    --------

    >>> annotated_text(
    ...     "This ",
    ...     ("is", "verb", "#8ef"),
    ...     " some ",
    ...     ("annotated", "adj", "#faa"),
    ...     ("text", "noun", "#afa"),
    ...     " for those of ",
    ...     ("you", "pronoun", "#fea"),
    ...     " who ",
    ...     ("like", "verb", "#8ef"),
    ...     " this sort of ",
    ...     ("thing", "noun", "#afa"),
    ... )

    >>> annotated_text(
    ...     "Hello ",
    ...     annotation("world!", "noun", color="#8ef", border="1px dashed red"),
    ... )

    """
    st.markdown(
        util.get_annotated_html(*args),
        unsafe_allow_html=True,
    )
