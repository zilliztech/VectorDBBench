import html

from htbuilder import H, HtmlElement, styles
from htbuilder.units import unit

# This works in 3.7+:
# from htbuilder import span
#
# ...so we use the 3.7 version of the code above here:
span = H.span

import annotated_text.parameters as p

def annotation(body, label="", background=None, color=None, **style):
    """Build an HtmlElement span object with the given body and annotation label.

    The end result will look something like this:

        [body | label]

    Parameters
    ----------
    body : string
        The string to put in the "body" part of the annotation.
    label : string
        The string to put in the "label" part of the annotation.
    background : string or None
        The color to use for the background "chip" containing this annotation.
        If None, will use a random color based on the label.
    color : string or None
        The color to use for the body and label text.
        If None, will use the document's default text color.
    style : dict
        Any CSS you want to apply to the containing "chip". This is useful for things like


    Examples
    --------

    Produce a simple annotation with default colors:

    >>> annotation("apple", "fruit")

    Produce an annotation with custom colors:

    >>> annotation("apple", "fruit", background="#FF0", color="black")

    Produce an annotation with crazy CSS:

    >>> annotation("apple", "fruit", background="#FF0", border="1px dashed red")

    """

    color_style = {}

    if color:
        color_style['color'] = color

    if background:
        background_color = background
    else:
        label_sum = sum(ord(c) for c in label)
        background_color = p.PALETTE[label_sum % len(p.PALETTE)]
        background_opacity = p.OPACITIES[label_sum % len(p.OPACITIES)]
        background = background_color + background_opacity

    label_element = ""

    if label:
        separator = ""

        if p.SHOW_LABEL_SEPARATOR:
            separator = span(
                style=styles(
                    border_left=f"1px solid",
                    opacity=0.1,
                    margin_left=p.LABEL_SPACING,
                    align_self="stretch",
                )
            ),

        label_element = (
            separator,
            span(
                style=styles(
                    margin_left=p.LABEL_SPACING,
                    font_size=p.LABEL_FONT_SIZE,
                    opacity=p.LABEL_OPACITY,
                )
            )(
                html.escape(label),
            )
        )

    return (
        span(
            style=styles(
                display="inline-flex",
                flex_direction="row",
                align_items="center",
                background=background,
                border_radius=p.BORDER_RADIUS,
                padding=p.PADDING,
                overflow="hidden",
                line_height=1,
                **color_style,
                **style,)
        )(
            html.escape(body),
            label_element,
        )
    )


def get_annotated_html(*args):
    """Writes text with annotations into an HTML string.

    Parameters
    ----------
    *args : see annotated_text()

    Returns
    -------
    str
        An HTML string.
    """

    return str(get_annotated_element(*args))


def get_annotated_element(*args):
    """Writes text with annotations into an HTBuilder HtmlElement object.

    Parameters
    ----------
    *args : see annotated_text()

    Returns
    -------
    HtmlElement
        An HTBuilder HtmlElement object.
    """

    out = span()

    for arg in args:
        if isinstance(arg, str):
            out(html.escape(arg))

        elif isinstance(arg, HtmlElement):
            out(arg)

        elif isinstance(arg, tuple):
            out(annotation(*arg))

        elif isinstance(arg, list):
            out(get_annotated_element(*arg))

        else:
            raise Exception("Oh noes!")

    return out
