import re
import xml.etree.ElementTree as etree
import xml.etree.ElementTree as ET
from typing import Tuple

from htbuilder import span
from markdown.inlinepatterns import InlineProcessor
from streamlit_extras.colored_header import ST_COLOR_PALETTE

SUPPORTED_COLORS = "|".join(ST_COLOR_PALETTE.keys())
COLOR_RE = rf"\[(?P<color_open>{SUPPORTED_COLORS})\](?P<content>[^\[]+)\[\/(?P<color_close>{SUPPORTED_COLORS})\]"


class ColorProcessor(InlineProcessor):
    """Transforms '[red]Test[/red]' into HTML."""

    def handleMatch(self, m: re.Match, data=None) -> Tuple[etree.Element, int, int]:
        """This function is called whenever a match is found.

        Args:
            m (re.Match): Match object
            data (_type_): [Not used - not sure what this does]

        Returns:
            (etree.Element, int, int): HTML element, with its starting and ending index.
        """

        groups = list(m.groups())
        color = groups[0]
        content = groups[1]
        color_hex = ST_COLOR_PALETTE[color]["70"]
        html = span(style=f"color:{color_hex};")(content)
        el = ET.ElementTree(ET.fromstring(str(html))).getroot()
        return el, m.start(0), m.end(0)
