import re
import xml.etree.ElementTree as etree
import xml.etree.ElementTree as ET
from typing import Tuple

from htbuilder import span
from markdown.inlinepatterns import InlineProcessor

SUPPORTED_ALIGNS = "|".join(["left", "right", "center", "justify"])
ALIGN_RE = rf"\[(?P<align_open>{SUPPORTED_ALIGNS})\](?P<content>[^\[]+)\[\/(?P<align_close>{SUPPORTED_ALIGNS})\]"


class AlignProcessor(InlineProcessor):
    """Transforms '[right]Test[/right]' into HTML."""

    def handleMatch(self, m: re.Match, data=None) -> Tuple[etree.Element, int, int]:
        """This function is called whenever a match is found.

        Args:
            m (re.Match): Match object
            data (_type_): [Not used - not sure what this does]

        Returns:
            (etree.Element, int, int): HTML element, with its starting and ending index.
        """

        groups = list(m.groups())
        align = groups[0]
        content = groups[1]
        html = span(style=f"text-align: {align};")(content)
        el = ET.ElementTree(ET.fromstring(str(html))).getroot()
        return el, m.start(0), m.end(0)
