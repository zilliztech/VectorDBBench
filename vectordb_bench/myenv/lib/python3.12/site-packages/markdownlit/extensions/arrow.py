import re
import xml.etree.ElementTree as etree
from typing import Tuple

from markdown.inlinepatterns import InlineProcessor

ARROW_RE = r"(->|<-)"


class ArrowProcessor(InlineProcessor):
    def handleMatch(self, m: re.Match, data) -> Tuple[etree.Element, int, int]:
        """This function is called whenever a match is found.
        It will replace all arrows -> by the → character.
        """
        el = etree.Element("span")

        match = m.groups()[0]
        if match == "->":
            el.text = "→"
        elif match == "<-":
            el.text = "←"
        return el, m.start(0), m.end(0)
