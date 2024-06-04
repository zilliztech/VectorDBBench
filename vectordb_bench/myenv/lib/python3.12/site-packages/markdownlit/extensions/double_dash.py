import re
import xml.etree.ElementTree as etree
from typing import Tuple

from markdown.inlinepatterns import InlineProcessor

DOUBLE_DASH_RE = r"--"


class DoubleDashProcessor(InlineProcessor):
    def handleMatch(self, m: re.Match, data) -> Tuple[etree.Element, int, int]:
        """This function is called whenever a match is found.
        It will replace all double dashes -- by the — character.
        """
        el = etree.Element("span")
        el.text = "—"
        return el, m.start(0), m.end(0)
