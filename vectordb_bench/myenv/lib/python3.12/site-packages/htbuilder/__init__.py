# Copyright 2020 Thiago Teixeira
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
htbuilder -- Tiny HTML string builder for Python
===========================================

Build HTML strings using a purely functional syntax:

Example
-------

If using Python 3.7+:

>>> from htbuilder import div, ul, li, img  # This syntax requires Python 3.7+
>>>
>>> image_paths = [
...   "http://...",
...   "http://...",
...   "http://...",
... ]
>>>
>>> out = div(id="container")(
...   ul(_class="image-list")(
...     [
...       li(img(src=image_path, _class="large-image"))
...       for image_path in image_paths
...     ]
...   )
... )
>>>
>>> print(out)
>>>
>>> # Or convert to string with:
>>> x = str(out)


If using Python < 3.7, the import should look like this instead:

>>> from htbuilder import H
>>>
>>> div = H.div
>>> ul = H.ul
>>> li = H.li
>>> img = H.img
>>>
>>> # ...then the rest is the same as in the previous example.

"""

from more_itertools import collapse

from .funcs import func
from .units import unit
from .utils import classes, fonts, rule, styles

EMPTY_ELEMENTS = set(
    [
        # https://developer.mozilla.org/en-US/docs/Glossary/Empty_element
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "keygen",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
        # SVG
        "circle",
        "line",
        "path",
        "polygon",
        "polyline",
        "rect",
    ]
)


class _ElementCreator(object):
    def __getattr__(self, tag):
        return HtmlTag(tag)


class HtmlTag(object):
    def __init__(self, tag):
        """HTML element builder."""
        self._tag = tag

    def __call__(self, *args, **kwargs):
        el = HtmlElement(self._tag)
        el(*args, **kwargs)
        return el


class HtmlElement(object):
    def __init__(self, tag, attrs={}, children=[]):
        """An HTML element."""
        self._tag = tag.lower()
        self._attrs = attrs
        self._children = children
        self._is_empty = tag in EMPTY_ELEMENTS

    def __call__(self, *children, **attrs):
        if children:
            if self._is_empty:
                raise TypeError("<%s> cannot have children" % self._tag)
            self._children = list(collapse([*self._children, *children]))

        if attrs:
            self._attrs = {**self._attrs, **attrs}

        return self

    def __getattr__(self, name):
        if name in self._attrs:
            return self._attrs[name]
        raise AttributeError("No such attribute %s" % name)

    def __setattr__(self, name, value):
        if name.startswith("_"):
            object.__setattr__(self, name, value)
            return
        self._attrs[name] = value

    def __delattr__(self, name):
        del self._attrs[name]

    def __str__(self):
        args = {
            "tag": _clean_name(self._tag),
            "attrs": " ".join(
                [f'{_clean_name(k)}="{v}"' for k, v in self._attrs.items()]
            ),
            "children": "".join([str(c) for c in self._children]),
        }

        if self._is_empty:
            if self._attrs:
                return "<%(tag)s %(attrs)s/>" % args
            else:
                return "<%(tag)s/>" % args
        else:
            if self._attrs:
                return "<%(tag)s %(attrs)s>%(children)s</%(tag)s>" % args
            else:
                return "<%(tag)s>%(children)s</%(tag)s>" % args


def _clean_name(k):
    # This allows you to use reserved words by prepending/appending underscores.
    # For example, "_class" instead of "class".
    return k.strip("_").replace("_", "-")


def fragment(*args):
    return "".join(str(arg) for arg in args)


# Python >= 3.7
# https://docs.python.org/3/reference/datamodel.html#customizing-module-attribute-access
def __getattr__(tag):
    return HtmlTag(tag)


# For Python < 3.7
H = _ElementCreator()
