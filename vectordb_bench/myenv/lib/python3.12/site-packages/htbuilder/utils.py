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

def classes(*names, convert_underscores=True, **names_and_bools):
    """Join multiple class names with spaces between them.

    Example
    -------

    >>> classes("foo", "bar", baz=False, boz=True, long_name=True)
    "foo bar boz long-name"

    Or, if you want to keep the underscores:

    >>> classes("foo", "bar", long_name=True, convert_underscores=False)
    "foo bar long_name"

    """
    if convert_underscores:
        def clean(name):
            return name.replace("_", "-")
    else:
        def clean(name):
            return name

    classes = [clean(name) for name in names]

    for name, include in names_and_bools.items():
        if include:
            classes.append(clean(name))

    return " ".join(classes)


def styles(**rules):
    """Create a style string from Python objects.

    For rules that have multiple components use tuples or lists. Tuples are
    joined with spaces " ", lists are joined with commas ",". And although you
    can use lists for font-family rules, we also provide a helper called
    `fonts()` that wraps font names in quotes as well. See example below.

    Example
    -------

    >>> px = unit.px
    >>> rgba = func.rgba
    >>> bottom_margin = 10
    >>>
    >>> styles(
    ...     color="black",
    ...     font_family=fonts("Comic Sans", "sans"),
    ...     margin=(0, 0, px(bottom_margin), 0),
    ...     box_shadow=[
    ...         (0, 0, "10px", rgba(0, 0, 0, 0.1)),
    ...         (0, 0, "2px", rgba(0, 0, 0, 0.5)),
    ...     ],
    ... )
    ...
    "color:black;font-family:\"Comic Sans\",\"sans\";margin:0 0 10px 0;
    box-shadow:0 0 10px rgba(0,0,0,0.1),0 0 2px rgba(0,0,0,0.5)"

    """
    if not isinstance(rules, dict):
        raise TypeError("Style must be a dict")

    return ";".join(
        "%s:%s" % (k.replace("_", "-"), _parse_style_value(v))
        for (k, v) in rules.items()
    )

    return _parse_style_value(v)


def _parse_style_value(style):
    if isinstance(style, tuple):
        return " ".join(_parse_style_value(x) for x  in style)

    if isinstance(style, list):
        return ",".join(_parse_style_value(x) for x  in style)

    return str(style)


def fonts(*names):
    """Join fonts with quotes and commas.

    >>> fonts("Comic Sans, "sans")
    "\"Comic Sans\", \"Sans\""
    """
    return ",".join('"%s"' % name for name in names)


def rule(*selectors, **properties):
    return "%s {%s}" % (
        ",".join(selectors),
        styles(**properties),
    )
