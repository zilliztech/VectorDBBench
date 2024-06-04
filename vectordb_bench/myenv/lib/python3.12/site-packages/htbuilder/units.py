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
Easily add units to CSS properties.

Usage
-----

>>> from htbuilder.units import px, em, percent
>>>
>>> px(10)
("10px",)
>>> em(0, 1, 2, 3)
("0", "1em", "2em", "3em")
>>> percent(10)
("10%",)
"""

class _UnitBuilder(object):
    def __getattr__(self, name):
        def maybe_add_unit(x):
            if x == 0:
                return str(x)
            return "%s%s" % (x, name)

        def out(*args):
            return tuple(maybe_add_unit(x) for x in args)
        return out


# For Python < 3.7
unit = _UnitBuilder()


percent = unit.__getattr__("%")


# Python >= 3.7
# https://docs.python.org/3/reference/datamodel.html#customizing-module-attribute-access
def __getattr__(name):
    return unit.__getattr__(name)
