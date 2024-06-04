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
Easily write CSS functions

Usage
-----

>>> from htbuilder.funcs import rgba, hsl
>>>
>>> rgba(0, 0, 0, 0.1)
"rgba(0, 0, 0, 0.1)"
>>> hsl(270, "60%", "70%")
"hsl(270, 60%, 70%)"
"""

class _FuncBuilder(object):
    def __getattr__(self, name):
        def out(*args):
            return "%s(%s)" % (name, ",".join(str(x) for x in args))
        return out


# For Python < 3.7
func = _FuncBuilder()


# Python >= 3.7
# https://docs.python.org/3/reference/datamodel.html#customizing-module-attribute-access
def __getattr__(name):
    return func.__getattr__(name)
