# Copyright 2019 Alibaba Cloud Inc. All rights reserved.
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

import re

from aliyunsdkcore.acs_exception.exceptions import ClientException
import aliyunsdkcore.acs_exception.error_code as error_code


def assert_integer_positive(integer, name):
    if isinstance(integer, int) and integer > 0:
        return
    raise ClientException(error_code.SDK_INVALID_PARAMETER,
                          "{0} should be a positive integer.".format(name))


def validate_pattern(prop, prop_name, pattern):
    match_obj = re.search(pattern, prop, re.M | re.I)
    if not match_obj:
        raise ClientException(error_code.SDK_INVALID_PARAMETER,
                              'The parameter %s not match with %s' % (prop_name, pattern))
