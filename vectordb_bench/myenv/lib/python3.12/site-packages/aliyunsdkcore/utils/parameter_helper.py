# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

# coding=utf-8

import hashlib
import base64
import time
import random
import threading
import xml.etree.cElementTree as EleTree

from aliyunsdkcore.compat import ensure_bytes, ensure_string

TIME_ZONE = "GMT"
FORMAT_ISO_8601 = "%Y-%m-%dT%H:%M:%SZ"
FORMAT_RFC_2616 = "%a, %d %b %Y %H:%M:%S GMT"

_process_start_time = int(time.time() * 1000)
_seqId = 0


def get_uuid():
    global _seqId
    thread_id = threading.current_thread().ident
    current_time = int(time.time() * 1000)
    seq = _seqId
    _seqId += 1
    randNum = random.getrandbits(64)
    msg = '%d-%d-%d-%d-%d' % (_process_start_time, thread_id, current_time, seq, randNum)
    md5 = hashlib.md5()
    md5.update(msg.encode('utf-8'))
    return md5.hexdigest()


def get_iso_8061_date():
    return time.strftime(FORMAT_ISO_8601, time.gmtime())


def get_rfc_2616_date():
    return time.strftime(FORMAT_RFC_2616, time.gmtime())


def md5_sum(content):
    content_bytes = ensure_bytes(content)
    md5_bytes = hashlib.md5(content_bytes).digest()
    return ensure_string(base64.standard_b64encode(md5_bytes))


def _get_xml_by_list(elem, val, parent_element):
    i = 0
    tag_name = elem.tag
    if val.__len__() > 0:
        _get_xml_factory(elem, val[0], parent_element)

    for item in val:
        if i > 0:
            sub_elem = EleTree.SubElement(parent_element, tag_name)
            _get_xml_factory(sub_elem, item, parent_element)
        i = i + 1


def _get_xml_by_dict(elem, val):
    for k in val:
        sub_elem = EleTree.SubElement(elem, k)
        _get_xml_factory(sub_elem, val[k], elem)


def _get_xml_factory(elem, val, parent_element=None):
    if val is None:
        return

    if isinstance(val, dict):
        _get_xml_by_dict(elem, val)
    elif isinstance(val, (list, tuple, set)):
        if parent_element is None:
            raise ValueError("parent_element")
        _get_xml_by_list(elem, val, parent_element)
    else:
        elem.text = str(val)


def to_xml(dic):
    if dic is None:
        return

    if dic.__len__() == 0:
        return ""
    else:
        result_xml = ""
        for k in dic:
            elem = EleTree.Element(k)
            _get_xml_factory(elem, dic[k])
            result_xml += bytes.decode(EleTree.tostring(elem), encoding="utf-8")
        return result_xml
