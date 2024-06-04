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

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import padding
from cryptography.hazmat.primitives.serialization import load_der_private_key

from aliyunsdkcore.compat import ensure_string
from aliyunsdkcore.compat import ensure_bytes
from aliyunsdkcore.compat import b64_encode_bytes
from aliyunsdkcore.compat import b64_decode_bytes


def get_sign_string(source, access_secret):
    key = load_der_private_key(
        b64_decode_bytes(ensure_bytes(access_secret)),
        password=None,
        backend=default_backend()
    )
    signed_bytes = key.sign(
        ensure_bytes(source),
        padding.PKCS1v15(),
        hashes.SHA256()
    )
    signed_base64 = b64_encode_bytes(signed_bytes)
    signature = ensure_string(signed_base64).replace('\n', '')
    return signature


def get_signer_name():
    return "SHA256withRSA"


def get_signer_version():
    return "1.0"


def get_signer_type():
    return "PRIVATEKEY"
