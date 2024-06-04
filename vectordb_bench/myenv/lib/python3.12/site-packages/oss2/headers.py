# -*- coding: utf-8 -*-
"""
oss2.headers
~~~~~~~~
这个模块包含http请求里header的key定义
同时包含了发送http请求的header, 类型为dict
"""
OSS_USER_METADATA_PREFIX = "x-oss-meta-"

OSS_CANNED_ACL = "x-oss-acl"

IF_UNMODIFIED_SINCE = "If-Unmodified-Since"
IF_MATCH = "If-Match"

OSS_COPY_OBJECT_SOURCE = "x-oss-copy-source"
OSS_COPY_OBJECT_SOURCE_RANGE = "x-oss-copy-source-range"

OSS_REQUEST_ID = "x-oss-request-id"

OSS_SECURITY_TOKEN = "x-oss-security-token"

OSS_NEXT_APPEND_POSITION = "x-oss-next-append-position"
OSS_HASH_CRC64_ECMA = "x-oss-hash-crc64ecma"
OSS_OBJECT_TYPE = "x-oss-object-type"

OSS_OBJECT_ACL = "x-oss-object-acl"

OSS_SYMLINK_TARGET = "x-oss-symlink-target"

OSS_SERVER_SIDE_ENCRYPTION = "x-oss-server-side-encryption"
OSS_SERVER_SIDE_ENCRYPTION_KEY_ID = "x-oss-server-side-encryption-key-id"

OSS_CLIENT_SIDE_ENCRYPTION_KEY = "x-oss-meta-client-side-encryption-key"
OSS_CLIENT_SIDE_ENCRYPTION_START = "x-oss-meta-client-side-encryption-start"
OSS_CLIENT_SIDE_ENCRYPTION_CEK_ALG = "x-oss-meta-client-side-encryption-cek-alg"
OSS_CLIENT_SIDE_ENCRYPTION_WRAP_ALG = "x-oss-meta-client-side-encryption-wrap-alg"
OSS_CLIENT_SIDE_ENCRYTPION_MATDESC = "x-oss-meta-client-side-encryption-matdesc"
OSS_CLIENT_SIDE_ENCRYPTION_UNENCRYPTED_CONTENT_LENGTH = "x-oss-meta-client-side-encryption-unencrypted-content-length"
OSS_CLIENT_SIDE_ENCRYPTION_UNENCRYPTED_CONTENT_MD5 = "x-oss-meta-client-side-encryption-unencrypted-content-md5"
OSS_CLIENT_SIDE_ENCRYPTION_DATA_SIZE = "x-oss-meta-client-side-encryption-data-size"
OSS_CLIENT_SIDE_ENCRYPTION_PART_SIZE = "x-oss-meta-client-side-encryption-part-size"

DEPRECATED_CLIENT_SIDE_ENCRYPTION_KEY = "x-oss-meta-oss-crypto-key"
DEPRECATED_CLIENT_SIDE_ENCRYPTION_START = "x-oss-meta-oss-crypto-start"
DEPRECATED_CLIENT_SIDE_ENCRYPTION_CEK_ALG = "x-oss-meta-oss-cek-alg"
DEPRECATED_CLIENT_SIDE_ENCRYPTION_WRAP_ALG = "x-oss-meta-oss-wrap-alg"
DEPRECATED_CLIENT_SIDE_ENCRYTPION_MATDESC = "x-oss-meta-oss-crypto-matdesc"
DEPRECATED_CLIENT_SIDE_ENCRYPTION_UNENCRYPTED_CONTENT_LENGTH = "x-oss-meta-oss-crypto-unencrypted-content-length"
DEPRECATED_CLIENT_SIDE_ENCRYPTION_UNENCRYPTED_CONTENT_MD5 = "x-oss-meta-oss-crypto-unencrypted-content-md5"

OSS_OBJECT_TAGGING = "x-oss-tagging"
OSS_OBJECT_TAGGING_COPY_DIRECTIVE = "x-oss-tagging-directive"

OSS_REQUEST_PAYER = 'x-oss-request-payer'

OSS_TRAFFIC_LIMIT = 'x-oss-traffic-limit'

RSA_NONE_PKCS1Padding_WRAP_ALGORITHM = 'RSA/NONE/PKCS1Padding'
RSA_NONE_OAEPWithSHA1AndMGF1Padding = 'RSA/NONE/OAEPWithSHA-1AndMGF1Padding'
KMS_ALI_WRAP_ALGORITHM = 'KMS/ALICLOUD'
OSS_ENCRYPTION_CLIENT = 'OssEncryptionClient'
OSS_TASK_ID = 'x-oss-task-id'

OSS_SERVER_SIDE_ENCRYPTION = "x-oss-server-side-encryption"
OSS_SERVER_SIDE_ENCRYPTION_KEY_ID = "x-oss-server-side-encryption-key-id"
OSS_SERVER_SIDE_DATA_ENCRYPTION = "x-oss-server-side-data-encryption"

OSS_METADATA_DIRECTIVE = 'x-oss-metadata-directive'
OSS_ALLOW_ACTION_OVERLAP = "x-oss-allow-same-action-overlap"

class RequestHeader(dict):
    def __init__(self, *arg, **kw):
        super(RequestHeader, self).__init__(*arg, **kw)

    def set_server_side_encryption(self, algorithm=None, cmk_id=None):
        if OSS_SERVER_SIDE_ENCRYPTION in self:
            del self[OSS_SERVER_SIDE_ENCRYPTION]
        if OSS_SERVER_SIDE_ENCRYPTION_KEY_ID in self:
            del self[OSS_SERVER_SIDE_ENCRYPTION_KEY_ID]

        if algorithm == "AES256":
            self[OSS_SERVER_SIDE_ENCRYPTION] = "AES256"
        elif algorithm == "KMS":
            self[OSS_SERVER_SIDE_ENCRYPTION] = "KMS"
            if cmk_id is not None:
                self[OSS_SERVER_SIDE_ENCRYPTION_KEY_ID] = cmk_id
