# -*- coding: utf-8 -*-

"""
oss2.utils
----------

工具函数模块。
"""

from email.utils import formatdate

import logging
import os.path
import mimetypes
import socket
import hashlib
import base64
import threading
import calendar
import datetime
import time
import errno

import binascii
import crcmod
import re
import sys
import random
import abc, six
import struct

from Crypto.Cipher import AES
from Crypto import Random
from Crypto.Util import Counter

from .crc64_combine import mkCombineFun
from .compat import to_string, to_bytes, urlparse
from .exceptions import ClientError, InconsistentError, RequestError, OpenApiFormatError
from . import defaults

logger = logging.getLogger(__name__)

_EXTRA_TYPES_MAP = {
    ".js": "application/javascript",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".xltx": "application/vnd.openxmlformats-officedocument.spreadsheetml.template",
    ".potx": "application/vnd.openxmlformats-officedocument.presentationml.template",
    ".ppsx": "application/vnd.openxmlformats-officedocument.presentationml.slideshow",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".sldx": "application/vnd.openxmlformats-officedocument.presentationml.slide",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".dotx": "application/vnd.openxmlformats-officedocument.wordprocessingml.template",
    ".xlam": "application/vnd.ms-excel.addin.macroEnabled.12",
    ".xlsb": "application/vnd.ms-excel.sheet.binary.macroEnabled.12",
    ".apk": "application/vnd.android.package-archive"
}


def b64encode_as_string(data):
    return to_string(base64.b64encode(to_bytes(data)))


def b64decode_from_string(data):
    try:
        return base64.b64decode(to_string(data))
    except (TypeError, binascii.Error) as e:
        raise OpenApiFormatError('Base64 Error: ' + to_string(data))


def content_md5(data):
    """计算data的MD5值，经过Base64编码并返回str类型。

    返回值可以直接作为HTTP Content-Type头部的值
    """
    m = hashlib.md5(to_bytes(data))
    return b64encode_as_string(m.digest())


def md5_string(data):
    """返回 `data` 的MD5值，以十六进制可读字符串（32个小写字符）的方式。"""
    return hashlib.md5(to_bytes(data)).hexdigest()


def content_type_by_name(name):
    """根据文件名，返回Content-Type。"""
    ext = os.path.splitext(name)[1].lower()
    if ext in _EXTRA_TYPES_MAP:
        return _EXTRA_TYPES_MAP[ext]

    return mimetypes.guess_type(name)[0]


def set_content_type(headers, name):
    """根据文件名在headers里设置Content-Type。如果headers中已经存在Content-Type，则直接返回。"""
    headers = headers or {}

    if 'Content-Type' in headers:
        return headers

    content_type = content_type_by_name(name)
    if content_type:
        headers['Content-Type'] = content_type

    return headers


def is_ip_or_localhost(netloc):
    """判断网络地址是否为IP或localhost。"""
    is_ipv6 = False
    right_bracket_index = netloc.find(']')
    if netloc[0] == '[' and right_bracket_index > 0:
        loc = netloc[1:right_bracket_index]
        is_ipv6 = True
    else:
        loc = netloc.split(':')[0]

    if loc == 'localhost':
        return True

    try:
        if is_ipv6:
            socket.inet_pton(socket.AF_INET6, loc)  # IPv6
        else:
            socket.inet_aton(loc)  # Only IPv4
    except socket.error:
        return False

    return True


_ALPHA_NUM = 'abcdefghijklmnopqrstuvwxyz0123456789'
_HYPHEN = '-'
_BUCKET_NAME_CHARS = set(_ALPHA_NUM + _HYPHEN)
_MAX_UINT32 = 2 ** 32 - 1
_MAX_UINT64 = 2 ** 64 - 1


def is_valid_bucket_name(name):
    """判断是否为合法的Bucket名"""
    if len(name) < 3 or len(name) > 63:
        return False

    if name[-1] == _HYPHEN:
        return False

    if name[0] not in _ALPHA_NUM:
        return False

    return set(name) <= _BUCKET_NAME_CHARS

def is_valid_endpoint(endpoint):
    """判断是否为合法的endpoint"""
    if endpoint is None:
        return False

    pattern = '^([a-zA-Z]+://)?[\w.-]+(:\d+)?$'
    if re.match(pattern, endpoint):
        return True

    return False

def change_endianness_if_needed(bytes_array):
    if sys.byteorder == 'little':
        bytes_array.reverse();


class SizedFileAdapter(object):
    """通过这个适配器（Adapter），可以把原先的 `file_object` 的长度限制到等于 `size`。"""
    def __init__(self, file_object, size):
        self.file_object = file_object
        self.size = size
        self.offset = 0

    def read(self, amt=None):
        if self.offset >= self.size:
            return ''

        if (amt is None or amt < 0) or (amt + self.offset >= self.size):
            data = self.file_object.read(self.size - self.offset)
            self.offset = self.size
            return data

        self.offset += amt
        return self.file_object.read(amt)

    @property
    def len(self):
        return self.size


def how_many(m, n):
    return (m + n - 1) // n


def file_object_remaining_bytes(fileobj):
    current = fileobj.tell()

    fileobj.seek(0, os.SEEK_END)
    end = fileobj.tell()
    fileobj.seek(current, os.SEEK_SET)

    return end - current


def _has_data_size_attr(data):
    return hasattr(data, '__len__') or hasattr(data, 'len') or (hasattr(data, 'seek') and hasattr(data, 'tell'))


def _get_data_size(data):
    if hasattr(data, '__len__'):
        return len(data)

    if hasattr(data, 'len'):
        return data.len

    if hasattr(data, 'seek') and hasattr(data, 'tell'):
        return file_object_remaining_bytes(data)

    return None


_CHUNK_SIZE = 8 * 1024


def make_progress_adapter(data, progress_callback, size=None):
    """返回一个适配器，从而在读取 `data` ，即调用read或者对其进行迭代的时候，能够
     调用进度回调函数。当 `size` 没有指定，且无法确定时，上传回调函数返回的总字节数为None。

    :param data: 可以是bytes、file object或iterable
    :param progress_callback: 进度回调函数，参见 :ref:`progress_callback`
    :param size: 指定 `data` 的大小，可选

    :return: 能够调用进度回调函数的适配器
    """
    data = to_bytes(data)

    if size is None:
        size = _get_data_size(data)

    if size is None:
        if hasattr(data, 'read'):
            return _FileLikeAdapter(data, progress_callback)
        elif hasattr(data, '__iter__'):
            return _IterableAdapter(data, progress_callback)
        else:
            raise ClientError('{0} is not a file object, nor an iterator'.format(data.__class__.__name__))
    else:
        return _BytesAndFileAdapter(data, progress_callback, size)


def make_crc_adapter(data, init_crc=0, discard=0):
    """返回一个适配器，从而在读取 `data` ，即调用read或者对其进行迭代的时候，能够计算CRC。

    :param discard:
    :return:
    :param data: 可以是bytes、file object或iterable
    :param init_crc: 初始CRC值，可选

    :return: 能够调用计算CRC函数的适配器
    """
    data = to_bytes(data)

    # bytes or file object
    if _has_data_size_attr(data):
        if discard:
            raise ClientError('Bytes of file object adapter does not support discard bytes')
        return _BytesAndFileAdapter(data, size=_get_data_size(data), crc_callback=Crc64(init_crc))
    # file-like object
    elif hasattr(data, 'read'):
        return _FileLikeAdapter(data, crc_callback=Crc64(init_crc), discard=discard)
    # iterator
    elif hasattr(data, '__iter__'):
        if discard:
            raise ClientError('Iterator adapter does not support discard bytes')
        return _IterableAdapter(data, crc_callback=Crc64(init_crc))
    else:
        raise ClientError('{0} is not a file object, nor an iterator'.format(data.__class__.__name__))


def calc_obj_crc_from_parts(parts, init_crc=0):
    object_crc = 0
    crc_obj = Crc64(init_crc)
    for part in parts:
        if not part.part_crc or not part.size:
            return None
        else:
            object_crc = crc_obj.combine(object_crc, part.part_crc, part.size)
    return object_crc


def make_cipher_adapter(data, cipher_callback, discard=0):
    """返回一个适配器，从而在读取 `data` ，即调用read或者对其进行迭代的时候，能够进行加解密操作。

        :param encrypt:
        :param cipher_callback:
        :param discard: 读取时需要丢弃的字节
        :param data: 可以是bytes、file object或iterable

        :return: 能够客户端加密函数的适配器
        """
    data = to_bytes(data)

    # bytes or file object
    if _has_data_size_attr(data):
        if discard:
            raise ClientError('Bytes of file object adapter does not support discard bytes')
        return _BytesAndFileAdapter(data, size=_get_data_size(data), cipher_callback=cipher_callback)
    if hasattr(data, 'read'):
        return _FileLikeAdapter(data, cipher_callback=cipher_callback, discard=discard)
    # iterator
    elif hasattr(data, '__iter__'):
        if discard:
            raise ClientError('Iterator adapter does not support discard bytes')
        return _IterableAdapter(data, cipher_callback=cipher_callback)
    else:
        raise ClientError('{0} is not a file object'.format(data.__class__.__name__))


def check_crc(operation, client_crc, oss_crc, request_id):
    if client_crc is not None and oss_crc is not None and client_crc != oss_crc:
        e = InconsistentError("InconsistentError: req_id: {0}, operation: {1}, CRC checksum of client: {2} is mismatch "
                              "with oss: {3}".format(request_id, operation, client_crc, oss_crc))
        logger.error("Exception: {0}".format(e))
        raise e


def _invoke_crc_callback(crc_callback, content, discard=0):
    if crc_callback:
        crc_callback(content[discard:])


def _invoke_progress_callback(progress_callback, consumed_bytes, total_bytes):
    if progress_callback:
        progress_callback(consumed_bytes, total_bytes)


def _invoke_cipher_callback(cipher_callback, content, discard=0):
    if cipher_callback:
        content = cipher_callback(content)
        return content[discard:]
    return content


class _IterableAdapter(object):
    def __init__(self, data, progress_callback=None, crc_callback=None, cipher_callback=None):
        self.iter = iter(data)
        self.progress_callback = progress_callback
        self.offset = 0
        
        self.crc_callback = crc_callback
        self.cipher_callback = cipher_callback

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):            
        _invoke_progress_callback(self.progress_callback, self.offset, None)

        content = next(self.iter)
        self.offset += len(content)
                
        _invoke_crc_callback(self.crc_callback, content)

        content = _invoke_cipher_callback(self.cipher_callback, content)

        return content
    
    @property
    def crc(self):
        if self.crc_callback:
            return self.crc_callback.crc
        elif self.iter:
            return self.iter.crc
        else:
            return None


class _FileLikeAdapter(object):
    """通过这个适配器，可以给无法确定内容长度的 `fileobj` 加上进度监控。

    :param fileobj: file-like object，只要支持read即可
    :param progress_callback: 进度回调函数
    """

    def __init__(self, fileobj, progress_callback=None, crc_callback=None, cipher_callback=None, discard=0):
        self.fileobj = fileobj
        self.progress_callback = progress_callback
        self.offset = 0

        self.crc_callback = crc_callback
        self.cipher_callback = cipher_callback
        self.discard = discard
        self.read_all = False

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.read_all:
            raise StopIteration

        content = self.read(_CHUNK_SIZE)

        if content:
            return content
        else:
            raise StopIteration

    def read(self, amt=None):
        offset_start = self.offset
        if offset_start < self.discard and amt and self.cipher_callback:
            amt += self.discard

        content = self.fileobj.read(amt)
        if not content:
            self.read_all = True
            _invoke_progress_callback(self.progress_callback, self.offset, None)
        else:
            _invoke_progress_callback(self.progress_callback, self.offset, None)

            self.offset += len(content)

            real_discard = 0
            if offset_start < self.discard:
                if len(content) <= self.discard:
                    real_discard = len(content)
                else:
                    real_discard = self.discard

            _invoke_crc_callback(self.crc_callback, content, real_discard)
            content = _invoke_cipher_callback(self.cipher_callback, content, real_discard)

            self.discard -= real_discard
        return content

    @property
    def crc(self):
        if self.crc_callback:
            return self.crc_callback.crc
        elif self.fileobj:
            return self.fileobj.crc
        else:
            return None


class _BytesAndFileAdapter(object):
    """通过这个适配器，可以给 `data` 加上进度监控。

    :param data: 可以是unicode字符串（内部会转换为UTF-8编码的bytes）、bytes或file object
    :param progress_callback: 用户提供的进度报告回调，形如 callback(bytes_read, total_bytes)。
        其中bytes_read是已经读取的字节数；total_bytes是总的字节数。
    :param int size: `data` 包含的字节数。
    """
    def __init__(self, data, progress_callback=None, size=None, crc_callback=None, cipher_callback=None):
        self.data = to_bytes(data)
        self.progress_callback = progress_callback
        self.size = size
        self.offset = 0
        
        self.crc_callback = crc_callback
        self.cipher_callback = cipher_callback

    @property
    def len(self):
        return self.size

    # for python 2.x
    def __bool__(self):
        return True
    # for python 3.x
    __nonzero__=__bool__

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        content = self.read(_CHUNK_SIZE)

        if content:
            return content
        else:
            raise StopIteration

    def read(self, amt=None):
        if self.offset >= self.size:
            return to_bytes('')

        if amt is None or amt < 0:
            bytes_to_read = self.size - self.offset
        else:
            bytes_to_read = min(amt, self.size - self.offset)

        if isinstance(self.data, bytes):
            content = self.data[self.offset:self.offset+bytes_to_read]
        else:
            content = self.data.read(bytes_to_read)

        self.offset += bytes_to_read
            
        _invoke_progress_callback(self.progress_callback, min(self.offset, self.size), self.size)

        _invoke_crc_callback(self.crc_callback, content)

        content = _invoke_cipher_callback(self.cipher_callback, content)

        return content
    
    @property
    def crc(self):
        if self.crc_callback:
            return self.crc_callback.crc
        elif self.data:
            return self.data.crc
        else:
            return None


class Crc64(object):

    _POLY = 0x142F0E1EBA9EA3693
    _XOROUT = 0XFFFFFFFFFFFFFFFF
    
    def __init__(self, init_crc=0):
        self.crc64 = crcmod.Crc(self._POLY, initCrc=init_crc, rev=True, xorOut=self._XOROUT)

        self.crc64_combineFun = mkCombineFun(self._POLY, initCrc=init_crc, rev=True, xorOut=self._XOROUT)

    def __call__(self, data):
        self.update(data)
    
    def update(self, data):
        self.crc64.update(data)

    def combine(self, crc1, crc2, len2):
        return self.crc64_combineFun(crc1, crc2, len2)
    
    @property
    def crc(self):
        return self.crc64.crcValue

class Crc32(object):
    _POLY = 0x104C11DB7
    _XOROUT = 0xFFFFFFFF
    
    def __init__(self, init_crc=0):
        self.crc32 = crcmod.Crc(self._POLY, initCrc=init_crc, rev=True, xorOut=self._XOROUT)

    def __call__(self, data):
        self.update(data)
    
    def update(self, data):
        self.crc32.update(data)
    
    @property
    def crc(self):
        return self.crc32.crcValue


_AES_256_KEY_SIZE = 32
_AES_BLOCK_LEN = 16
_AES_BLOCK_BITS_LEN = 8 * 16

AES_GCM = 'AES/GCM/NoPadding'
AES_CTR = 'AES/CTR/NoPadding'


@six.add_metaclass(abc.ABCMeta)
class AESCipher(object):
    """AES256 加密实现。
            :param str key: 对称加密数据密钥
            :param str start: 对称加密初始随机值
        .. note::
            用户可自行实现对称加密算法，需服务如下规则：
            1、提供对称加密算法名，ALGORITHM
            2、提供静态方法，返回加密密钥和初始随机值（若算法不需要初始随机值，也需要提供）
            3、提供加密解密方法
    """

    # aes 256, key always is 32 bytes
    def __init__(self):
        self.alg = None
        self.key_len = _AES_256_KEY_SIZE
        self.block_size_len = _AES_BLOCK_LEN
        self.block_size_len_in_bits = _AES_BLOCK_BITS_LEN

    @abc.abstractmethod
    def get_key(self):
        pass

    @abc.abstractmethod
    def get_iv(self):
        pass

    @abc.abstractmethod
    def initialize(self, key, iv, off=0):
        pass

    @abc.abstractmethod
    def encrypt(self, raw):
        pass

    @abc.abstractmethod
    def decrypt(self, enc):
        pass

    @abc.abstractmethod
    def determine_part_size(self, data_size, excepted_part_size=None):
        pass

    def adjust_range(self, start, end):
        return start, end

    def is_block_aligned(self, offset):
        if offset is None:
            offset = 0
        return 0 == offset % self.block_size_len

    def is_valid_part_size(self, part_size, data_size=None):
        return True


class AESCTRCipher(AESCipher):
    """AES256 加密实现。
        :param str key: 对称加密数据密钥
        :param str start: 对称加密初始随机值
    .. note::
        用户可自行实现对称加密算法，需服务如下规则：
        1、提供对称加密算法名，ALGORITHM
        2、提供静态方法，返回加密密钥和初始随机值（若算法不需要初始随机值，也需要提供）
        3、提供加密解密方法
    """

    def __init__(self):
        super(AESCTRCipher, self).__init__()
        self.alg = AES_CTR
        self.__cipher = None

    def get_key(self):
        return random_key(self.key_len)

    def get_iv(self):
        return random_iv()

    def initialize(self, key, iv, offset=0):
        counter = iv_to_big_int(iv) + offset
        self.initial_by_counter(key, counter)

    def initial_by_counter(self, key, counter):
        ctr = Counter.new(self.block_size_len_in_bits, initial_value=counter)
        self.__cipher = AES.new(key, AES.MODE_CTR, counter=ctr)

    def encrypt(self, raw):
        return self.__cipher.encrypt(raw)

    def decrypt(self, enc):
        return self.__cipher.encrypt(enc)

    def adjust_range(self, start, end):
        if start:
            if end:
                if start <= end:
                    start = (start // self.block_size_len) * self.block_size_len
            else:
                start = (start // self.block_size_len) * self.block_size_len
        return start, end

    def is_valid_part_size(self, part_size, data_size):
        if not self.is_block_aligned(part_size) or part_size < defaults.min_part_size:
            return False

        if part_size * defaults.max_part_count < data_size:
            return False
        return True

    def calc_offset(self, offset):
        if not self.is_block_aligned(offset):
            raise ClientError('offset is not align to encrypt block')
        return offset // self.block_size_len

    def determine_part_size(self, data_size, excepted_part_size=None):
        if excepted_part_size:
            if self.is_valid_part_size(excepted_part_size, data_size):
                return excepted_part_size
            # excepted_part_size is not aligned
            elif excepted_part_size * defaults.max_part_count >= data_size:
                part_size = int(excepted_part_size / self.block_size_len + 1) * self.block_size_len
                return part_size

        # if excepted_part_size is None or is too small, calculate a correct part_size
        part_size = defaults.part_size
        while part_size * defaults.max_part_count < data_size:
            part_size = part_size * 2

        if not self.is_block_aligned(part_size):
            part_size = int(part_size / self.block_size_len + 1) * self.block_size_len

        return part_size


def random_key(key_len):
    return Random.new().read(key_len)


def random_iv():
    iv = Random.new().read(16)
    safe_iv = iv[0:8] + struct.pack(">L", 0) + iv[12:]
    return safe_iv


def iv_to_big_int(iv):
    iv_high_low_pair = struct.unpack(">QQ", iv)
    iv_big_int = iv_high_low_pair[0] << 64 | iv_high_low_pair[1]
    return iv_big_int


_STRPTIME_LOCK = threading.Lock()

_ISO8601_FORMAT = "%Y-%m-%dT%H:%M:%S.000Z"

# A regex to match HTTP Last-Modified header, whose format is 'Sat, 05 Dec 2015 11:10:29 GMT'.
# Its strftime/strptime format is '%a, %d %b %Y %H:%M:%S GMT'

_HTTP_GMT_RE = re.compile(
    r'(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun), (?P<day>0[1-9]|([1-2]\d)|(3[0-1])) (?P<month>Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) (?P<year>\d+) (?P<hour>([0-1]\d)|(2[0-3])):(?P<minute>[0-5]\d):(?P<second>[0-5]\d) GMT$'
)

_ISO8601_RE = re.compile(
    r'(?P<year>\d+)-(?P<month>01|02|03|04|05|06|07|08|09|10|11|12)-(?P<day>0[1-9]|([1-2]\d)|(3[0-1]))T(?P<hour>([0-1]\d)|(2[0-3])):(?P<minute>[0-5]\d):(?P<second>[0-5]\d)\.000Z$'
)

_MONTH_MAPPING = {
    'Jan': 1,
    'Feb': 2,
    'Mar': 3,
    'Apr': 4,
    'May': 5,
    'Jun': 6,
    'Jul': 7,
    'Aug': 8,
    'Sep': 9,
    'Oct': 10,
    'Nov': 11,
    'Dec': 12
}


def to_unixtime(time_string, format_string):
    with _STRPTIME_LOCK:
        return int(calendar.timegm(time.strptime(time_string, format_string)))


def http_date(timeval=None):
    """返回符合HTTP标准的GMT时间字符串，用strftime的格式表示就是"%a, %d %b %Y %H:%M:%S GMT"。
    但不能使用strftime，因为strftime的结果是和locale相关的。
    """
    return formatdate(timeval, usegmt=True)


def http_to_unixtime(time_string):
    """把HTTP Date格式的字符串转换为UNIX时间（自1970年1月1日UTC零点的秒数）。

    HTTP Date形如 `Sat, 05 Dec 2015 11:10:29 GMT` 。
    """
    m = _HTTP_GMT_RE.match(time_string)

    if not m:
        raise ValueError(time_string + " is not in valid HTTP date format")

    day = int(m.group('day'))
    month = _MONTH_MAPPING[m.group('month')]
    year = int(m.group('year'))
    hour = int(m.group('hour'))
    minute = int(m.group('minute'))
    second = int(m.group('second'))

    tm = datetime.datetime(year, month, day, hour, minute, second).timetuple()

    return calendar.timegm(tm)


def iso8601_to_unixtime(time_string):
    """把ISO8601时间字符串（形如，2012-02-24T06:07:48.000Z）转换为UNIX时间，精确到秒。"""

    m = _ISO8601_RE.match(time_string)

    if not m:
        raise ValueError(time_string + " is not in valid ISO8601 format")

    day = int(m.group('day'))
    month = int(m.group('month'))
    year = int(m.group('year'))
    hour = int(m.group('hour'))
    minute = int(m.group('minute'))
    second = int(m.group('second'))

    tm = datetime.datetime(year, month, day, hour, minute, second).timetuple()

    return calendar.timegm(tm)


def date_to_iso8601(d):
    return d.strftime(_ISO8601_FORMAT)  # It's OK to use strftime, since _ISO8601_FORMAT is not locale dependent


def iso8601_to_date(time_string):
    timestamp = iso8601_to_unixtime(time_string)
    return datetime.date.fromtimestamp(timestamp)


def makedir_p(dirpath):
    try:
        os.makedirs(dirpath)
    except os.error as e:
        if e.errno != errno.EEXIST:
            raise


def silently_remove(filename):
    """删除文件，如果文件不存在也不报错。"""
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:
            raise


def force_rename(src, dst):
    try:
        os.rename(src, dst)
    except OSError as e:
        if e.errno == errno.EEXIST:
            silently_remove(dst)
            os.rename(src, dst)
        else:
            raise


def copyfileobj_and_verify(fsrc, fdst, expected_len,
                           chunk_size=16*1024,
                           request_id=''):
    """copy data from file-like object fsrc to file-like object fdst, and verify length"""

    num_read = 0

    while 1:
        buf = fsrc.read(chunk_size)
        if not buf:
            break

        num_read += len(buf)
        fdst.write(buf)

    if num_read != expected_len:
        raise InconsistentError("IncompleteRead from source", request_id)

def _make_line_range_string(range):
    if range is None:
        return ''

    start = range[0]
    last = range[1]

    if start is None and last is None:
        return ''

    return 'line-range=' + _range_internal(start, last)

def _make_split_range_string(range):
    if range is None:
        return ''

    start = range[0]
    last = range[1]

    if start is None and last is None:
        return ''

    return 'split-range=' + _range_internal(start, last)

def _range_internal(start, last):
    def to_str(pos):
        if pos is None:
            return ''
        else:
            return str(pos)

    return to_str(start) + '-' + to_str(last)
