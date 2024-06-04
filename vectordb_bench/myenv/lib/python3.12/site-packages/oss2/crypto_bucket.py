# -*- coding: utf-8 -*-

from . import http
from . import exceptions
from . import Bucket

from .api import _make_range_string
from .models import *
from .compat import to_string, urlsplit, parse_qs
from .crypto import BaseCryptoProvider
from .exceptions import ClientError
import copy
import threading

logger = logging.getLogger(__name__)


class CryptoBucket(Bucket):
    """用于加密Bucket和Object操作的类，诸如上传、下载Object等。创建、删除bucket的操作需使用Bucket类接口。

    用法（假设Bucket属于杭州区域） ::

        >>> import oss2
        >>> auth = oss2.Auth('your-access-key-id', 'your-access-key-secret')
        >>> bucket = oss2.CryptoBucket(auth, 'http://oss-cn-hangzhou.aliyuncs.com', 'your-bucket', oss2.LocalRsaProvider())
        >>> bucket.put_object('readme.txt', 'content of the object')
        <oss2.models.PutObjectResult object at 0x029B9930>

    :param auth: 包含了用户认证信息的Auth对象
    :type auth: oss2.Auth

    :param str endpoint: 访问域名或者CNAME
    :param str bucket_name: Bucket名
    :param crypto_provider: 客户端加密类。该参数默认为空
    :type crypto_provider: oss2.crypto.BaseCryptoProvider
    :param bool is_cname: 如果endpoint是CNAME则设为True；反之，则为False。

    :param session: 会话。如果是None表示新开会话，非None则复用传入的会话
    :type session: oss2.Session

    :param float connect_timeout: 连接超时时间，以秒为单位。

    :param str app_name: 应用名。该参数不为空，则在User Agent中加入其值。
        注意到，最终这个字符串是要作为HTTP Header的值传输的，所以必须要遵循HTTP标准。

    :param bool enable_crc: 如果开启crc校验则设为True；反之，则为False

    """

    def __init__(self, auth, endpoint, bucket_name, crypto_provider,
                 is_cname=False,
                 session=None,
                 connect_timeout=None,
                 app_name='',
                 enable_crc=True,
                 proxies=None,
                 region=None,
                 cloudbox_id=None,
                 is_path_style=False
                 ):

        if not isinstance(crypto_provider, BaseCryptoProvider):
            raise ClientError('crypto_provider must be an instance of BaseCryptoProvider')

        logger.debug("Init CryptoBucket: {0}".format(bucket_name))
        super(CryptoBucket, self).__init__(auth, endpoint, bucket_name, is_cname, session, connect_timeout, app_name,
                                           enable_crc, proxies=proxies, region=region, cloudbox_id=cloudbox_id, is_path_style= is_path_style)

        self.crypto_provider = crypto_provider
        self.upload_contexts = {}
        self.upload_contexts_lock = threading.Lock()

        if self.app_name:
            self.user_agent = http.USER_AGENT + '/' + self.app_name + '/' + OSS_ENCRYPTION_CLIENT
        else:
            self.user_agent = http.USER_AGENT + '/' + OSS_ENCRYPTION_CLIENT

    def _init_user_agent(self, headers):
        if 'User-Agent' not in headers:
            headers['User-Agent'] = self.user_agent
        else:
            headers['User-Agent'] += '/' + OSS_ENCRYPTION_CLIENT

    def put_object(self, key, data,
                   headers=None,
                   progress_callback=None):
        """上传一个普通文件。

        用法 ::
            >>> bucket.put_object('readme.txt', 'content of readme.txt')
            >>> with open(u'local_file.txt', 'rb') as f:
            >>>     bucket.put_object('remote_file.txt', f)

        :param mat_desc: map，对象文件的description
        :param key: 上传到OSS的文件名

        :param data: 待上传的内容。
        :type data: bytes，str或file-like object

        :param headers: 用户指定的HTTP头部。可以指定Content-Type、Content-MD5、x-oss-meta-开头的头部等
        :type headers: 可以是dict，建议是oss2.CaseInsensitiveDict

        :param progress_callback: 用户指定的进度回调函数。可以用来实现进度条等功能。参考 :ref:`progress_callback` 。

        :return: :class:`PutObjectResult <oss2.models.PutObjectResult>`
        """
        logger.debug("Start to put object to CryptoBucket")

        headers = http.CaseInsensitiveDict(headers)
        self._init_user_agent(headers)
        content_crypto_material = self.crypto_provider.create_content_material()
        data = self.crypto_provider.make_encrypt_adapter(data, content_crypto_material.cipher)
        headers = content_crypto_material.to_object_meta(headers)

        return super(CryptoBucket, self).put_object(key, data, headers, progress_callback)

    def put_object_with_url(self, sign_url, data, headers=None, progress_callback=None):

        """ 使用加签的url上传对象

        :param sign_url: 加签的url
        :param data: 待上传的数据
        :param headers: 用户指定的HTTP头部。可以指定Content-Type、Content-MD5、x-oss-meta-开头的头部等，必须和签名时保持一致
        :param progress_callback: 用户指定的进度回调函数。参考 :ref:`progress_callback`
        :return:
        """
        raise ClientError("The operation is not support for CryptoBucket now")

    def append_object(self, key, position, data,
                      headers=None,
                      progress_callback=None,
                      init_crc=None):
        raise ClientError("The operation is not support for CryptoBucket")

    def get_object(self, key,
                   byte_range=None,
                   headers=None,
                   progress_callback=None,
                   process=None,
                   params=None):
        """下载一个文件。

        用法 ::

            >>> result = bucket.get_object('readme.txt')
            >>> print(result.read())
            'hello world'

        :param key: 文件名
        :param byte_range: 指定下载范围。参见 :ref:`byte_range`

        :param headers: HTTP头部
        :type headers: 可以是dict，建议是oss2.CaseInsensitiveDict

        :param progress_callback: 用户指定的进度回调函数。参考 :ref:`progress_callback`
        :param process: oss文件处理，如图像服务等。指定后process，返回的内容为处理后的文件。

        :param params: http 请求的查询字符串参数
        :type params: dict

        :return: file-like object

        :raises: 如果文件不存在，则抛出 :class:`NoSuchKey <oss2.exceptions.NoSuchKey>` ；还可能抛出其他异常
        """
        if process:
            raise ClientError("Process object operation is not support for Crypto Bucket")

        headers = http.CaseInsensitiveDict(headers)
        self._init_user_agent(headers)

        discard = 0
        range_string = ''

        if byte_range:
            if byte_range[0] is None and byte_range[1]:
                raise ClientError("Don't support range get while start is none and end is not")
            start, end = self.crypto_provider.adjust_range(byte_range[0], byte_range[1])
            adjust_byte_range = (start, end)

            range_string = _make_range_string(adjust_byte_range)
            if range_string:
                headers['range'] = range_string

            if byte_range[0] and adjust_byte_range[0] < byte_range[0]:
                discard = byte_range[0] - adjust_byte_range[0]
            logger.debug("adjust range of get object, byte_range: {0}, adjust_byte_range: {1}, discard: {2}".format(
                byte_range, adjust_byte_range, discard))

        logger.debug(
            "Start to get object from CryptoBucket: {0}, key: {1}, range: {2}, headers: {3}, params: {4}".format(
                self.bucket_name, to_string(key), range_string, headers, params))
        resp = self._do('GET', self.bucket_name, key, headers=headers, params=params)
        logger.debug("Get object done, req_id: {0}, status_code: {1}".format(resp.request_id, resp.status))

        return GetObjectResult(resp, progress_callback, self.enable_crc, crypto_provider=self.crypto_provider,
                               discard=discard)

    def get_object_with_url(self, sign_url,
                            byte_range=None,
                            headers=None,
                            progress_callback=None):
        """使用加签的url下载文件

        :param sign_url: 加签的url
        :param byte_range: 指定下载范围。参见 :ref:`byte_range`

        :param headers: HTTP头部
        :type headers: 可以是dict，建议是oss2.CaseInsensitiveDict，必须和签名时保持一致

        :param progress_callback: 用户指定的进度回调函数。参考 :ref:`progress_callback`

        :return: file-like object

        :raises: 如果文件不存在，则抛出 :class:`NoSuchKey <oss2.exceptions.NoSuchKey>` ；还可能抛出其他异常
        """
        query = parse_qs(urlsplit(sign_url).query)
        if query and (Bucket.PROCESS in query):
            raise ClientError("Process object operation is not support for Crypto Bucket")

        headers = http.CaseInsensitiveDict(headers)
        self._init_user_agent(headers)

        discard = 0
        range_string = ''

        if byte_range:
            if not byte_range[0] and byte_range[1]:
                raise ClientError("Don't support range get while start is none and end is not")
            start, end = self.crypto_provider.adjust_range(byte_range[0], byte_range[1])
            adjust_byte_range = (start, end)

            range_string = _make_range_string(adjust_byte_range)
            if range_string:
                headers['range'] = range_string

            if byte_range[0] and adjust_byte_range[0] < byte_range[0]:
                discard = byte_range[0] - adjust_byte_range[0]
            logger.debug("adjust range of get object, byte_range: {0}, adjust_byte_range: {1}, discard: {2}".format(
                byte_range, adjust_byte_range, discard))

        logger.debug(
            "Start to get object with url from CryptoBucket: {0}, sign_url: {1}, range: {2}, headers: {3}".format(
                self.bucket_name, sign_url, range_string, headers))
        resp = self._do_url('GET', sign_url, headers=headers)
        return GetObjectResult(resp, progress_callback, self.enable_crc,
                               crypto_provider=self.crypto_provider, discard=discard)

    def create_select_object_meta(self, key, select_meta_params=None):
        raise ClientError("The operation is not support for Crypto Bucket")

    def select_object(self, key, sql,
                      progress_callback=None,
                      select_params=None
                      ):
        raise ClientError("The operation is not support for CryptoBucket")

    def init_multipart_upload(self, key, headers=None, params=None, upload_context=None):
        """客户端加密初始化分片上传。

        :param params
        :param upload_context

        :param headers: HTTP头部
        :type headers: 可以是dict，建议是oss2.CaseInsensitiveDict

        :return: :class:`InitMultipartUploadResult <oss2.models.InitMultipartUploadResult>`
        返回值中的 `crypto_multipart_context` 记录了加密Meta信息，在upload_part时需要一并传入
        """

        headers = http.CaseInsensitiveDict(headers)
        self._init_user_agent(headers)
        if not upload_context or not upload_context.data_size:
            raise ClientError("It is not support none upload_context and must specify data_size of upload_context ")

        logger.info("Start to init multipart upload by CryptoBucket, data_size: {0}, part_size: {1}".format(
            upload_context.data_size, upload_context.part_size))

        if upload_context.part_size:
            res = self.crypto_provider.cipher.is_valid_part_size(upload_context.part_size, upload_context.data_size)
            if not res:
                raise ClientError("part_size is invalid for multipart upload for CryptoBucket")
        else:
            upload_context.part_size = self.crypto_provider.cipher.determine_part_size(upload_context.data_size)

        content_crypto_material = self.crypto_provider.create_content_material()

        upload_context.content_crypto_material = content_crypto_material

        headers = content_crypto_material.to_object_meta(headers, upload_context)

        resp = super(CryptoBucket, self).init_multipart_upload(key, headers)

        return resp

    def upload_part(self, key, upload_id, part_number, data, progress_callback=None, headers=None, upload_context=None):
        """客户端加密上传一个分片。

        :param upload_context:
        :param str key: 待上传文件名，这个文件名要和 :func:`init_multipart_upload` 的文件名一致。
        :param str upload_id: 分片上传ID
        :param int part_number: 分片号，最小值是1.
        :param data: 待上传数据。
        :param progress_callback: 用户指定进度回调函数。可以用来实现进度条等功能。参考 :ref:`progress_callback` 。

        :param headers: 用户指定的HTTP头部。可以指定Content-MD5头部等
        :type headers: 可以是dict，建议是oss2.CaseInsensitiveDict

        :return: :class:`PutObjectResult <oss2.models.PutObjectResult>`
        """
        logger.info(
            "Start to upload multipart of CryptoBucket, upload_id = {0}, part_number = {1}".format(upload_id,
                                                                                                   part_number))
        headers = http.CaseInsensitiveDict(headers)
        self._init_user_agent(headers)
        if upload_context:
            context = upload_context
        else:
            raise ClientError("Could not init upload context, upload contexts flag is False and upload context is none")

        content_crypto_material = context.content_crypto_material

        if content_crypto_material.cek_alg != self.crypto_provider.cipher.alg or content_crypto_material.wrap_alg != \
                self.crypto_provider.wrap_alg:
            err_msg = 'Envelope or data encryption/decryption algorithm is inconsistent'
            raise InconsistentError(err_msg, self)

        headers = content_crypto_material.to_object_meta(headers, context)

        plain_key = self.crypto_provider.decrypt_encrypted_key(content_crypto_material.encrypted_key)
        plain_iv = self.crypto_provider.decrypt_encrypted_iv(content_crypto_material.encrypted_iv)

        offset = context.part_size * (part_number - 1)
        counter = self.crypto_provider.cipher.calc_offset(offset)

        cipher = copy.copy(content_crypto_material.cipher)
        cipher.initialize(plain_key, plain_iv, counter)
        data = self.crypto_provider.make_encrypt_adapter(data, cipher)
        resp = super(CryptoBucket, self).upload_part(key, upload_id, part_number, data, progress_callback, headers)

        return resp

    def complete_multipart_upload(self, key, upload_id, parts, headers=None):
        """客户端加密完成分片上传，创建文件。
        当所有分片均已上传成功，才可以调用此函数

        :param str key: 待上传的文件名，这个文件名要和 :func:`init_multipart_upload` 的文件名一致。
        :param str upload_id: 分片上传ID

        :param parts: PartInfo列表。PartInfo中的part_number和etag是必填项。其中的etag可以从 :func:`upload_part` 的返回值中得到。
        :type parts: list of `PartInfo <oss2.models.PartInfo>`

        :param headers: HTTP头部
        :type headers: 可以是dict，建议是oss2.CaseInsensitiveDict

        :return: :class:`PutObjectResult <oss2.models.PutObjectResult>`
        """
        logger.info("Start to complete multipart upload of CryptoBucket, upload_id = {0}".format(upload_id))

        headers = http.CaseInsensitiveDict(headers)
        self._init_user_agent(headers)
        try:
            resp = super(CryptoBucket, self).complete_multipart_upload(key, upload_id, parts, headers)
        except exceptions as e:
            raise e

        return resp

    def abort_multipart_upload(self, key, upload_id, headers=None):
        """取消分片上传。

        :param headers: 可以是dict，建议是oss2.CaseInsensitiveDict
        :param str key: 待上传的文件名，这个文件名要和 :func:`init_multipart_upload` 的文件名一致。
        :param str upload_id: 分片上传ID

        :return: :class:`RequestResult <oss2.models.RequestResult>`
        """
        logger.info("Start to abort multipart upload of CryptoBucket, upload_id = {0}".format(upload_id))

        headers = http.CaseInsensitiveDict(headers)
        self._init_user_agent(headers)
        try:
            resp = super(CryptoBucket, self).abort_multipart_upload(key, upload_id)
        except exceptions as e:
            raise e

        return resp

    def upload_part_copy(self, source_bucket_name, source_key, byte_range, target_key, target_upload_id,
                         target_part_number, headers=None):
        """分片拷贝。把一个已有文件的一部分或整体拷贝成目标文件的一个分片。

        :param target_part_number:
        :param target_upload_id:
        :param target_key:
        :param source_key:
        :param source_bucket_name:
        :param byte_range: 指定待拷贝内容在源文件里的范围。参见 :ref:`byte_range`

        :param headers: HTTP头部
        :type headers: 可以是dict，建议是oss2.CaseInsensitiveDict

        :return: :class:`PutObjectResult <oss2.models.PutObjectResult>`
        """
        raise ClientError("The operation is not support for CryptoBucket now")

    def process_object(self, key, process):
        raise ClientError("The operation is not support for CryptoBucket")
