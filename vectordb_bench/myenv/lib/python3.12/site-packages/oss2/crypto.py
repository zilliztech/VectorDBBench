# -*- coding: utf-8 -*-

"""
oss2.encryption
~~~~~~~~~~~~~~

该模块包含了客户端加解密相关的函数和类。
"""
import abc
import hashlib
import json
import os
import copy
import logging
import struct
from functools import partial

import six
from Crypto.Cipher import PKCS1_OAEP, PKCS1_v1_5
from Crypto.PublicKey import RSA
from aliyunsdkcore import client
from aliyunsdkcore.acs_exception.exceptions import ServerException, ClientException
from aliyunsdkcore.http import format_type, method_type
from aliyunsdkkms.request.v20160120 import GenerateDataKeyRequest, DecryptRequest, EncryptRequest

from . import models
from . import headers
from . import utils
from .utils import b64decode_from_string, b64encode_as_string
from .compat import to_bytes, to_unicode
from .exceptions import ClientError, OpenApiFormatError, OpenApiServerError

logger = logging.getLogger(__name__)


class EncryptionMaterials(object):
    def __init__(self, desc, key_pair=None, custom_master_key_id=None, passphrase=None):
        self.desc = {}
        if desc:
            if isinstance(desc, dict):
                self.desc = desc
            else:
                raise ClientError('Invalid type, the type of mat_desc must be dict!')
        if key_pair and custom_master_key_id:
            raise ClientError('Both key_pair and custom_master_key_id are not none')

        if key_pair and not isinstance(key_pair, dict):
            raise ClientError('Invalid type, the type of key_pair must be dict!')

        self.key_pair = key_pair
        self.custom_master_key_id = custom_master_key_id
        self.passphrase = passphrase

    def add_description(self, key, value):
        self.desc[key] = value

    def add_descriptions(self, descriptions):
        for key in descriptions:
            self.desc[key] = descriptions[key]


@six.add_metaclass(abc.ABCMeta)
class BaseCryptoProvider(object):
    """CryptoProvider 基类，提供基础的数据加密解密adapter

    """

    def __init__(self, cipher, mat_desc=None):
        if not cipher:
            raise ClientError('Please initialize the value of cipher!')
        self.cipher = cipher
        self.cek_alg = None
        self.wrap_alg = None
        self.mat_desc = None
        self.encryption_materials_dict = {}
        if mat_desc:
            if isinstance(mat_desc, dict):
                self.mat_desc = mat_desc
            else:
                raise ClientError('Invalid type, the type of mat_desc must be dict!')

    @abc.abstractmethod
    def get_key(self):
        pass

    def get_iv(self):
        return self.cipher.get_iv()

    @staticmethod
    def make_encrypt_adapter(stream, cipher):
        return utils.make_cipher_adapter(stream, partial(cipher.encrypt))

    @staticmethod
    def make_decrypt_adapter(stream, cipher, discard=0):
        return utils.make_cipher_adapter(stream, partial(cipher.decrypt), discard)

    @abc.abstractmethod
    def decrypt_encrypted_key(self, encrypted_key):
        pass

    @abc.abstractmethod
    def decrypt_encrypted_iv(self, encrypted_iv):
        pass

    @abc.abstractmethod
    def reset_encryption_materials(self, encryption_materials):
        pass

    def adjust_range(self, start, end):
        return self.cipher.adjust_range(start, end)

    @abc.abstractmethod
    def create_content_material(self):
        pass

    def add_encryption_materials(self, encryption_materials):
        if encryption_materials.desc:
            key = frozenset(encryption_materials.desc.items())
            self.encryption_materials_dict[key] = encryption_materials

    def get_encryption_materials(self, desc):
        if desc:
            key = frozenset(desc.items())
            if key in self.encryption_materials_dict.keys():
                return self.encryption_materials_dict[key]


_LOCAL_RSA_TMP_DIR = '.oss-local-rsa'


@six.add_metaclass(abc.ABCMeta)
class LocalRsaProvider(BaseCryptoProvider):
    """使用本地RSA加密数据密钥。

        :param str dir: 本地RSA公钥私钥存储路径
        :param str key: 本地RSA公钥私钥名称前缀
        :param str passphrase: 本地RSA公钥私钥密码
        :param class cipher: 数据加密，默认aes256，用户可自行实现对称加密算法，需符合AESCipher注释规则
    """

    DEFAULT_PUB_KEY_SUFFIX = '.public_key.pem'
    DEFAULT_PRIV_KEY_SUFFIX = '.private_key.pem'

    def __init__(self, dir=None, key='', passphrase=None, cipher=utils.AESCTRCipher(),
                 pub_key_suffix=DEFAULT_PUB_KEY_SUFFIX, private_key_suffix=DEFAULT_PRIV_KEY_SUFFIX):

        super(LocalRsaProvider, self).__init__(cipher=cipher)

        self.wrap_alg = headers.RSA_NONE_OAEPWithSHA1AndMGF1Padding
        keys_dir = dir or os.path.join(os.path.expanduser('~'), _LOCAL_RSA_TMP_DIR)

        priv_key_path = os.path.join(keys_dir, key + private_key_suffix)
        pub_key_path = os.path.join(keys_dir, key + pub_key_suffix)
        try:
            if os.path.exists(priv_key_path) and os.path.exists(pub_key_path):
                with open(priv_key_path, 'rb') as f:
                    self.__decrypt_obj = PKCS1_OAEP.new(RSA.importKey(f.read(), passphrase=passphrase))

                with open(pub_key_path, 'rb') as f:
                    self.__encrypt_obj = PKCS1_OAEP.new(RSA.importKey(f.read(), passphrase=passphrase))

            else:
                logger.warn('The file path of private key or public key is not exist, will generate key pair')
                private_key = RSA.generate(2048)
                public_key = private_key.publickey()

                self.__encrypt_obj = PKCS1_OAEP.new(public_key)
                self.__decrypt_obj = PKCS1_OAEP.new(private_key)

                utils.makedir_p(keys_dir)
                with open(priv_key_path, 'wb') as f:
                    f.write(private_key.exportKey(passphrase=passphrase))

                with open(pub_key_path, 'wb') as f:
                    f.write(public_key.exportKey(passphrase=passphrase))
        except (ValueError, TypeError, IndexError) as e:
            raise ClientError(str(e))

    def get_key(self):
        return self.cipher.get_key()

    def decrypt_encrypted_key(self, encrypted_key):
        try:
            return self.__decrypt_data(encrypted_key)
        except (TypeError, ValueError) as e:
            raise ClientError(str(e))

    def decrypt_encrypted_iv(self, encrypted_iv):
        try:
            return self.__decrypt_data(encrypted_iv)
        except (TypeError, ValueError) as e:
            raise ClientError(str(e))

    def reset_encryption_materials(self, encryption_materials):
        raise ClientError("do not support reset_encryption_materials!")

    def create_content_material(self):
        plain_key = self.get_key()
        encrypted_key = self.__encrypt_data(plain_key)
        plain_iv = self.get_iv()
        encrypted_iv = self.__encrypt_data(plain_iv)
        cipher = copy.copy(self.cipher)
        wrap_alg = self.wrap_alg
        mat_desc = self.mat_desc

        cipher.initialize(plain_key, plain_iv)

        content_crypto_material = models.ContentCryptoMaterial(cipher, wrap_alg, encrypted_key, encrypted_iv,
                                                               mat_desc)
        return content_crypto_material

    def __encrypt_data(self, data):
        return self.__encrypt_obj.encrypt(data)

    def __decrypt_data(self, data):
        return self.__decrypt_obj.decrypt(data)


@six.add_metaclass(abc.ABCMeta)
class RsaProvider(BaseCryptoProvider):
    """使用本地RSA加密数据密钥。

        :param str dir: 本地RSA公钥私钥存储路径
        :param str key: 本地RSA公钥私钥名称前缀
        :param str passphrase: 本地RSA公钥私钥密码
        :param class cipher: 数据加密，默认aes256，用户可自行实现对称加密算法，需符合AESCipher注释规则
    """

    def __init__(self, key_pair, passphrase=None, cipher=utils.AESCTRCipher(), mat_desc=None):

        super(RsaProvider, self).__init__(cipher=cipher, mat_desc=mat_desc)
        self.wrap_alg = headers.RSA_NONE_PKCS1Padding_WRAP_ALGORITHM

        if key_pair and not isinstance(key_pair, dict):
            raise ClientError('Invalid type, the type of key_pair must be dict!')

        try:
            if 'public_key' in key_pair:
                self.__encrypt_obj = PKCS1_v1_5.new(RSA.importKey(key_pair['public_key'], passphrase=passphrase))

            if 'private_key' in key_pair:
                self.__decrypt_obj = PKCS1_v1_5.new(RSA.importKey(key_pair['private_key'], passphrase=passphrase))
        except (ValueError, TypeError) as e:
            raise ClientError(str(e))

    def get_key(self):
        return self.cipher.get_key()

    def decrypt_encrypted_key(self, encrypted_key):
        try:
            return self.__decrypt_data(encrypted_key)
        except (TypeError, ValueError) as e:
            raise ClientError(str(e))

    def decrypt_encrypted_iv(self, encrypted_iv):
        try:
            return self.__decrypt_data(encrypted_iv)
        except (TypeError, ValueError) as e:
            raise ClientError(str(e))

    def reset_encryption_materials(self, encryption_materials):
        return RsaProvider(encryption_materials.key_pair, encryption_materials.passphrase, self.cipher,
                           encryption_materials.desc)

    def create_content_material(self):
        plain_key = self.get_key()
        encrypted_key = self.__encrypt_data(plain_key)
        plain_iv = self.get_iv()
        encrypted_iv = self.__encrypt_data(plain_iv)
        cipher = copy.copy(self.cipher)
        wrap_alg = self.wrap_alg
        mat_desc = self.mat_desc

        cipher.initialize(plain_key, plain_iv)

        content_crypto_material = models.ContentCryptoMaterial(cipher, wrap_alg, encrypted_key, encrypted_iv,
                                                               mat_desc)
        return content_crypto_material

    def __encrypt_data(self, data):
        return self.__encrypt_obj.encrypt(data)

    def __decrypt_data(self, data):
        decrypted_data = self.__decrypt_obj.decrypt(data, object)
        if decrypted_data == object:
            raise ClientError('Decrypted data error, please check you key pair!')
        return decrypted_data


class AliKMSProvider(BaseCryptoProvider):
    """使用aliyun kms服务加密数据密钥。kms的详细说明参见
        https://help.aliyun.com/product/28933.html?spm=a2c4g.11186623.3.1.jlYT4v
        此接口在py3.3下暂时不可用，详见
        https://github.com/aliyun/aliyun-openapi-python-sdk/issues/61

        :param str access_key_id: 可以访问kms密钥服务的access_key_id
        :param str access_key_secret: 可以访问kms密钥服务的access_key_secret
        :param str region: kms密钥服务地区
        :param str cmkey: 用户主密钥
        :param str sts_token: security token，如果使用的是临时AK需提供
        :param str passphrase: kms密钥服务密码
        :param class cipher: 数据加密，默认aes256，当前仅支持默认实现
    """

    def __init__(self, access_key_id, access_key_secret, region, cmk_id, sts_token=None, passphrase=None,
                 cipher=utils.AESCTRCipher(), mat_desc=None):

        super(AliKMSProvider, self).__init__(cipher=cipher, mat_desc=mat_desc)
        if not isinstance(cipher, utils.AESCTRCipher):
            raise ClientError('AliKMSProvider only support AES256 cipher now')
        self.wrap_alg = headers.KMS_ALI_WRAP_ALGORITHM
        self.custom_master_key_id = cmk_id
        self.sts_token = sts_token
        self.context = '{"x-passphrase":"' + passphrase + '"}' if passphrase else ''
        self.kms_client = client.AcsClient(access_key_id, access_key_secret, region)

    def get_key(self):
        plain_key, encrypted_key = self.__generate_data_key()
        return plain_key, encrypted_key

    def decrypt_encrypted_key(self, encrypted_key):
        return b64decode_from_string(self.__decrypt_data(encrypted_key))

    def decrypt_encrypted_iv(self, encrypted_iv, deprecated=False):
        if deprecated:
            return self.__decrypt_data(encrypted_iv)
        return b64decode_from_string(self.__decrypt_data(encrypted_iv))

    def reset_encryption_materials(self, encryption_materials):
        provider = copy.copy(self)
        provider.custom_master_key_id = encryption_materials.custom_master_key_id
        provider.context = '{"x-passphrase":"' + encryption_materials.passphrase + '"}' if encryption_materials.passphrase else ''
        provider.mat_desc = encryption_materials.desc
        return provider

    def create_content_material(self):
        plain_key, encrypted_key = self.get_key()
        plain_iv = self.get_iv()
        encrypted_iv = self.__encrypt_data(b64encode_as_string(plain_iv))
        cipher = copy.copy(self.cipher)
        wrap_alg = self.wrap_alg
        mat_desc = self.mat_desc

        cipher.initialize(plain_key, plain_iv)

        content_crypto_material = models.ContentCryptoMaterial(cipher, wrap_alg, encrypted_key, encrypted_iv,
                                                               mat_desc)
        return content_crypto_material

    def __generate_data_key(self):
        req = GenerateDataKeyRequest.GenerateDataKeyRequest()

        req.set_accept_format(format_type.JSON)
        req.set_method(method_type.POST)

        req.set_KeyId(self.custom_master_key_id)
        req.set_KeySpec('AES_256')
        req.set_NumberOfBytes(32)
        req.set_EncryptionContext(self.context)
        if self.sts_token:
            req.set_STSToken(self.sts_token)

        resp = self.__do(req)

        return b64decode_from_string(resp['Plaintext']), resp['CiphertextBlob']

    def __encrypt_data(self, data):
        req = EncryptRequest.EncryptRequest()

        req.set_accept_format(format_type.JSON)
        req.set_method(method_type.POST)
        req.set_KeyId(self.custom_master_key_id)
        req.set_Plaintext(data)
        req.set_EncryptionContext(self.context)
        if self.sts_token:
            req.set_STSToken(self.sts_token)

        resp = self.__do(req)

        return resp['CiphertextBlob']

    def __decrypt_data(self, data):
        req = DecryptRequest.DecryptRequest()

        req.set_accept_format(format_type.JSON)
        req.set_method(method_type.POST)
        req.set_CiphertextBlob(data)
        req.set_EncryptionContext(self.context)
        if self.sts_token:
            req.set_STSToken(self.sts_token)

        resp = self.__do(req)
        return resp['Plaintext']

    def __do(self, req):

        try:
            body = self.kms_client.do_action_with_exception(req)
            return json.loads(to_unicode(body))
        except ServerException as e:
            raise OpenApiServerError(e.http_status, e.request_id, e.message, e.error_code)
        except ClientException as e:
            raise ClientError(e.message)
        except (KeyError, ValueError, TypeError) as e:
            raise OpenApiFormatError('Json Error: ' + str(e))
