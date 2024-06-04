# -*- coding: utf-8 -*-
import os
import time
import requests
import json
import logging
import threading
from .exceptions import ClientError
from .utils import to_unixtime
from .compat import to_unicode

logger = logging.getLogger(__name__)


class Credentials(object):
    def __init__(self, access_key_id="", access_key_secret="", security_token=""):
        self.access_key_id = access_key_id
        self.access_key_secret = access_key_secret
        self.security_token = security_token

    def get_access_key_id(self):
        return self.access_key_id

    def get_access_key_secret(self):
        return self.access_key_secret

    def get_security_token(self):
        return self.security_token


DEFAULT_ECS_SESSION_TOKEN_DURATION_SECONDS = 3600 * 6
DEFAULT_ECS_SESSION_EXPIRED_FACTOR = 0.85


class EcsRamRoleCredential(Credentials):
    def __init__(self,
                 access_key_id,
                 access_key_secret,
                 security_token,
                 expiration,
                 duration,
                 expired_factor=None):
        self.access_key_id = access_key_id
        self.access_key_secret = access_key_secret
        self.security_token = security_token
        self.expiration = expiration
        self.duration = duration
        self.expired_factor = expired_factor or DEFAULT_ECS_SESSION_EXPIRED_FACTOR

    def get_access_key_id(self):
        return self.access_key_id

    def get_access_key_secret(self):
        return self.access_key_secret

    def get_security_token(self):
        return self.security_token

    def will_soon_expire(self):
        now = int(time.time())
        return self.duration * (1.0 - self.expired_factor) > self.expiration - now


class CredentialsProvider(object):
    def get_credentials(self):
        return


class StaticCredentialsProvider(CredentialsProvider):
    def __init__(self, access_key_id="", access_key_secret="", security_token=""):
        self.credentials = Credentials(access_key_id, access_key_secret, security_token)

    def get_credentials(self):
        return self.credentials


class EcsRamRoleCredentialsProvider(CredentialsProvider):
    def __init__(self, auth_host, max_retries=3, timeout=10):
        self.fetcher = EcsRamRoleCredentialsFetcher(auth_host)
        self.max_retries = max_retries
        self.timeout = timeout
        self.credentials = None
        self.__lock = threading.Lock()

    def get_credentials(self):
        if self.credentials is None or self.credentials.will_soon_expire():
            with self.__lock:
                if self.credentials is None or self.credentials.will_soon_expire():
                    try:
                        self.credentials = self.fetcher.fetch(self.max_retries, self.timeout)
                    except Exception as e:
                        logger.error("Exception: {0}".format(e))
                        if self.credentials is None:
                            raise

        return self.credentials


class EcsRamRoleCredentialsFetcher(object):
    def __init__(self, auth_host):
        self.auth_host = auth_host

    def fetch(self, retry_times=3, timeout=10):
        for i in range(0, retry_times):
            try:
                response = requests.get(self.auth_host, timeout=timeout)
                if response.status_code != 200:
                    raise ClientError(
                        "Failed to fetch credentials url, http code:{0}, msg:{1}".format(response.status_code,
                                                                                         response.text))
                dic = json.loads(to_unicode(response.content))
                code = dic.get('Code')
                access_key_id = dic.get('AccessKeyId')
                access_key_secret = dic.get('AccessKeySecret')
                security_token = dic.get('SecurityToken')
                expiration_date = dic.get('Expiration')
                last_updated_date = dic.get('LastUpdated')

                if code != "Success":
                    raise ClientError("Get credentials from ECS metadata service error, code: {0}".format(code))

                expiration_stamp = to_unixtime(expiration_date, "%Y-%m-%dT%H:%M:%SZ")
                duration = DEFAULT_ECS_SESSION_TOKEN_DURATION_SECONDS
                if last_updated_date is not None:
                    last_updated_stamp = to_unixtime(last_updated_date, "%Y-%m-%dT%H:%M:%SZ")
                    duration = expiration_stamp - last_updated_stamp
                return EcsRamRoleCredential(access_key_id, access_key_secret, security_token, expiration_stamp,
                                            duration, DEFAULT_ECS_SESSION_EXPIRED_FACTOR)
            except Exception as e:
                if i == retry_times - 1:
                    logger.error("Exception: {0}".format(e))
                    raise ClientError("Failed to get credentials from ECS metadata service. {0}".format(e))


class EnvironmentVariableCredentialsProvider(CredentialsProvider):
    def __init__(self):
        self.access_key_id = ""
        self.access_key_secret = ""
        self.security_token = ""

    def get_credentials(self):
        access_key_id = os.getenv('OSS_ACCESS_KEY_ID')
        access_key_secret = os.getenv('OSS_ACCESS_KEY_SECRET')
        security_token = os.getenv('OSS_SESSION_TOKEN')

        if not access_key_id:
            raise ClientError("Access key id should not be null or empty.")
        if not access_key_secret:
            raise ClientError("Secret access key should not be null or empty.")
        return Credentials(access_key_id, access_key_secret, security_token)
