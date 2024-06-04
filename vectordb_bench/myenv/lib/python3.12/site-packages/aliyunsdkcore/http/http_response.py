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

import os
import logging

from aliyunsdkcore.vendored.requests import Session, Request
from aliyunsdkcore.http.http_request import HttpRequest
from aliyunsdkcore.http import protocol_type as PT

from aliyunsdkcore.vendored.requests import status_codes

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
logger.addHandler(ch)

DEFAULT_CONNECT_TIMEOUT = 5


class HttpResponse(HttpRequest):
    def __init__(
            self,
            host="",
            url="/",
            method="GET",
            headers={},
            protocol=PT.HTTP,
            content=None,
            port=None,
            key_file=None,
            cert_file=None,
            read_timeout=None,
            connect_timeout=None,
            verify=None,
            session=None,
            proxy=None
    ):
        HttpRequest.__init__(
            self,
            host=host,
            url=url,
            method=method,
            headers=headers)
        self.__ssl_enable = False
        if protocol is PT.HTTPS:
            self.__ssl_enable = True
        self.__key_file = key_file
        self.__cert_file = cert_file
        self.__port = port
        self.__connection = None
        self.__read_timeout = read_timeout
        self.__connect_timeout = connect_timeout
        self.__verify = verify
        self.__session = session
        if session is None:
            self.__session = Session()
        self.set_body(content)
        self.proxy = proxy

    def set_ssl_enable(self, enable):
        self.__ssl_enable = enable

    def get_ssl_enabled(self):
        return self.__ssl_enable

    @staticmethod
    def prepare_http_debug(request, symbol):
        base = ''
        for key, value in request.headers.items():
            base += '\n%s %s : %s' % (symbol, key, value)
        return base

    def do_http_debug(self, request, response):
        # logger the request
        request_base = '\n> %s %s HTTP/1.1' % (self.get_method().upper(), self.get_url())
        request_base += '\n> Host : %s' % self.get_host()
        logger.debug(request_base + self.prepare_http_debug(request, '>'))

        # logger the response
        response_base = '\n< HTTP/1.1 %s %s' % (
            response.status_code, status_codes._codes.get(response.status_code)[0].upper())
        logger.debug(response_base + self.prepare_http_debug(response, '<'))

    def get_verify_value(self):
        if self.__verify is not None:
            return self.__verify
        return os.environ.get('ALIBABA_CLOUD_CA_BUNDLE', True)

    def get_response_object(self):
        current_protocol = 'https://' if self.get_ssl_enabled() else 'http://'
        host = self.get_host()
        if host.startswith('https://') or\
                not host.startswith('https://') and current_protocol == 'https://':
            port = ':%s' % self.__port if self.__port != 80 and self.__port != 443 else ''
        else:
            port = ':%s' % self.__port if self.__port != 80 else ''

        if host.startswith('http://') or host.startswith('https://'):
            url = host + port + self.get_url()
        else:
            url = current_protocol + host + port + self.get_url()

        self.__session.cookies.clear()

        req = Request(method=self.get_method(), url=url,
                      data=self.get_body(),
                      headers=self.get_headers(),
                      )
        prepped = self.__session.prepare_request(req)
        if not self.proxy:
            proxy_https = os.environ.get('HTTPS_PROXY') or os.environ.get(
                'https_proxy')
            proxy_http = os.environ.get(
                'HTTP_PROXY') or os.environ.get('http_proxy')

            self.proxy = {}
            if proxy_http:
                self.proxy['http'] = proxy_http
            if proxy_https:
                self.proxy['https'] = proxy_https

        response = self.__session.send(
            prepped,
            proxies=self.proxy,
            timeout=(self.__connect_timeout, self.__read_timeout),
            allow_redirects=False,
            verify=self.get_verify_value()
        )

        http_debug = os.environ.get('DEBUG')

        if http_debug is not None and http_debug.lower() == 'sdk':
            # http debug information
            self.do_http_debug(prepped, response)

        return response.status_code, response.headers, response.content
