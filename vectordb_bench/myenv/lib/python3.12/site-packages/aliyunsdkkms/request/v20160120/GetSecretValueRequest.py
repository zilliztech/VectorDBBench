# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

from aliyunsdkcore.request import RpcRequest
from aliyunsdkkms.endpoint import endpoint_data

class GetSecretValueRequest(RpcRequest):

	def __init__(self):
		RpcRequest.__init__(self, 'Kms', '2016-01-20', 'GetSecretValue','kms')
		self.set_protocol_type('https')
		self.set_method('POST')

		if hasattr(self, "endpoint_map"):
			setattr(self, "endpoint_map", endpoint_data.getEndpointMap())
		if hasattr(self, "endpoint_regional"):
			setattr(self, "endpoint_regional", endpoint_data.getEndpointRegional())

	def get_VersionId(self): # String
		return self.get_query_params().get('VersionId')

	def set_VersionId(self, VersionId):  # String
		self.add_query_param('VersionId', VersionId)
	def get_VersionStage(self): # String
		return self.get_query_params().get('VersionStage')

	def set_VersionStage(self, VersionStage):  # String
		self.add_query_param('VersionStage', VersionStage)
	def get_SecretName(self): # String
		return self.get_query_params().get('SecretName')

	def set_SecretName(self, SecretName):  # String
		self.add_query_param('SecretName', SecretName)
	def get_FetchExtendedConfig(self): # Boolean
		return self.get_query_params().get('FetchExtendedConfig')

	def set_FetchExtendedConfig(self, FetchExtendedConfig):  # Boolean
		self.add_query_param('FetchExtendedConfig', FetchExtendedConfig)
