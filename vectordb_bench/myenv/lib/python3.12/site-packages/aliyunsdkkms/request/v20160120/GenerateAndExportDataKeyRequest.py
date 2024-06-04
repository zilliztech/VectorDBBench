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

class GenerateAndExportDataKeyRequest(RpcRequest):

	def __init__(self):
		RpcRequest.__init__(self, 'Kms', '2016-01-20', 'GenerateAndExportDataKey','kms')
		self.set_protocol_type('https')
		self.set_method('POST')

		if hasattr(self, "endpoint_map"):
			setattr(self, "endpoint_map", endpoint_data.getEndpointMap())
		if hasattr(self, "endpoint_regional"):
			setattr(self, "endpoint_regional", endpoint_data.getEndpointRegional())

	def get_KeyId(self): # String
		return self.get_query_params().get('KeyId')

	def set_KeyId(self, KeyId):  # String
		self.add_query_param('KeyId', KeyId)
	def get_KeySpec(self): # String
		return self.get_query_params().get('KeySpec')

	def set_KeySpec(self, KeySpec):  # String
		self.add_query_param('KeySpec', KeySpec)
	def get_NumberOfBytes(self): # Integer
		return self.get_query_params().get('NumberOfBytes')

	def set_NumberOfBytes(self, NumberOfBytes):  # Integer
		self.add_query_param('NumberOfBytes', NumberOfBytes)
	def get_PublicKeyBlob(self): # String
		return self.get_query_params().get('PublicKeyBlob')

	def set_PublicKeyBlob(self, PublicKeyBlob):  # String
		self.add_query_param('PublicKeyBlob', PublicKeyBlob)
	def get_EncryptionContext(self): # String
		return self.get_query_params().get('EncryptionContext')

	def set_EncryptionContext(self, EncryptionContext):  # String
		self.add_query_param('EncryptionContext', EncryptionContext)
	def get_WrappingAlgorithm(self): # String
		return self.get_query_params().get('WrappingAlgorithm')

	def set_WrappingAlgorithm(self, WrappingAlgorithm):  # String
		self.add_query_param('WrappingAlgorithm', WrappingAlgorithm)
	def get_WrappingKeySpec(self): # String
		return self.get_query_params().get('WrappingKeySpec')

	def set_WrappingKeySpec(self, WrappingKeySpec):  # String
		self.add_query_param('WrappingKeySpec', WrappingKeySpec)
