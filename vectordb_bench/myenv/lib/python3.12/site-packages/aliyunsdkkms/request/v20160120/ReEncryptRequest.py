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

class ReEncryptRequest(RpcRequest):

	def __init__(self):
		RpcRequest.__init__(self, 'Kms', '2016-01-20', 'ReEncrypt','kms')
		self.set_protocol_type('https')
		self.set_method('POST')

		if hasattr(self, "endpoint_map"):
			setattr(self, "endpoint_map", endpoint_data.getEndpointMap())
		if hasattr(self, "endpoint_regional"):
			setattr(self, "endpoint_regional", endpoint_data.getEndpointRegional())

	def get_DestinationEncryptionContext(self): # String
		return self.get_query_params().get('DestinationEncryptionContext')

	def set_DestinationEncryptionContext(self, DestinationEncryptionContext):  # String
		self.add_query_param('DestinationEncryptionContext', DestinationEncryptionContext)
	def get_SourceKeyId(self): # String
		return self.get_query_params().get('SourceKeyId')

	def set_SourceKeyId(self, SourceKeyId):  # String
		self.add_query_param('SourceKeyId', SourceKeyId)
	def get_SourceEncryptionAlgorithm(self): # String
		return self.get_query_params().get('SourceEncryptionAlgorithm')

	def set_SourceEncryptionAlgorithm(self, SourceEncryptionAlgorithm):  # String
		self.add_query_param('SourceEncryptionAlgorithm', SourceEncryptionAlgorithm)
	def get_SourceKeyVersionId(self): # String
		return self.get_query_params().get('SourceKeyVersionId')

	def set_SourceKeyVersionId(self, SourceKeyVersionId):  # String
		self.add_query_param('SourceKeyVersionId', SourceKeyVersionId)
	def get_DestinationKeyId(self): # String
		return self.get_query_params().get('DestinationKeyId')

	def set_DestinationKeyId(self, DestinationKeyId):  # String
		self.add_query_param('DestinationKeyId', DestinationKeyId)
	def get_SourceEncryptionContext(self): # String
		return self.get_query_params().get('SourceEncryptionContext')

	def set_SourceEncryptionContext(self, SourceEncryptionContext):  # String
		self.add_query_param('SourceEncryptionContext', SourceEncryptionContext)
	def get_CiphertextBlob(self): # String
		return self.get_query_params().get('CiphertextBlob')

	def set_CiphertextBlob(self, CiphertextBlob):  # String
		self.add_query_param('CiphertextBlob', CiphertextBlob)
