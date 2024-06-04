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

class CreateSecretRequest(RpcRequest):

	def __init__(self):
		RpcRequest.__init__(self, 'Kms', '2016-01-20', 'CreateSecret','kms')
		self.set_protocol_type('https')
		self.set_method('POST')

		if hasattr(self, "endpoint_map"):
			setattr(self, "endpoint_map", endpoint_data.getEndpointMap())
		if hasattr(self, "endpoint_regional"):
			setattr(self, "endpoint_regional", endpoint_data.getEndpointRegional())

	def get_SecretType(self): # String
		return self.get_query_params().get('SecretType')

	def set_SecretType(self, SecretType):  # String
		self.add_query_param('SecretType', SecretType)
	def get_Description(self): # String
		return self.get_query_params().get('Description')

	def set_Description(self, Description):  # String
		self.add_query_param('Description', Description)
	def get_RotationInterval(self): # String
		return self.get_query_params().get('RotationInterval')

	def set_RotationInterval(self, RotationInterval):  # String
		self.add_query_param('RotationInterval', RotationInterval)
	def get_EnableAutomaticRotation(self): # Boolean
		return self.get_query_params().get('EnableAutomaticRotation')

	def set_EnableAutomaticRotation(self, EnableAutomaticRotation):  # Boolean
		self.add_query_param('EnableAutomaticRotation', EnableAutomaticRotation)
	def get_EncryptionKeyId(self): # String
		return self.get_query_params().get('EncryptionKeyId')

	def set_EncryptionKeyId(self, EncryptionKeyId):  # String
		self.add_query_param('EncryptionKeyId', EncryptionKeyId)
	def get_Tags(self): # String
		return self.get_query_params().get('Tags')

	def set_Tags(self, Tags):  # String
		self.add_query_param('Tags', Tags)
	def get_ExtendedConfig(self): # String
		return self.get_query_params().get('ExtendedConfig')

	def set_ExtendedConfig(self, ExtendedConfig):  # String
		self.add_query_param('ExtendedConfig', ExtendedConfig)
	def get_VersionId(self): # String
		return self.get_query_params().get('VersionId')

	def set_VersionId(self, VersionId):  # String
		self.add_query_param('VersionId', VersionId)
	def get_DKMSInstanceId(self): # String
		return self.get_query_params().get('DKMSInstanceId')

	def set_DKMSInstanceId(self, DKMSInstanceId):  # String
		self.add_query_param('DKMSInstanceId', DKMSInstanceId)
	def get_SecretData(self): # String
		return self.get_query_params().get('SecretData')

	def set_SecretData(self, SecretData):  # String
		self.add_query_param('SecretData', SecretData)
	def get_SecretName(self): # String
		return self.get_query_params().get('SecretName')

	def set_SecretName(self, SecretName):  # String
		self.add_query_param('SecretName', SecretName)
	def get_SecretDataType(self): # String
		return self.get_query_params().get('SecretDataType')

	def set_SecretDataType(self, SecretDataType):  # String
		self.add_query_param('SecretDataType', SecretDataType)
	def get_Policy(self): # String
		return self.get_query_params().get('Policy')

	def set_Policy(self, Policy):  # String
		self.add_query_param('Policy', Policy)
