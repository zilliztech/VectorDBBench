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

class CreateKeyRequest(RpcRequest):

	def __init__(self):
		RpcRequest.__init__(self, 'Kms', '2016-01-20', 'CreateKey','kms')
		self.set_protocol_type('https')
		self.set_method('POST')

		if hasattr(self, "endpoint_map"):
			setattr(self, "endpoint_map", endpoint_data.getEndpointMap())
		if hasattr(self, "endpoint_regional"):
			setattr(self, "endpoint_regional", endpoint_data.getEndpointRegional())

	def get_Origin(self): # String
		return self.get_query_params().get('Origin')

	def set_Origin(self, Origin):  # String
		self.add_query_param('Origin', Origin)
	def get_Description(self): # String
		return self.get_query_params().get('Description')

	def set_Description(self, Description):  # String
		self.add_query_param('Description', Description)
	def get_KeySpec(self): # String
		return self.get_query_params().get('KeySpec')

	def set_KeySpec(self, KeySpec):  # String
		self.add_query_param('KeySpec', KeySpec)
	def get_RotationInterval(self): # String
		return self.get_query_params().get('RotationInterval')

	def set_RotationInterval(self, RotationInterval):  # String
		self.add_query_param('RotationInterval', RotationInterval)
	def get_EnableAutomaticRotation(self): # Boolean
		return self.get_query_params().get('EnableAutomaticRotation')

	def set_EnableAutomaticRotation(self, EnableAutomaticRotation):  # Boolean
		self.add_query_param('EnableAutomaticRotation', EnableAutomaticRotation)
	def get_Tags(self): # String
		return self.get_query_params().get('Tags')

	def set_Tags(self, Tags):  # String
		self.add_query_param('Tags', Tags)
	def get_ProtectionLevel(self): # String
		return self.get_query_params().get('ProtectionLevel')

	def set_ProtectionLevel(self, ProtectionLevel):  # String
		self.add_query_param('ProtectionLevel', ProtectionLevel)
	def get_KeyUsage(self): # String
		return self.get_query_params().get('KeyUsage')

	def set_KeyUsage(self, KeyUsage):  # String
		self.add_query_param('KeyUsage', KeyUsage)
	def get_DKMSInstanceId(self): # String
		return self.get_query_params().get('DKMSInstanceId')

	def set_DKMSInstanceId(self, DKMSInstanceId):  # String
		self.add_query_param('DKMSInstanceId', DKMSInstanceId)
	def get_Policy(self): # String
		return self.get_query_params().get('Policy')

	def set_Policy(self, Policy):  # String
		self.add_query_param('Policy', Policy)
