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

class CreatePolicyRequest(RpcRequest):

	def __init__(self):
		RpcRequest.__init__(self, 'Kms', '2016-01-20', 'CreatePolicy','kms')
		self.set_protocol_type('https')
		self.set_method('POST')

		if hasattr(self, "endpoint_map"):
			setattr(self, "endpoint_map", endpoint_data.getEndpointMap())
		if hasattr(self, "endpoint_regional"):
			setattr(self, "endpoint_regional", endpoint_data.getEndpointRegional())

	def get_AccessControlRules(self): # String
		return self.get_query_params().get('AccessControlRules')

	def set_AccessControlRules(self, AccessControlRules):  # String
		self.add_query_param('AccessControlRules', AccessControlRules)
	def get_Description(self): # String
		return self.get_query_params().get('Description')

	def set_Description(self, Description):  # String
		self.add_query_param('Description', Description)
	def get_Resources(self): # String
		return self.get_query_params().get('Resources')

	def set_Resources(self, Resources):  # String
		self.add_query_param('Resources', Resources)
	def get_KmsInstance(self): # String
		return self.get_query_params().get('KmsInstance')

	def set_KmsInstance(self, KmsInstance):  # String
		self.add_query_param('KmsInstance', KmsInstance)
	def get_Permissions(self): # String
		return self.get_query_params().get('Permissions')

	def set_Permissions(self, Permissions):  # String
		self.add_query_param('Permissions', Permissions)
	def get_Name(self): # String
		return self.get_query_params().get('Name')

	def set_Name(self, Name):  # String
		self.add_query_param('Name', Name)
