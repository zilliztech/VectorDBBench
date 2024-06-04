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

class GetRandomPasswordRequest(RpcRequest):

	def __init__(self):
		RpcRequest.__init__(self, 'Kms', '2016-01-20', 'GetRandomPassword','kms')
		self.set_protocol_type('https')
		self.set_method('POST')

		if hasattr(self, "endpoint_map"):
			setattr(self, "endpoint_map", endpoint_data.getEndpointMap())
		if hasattr(self, "endpoint_regional"):
			setattr(self, "endpoint_regional", endpoint_data.getEndpointRegional())

	def get_ExcludeCharacters(self): # String
		return self.get_query_params().get('ExcludeCharacters')

	def set_ExcludeCharacters(self, ExcludeCharacters):  # String
		self.add_query_param('ExcludeCharacters', ExcludeCharacters)
	def get_PasswordLength(self): # String
		return self.get_query_params().get('PasswordLength')

	def set_PasswordLength(self, PasswordLength):  # String
		self.add_query_param('PasswordLength', PasswordLength)
	def get_ExcludePunctuation(self): # String
		return self.get_query_params().get('ExcludePunctuation')

	def set_ExcludePunctuation(self, ExcludePunctuation):  # String
		self.add_query_param('ExcludePunctuation', ExcludePunctuation)
	def get_RequireEachIncludedType(self): # String
		return self.get_query_params().get('RequireEachIncludedType')

	def set_RequireEachIncludedType(self, RequireEachIncludedType):  # String
		self.add_query_param('RequireEachIncludedType', RequireEachIncludedType)
	def get_ExcludeNumbers(self): # String
		return self.get_query_params().get('ExcludeNumbers')

	def set_ExcludeNumbers(self, ExcludeNumbers):  # String
		self.add_query_param('ExcludeNumbers', ExcludeNumbers)
	def get_ExcludeLowercase(self): # String
		return self.get_query_params().get('ExcludeLowercase')

	def set_ExcludeLowercase(self, ExcludeLowercase):  # String
		self.add_query_param('ExcludeLowercase', ExcludeLowercase)
	def get_ExcludeUppercase(self): # String
		return self.get_query_params().get('ExcludeUppercase')

	def set_ExcludeUppercase(self, ExcludeUppercase):  # String
		self.add_query_param('ExcludeUppercase', ExcludeUppercase)
