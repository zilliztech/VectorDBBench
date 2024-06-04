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

class CreateCertificateRequest(RpcRequest):

	def __init__(self):
		RpcRequest.__init__(self, 'Kms', '2016-01-20', 'CreateCertificate','kms')
		self.set_protocol_type('https')
		self.set_method('POST')

		if hasattr(self, "endpoint_map"):
			setattr(self, "endpoint_map", endpoint_data.getEndpointMap())
		if hasattr(self, "endpoint_regional"):
			setattr(self, "endpoint_regional", endpoint_data.getEndpointRegional())

	def get_Subject(self): # String
		return self.get_query_params().get('Subject')

	def set_Subject(self, Subject):  # String
		self.add_query_param('Subject', Subject)
	def get_KeySpec(self): # String
		return self.get_query_params().get('KeySpec')

	def set_KeySpec(self, KeySpec):  # String
		self.add_query_param('KeySpec', KeySpec)
	def get_ExportablePrivateKey(self): # Boolean
		return self.get_query_params().get('ExportablePrivateKey')

	def set_ExportablePrivateKey(self, ExportablePrivateKey):  # Boolean
		self.add_query_param('ExportablePrivateKey', ExportablePrivateKey)
	def get_SubjectAlternativeNames(self): # Json
		return self.get_query_params().get('SubjectAlternativeNames')

	def set_SubjectAlternativeNames(self, SubjectAlternativeNames):  # Json
		self.add_query_param('SubjectAlternativeNames', SubjectAlternativeNames)
