#!/usr/bin/env python
# -*- coding: utf-8 -*-

#
#  Copyright 2019 The FATE Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

################################################################################
#
# AUTO GENERATED TRANSFER VARIABLE CLASS. DO NOT MODIFY
#
################################################################################

from federatedml.transfer_variable.transfer_class.base_transfer_variable import BaseTransferVariable, Variable


# noinspection PyAttributeOutsideInit
class HeteroNCFTransferVariable(BaseTransferVariable):
    def define_transfer_variable(self):
        self.guest_uuid = Variable(name='HeteroNCFTransferVariable.guest_uuid', auth=dict(src='guest', dst=['arbiter']), transfer_variable=self)
        self.host_uuid = Variable(name='HeteroNCFTransferVariable.host_uuid', auth=dict(src='host', dst=['arbiter']), transfer_variable=self)
        self.uuid_conflict_flag = Variable(name='HeteroNCFTransferVariable.uuid_conflict_flag', auth=dict(src='arbiter', dst=['guest', 'host']), transfer_variable=self)
        self.dh_pubkey = Variable(name='HeteroNCFTransferVariable.dh_pubkey', auth=dict(src='arbiter', dst=['guest', 'host']), transfer_variable=self)
        self.dh_ciphertext_host = Variable(name='HeteroNCFTransferVariable.dh_ciphertext_host', auth=dict(src='host', dst=['arbiter']), transfer_variable=self)
        self.dh_ciphertext_guest = Variable(name='HeteroNCFTransferVariable.dh_ciphertext_guest', auth=dict(src='guest', dst=['arbiter']), transfer_variable=self)
        self.dh_ciphertext_bc = Variable(name='HeteroNCFTransferVariable.dh_ciphertext_bc', auth=dict(src='arbiter', dst=['guest', 'host']), transfer_variable=self)
        self.paillier_pubkey = Variable(name='HeteroNCFTransferVariable.paillier_pubkey', auth=dict(src='arbiter', dst=['host']), transfer_variable=self)
        self.guest_model = Variable(name='HeteroNCFTransferVariable.guest_model', auth=dict(src='guest', dst=['arbiter']), transfer_variable=self)
        self.host_model = Variable(name='HeteroNCFTransferVariable.host_model', auth=dict(src='host', dst=['arbiter']), transfer_variable=self)
        self.aggregated_model = Variable(name='HeteroNCFTransferVariable.aggregated_model', auth=dict(src='arbiter', dst=['guest', 'host']), transfer_variable=self)
        self.is_converge = Variable(name='HeteroNCFTransferVariable.is_converge', auth=dict(src='arbiter', dst=['guest', 'host']), transfer_variable=self)
        self.guest_loss = Variable(name='HeteroNCFTransferVariable.guest_loss', auth=dict(src='guest', dst=['arbiter']), transfer_variable=self)
        self.host_loss = Variable(name='HeteroNCFTransferVariable.host_loss', auth=dict(src='host', dst=['arbiter']), transfer_variable=self)
        self.host_user_ids = Variable(name='HeteroNCFTransferVariable.host_user_ids', auth=dict(src='host', dst=['guest']), transfer_variable=self)
        self.guest_user_ids = Variable(name='HeteroNCFTransferVariable.guest_user_ids', auth=dict(src='guest', dst=['host']), transfer_variable=self)
        self.host_user_num = Variable(name='HeteroNCFTransferVariable.host_user_num',
                                      auth=dict(src='host', dst=['guest']), transfer_variable=self)
        self.guest_user_num = Variable(name='HeteroNCFTransferVariable.guest_user_num',
                                       auth=dict(src='guest', dst=['host']), transfer_variable=self)

        pass
