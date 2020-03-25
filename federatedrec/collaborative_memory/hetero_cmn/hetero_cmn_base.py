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

from arch.api.utils import log_utils
from federatedml.model_base import ModelBase
from federatedml.framework.homo.procedure import aggregator
from federatedrec.param.collaborative_memory_param import HeteroCMNParam
from federatedml.transfer_variable.transfer_class.hetero_cmn_transfer_variable import HeteroCMNTransferVariable

LOGGER = log_utils.getLogger()


class HeteroCMNBase(ModelBase):
    def __init__(self):
        super(HeteroCMNBase, self).__init__()
        self.model_name = 'HeteroCMNModel'
        self.model_param_name = 'CMNModelParam'
        self.model_meta_name = 'CMNModelMeta'

        self.model_param = HeteroCMNParam()
        self.aggregator = None
        self.user_num_sync = None

    def _iter_suffix(self):
        return self.aggregator_iter,

    def _init_model(self, params):
        super(HeteroCMNBase, self)._init_model(params)
        self.params = params
        self.transfer_variable = HeteroCMNTransferVariable()
        secure_aggregate = params.secure_aggregate
        self.aggregator = aggregator.with_role(role=self.role,
                                               transfer_variable=self.transfer_variable,
                                               enable_secure_aggregate=secure_aggregate)
        self.max_iter = params.max_iter
        self.aggregator_iter = 0