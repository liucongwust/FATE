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
from federatedrec.param.collaborative_deeplearning_param import HeteroCDLParam
from federatedrec.transfer_variable.transfer_class.hetero_cdl_transfer_variable import HeteroCDLTransferVariable

LOGGER = log_utils.getLogger()


class HeteroCDLBase(ModelBase):
    def __init__(self):
        super(HeteroCDLBase, self).__init__()
        self.model_param_name = 'CDLModelParam'
        self.model_meta_name = 'CDLModelMeta'

        self.model_param = HeteroCDLParam()
        self.aggregator = None
        self.user_ids_sync = None

    def _iter_suffix(self):
        return self.aggregator_iter,

    def _init_model(self, params):
        super(HeteroCDLBase, self)._init_model(params)
        self.params = params
        self.transfer_variable = HeteroCDLTransferVariable()
        secure_aggregate = params.secure_aggregate
        self.aggregator = aggregator.with_role(role=self.role,
                                               transfer_variable=self.transfer_variable,
                                               enable_secure_aggregate=secure_aggregate)
        self.max_iter = params.max_iter
        self.aggregator_iter = 0

    @staticmethod
    def extract_ids(data_instances):
        user_ids = data_instances.map(lambda k, v: (v.features.get_data(0), None))
        item_ids = data_instances.map(lambda k, v: (v.features.get_data(1), None))
        return user_ids, item_ids

    def _run_data(self, data_sets=None, stage=None):
        train_data = None
        eval_data = None
        data = None

        for data_key in data_sets:
            if data_sets[data_key].get("train_data", None):
                if train_data is None:
                    train_data = []
                train_data.append(data_sets[data_key]["train_data"])

            if data_sets[data_key].get("eval_data", None):
                eval_data = data_sets[data_key]["eval_data"]

            if data_sets[data_key].get("data", None):
                data = data_sets[data_key]["data"]

        if not self.need_run:
            self.data_output = data
            return data

        if stage == 'cross_validation':
            LOGGER.info("Need cross validation.")
            self.cross_validation(train_data)

        elif train_data is not None:
            train_data, itemfea_data = train_data[0], train_data[1]
            self.set_flowid('fit')
            self.fit(train_data, itemfea_data, eval_data)
            self.set_flowid('predict')
            self.data_output = self.predict(train_data)

            if self.data_output:
                self.data_output = self.data_output.mapValues(lambda value: value + ["train"])

            if eval_data:
                self.set_flowid('validate')
                eval_data_output = self.predict(eval_data)

                if eval_data_output:
                    eval_data_output = eval_data_output.mapValues(lambda value: value + ["validation"])

                if self.data_output and eval_data_output:
                    self.data_output = self.data_output.union(eval_data_output)
                elif not self.data_output and eval_data_output:
                    self.data_output = eval_data_output

            self.set_predict_data_schema(self.data_output, train_data.schema)

        elif eval_data is not None:
            self.set_flowid('predict')
            self.data_output = self.predict(eval_data)

            if self.data_output:
                self.data_output = self.data_output.mapValues(lambda value: value + ["test"])

            self.set_predict_data_schema(self.data_output, eval_data.schema)

        else:
            if stage == "fit":
                self.set_flowid('fit')
                self.data_output = self.fit(data)
            else:
                self.set_flowid('transform')
                self.data_output = self.transform(data)

        if self.data_output:
            # LOGGER.debug("data is {}".format(self.data_output.first()[1].features))
            LOGGER.debug("In model base, data_output schema: {}".format(self.data_output.schema))