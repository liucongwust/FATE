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

import typing
import functools

import numpy as np
from federatedml.util import consts
from arch.api.utils import log_utils
from arch.api import session as fate_session
from federatedml.statistic import data_overview
from federatedrec.optim.sync import user_ids_transfer_sync
from federatedrec.collaborative_deeplearning.hetero_cdl.hetero_cdl_base import HeteroCDLBase
from federatedrec.collaborative_deeplearning.hetero_cdl.backend import KerasSeqDataConverter, CDLModel

LOGGER = log_utils.getLogger()


class HeteroCDLClient(HeteroCDLBase):
    def __init__(self):
        super(HeteroCDLClient, self).__init__()

        self._model = None
        self.feature_shape = None

    def _init_model(self, param):
        super()._init_model(param)

        self.batch_size = param.batch_size
        self.aggregate_every_n_epoch = 1
        self.optimizer = param.optimizer
        self.loss = param.loss
        self.metrics = param.metrics
        self.data_converter = KerasSeqDataConverter()
        self.user_ids_sync.register_user_ids_transfer(self.transfer_variable)

    def _check_monitored_status(self, data, epoch_degree):
        metrics = self._model.evaluate(data)
        LOGGER.info(f"metrics at iter {self.aggregator_iter}: {metrics}")
        loss = metrics["loss"]
        self.aggregator.send_loss(loss=loss,
                                  degree=epoch_degree,
                                  suffix=self._iter_suffix())
        return self.aggregator.get_converge_status(suffix=self._iter_suffix())

    def send_user_ids(self, data):
        pass

    def get_user_ids(self):
        pass

    def get_features_shape(self, data_instances):
        if self.feature_shape is not None:
            return self.feature_shape
        return data_overview.get_features_shape(data_instances)

    def fit(self, data, validate_data=None):
        data_instances = data[0]
        itemfea_instances = data[1]
        LOGGER.debug("Start data count: {}".format(data_instances.count()))

        # self._abnormal_detection(data_instances)
        # self.init_schema(data_instances)
        # validation_strategy = self.init_validation_strategy(data_instances, validate_data)

        item_fea_dim = self.get_features_shape(itemfea_instances)
        user_ids_table, item_ids_table = self.extract_ids(data_instances)
        self.send_user_ids(user_ids_table)
        received_user_ids = self.get_user_ids()
        join_user_ids = user_ids_table.union(received_user_ids)
        user_ids = sorted([_id for (_id, _) in join_user_ids.collect()])
        LOGGER.debug(f"after join, get user ids {user_ids}")
        item_ids = [_id for (_id, _) in item_ids_table.collect()]
        # user_num, item_num = len(user_ids), len(item_ids)

        # Convert data_inst to keras sequence data
        itemfea_dict = {item_id:inst.features for item_id,inst in itemfea_instances.collect()}
        data = self.data_converter.convert(data_instances, user_ids, item_ids, itemfea_dict,
                                           batch_size=self.batch_size)
        self._model = CDLModel.build_model(user_ids, item_ids, self.params.init_param.embed_dim,
                                           item_fea_dim, self.loss, self.optimizer, self.metrics)

        itemfea_array = [inst.features for item_id,inst in itemfea_instances.collect()]
        itemfea_array = np.array(itemfea_array)
        self._model.pretrain_autoencoder(itemfea_array)

        epoch_degree = float(len(data))

        while self.aggregator_iter < self.max_iter:
            LOGGER.info(f"start {self.aggregator_iter}_th aggregation")

            # train
            self._model.train(data, aggregate_every_n_epoch=self.aggregate_every_n_epoch)

            # send model for aggregate, then set aggregated model to local
            modify_func: typing.Callable = functools.partial(self.aggregator.aggregate_then_get,
                                                             degree=epoch_degree * self.aggregate_every_n_epoch,
                                                             suffix=self._iter_suffix())
            self._model.modify(modify_func)

            # calc loss and check convergence
            if self._check_monitored_status(data, epoch_degree):
                LOGGER.info(f"early stop at iter {self.aggregator_iter}")
                break

            LOGGER.info(f"role {self.role} finish {self.aggregator_iter}_th aggregation")
            self.aggregator_iter += 1
        else:
            LOGGER.warn(f"reach max iter: {self.aggregator_iter}, not converged")

    def export_model(self):
        meta = self._get_meta()
        param = self._get_param()
        return {self.model_meta_name: meta, self.model_param_name: param}

    def _get_meta(self):
        from federatedrec.protobuf.generated import cdl_model_meta_pb2
        meta_pb = cdl_model_meta_pb2.CDLModelMeta()
        meta_pb.params.CopyFrom(self.model_param.generate_pb())
        meta_pb.aggregate_iter = self.aggregator_iter
        return meta_pb

    def _get_param(self):
        from federatedrec.protobuf.generated import cdl_model_param_pb2
        param_pb = cdl_model_param_pb2.CDLModelParam()
        param_pb.saved_model_bytes = self._model.export_model()
        param_pb.user_ids.extend(self._model.user_ids)
        param_pb.item_ids.extend(self._model.item_ids)

        return param_pb

    def predict(self, data_inst):
        # user_ids_table, item_ids_table = self.extract_ids(data_inst)
        # user_ids = sorted([_id for (_id, _) in user_ids_table.collect()])
        # item_ids = [_id for (_id, _) in item_ids_table.collect()]
        if isinstance(data_inst, list):
            data_inst = data_inst[0]

        data = self.data_converter.convert(data_inst, self._model.user_ids, self._model.item_ids, None,
                                           self.batch_size)
        LOGGER.info(f"data_inst count: {data_inst.count()}")
        label_data = data_inst.map(lambda k, v: (k, v.features.get_data(2)))
        LOGGER.info(f"label_data count: {label_data.count()}")
        predict = self._model.predict(data)
        threshold = self.params.predict_param.threshold

        kv = [(x[0], (0 if x[1][0] <= threshold else 1, x[1][0].item())) for x in zip(data.get_keys(), predict)]
        pred_tbl = fate_session.parallelize(kv, include_key=True)
        return label_data.join(pred_tbl, lambda d, pred: [d, pred[0], pred[1], {"label": pred[0]}])

    def load_model(self, model_dict):
        model_dict = list(model_dict["model"].values())[0]
        model_obj = model_dict.get(self.model_param_name)
        meta_obj = model_dict.get(self.model_meta_name)
        self.model_param.restore_from_pb(meta_obj.params)
        self._init_model(self.model_param)
        self.aggregator_iter = meta_obj.aggregate_iter
        # self.nn_model = restore_nn_model(self.config_type, model_obj.saved_model_bytes)
        self._model = CDLModel.restore_model(model_obj.saved_model_bytes)
        self._model.set_user_ids(model_obj.user_ids)
        self._model.set_item_ids(model_obj.item_ids)


class HeteroCDLHost(HeteroCDLClient):

    def __init__(self):
        super().__init__()
        self.role = consts.HOST
        self.user_ids_sync = user_ids_transfer_sync.Host()

    def send_user_ids(self, data):
        self.user_ids_sync.send_host_user_ids(data)

    def get_user_ids(self):
        return self.user_ids_sync.get_guest_user_ids()


class HeteroCDLGuest(HeteroCDLClient):

    def __init__(self):
        super().__init__()
        self.role = consts.GUEST
        self.user_ids_sync = user_ids_transfer_sync.Guest()

    def send_user_ids(self, data):
        self.user_ids_sync.send_guest_user_ids(data)

    def get_user_ids(self):
        return self.user_ids_sync.get_host_user_ids()
