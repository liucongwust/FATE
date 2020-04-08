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
import traceback

import numpy as np

from federatedml.util import consts
from arch.api.utils import log_utils
from arch.api import session as fate_session
from federatedml.statistic import data_overview
from federatedrec.optim.sync import dnnrec_embedding_transfer_sync
from federatedrec.dnn_rec.hetero_dnnrec.hetero_dnnrec_base import HeteroDNNRecBase
from federatedrec.dnn_rec.hetero_dnnrec.backend import DNNRecModel
from federatedrec.dnn_rec.hetero_dnnrec.dnnrec_data_convertor import DNNRecDataConverter

LOGGER = log_utils.getLogger()


class HeteroDNNRecClient(HeteroDNNRecBase):
    def __init__(self):
        super(HeteroDNNRecClient, self).__init__()

        self.max_clk_num = None
        self.genres_dim = None
        self.tags_dim = None
        self.title_dim = None
        self.max_title_len = None
        self.user_items = {}
        self._model = None
        self.feature_shape = None
        self.user_num = None
        self.item_num = None
        self.aggregator_iter = None

    def _init_model(self, param):
        super()._init_model(param)

        self.batch_size = param.batch_size
        self.aggregate_every_n_epoch = 1
        self.optimizer = param.optimizer
        self.loss = param.loss
        self.metrics = param.metrics
        self.data_converter = DNNRecDataConverter()
        self.user_num_sync.register_user_num_transfer(self.transfer_variable)

    def _check_monitored_status(self, data, epoch_degree):
        """
        check the model whether is converged or not
        :param data:
        :param epoch_degree:
        :return:
        """
        # metrics = self._model.evaluate(data)
        # user_ids = data.user_ids
        #
        # LOGGER.info(f"metrics at iter {self.aggregator_iter}: {metrics}")
        # loss = metrics["loss"]
        # self.aggregator.send_loss(loss=loss,
        #                           degree=epoch_degree,
        #                           suffix=self._iter_suffix())
        # return self.aggregator.get_converge_status(suffix=self._iter_suffix())
        return False

    def send_user_num(self, data):
        pass

    def get_user_num(self):
        pass

    def send_user_ids(self, data):
        pass

    def get_user_ids(self):
        pass

    def send_item_embedding(self, data):
        pass

    def get_item_embedding(self):
        pass

    def get_features_shape(self, data_instances):
        if self.feature_shape is not None:
            return self.feature_shape
        return data_overview.get_features_shape(data_instances)

    def get_user_clicks(self, user_ids):
        user_nums = len(user_ids)
        clk_dims = self.max_clk_num
        user_clk_items = np.zeros((user_nums, clk_dims), dtype=np.int)

        for idx, uid in enumerate(user_ids):
            clk_items = np.array(self.user_items.get(uid, []))
            user_clk_items[idx, :len(clk_items)] = clk_items
        return user_clk_items

    def fit(self, data_instances, validate_data=None):
        """
        train model
        :param data_instances: training data
        :param validate_data:  validation data
        :return:
        """
        try:
            data = self.data_converter.convert(data_instances, batch_size=self.batch_size)

            user_num = data.user_count
            item_num = data.item_count
            title_dim = data.title_dim
            genres_dim = data.genres_dim
            tags_dim = data.tag_dim
            max_clk_num = data.max_clicks
            max_title_len = data.max_title_len

            LOGGER.info(f'send user_num')
            self.send_user_num(user_num)
            LOGGER.info(f'get remote user_num')
            remote_user_num = self.get_user_num()
            LOGGER.info(f'local user num: {user_num}, remote user num: {remote_user_num}')
            self.user_num = max(remote_user_num, user_num)
            self.item_num = item_num
            self.max_clk_num = max_clk_num
            self.genres_dim = genres_dim
            self.tags_dim = tags_dim
            self.title_dim = title_dim
            self.user_items = data.user_click_items
            self.max_title_len = max_title_len

            self._model = DNNRecModel.build_model(user_num=self.user_num, item_num=item_num,
                                                  embedding_dim=self.params.init_param.embed_dim,
                                                  title_dim=title_dim, max_title_len=max_title_len,
                                                  genres_dim=genres_dim, tags_dim=tags_dim,
                                                  max_clk_num=max_clk_num,
                                                  mlp_params=self.model_param.mlp_params,
                                                  loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)

            epoch_degree = float(len(data))

            user_ids = data.user_ids
            user_idx_dict = dict([(user_id, u_idx) for u_idx, user_id in enumerate(user_ids)])
            LOGGER.info(f'send user_ids, user_ids len: {len(user_ids)}, max: {max(user_ids)}')
            self.send_user_ids(user_ids)
            LOGGER.info(f'get remote user_ids')
            remote_user_ids = self.get_user_ids()
            LOGGER.info(f"remote_user_ids len: {len(remote_user_ids)}")
            # LOGGER.info(f'local user ids: {user_ids}, \nremote user ids: {remote_user_ids}')
            user_clicked_items = self.get_user_clicks(remote_user_ids)

            while self.aggregator_iter < self.max_iter:
                LOGGER.info(f"start {self.aggregator_iter}_th aggregation")

                avg_item_embeddings = self._model.generate_item_embedding(user_clicked_items)
                LOGGER.info(f"avg_item_embeddings shape: {avg_item_embeddings.shape}")
                LOGGER.info(f"avg_item_embed: {avg_item_embeddings[:2, :]}")

                LOGGER.info(f'send item_embeddings')
                self.send_item_embedding(avg_item_embeddings)
                LOGGER.info(f'get remote item_embeddings')
                remote_item_embeddings = self.get_item_embedding()[:len(user_ids), :]
                LOGGER.info(f"remote_item_embeddings shape: {remote_item_embeddings.shape}")
                LOGGER.info(f"remote item embeddings:{remote_item_embeddings[:2, :]}")

                for idx in range(data.__len__()):
                    x, y = data.__getitem__(index=idx)
                    tmp_user_ids = x[0].astype(int).tolist()
                    LOGGER.debug(f"begin train tmp_user_ids len: {len(tmp_user_ids)}")

                    tmp_idx = [user_idx_dict.get(uid, 0) for uid in tmp_user_ids]
                    tmp_remote_embeddings = remote_item_embeddings[tmp_idx, :]
                    x.append(tmp_remote_embeddings)
                    # train
                    LOGGER.debug(f"begin train idx: {idx}")
                    self._model.train_on_batch(x=x, y=y, aggregate_every_n_epoch=self.aggregate_every_n_epoch)

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
        except Exception as e:
            LOGGER.info(traceback.format_exc())


    # def fit(self, data_instances, validate_data=None):
    #     """
    #     train model
    #     :param data_instances: training data
    #     :param validate_data:  validation data
    #     :return:
    #     """
    #     data = self.data_converter.convert(data_instances, batch_size=self.batch_size)
    #
    #     user_num = data.user_count
    #     item_num = data.item_count
    #     title_dim = data.title_dim
    #     genres_dim = data.genres_dim
    #     tags_dim = data.tag_dim
    #     max_clk_num = data.max_clicks
    #
    #     LOGGER.info(f'send user_num')
    #     self.send_user_num(user_num)
    #     LOGGER.info(f'get remote user_num')
    #     remote_user_num = self.get_user_num()
    #     LOGGER.info(f'local user num: {user_num}, remote user num: {remote_user_num}')
    #     self.user_num = max(remote_user_num, user_num)
    #     self.item_num = item_num
    #     self.max_clk_num = max_clk_num
    #     self.genres_dim = genres_dim
    #     self.tags_dim = tags_dim
    #     self.title_dim = title_dim
    #
    #     self._model = DNNRecModel.build_model(user_num=self.user_num, item_num=item_num,
    #                                           embedding_dim=self.params.init_param.embed_dim,
    #                                           title_dim=title_dim, genres_dim=genres_dim, tags_dim=tags_dim,
    #                                           max_clk_num=max_clk_num,
    #                                           mlp_params=self.model_param.mlp_params,
    #                                           loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
    #
    #     epoch_degree = float(len(data))
    #
    #     while self.aggregator_iter < self.max_iter:
    #         LOGGER.info(f"start {self.aggregator_iter}_th aggregation")
    #
    #         # train
    #         LOGGER.debug(f"begin train")
    #         self._model.train(data, aggregate_every_n_epoch=self.aggregate_every_n_epoch)
    #         LOGGER.debug(f"after train")
    #
    #         # send model for aggregate, then set aggregated model to local
    #         modify_func: typing.Callable = functools.partial(self.aggregator.aggregate_then_get,
    #                                                          degree=epoch_degree * self.aggregate_every_n_epoch,
    #                                                          suffix=self._iter_suffix())
    #         self._model.modify(modify_func)
    #
    #         # calc loss and check convergence
    #         if self._check_monitored_status(data, epoch_degree):
    #             LOGGER.info(f"early stop at iter {self.aggregator_iter}")
    #             break
    #
    #         LOGGER.info(f"role {self.role} finish {self.aggregator_iter}_th aggregation")
    #         self.aggregator_iter += 1
    #     else:
    #         LOGGER.warn(f"reach max iter: {self.aggregator_iter}, not converged")

    def export_model(self):
        """
        export model
        :return: saved model dict
        """
        meta = self._get_meta()
        param = self._get_param()
        model_dict = {self.model_meta_name: meta, self.model_param_name: param}
        LOGGER.info(f"model_dict keys: {model_dict.keys()}")
        return model_dict

    def _get_meta(self):
        """
        get meta data for saving model
        :return:
        """
        from federatedml.protobuf.generated import dnnrec_model_meta_pb2
        LOGGER.info(f"_get_meta")
        meta_pb = dnnrec_model_meta_pb2.DNNRecModelMeta()
        meta_pb.params.CopyFrom(self.model_param.generate_pb())
        meta_pb.aggregate_iter = self.aggregator_iter
        return meta_pb

    def _get_param(self):
        """
        get model param for saving model
        :return:
        """
        from federatedml.protobuf.generated import dnnrec_model_param_pb2
        LOGGER.info(f"_get_param")
        param_pb = dnnrec_model_param_pb2.DNNRecModelParam()
        param_pb.saved_model_bytes = self._model.export_model()
        param_pb.user_num = self.user_num
        param_pb.item_num = self.item_num
        param_pb.title_dim = self.title_dim
        param_pb.genres_dim = self.genres_dim
        param_pb.tags_dim = self.tags_dim
        param_pb.max_clk_num = self.max_clk_num
        return param_pb

    def predict(self, data_inst):
        """
        predicton function. Note that: NCF model use different DataConverter in evaluation and prediction procedure.
        :param data_inst: data instance
        :return: the prediction results
        """
        LOGGER.info(f"data_inst type: {type(data_inst)}, size: {data_inst.count()}, table name: {data_inst.get_name()}")
        LOGGER.info(f"current flowid: {self.flowid}")

        data = self.data_converter.convert(data_inst, batch_size=self.batch_size)

        LOGGER.info(f"data example: {data_inst.first()[1].features.astype(str)}")
        LOGGER.info(f"converted data, size :{data.size}")
        predict = None

        user_ids = data.user_ids
        user_idx_dict = dict([(user_id, idx) for idx, user_id in enumerate(user_ids)])
        LOGGER.info(f'send user_ids, user_ids len: {len(user_ids)}, max: {max(user_ids)}')
        self.send_user_ids(user_ids)
        LOGGER.info(f'get remote user_ids')
        remote_user_ids = self.get_user_ids()
        LOGGER.info(f"remote_user_ids len: {len(remote_user_ids)}")
        # LOGGER.info(f'local user ids: {user_ids}, \nremote user ids: {remote_user_ids}')
        user_clicked_items = self.get_user_clicks(remote_user_ids)

        avg_item_embeddings = self._model.generate_item_embedding(user_clicked_items)
        LOGGER.info(f"avg_item_embeddings shape: {avg_item_embeddings.shape}")
        LOGGER.info(f"avg_item_embed: {avg_item_embeddings[:2, :]}")

        LOGGER.info(f'send item_embeddings')
        self.send_item_embedding(avg_item_embeddings)
        LOGGER.info(f'get remote item_embeddings')
        remote_item_embeddings = self.get_item_embedding()[:len(user_ids), :]
        LOGGER.info(f"remote_item_embeddings shape: {remote_item_embeddings.shape}")
        LOGGER.info(f"remote item embeddings:{remote_item_embeddings[:2, :]}")

        for idx in range(data.__len__()):
            x, y = data.__getitem__(index=idx)
            tmp_user_ids = x[0].astype(int).tolist()
            tmp_idx = [user_idx_dict.get(uid, 0) for uid in tmp_user_ids]
            tmp_remote_embeddings = remote_item_embeddings[tmp_idx, :]
            x.append(tmp_remote_embeddings)

            tmp_predict = self._model.predict_on_batch(data=x)
            LOGGER.info(f"predict shape:{tmp_predict.shape}")

            if predict is not None:
                predict = np.concatenate((predict, tmp_predict), axis=0)
            else:
                predict = tmp_predict

        LOGGER.info(f"predict shape: {predict.shape}")

        threshold = self.params.predict_param.threshold

        kv = [(x[0], (0 if x[1] <= threshold else 1, x[1].item())) for x in zip(data.get_keys(), predict)]
        pred_tbl = fate_session.parallelize(kv, include_key=True)
        pred_data = data_inst.join(pred_tbl, lambda d, pred: [d.label, pred[0], pred[1], {"label": pred[0]}])
        LOGGER.info(f"pred_data sample: {pred_data.take(20)}")
        return pred_data

    # def predict(self, data_inst):
    #     """
    #     predicton function. Note that: NCF model use different DataConverter in evaluation and prediction procedure.
    #     :param data_inst: data instance
    #     :return: the prediction results
    #     """
    #     LOGGER.info(f"data_inst type: {type(data_inst)}, size: {data_inst.count()}, table name: {data_inst.get_name()}")
    #     LOGGER.info(f"current flowid: {self.flowid}")
    #
    #     data = self.data_converter.convert(data_inst, batch_size=self.batch_size)
    #
    #     LOGGER.info(f"data example: {data_inst.first()[1].features.astype(str)}")
    #     LOGGER.info(f"converted data, size :{data.size}")
    #     predict = self._model.predict(data)
    #     LOGGER.info(f"predict shape: {predict.shape}")
    #     threshold = self.params.predict_param.threshold
    #
    #     kv = [(x[0], (0 if x[1] <= threshold else 1, x[1].item())) for x in zip(data.get_keys(), predict)]
    #     pred_tbl = fate_session.parallelize(kv, include_key=True)
    #     pred_data = data_inst.join(pred_tbl, lambda d, pred: [d.label, pred[0], pred[1], {"label": pred[0]}])
    #     LOGGER.info(f"pred_data sample: {pred_data.take(20)}")
    #     return pred_data

    def load_model(self, model_dict):
        """
        load model from saved model, and initialize the model params
        :param model_dict:
        :return:
        """
        model_dict = list(model_dict["model"].values())[0]
        model_obj = model_dict.get(self.model_param_name)
        meta_obj = model_dict.get(self.model_meta_name)
        self.user_num = model_obj.user_num
        self.item_num = model_obj.item_num
        self.title_dim = model_obj.title_dim
        self.genres_dim = model_obj.genres_dim
        self.tags_dim = model_obj.tags_dim
        self.max_clk_num = model_obj.max_clk_num
        self.model_param.restore_from_pb(meta_obj.params)
        self._init_model(self.model_param)
        self.aggregator_iter = meta_obj.aggregate_iter
        LOGGER.info(f"title_dim: {model_obj.title_dim}, genres_dim: {model_obj.genres_dim},"
                    f"tags_dim: {model_obj.tags_dim}, max_clk_num: {model_obj.max_clk_num}")
        self._model = DNNRecModel.restore_model(model_obj.saved_model_bytes, model_obj.user_num, model_obj.item_num,
                                                self.model_param.init_param.embed_dim, model_obj.title_dim,
                                                model_obj.genres_dim, model_obj.tags_dim, model_obj.max_clk_num)
        self._model.set_user_num(model_obj.user_num)
        self._model.set_item_num(model_obj.item_num)


class HeteroDNNRecHost(HeteroDNNRecClient):
    """
    Host HeteroDNNRec Class, implement the get_user_num and send_user_num function
    """

    def __init__(self):
        super().__init__()
        self.role = consts.HOST
        self.user_num_sync = dnnrec_embedding_transfer_sync.Host()

    def send_user_num(self, data):
        self.user_num_sync.send_host_user_num(data)

    def get_user_num(self):
        return self.user_num_sync.get_guest_user_num()

    def send_user_ids(self, data):
        self.user_num_sync.send_host_user_ids(data)

    def get_user_ids(self):
        return self.user_num_sync.get_guest_user_ids()

    def send_item_embedding(self, data):
        self.user_num_sync.send_host_item_embedding(data)

    def get_item_embedding(self):
        return self.user_num_sync.get_guest_item_embedding()


class HeteroDNNRecGuest(HeteroDNNRecClient):
    """
    Guest HeteroDNNRec Class, implement the get_user_num and send_user_num function
    """

    def __init__(self):
        super().__init__()
        self.role = consts.GUEST
        self.user_num_sync = dnnrec_embedding_transfer_sync.Guest()

    def send_user_num(self, data):
        self.user_num_sync.send_guest_user_num(data)

    def get_user_num(self):
        return self.user_num_sync.get_host_user_num()

    def send_user_ids(self, data):
        self.user_num_sync.send_guest_user_ids(data)

    def get_user_ids(self):
        return self.user_num_sync.get_host_user_ids()

    def send_item_embedding(self, data):
        self.user_num_sync.send_guest_item_embedding(data)

    def get_item_embedding(self):
        return self.user_num_sync.get_host_item_embedding()
