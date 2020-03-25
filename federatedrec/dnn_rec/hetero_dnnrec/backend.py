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

import io
import os
import copy
import typing
import zipfile
import tempfile
import traceback

import tensorflow as tf
from tensorflow.keras.losses import MSE as MSE
from tensorflow.keras import Model
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Input, Embedding, Lambda, Dense, Multiply, Concatenate, Flatten
from tensorflow_core.python.keras.regularizers import l2

from arch.api.utils import log_utils
from federatedrec.utils import zip_dir_as_bytes
from federatedml.framework.weights import OrderDictWeights, Weights

LOGGER = log_utils.getLogger()

default_mlp_param = {
    "embed_dim": 32,
    "num_layer": 3,
    "layer_dim": [32, 32, 32],
    "reg_layers": [0.01, 0.01, 0.01],
}


class DNNRecModel:
    """
    General Matrix Factorization model
    """

    def __init__(self, user_num=None, item_num=None, embedding_dim=10, mlp_params=default_mlp_param):
        self.user_num = user_num
        self.item_num = item_num
        self.embedding_dim = embedding_dim
        self.mlp_params = mlp_params
        self._trainable_weights = None
        self._aggregate_weights = None
        self._model = None
        self._sess = None

    def train(self, data: tf.keras.utils.Sequence, **kwargs):
        """
        Train model on input data.
        :param data: input data.
        :param kwargs: other params.
        :return: Training steps.
        """
        epochs = 1
        left_kwargs = copy.deepcopy(kwargs)
        if "aggregate_every_n_epoch" in kwargs:
            epochs = kwargs["aggregate_every_n_epoch"]
            del left_kwargs["aggregate_every_n_epoch"]
        self._model.fit(x=data, epochs=epochs, verbose=1, shuffle=True, **left_kwargs)
        return epochs * len(data)

    def _set_model(self, _model):
        """
        Set _model as trainning model.
        :param _model: training model.
        :return:
        """
        self._model = _model

    def _set_model(self, _model):
        """
        Set _model as prediction model.
        :param _model: prediction model.
        :return:
        """
        self._model = _model

    def modify(self, func: typing.Callable[[Weights], Weights]) -> Weights:
        """
        Apply func on model weights.
        :param func: operator to apply.
        :return: updated weights.
        """
        weights = self.get_model_weights()
        self.set_model_weights(func(weights))
        return weights

    def get_model_weights(self) -> OrderDictWeights:
        """
        Return model's weights as OrderDictWeights.
        :return: model's weights.
        """
        return OrderDictWeights(self.session.run(self._aggregate_weights))

    def set_model_weights(self, weights: Weights):
        """
        Set model's weights with input weights.
        :param weights: input weights.
        :return: updated weights.
        """
        unboxed = weights.unboxed
        self.session.run([tf.assign(v, unboxed[name]) for name, v in self._aggregate_weights.items()])

    def evaluate(self, data: tf.keras.utils.Sequence):
        """
        Evaluate on input data and return evaluation results.
        :param data: input data sequence.
        :return: evaluation results.
        """
        names = self._model.metrics_names
        values = self._model.evaluate(x=data, verbose=1)
        if not isinstance(values, list):
            values = [values]
        return dict(zip(names, values))

    def predict(self, data: tf.keras.utils.Sequence, **kwargs):
        """
        Predict on input data and return prediction results which used in prediction.
        :param data: input data.
        :return: prediction results.
        """
        return self._model.predict(data)

    @classmethod
    def restore_model(cls, model_bytes, user_num, item_num, embedding_dim):
        """
        Restore model from model bytes.
        :param model_bytes: model bytes of saved model.
        :param user_num: user num
        :param item_num: item num
        :param embedding_dim: embedding dimension
        :return:restored model object.
        """
        LOGGER.info("begin restore_model")
        with tempfile.TemporaryDirectory() as tmp_path:
            with io.BytesIO(model_bytes) as bytes_io:
                with zipfile.ZipFile(bytes_io, 'r', zipfile.ZIP_DEFLATED) as file:
                    file.extractall(tmp_path)

            keras_model = tf.keras.experimental.load_from_saved_model(
                saved_model_path=tmp_path)
        model = cls(user_num=user_num, item_num=item_num, embedding_dim=embedding_dim)
        model._set_model(keras_model)
        return model

    def export_model(self):
        """
        Export model to bytes.
        :return: bytes of saved model.
        """
        model_bytes = None
        with tempfile.TemporaryDirectory() as tmp_path:
            LOGGER.info(f"tmp_path: {tmp_path}")
            tf.keras.experimental.export_saved_model(
                    self._model, saved_model_path=tmp_path)

            model_bytes = zip_dir_as_bytes(tmp_path)

        return model_bytes

    def build(self, user_num, item_num, embedding_dim, mlp_params={}, optimizer='rmsprop', loss='mse', metrics='mse'):
        """
        build network graph of model
        :param user_num: user num
        :param item_num: item num
        :param embedding_dim: embedding dimension
        :param optimizer: optimizer method
        :param loss:  loss methods
        :param metrics: metric methods
        :return:
        """
        sess = self.session
        users_input = Input(shape=(1,), dtype='int32', name='user_input')
        items_input = Input(shape=(1,), dtype='int32', name='item_input')

        items = Lambda(
            lambda x: tf.strings.to_hash_bucket(tf.strings.as_string(x), item_num))(items_input)

        mf_user_embed_layer = Embedding(user_num, embedding_dim,
                                        embeddings_initializer=RandomNormal(stddev=0.1),
                                        name='mf_user_embedding')

        mf_item_embed_layer = Embedding(item_num, embedding_dim,
                                        embeddings_initializer=RandomNormal(stddev=0.1),
                                        name='mf_item_embedding')

        mlp_user_embed_layer = Embedding(user_num, mlp_params["embed_dim"],
                                         embeddings_initializer=RandomNormal(stddev=0.1),
                                         name='mlp_user_embedding')

        mlp_item_embed_layer = Embedding(item_num, mlp_params["embed_dim"],
                                         embeddings_initializer=RandomNormal(stddev=0.1),
                                         name='mlp_item_embedding')

        mf_user_embed = mf_user_embed_layer(users_input)
        mf_user_embed = Flatten()(mf_user_embed)
        mf_item_embed = mf_item_embed_layer(items)
        mf_item_embed = Flatten()(mf_item_embed)

        mlp_user_embed = mlp_user_embed_layer(users_input)
        mlp_user_embed = Flatten()(mlp_user_embed)
        mlp_item_embed = mlp_item_embed_layer(items)
        mlp_item_embed = Flatten()(mlp_item_embed)

        mf_vector = Multiply()([mf_user_embed, mf_item_embed])
        mlp_vector = Concatenate(axis=-1)([mlp_user_embed, mlp_item_embed])

        for idx in range(1, mlp_params["num_layer"]):
            layer = Dense(mlp_params["layer_dim"][idx]
                          , kernel_regularizer=l2(mlp_params["reg_layers"][idx])
                          , activation='relu'
                          , name="layer%d" % idx)
            mlp_vector = layer(mlp_vector)

        predict_vector = Concatenate(axis=-1)([mf_vector, mlp_vector])
        prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(
            predict_vector)

        optimizer_instance = getattr(tf.keras.optimizers, optimizer.optimizer)(**optimizer.kwargs)

        self._model = Model(inputs=[users_input, items_input],
                            outputs=prediction)
        # model for prediction
        LOGGER.info(f"model output names {self._model.output_names}")
        self._model.compile(optimizer=optimizer_instance,
                            loss=loss, metrics=metrics)

        # pick user_embedding for aggregating
        self._trainable_weights = {v.name.split("/")[0]: v for v in self._model.trainable_weights}
        self._aggregate_weights = {"mf_user_embedding": self._trainable_weights["mf_user_embedding"],
                                   "mlp_user_embedding": self._trainable_weights["mlp_user_embedding"]}
        LOGGER.info(f"finish building model, in {self.__class__.__name__} _build function")

    @classmethod
    def build_model(cls, user_num, item_num, embedding_dim, mlp_params, loss, optimizer, metrics):
        """
        build model
        :param user_num: user num
        :param item_num:  item num
        :param embedding_dim: embedding dimension
        :param loss: loss func
        :param optimizer: optimization methods
        :param metrics: metrics
        :return: model object
        """
        if mlp_params.__sizeof__() == 0:
            mlp_params = default_mlp_param
            LOGGER.info(f"build model default mlp_params: {mlp_params}")
        LOGGER.info(f"build model mlp_params: {mlp_params}")
        model = cls(user_num=user_num, item_num=item_num, embedding_dim=embedding_dim, mlp_params=mlp_params)
        model.build(user_num=user_num, item_num=item_num, embedding_dim=embedding_dim, mlp_params=mlp_params,
                    loss=loss, optimizer=optimizer, metrics=metrics)
        return model


    @property
    def session(self):
        """
        If session not created, then init a tensorflow session and return.
        :return: tensorflow session.
        """
        if self._sess is None:
            sess = tf.Session()
            tf.get_default_graph()
            set_session(sess)
            self._sess = sess
        return self._sess

    def set_user_num(self, user_num):
        """
        set user num
        :param user_num:
        :return:
        """
        self.user_num = user_num

    def set_item_num(self, item_num):
        """
        set item num
        :param item_num:
        :return:
        """
        self.item_num = item_num
