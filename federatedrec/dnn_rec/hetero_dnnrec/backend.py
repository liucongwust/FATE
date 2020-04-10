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
import copy
import typing
import zipfile
import tempfile

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.python.keras.backend import set_session
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.layers import Input, Embedding, Lambda, Dense, Concatenate, Dot, Flatten
from tensorflow_core.python.keras.regularizers import l2
from tensorflow.keras import Sequential

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

    def __init__(self, user_num=None, item_num=None, embedding_dim=10, title_dim=None,
                 genres_dim=None, tags_dim=None, max_clk_num=None, mlp_params=default_mlp_param):
        self.user_num = user_num
        self.item_num = item_num
        self.title_dim = title_dim
        self.genres_dim = genres_dim
        self.tags_dim = tags_dim
        self.max_clk_num = max_clk_num
        self.embedding_dim = embedding_dim
        self.mlp_params = mlp_params
        self._trainable_weights = None
        self._aggregate_weights = None
        self._model = None
        self._item_embed_model = None
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

    def train_on_batch(self, x, y=None, **kwargs):
        left_kwargs = copy.deepcopy(kwargs)
        if "aggregate_every_n_epoch" in kwargs:
            epochs = kwargs["aggregate_every_n_epoch"]
            del left_kwargs["aggregate_every_n_epoch"]
        self._model.train_on_batch(x=x, y=y)
        return epochs * len(x)

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
        LOGGER.info(f"modify weights")
        return weights

    def get_model_weights(self) -> OrderDictWeights:
        """
        Return model's weights as OrderDictWeights.
        :return: model's weights.
        """
        # LOGGER.info(f"model weights: {self.session.run(self._aggregate_weights)}")
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

    def predict_on_batch(self, data, **kwargs):
        """
        Predict on input data and return prediction results which used in prediction.
        :param data: input data.
        :return: prediction results.
        """
        return self._model.predict(data)

    @classmethod
    def restore_model(cls, model_bytes, user_num, item_num, embedding_dim, title_dim,
                      genres_dim, tags_dim, max_clk_num):
        """
        Restore model from model bytes.
        :param max_clk_num:
        :param tags_dim:
        :param genres_dim:
        :param title_dim:
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

        model = cls(user_num=user_num, item_num=item_num, embedding_dim=embedding_dim,
                    title_dim=title_dim, genres_dim=genres_dim, tags_dim=tags_dim,
                    max_clk_num=max_clk_num)
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

    def generate_item_embedding(self, data):
        """
        :param data: batch data of click_items
        :return: average embeddings of clicked items
        """
        pred_embedding = self._item_embed_model.predict_on_batch(data)
        return pred_embedding

    def build(self, user_num, item_num, embedding_dim, title_dim, max_title_len, genres_dim, tags_dim, max_clk_num,
              mlp_params={}, optimizer='rmsprop', loss='mse', metrics='mse'):
        """
        build network graph of model
        :param mlp_params: DNN params
        :param max_clk_num: maximum num of clicked items
        :param tags_dim: tags vocabulary size
        :param genres_dim: genres vocabulary size
        :param title_dim: title vocabulary size
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

        title_input = Input(shape=(max_title_len,), dtype='float32', name='title_input')
        genres_input = Input(shape=(genres_dim,), dtype='float32', name='genres_input')
        tags_input = Input(shape=(tags_dim,), dtype='float32', name='tags_input')
        clk_items_input = Input(shape=(max_clk_num,), dtype='int32', name='clk_items_input')
        remote_item_embed_input = Input(shape=(embedding_dim,), dtype='float32', name='emote_item_embed')

        users = Lambda(lambda x: tf.keras.backend.squeeze(x, -1))(users_input)
        items = Lambda(lambda x: tf.strings.to_hash_bucket(tf.strings.as_string(x), item_num))(items_input)

        user_embed_layer = Embedding(user_num, embedding_dim,
                                     embeddings_initializer=RandomNormal(stddev=0.1),
                                     name='user_embedding')

        item_embed_layer = Embedding(item_num, embedding_dim,
                                     embeddings_initializer=RandomNormal(stddev=0.1),
                                     mask_zero=True, name='item_embedding')

        title_embed_layer = Embedding(title_dim, embedding_dim,
                                      embeddings_initializer=RandomNormal(stddev=0.1),
                                      mask_zero=True, name='title_embedding')

        item_embed = item_embed_layer(items)
        flatten_item_embed = Flatten()(item_embed)

        title_embed = title_embed_layer(title_input)
        avg_title_embed = Lambda(lambda x: tf.keras.backend.mean(x, axis=1), name="avg_title_embed")(title_embed)
        LOGGER.info(f"shape of avg_title_embed: {avg_title_embed.shape}, genres: {genres_input.shape}, tag: {tags_input.shape}, "
                    f"clicked items: {clk_items_input.shape}, item embed: {item_embed.shape}, "
                    f"flatten item embed: {flatten_item_embed.shape}")
        item_dense = Concatenate(name="item_dense")([flatten_item_embed, genres_input, tags_input, avg_title_embed])

        user_embed = user_embed_layer(users)

        clk_items = Lambda(lambda x: tf.strings.to_hash_bucket(tf.strings.as_string(x), item_num))(clk_items_input)
        clk_items_embed = item_embed_layer(clk_items)
        avg_clk_embed = Lambda(lambda x: tf.keras.backend.mean(x, axis=1), name="avg_clk_embed")(clk_items_embed)
        LOGGER.info(f"shape of clk_items_embed: {clk_items_embed.shape}, avg_clk_embed: {avg_clk_embed.shape}")
        user_dense = Concatenate(name="user_dense")([user_embed, avg_clk_embed, remote_item_embed_input])

        item_dnn_squential = Sequential()
        for idx in range(mlp_params["num_layer"]):
            layer = Dense(mlp_params["layer_dim"][idx]
                          , kernel_regularizer=l2(mlp_params["reg_layers"][idx])
                          , activation='relu'
                          , name="item_layer_%d" % idx)
            item_dnn_squential.add(layer)

        user_dnn_squential = Sequential()
        for idx in range(mlp_params["num_layer"]):
            layer = Dense(mlp_params["layer_dim"][idx]
                          , kernel_regularizer=l2(mlp_params["reg_layers"][idx])
                          , activation='relu'
                          , name="user_layer_%d" % idx)
            user_dnn_squential.add(layer)

        user_output = user_dnn_squential(user_dense)
        item_output = item_dnn_squential(item_dense)
        LOGGER.info(f"shape, user_output: {user_output.shape}, item_output: {item_output.shape}")
        pred_out = Dot(axes=-1)([user_output, item_output])

        optimizer_instance = getattr(tf.keras.optimizers, optimizer.optimizer)(**optimizer.kwargs)

        self._model = Model(inputs=[users_input, items_input, title_input, genres_input, tags_input,
                                    clk_items_input, remote_item_embed_input],
                            outputs=pred_out)
        self._item_embed_model = Model(inputs=[clk_items_input], outputs=avg_clk_embed)

        # model for prediction
        LOGGER.info(f"model output names {self._model.output_names}")
        self._model.compile(optimizer=optimizer_instance,
                            loss=loss, metrics=metrics)

        # pick user_embedding for aggregating
        self._trainable_weights = {v.name.split("/")[0]: v for v in self._model.trainable_weights}
        self._aggregate_weights = {"user_embedding": self._trainable_weights["user_embedding"]}
        LOGGER.info(f"finish building model, in {self.__class__.__name__} _build function")

    @classmethod
    def build_model(cls, user_num, item_num, embedding_dim, title_dim, max_title_len, genres_dim, tags_dim, max_clk_num,
                    mlp_params={}, optimizer='rmsprop', loss='mse', metrics='mse'):
        """
        build model
        :param mlp_params: DNN params
        :param max_clk_num: maximum num of clicked items
        :param tags_dim: tags vocabulary size
        :param genres_dim: genres vocabulary size
        :param title_dim: title vocabulary size
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
                    title_dim=title_dim, genres_dim=genres_dim, tags_dim=tags_dim, max_clk_num=max_clk_num,
                    max_title_len=max_title_len, loss=loss, optimizer=optimizer, metrics=metrics)
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
