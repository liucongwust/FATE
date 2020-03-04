import io
import copy
import typing
import zipfile
import tempfile

import numpy as np
import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.layers.noise import GaussianNoise
from tensorflow.python.keras.initializers import RandomUniform, RandomNormal
from tensorflow.python.keras.layers import Input, Embedding, Dot, Flatten, Dense, Dropout, Lambda, Add, Subtract
from arch.api.utils import log_utils
from federatedrec.utils import zip_dir_as_bytes
from federatedml.framework.weights import OrderDictWeights, Weights

LOGGER = log_utils.getLogger()


class KerasSequenceData(tf.keras.utils.Sequence):

    def __init__(self, data_instances, user_ids, item_ids, itemfea_dict, batch_size):
        self.size = data_instances.count()
        if self.size <= 0:
            raise ValueError("empty data")

        if itemfea_dict is not None:
            one_data = next(iter(itemfea_dict.values()))
            itemfea_dim = len(one_data)
            self.itemfea_array = np.zeros((self.size, itemfea_dim))
        else:
            self.itemfea_array = None

        user_ids_map = {uid:i for i,uid in enumerate(user_ids)}
        item_ids_map = {iid:i for i,iid in enumerate(item_ids)}

        self.x = np.zeros((self.size, 2))
        self.y = np.zeros((self.size, 1))
        self._keys = []
        for index, (k, inst) in enumerate(data_instances.collect()):
            self._keys.append(k)
            uid = inst.features.get_data(0)
            iid = inst.features.get_data(1)
            rate = float(inst.features.get_data(2))
            self.x[index] = [user_ids_map[uid], item_ids_map[iid]]
            self.y[index] = rate
            if itemfea_dict is not None:
                self.itemfea_array[index] = itemfea_dict.get(iid)
            index += 1

        self.batch_size = batch_size if batch_size > 0 else self.size

    def __getitem__(self, index):
        """Gets batch at position `index`.

        # Arguments
            index: position of the batch in the Sequence.

        # Returns
            A batch
        """
        start = self.batch_size * index
        end = self.batch_size * (index + 1)
        if self.itemfea_array is not None:
            return [self.x[start: end, 0], self.x[start: end, 1], self.itemfea_array[start: end, :]], \
                   [self.y[start: end], self.itemfea_array[start: end, :], np.zeros((end - start, 1))]
        else:
            return [self.x[start: end, 0], self.x[start: end, 1]], self.y[start: end]

    def __len__(self):
        """Number of batch in the Sequence.

        # Returns
            The number of batches in the Sequence.
        """
        return int(np.ceil(self.size / float(self.batch_size)))

    def get_keys(self):
        return self._keys


class KerasSeqDataConverter:
    """
    Keras Sequence Data Converter
    """
    @staticmethod
    def convert(data, user_ids, item_ids, itemfea_dict, batch_size):
        return KerasSequenceData(data, user_ids, item_ids, itemfea_dict, batch_size)


class KerasModel:
    def __init__(self):
        self._aggregate_weights = None
        self.session = None
        self._predict_model = None

    def train(self, data: tf.keras.utils.Sequence, **kwargs):
        epochs = 1
        left_kwargs = copy.deepcopy(kwargs)
        if "aggregate_every_n_epoch" in kwargs:
            epochs = kwargs["aggregate_every_n_epoch"]
            del left_kwargs["aggregate_every_n_epoch"]
        self._model.fit(x=data, epochs=epochs, verbose=1, shuffle=True, **left_kwargs)
        return epochs * len(data)

    def _set_model(self, _model):
        self._model = _model

    def _set_predict_model(self, _predict_model):
        self._predict_model = _predict_model

    def modify(self, func: typing.Callable[[Weights], Weights]) -> Weights:
        weights = self.get_model_weights()
        self.set_model_weights(func(weights))
        return weights

    def get_model_weights(self) -> OrderDictWeights:
        return OrderDictWeights(self.session.run(self._aggregate_weights))

    def set_model_weights(self, weights: Weights):
        unboxed = weights.unboxed
        self.session.run([tf.assign(v, unboxed[name]) for name, v in self._aggregate_weights.items()])

    def evaluate(self, data: tf.keras.utils.Sequence):
        names = self._model.metrics_names
        values = self._model.evaluate(x=data, verbose=1)
        if not isinstance(values, list):
            values = [values]
        return dict(zip(names, values))

    def predict(self, data: tf.keras.utils.Sequence, **kwargs):
        if self._predict_model is not None:
            return self._predict_model.predict(data)
        else:
            return self._model.predict(data)

    @classmethod
    def restore_model(cls, model_bytes):  # todo: restore optimizer to support incremental learning
        model = cls()
        keras_model = None
        with tempfile.TemporaryDirectory() as tmp_path:
            with io.BytesIO(model_bytes) as bytes_io:
                with zipfile.ZipFile(bytes_io, 'r', zipfile.ZIP_DEFLATED) as f:
                    f.extractall(tmp_path)

            keras_model = tf.keras.experimental.load_from_saved_model(saved_model_path=tmp_path)
        model._set_predict_model(keras_model)
        return model

    def export_model(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            tf.keras.experimental.export_saved_model(self._predict_model, saved_model_path=tmp_path)
            model_bytes = zip_dir_as_bytes(tmp_path)

        return model_bytes


class AutoEncoder(KerasModel):
    def __init__(self, feature_dim, embedding_dim, session):
        super().__init__()
        self.feature_dim = feature_dim
        self.embedding_dim = embedding_dim
        self.session = session
        self.auto_encoders = None
        self.encoders = None
        self.decoders = None
        if embedding_dim is not None:
            self.hidden_layers = [feature_dim, 2*embedding_dim, embedding_dim]

    def build(self, lamda_w=0.1, encoder_noise=0.1, dropout_rate=0.1, activation='sigmoid'):
        '''
        layer-wise pretraining on item features (item_mat)
        '''
        self.auto_encoders = []
        self.encoders = []
        self.decoders = []
        for input_dim, hidden_dim in zip(self.hidden_layers[:-1], self.hidden_layers[1:]):
            LOGGER.info('Build auto_encoder layer: Input dim {} -> Output dim {}'.format(input_dim, hidden_dim))
            # autoencoder
            pretrain_input = Input(shape=(input_dim,))
            encoded = GaussianNoise(stddev=encoder_noise)(pretrain_input)
            encoded = Dropout(dropout_rate)(encoded)
            encoder = Dense(hidden_dim, activation=activation, kernel_regularizer=l2(lamda_w),
                            bias_regularizer=l2(lamda_w))(encoded)
            decoder = Dense(input_dim, activation=activation, kernel_regularizer=l2(lamda_w),
                            bias_regularizer=l2(lamda_w))(encoder)
            ae = Model(inputs=pretrain_input, outputs=decoder)

            # encoder
            ae_encoder = Model(inputs=pretrain_input, outputs=encoder)

            # decoder
            encoded_input = Input(shape=(hidden_dim,))
            decoder_layer = ae.layers[-1] # the last layer
            ae_decoder = Model(encoded_input, decoder_layer(encoded_input))

            self.auto_encoders.append(ae)
            self.encoders.append(ae_encoder)
            self.decoders.append(ae_decoder)

    def pretrain(self, item_mat, batch_size=64, epochs=10):
        '''
        layer-wise pretraining on item features (item_mat)
        '''
        x_train = item_mat
        for ae, ae_encoder in zip(self.auto_encoders, self.encoders):
            ae.compile(loss='mse', optimizer='rmsprop')
            history = ae.fit(x_train, x_train, batch_size=batch_size, epochs=epochs, verbose=2)
            LOGGER.info(f'Auto encoder training history {history.history}.')
            x_train = ae_encoder.predict(x_train)


class CDLModel(KerasModel):
    def __init__(self, user_ids=None, item_ids=None, embedding_dim=None, itemfea_dim=None):
        super().__init__()
        if user_ids is not None:
            self.user_num = len(user_ids)
        if item_ids is not None:
            self.item_num = len(item_ids)
        self.embedding_dim = embedding_dim
        self._sess = None
        self._trainable_weights = None
        self._aggregate_weights = None
        self.user_ids = user_ids
        self.item_ids = item_ids
        self.itemfea_dim = itemfea_dim
        self.session = self.init_session()
        self.auto_encoder = AutoEncoder(itemfea_dim, embedding_dim, self.session)

    def pretrain_autoencoder(self, item_features):
        self.auto_encoder.pretrain(item_features)

    def _build(self, lamda_u=0.1, lamda_v=0.1, optimizer='rmsprop', loss='mse', metrics='mse'):
        self.auto_encoder.build()
        # item autoencoder
        itemfeat_InputLayer = Input(shape=(self.itemfea_dim,), name='item_feat_input')
        encoded = self.auto_encoder.encoders[0](itemfeat_InputLayer)
        encoded = self.auto_encoder.encoders[1](encoded)
        decoded = self.auto_encoder.decoders[1](encoded)
        decoded = self.auto_encoder.decoders[0](decoded)

        # user embedding
        user_InputLayer = Input(shape=(1,), dtype='int32', name='user_input')
        user_EmbeddingLayer = Embedding(input_dim=self.user_num, output_dim=self.embedding_dim, input_length=1,
                                        name='user_embedding', embeddings_regularizer=l2(lamda_u),
                                        embeddings_initializer=RandomNormal(mean=0, stddev=1))(user_InputLayer)
        user_EmbeddingLayer = Flatten(name='user_flatten')(user_EmbeddingLayer)

        # item embedding
        item_InputLayer = Input(shape=(1,), dtype='int32', name='item_input')
        item_EmbeddingLayer = Embedding(input_dim=self.item_num, output_dim=self.embedding_dim, input_length=1,
                                        name='item_embedding', embeddings_regularizer=l2(lamda_v),
                                        embeddings_initializer=RandomNormal(mean=0, stddev=1))(item_InputLayer)
        item_EmbeddingLayer = Flatten(name='item_flatten')(item_EmbeddingLayer)

        # square sum of difference between item embedding and auto-encoder embedding
        diff_item_embedding = Subtract()([item_EmbeddingLayer, Flatten(name='encode_flatten')(encoded)])
        diff_item_embedding = tf.keras.backend.square(diff_item_embedding)
        diff_item_embedding = tf.keras.backend.mean(diff_item_embedding)

        # rating prediction
        dotLayer = Dot(axes=-1, name='dot_layer')([user_EmbeddingLayer, item_EmbeddingLayer])
        self._model = Model(inputs=[user_InputLayer, item_InputLayer, itemfeat_InputLayer],
                            outputs=[dotLayer, decoded, diff_item_embedding])
        # model for prediction
        self._predict_model = Model(inputs=[user_InputLayer, item_InputLayer], outputs=dotLayer)

        # compile model
        LOGGER.info(f"model output names {self._model.output_names}")
        optimizer_instance = getattr(tf.keras.optimizers, optimizer.optimizer)(**optimizer.kwargs)
        loss1 = getattr(tf.keras.losses, loss)
        losses = {self._model.output_names[0]: loss1,
                  self._model.output_names[1]: tf.keras.losses.MSE,
                  self._model.output_names[2]: tf.keras.losses.MSE}
        lossWeights = {self._model.output_names[0]: 1.,
                       self._model.output_names[1]: 1.,
                       self._model.output_names[2]: 1.}

        self._model.compile(optimizer=optimizer_instance,
                            loss_weights=lossWeights,
                            loss=losses, metrics=metrics)

        # pick user_embedding for aggregating
        self._trainable_weights = {v.name.split("/")[0]: v for v in self._model.trainable_weights}
        self._aggregate_weights = {"user_embedding": self._trainable_weights["user_embedding"]}

    @classmethod
    def build_model(cls, user_ids, item_ids, embedding_dim, itemfea_dim, loss, optimizer, metrics):
        model = cls(user_ids, item_ids, embedding_dim, itemfea_dim)
        model._build(loss=loss, optimizer=optimizer, metrics=metrics)
        return model

    def init_session(self):
        sess = tf.Session()
        tf.get_default_graph()
        set_session(sess)
        return sess

    def set_user_ids(self, user_ids):
        self.user_ids = user_ids

    def set_item_ids(self, item_ids):
        self.item_ids = item_ids
