import io
import copy
import typing
import zipfile
import tempfile

import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.keras.backend import set_session
from tensorflow.python.keras.initializers import TruncatedNormal
from tensorflow.python.keras.layers import Input, Embedding, Multiply, Dense, Lambda, Concatenate
from arch.api.utils import log_utils
from federatedrec.utils import zip_dir_as_bytes
from federatedml.framework.weights import OrderDictWeights, Weights

LOGGER = log_utils.getLogger()


class KerasModel:
    def __init__(self):
        self._aggregate_weights = None
        self.session = None
        self._model = None

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

    def _set_model(self, _model):
        self._model = _model

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
        if self._model is not None:
            return self._model.predict(data)
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

            try:
                keras_model = tf.keras.models.load_model(filepath=tmp_path)
            except IOError:
                import warnings
                warnings.warn('loading the model as SavedModel is still in experimental stages. '
                              'trying tf.keras.experimental.load_from_saved_model...')
                keras_model = tf.keras.experimental.load_from_saved_model(saved_model_path=tmp_path)
        model._set_model(keras_model)
        return model

    def export_model(self):
        with tempfile.TemporaryDirectory() as tmp_path:
            try:
                tf.keras.models.save_model(self._model, filepath=tmp_path, save_format="tf")
            except NotImplementedError:
                import warnings
                warnings.warn('Saving the model as SavedModel is still in experimental stages. '
                              'trying tf.keras.experimental.export_saved_model...')

                tf.keras.experimental.export_saved_model(self._model, saved_model_path=tmp_path)
            LOGGER.info(f"export saved model at path: {tmp_path}")
            model_bytes = zip_dir_as_bytes(tmp_path)

        return model_bytes


default_mlp_param = {
    "embed_dim": 32,
    "num_layer": 3,
    "layer_dim": [32, 32, 32],
    "reg_layers": [0.01, 0.01, 0.01],
}


class NCFModel(KerasModel):
    def __init__(self, user_num=10000, item_num=1000, embedding_dim=10, l2_coef=0.01, mlp_params=default_mlp_param):
        super().__init__()
        self.user_num = user_num
        self.item_num = item_num
        self.mlp_params = mlp_params
        self.embedding_dim = embedding_dim
        self._sess = None
        self._trainable_weights = None
        self._aggregate_weights = None
        self.session = self.init_session()
        self._embedding_initializer = TruncatedNormal(stddev=0.1)

    def _build(self, optimizer='rmsprop', loss='mse', metrics='mse'):
        users_input = Input(shape=(1,), dtype='int32', name='user_input')
        items_input = Input(shape=(1,), dtype='int32', name='item_input')

        users = Lambda(
            lambda x: tf.squeeze(x, 1))(users_input)
        items = Lambda(
            lambda x: tf.strings.to_hash_bucket(tf.strings.as_string(tf.squeeze(x, 1)), self.item_num))(items_input)

        mf_user_embed_layer = Embedding(self.user_num, self.embedding_dim,
                                        embeddings_initializer=self._embedding_initializer,
                                        name='mf_user_embedding')

        mf_item_embed_layer = Embedding(self.item_num, self.embedding_dim,
                                        embeddings_initializer=self._embedding_initializer,
                                        name='mf_item_embedding')

        mlp_user_embed_layer = Embedding(self.user_num, self.mlp_params["embed_dim"],
                                         embeddings_initializer=self._embedding_initializer,
                                         name='mlp_user_embedding')

        mlp_item_embed_layer = Embedding(self.item_num, self.mlp_params["embed_dim"],
                                         embeddings_initializer=self._embedding_initializer,
                                         name='mlp_item_embedding')

        mf_user_embed = mf_user_embed_layer(users)
        mf_item_embed = mf_item_embed_layer(items)

        mlp_user_embed = mlp_user_embed_layer(users)
        mlp_item_embed = mlp_item_embed_layer(items)

        # LOGGER.debug(f"users shapes: {users_input.shape}")
        # LOGGER.debug(f"items shapes: {items_input.shape}")
        # LOGGER.debug(f"embedding shapes, mf_user_embedding: {mf_user_embed.shape}")
        # LOGGER.debug(f"embedding shapes, mf_item_embedding: {mf_item_embed.shape}")
        # LOGGER.debug(f"embedding shapes, mlp_user_embedding: {mlp_user_embed.shape}")
        # LOGGER.debug(f"embedding shapes, mlp_item_embedding: {mlp_item_embed.shape}")

        mf_vector = Multiply()([mf_user_embed, mf_item_embed])
        mlp_vector = Concatenate(axis=-1)([mlp_user_embed, mlp_item_embed])

        for idx in range(1, self.mlp_params["num_layer"]):
            layer = Dense(self.mlp_params["layer_dim"][idx]
                          , kernel_regularizer=l2(self.mlp_params["reg_layers"][idx])
                          , activation='relu'
                          , name="layer%d" % idx)
            mlp_vector = layer(mlp_vector)

        predict_vector = Concatenate(axis=-1)([mf_vector, mlp_vector])
        prediction = Dense(1, activation='sigmoid', kernel_initializer='lecun_uniform', name="prediction")(predict_vector)

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
    def build_model(cls, user_num, item_num, embedding_dim, loss, optimizer, metrics, mlp_params={}):
        if mlp_params.__sizeof__() == 0:
            mlp_params = default_mlp_param
            LOGGER.info(f"build model default mlp_params: {mlp_params}")
        LOGGER.info(f"build model mlp_params: {mlp_params}")
        model = cls(user_num=user_num, item_num=item_num, embedding_dim=embedding_dim, mlp_params=mlp_params)
        model._build(loss=loss, optimizer=optimizer, metrics=metrics)
        return model

    def init_session(self):
        sess = tf.Session()
        tf.get_default_graph()
        set_session(sess)
        return sess
