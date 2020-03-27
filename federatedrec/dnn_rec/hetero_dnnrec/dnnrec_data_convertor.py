import random
from collections import defaultdict

import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from arch.api.utils import log_utils

LOGGER = log_utils.getLogger()


class DataConverter(object):
    def convert(self, data, *args, **kwargs):
        pass


class DNNRecDataConverter(DataConverter):
    def convert(self, data, *args, **kwargs):
        return DNNRecSequenceData(data, *args, **kwargs)


class DNNRecSequenceData(tf.keras.utils.Sequence):
    def __init__(self, data_instances, batch_size):
        """
        Wraps dataset and produces batches for the model to consume. The dataset only has positive clicked data,
        generate negative samples and use neg_count params to control the rate of negative samples.
        :param data_instances: data instances: Instance
        :param batch_size: batch size of data
        :param neg_count: num of negative items
        """

        self.batch_size = batch_size
        self.data_instances = data_instances
        self._keys = []
        self._user_ids = set()
        self._item_ids = set()
        self.tag_corpus = list()
        self.genres_corpus = list()
        self.title_corpus = list()
        self._keys = list()

        self.title_input = None
        self.genres_input = None
        self.tags_input = None
        self.clk_items_input = None

        print(f"initialize class, data type: {type(self.data_instances)}, count:{data_instances.first()}")
        self.size = self.data_instances.count()

        if self.size <= 0:
            raise ValueError("empty data")
        self.batch_size = batch_size if batch_size > 0 else self.size

        # Neighborhoods
        self.user_items = defaultdict(set)
        self.item_users = defaultdict(set)

        for key, instance in self.data_instances.collect():
            # features = instance.features
            features = np.array(instance.features).astype(np.str).squeeze().tolist()
            u = int(features[0])
            i = int(features[1])
            tags = features[2]
            title = features[3]
            genres = features[4]
            self.user_items[u].add(i)
            self.item_users[i].add(u)
            self._user_ids.add(u)
            self._item_ids.add(i)

            if tags is not None and tags != '' and tags != '0':
                self.tag_corpus.append(tags.replace("|", " "))

            if title is not None and title != '' and title != '0':
                self.title_corpus.append(title)

            if genres is not None and genres != '' and genres != '0':
                self.genres_corpus.append(genres.replace("|", " "))

        self._n_users, self._n_items = max(self._user_ids) + 1, self._item_ids.__len__()

        # Get a list version so we do not need to perform type casting
        self.user_items = dict(self.user_items)
        self.item_users = dict(self.item_users)

        self._max_clicks = max(list(map(lambda x: len(x[1]), self.user_items.items())))

        self.users = None
        self.items = None
        self.y = None

        self.title_count_vector = CountVectorizer(ngram_range=(1, 2))
        self.title_array = self.title_count_vector.fit_transform(self.title_corpus)
        self.genres_count_vector = CountVectorizer()
        self.genres_array = self.genres_count_vector.fit_transform(self.genres_corpus)
        self.tag_count_vector = CountVectorizer()
        self.tag_array = self.tag_count_vector.fit_transform(self.tag_corpus)

        self._tag_dim = self.tag_count_vector.vocabulary_.__len__()
        self._title_dim = self.title_count_vector.vocabulary_.__len__()
        self._genres_dim = self.genres_count_vector.vocabulary_.__len__()
        self.transfer_data()

    @property
    def data_size(self):
        """
        data size of corpus
        :return:
        """
        return len(self.size)

    @property
    def user_ids(self):
        """
        user Ids in corpus
        :return:
        """
        return list(self._user_ids)

    @property
    def item_ids(self):
        """
        item Ids in corpus
        :return:
        """
        return list(self._item_ids)

    @property
    def tag_dim(self):
        """
        vocabulary size of tag
        :return:
        """
        return self._tag_dim

    @property
    def title_dim(self):
        """
        vocabulary size of title
        :return:
        """
        return self._title_dim

    @property
    def genres_dim(self):
        """
        vocabulary size of genres
        :return:
        """
        return self._genres_dim

    @property
    def user_count(self):
        """
        Number of users in dataset
        """
        return self._n_users

    @property
    def item_count(self):
        """
        Number of items in dataset
        """
        return self._n_items

    @property
    def max_clicks(self):
        """
        Number of maximum clicked items
        :return:
        """
        return self._max_clicks

    def transfer_data(self):
        # Allocate inputs
        # print("transfer_data")
        size = self.size if self.size > 0 else self.batch_size
        users = np.zeros((size,), dtype=np.uint32)
        items = np.zeros((size,), dtype=np.uint32)
        title_input = np.zeros((size, self._title_dim), dtype=np.uint32)
        genres_input = np.zeros((size, self._genres_dim), dtype=np.uint32)
        tags_input = np.zeros((size, self._tag_dim), dtype=np.uint32)
        click_items = np.zeros((size, self._max_clicks), dtype=np.uint32)
        y = np.zeros((size,), dtype=np.float)

        idx = 0
        for key, instance in self.data_instances.collect():
            features = np.array(instance.features).astype(np.str).squeeze().tolist()
            user_idx = int(features[0])
            item_idx = int(features[1])
            tags = features[2] if features[2] and features[2] != "0" else ""
            title = features[3]
            genres = features[4] if features[4] and features[4] != "0" else ""
            users[idx] = user_idx
            items[idx] = item_idx
            y[idx] = instance.label

            # if tags:
            #     LOGGER.info(
            #         f"title {title}, genres: {genres}, tags: {tags}\n"
            #         f"transformed data, title: {self.title_count_vector.transform([title]).toarray().nonzero()} \n"
            #         f"tags : {self.tag_count_vector.transform([tags.replace('|', ' ')]).toarray().nonzero()} \n"
            #         f"genres: {self.genres_count_vector.transform([genres.replace('|', ' ')]).toarray().nonzero()}")

            title_input[idx, :] = self.title_count_vector.transform([title]).toarray()[0, :]
            tags_input[idx, :] = self.tag_count_vector.transform([tags.replace("|", " ")]).toarray()[0, :]
            genres_input[idx, :] = self.genres_count_vector.transform([genres.replace("|", " ")]).toarray()[0, :]
            clk_items = list(self.user_items[user_idx])
            click_items[idx, :len(clk_items)] = np.array(clk_items)

            idx += 1
            self._keys.append(key)

        shuffle_idx = [i for i in range(idx)]
        random.shuffle(shuffle_idx)
        # TODO: need to transfer into class Instance
        self.users = users[shuffle_idx]
        self.items = items[shuffle_idx]
        self.clk_items_input = click_items[shuffle_idx, :]
        self.genres_input = genres_input[shuffle_idx, :]
        self.title_input = title_input[shuffle_idx, :]
        self.tags_input = tags_input[shuffle_idx, :]
        self.y = y[shuffle_idx]

    def __getitem__(self, index):
        """Gets batch at position `index`.

        # Arguments
            index: position of the batch in the Sequence.

        # Returns
            A batch
        """
        start = self.batch_size * index
        end = self.batch_size * (index + 1)
        X = [self.users[start: end], self.items[start: end], self.title_input[start: end], self.genres_input[start:end],
             self.tags_input[start: end], self.clk_items_input[start: end]]
        y = self.y[start:end]
        return X, y

    def __len__(self):
        """Number of batch in the Sequence.

        # Returns
            The number of batches in the Sequence.
        """
        return int(np.ceil(self.size / float(self.batch_size)))

    def get_keys(self):
        return self._keys
