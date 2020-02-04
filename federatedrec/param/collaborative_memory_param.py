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
import copy
import json
import typing
from types import SimpleNamespace

from federatedml.param.base_param import BaseParam
from federatedml.param.cross_validation_param import CrossValidationParam
from federatedml.param.init_model_param import InitParam
from federatedml.param.predict_param import PredictParam
from federatedml.util import consts
from federatedrec.protobuf.generated import cmn_model_meta_pb2


class CMNInitParam(InitParam):
    def __init__(self, embed_dim=10, init_method='random_normal'):
        super(CMNInitParam, self).__init__()
        self.embed_dim = embed_dim
        self.init_method = init_method

    def check(self):
        return True

class CMNParam(BaseParam):
    """
    Parameters used for Logistic Regression both for Homo mode or Hetero mode.

    Parameters
    ----------
    penalty : str, 'L1' or 'L2'. default: 'L2'
        Penalty method used in LR. Please note that, when using encrypted version in HomoLR,
        'L1' is not supported.

    tol : float, default: 1e-5
        The tolerance of convergence

    alpha : float, default: 1.0
        Regularization strength coefficient.

    optimizer : str, 'sgd', 'rmsprop', 'adam', 'nesterov_momentum_sgd' or 'adagrad', default: 'sgd'
        Optimize method

    batch_size : int, default: -1
        Batch size when updating model. -1 means use all data in a batch. i.e. Not to use mini-batch strategy.

    learning_rate : float, default: 0.01
        Learning rate

    init_param: InitParam object, default: default InitParam object
        Init param method object.

    max_iter : int, default: 100
        The maximum iteration for training.

    early_stop : str, 'diff', 'weight_diff' or 'abs', default: 'diff'
        Method used to judge converge or not.
            a)	diff： Use difference of loss between two iterations to judge whether converge.
            b)  weight_diff: Use difference between weights of two consecutive iterations
            c)	abs: Use the absolute value of loss to judge whether converge. i.e. if loss < eps, it is converged.

    decay: int or float, default: 1
        Decay rate for learning rate. learning rate will follow the following decay schedule.
        lr = lr0/(1+decay*t) if decay_sqrt is False. If decay_sqrt is True, lr = lr0 / sqrt(1+decay*t)
        where t is the iter number.

    decay_sqrt: Bool, default: True
        lr = lr0/(1+decay*t) if decay_sqrt is False, otherwise, lr = lr0 / sqrt(1+decay*t)

    encrypt_param: EncryptParam object, default: default EncryptParam object

    predict_param: PredictParam object, default: default PredictParam object

    cv_param: CrossValidationParam object, default: default CrossValidationParam object

    hops:  hopt count of CMN model

    max_len: max length of neighbors

    l2_coef: l2 penalty coefficient
    """

    def __init__(self,
                 secure_aggregate: bool = True,
                 aggregate_every_n_epoch: int = 1,
                 early_stop: typing.Union[str, dict, SimpleNamespace] = "diff",
                 penalty='L2',
                 tol=1e-5,
                 alpha=1.0,
                 optimizer: typing.Union[str, dict, SimpleNamespace] = 'SGD',
                 batch_size=-1,
                 learning_rate=0.01,
                 init_param=CMNInitParam(),
                 max_iter=100,
                 predict_param=PredictParam(),
                 cv_param=CrossValidationParam(),
                 decay=1, decay_sqrt=True,
                 validation_freqs=None,
                 metrics: typing.Union[str, list] = None,
                 loss: str = 'mse',
                 hops=2, neg_count=4, l2_coef=0.1, max_len=20
                 ):
        super(CMNParam, self).__init__()
        self.penalty = penalty
        self.tol = tol
        self.alpha = alpha
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.init_param = copy.deepcopy(init_param)
        self.max_iter = max_iter
        self.early_stop = early_stop
        self.predict_param = copy.deepcopy(predict_param)
        self.cv_param = copy.deepcopy(cv_param)
        self.decay = decay
        self.decay_sqrt = decay_sqrt
        self.validation_freqs = validation_freqs
        self.hops = hops
        self.max_len = max_len
        self.neg_count = neg_count
        self.l2_coef = l2_coef
        self.secure_aggregate = secure_aggregate
        self.aggregate_every_n_epoch = aggregate_every_n_epoch
        self.loss = loss
        self.metrics = metrics

    def check(self):
        descr = "CMN(collaborative Memory Networks)'s"
        self.early_stop = self._parse_early_stop(self.early_stop)
        self.optimizer = self._parse_optimizer(self.optimizer)
        self.metrics = self._parse_metrics(self.metrics)

        if self.penalty is None:
            pass
        elif type(self.penalty).__name__ != "str":
            raise ValueError(
                "CMN(collaborative Memory Networks)'s penalty {} not supported, should be str type".format(
                    self.penalty))
        else:
            self.penalty = self.penalty.upper()
            if self.penalty not in [consts.L1_PENALTY, consts.L2_PENALTY, 'NONE']:
                raise ValueError(
                    "CMN(collaborative Memory Networks)'s penalty not supported, penalty should be 'L1', 'L2' or 'none'")

        if not isinstance(self.tol, (int, float)):
            raise ValueError(
                "CMN(collaborative Memory Networks)'s tol {} not supported, should be float type".format(self.tol))

        if type(self.alpha).__name__ not in ["float", 'int']:
            raise ValueError(
                "CMN(collaborative Memory Networks)'s alpha {} not supported, should be float or int type".format(
                    self.alpha))

        # if type(self.optimizer).__name__ != "dict":
        #     raise ValueError(
        #         "CMN(collaborative Memory Networks)'s optimizer {} not supported, should be str type".format(
        #             self.optimizer))
        # else:
        #     self.optimizer = self.optimizer
        #     if self.optimizer["optimizer"] not in ['sgd', 'rmsprop', 'adam', 'adagrad', 'nesterov_momentum_sgd']:
        #         raise ValueError(
        #             "CMN(collaborative Memory Networks)'s optimizer not supported, optimizer should be"
        #             " 'sgd', 'rmsprop', 'adam', 'nesterov_momentum_sgd' or 'adagrad'")

        if self.batch_size != -1:
            if type(self.batch_size).__name__ not in ["int"] \
                    or self.batch_size < consts.MIN_BATCH_SIZE:
                raise ValueError(descr + " {} not supported, should be larger than 10 or "
                                         "-1 represent for all data".format(self.batch_size))

        if type(self.learning_rate).__name__ != "float"\
                or (isinstance(self.learning_rate, float) and self.learning_rate < 0):
            raise ValueError(
                "CMN(collaborative Memory Networks)'s learning_rate {} not supported, "
                "should be float type and greater than 0".format(
                    self.learning_rate))

        self.init_param.check()

        if type(self.max_iter).__name__ != "int" or (isinstance(self.max_iter, int) and self.max_iter < 1):
            raise ValueError(
                "CMN(collaborative Memory Networks)'s max_iter {} not supported, "
                "should be int type and greater than 0".format(
                    self.max_iter))
        elif self.max_iter <= 0:
            raise ValueError(
                "CMN(collaborative Memory Networks)'s max_iter must be greater or equal to 1")

        # if type(self.early_stop).__name__ != "dict":
        #     raise ValueError(
        #         "CMN(collaborative Memory Networks)'s early_stop {} not supported, should be str type".format(
        #             self.early_stop))
        # else:
        #     # self.early_stop = self.early_stop.lower()
        #     if self.early_stop["early_stop"] not in ['diff', 'abs', 'weight_diff']:
        #         raise ValueError(
        #             "CMN(collaborative Memory Networks)'s early_stop not supported, converge_func should be"
        #             " 'diff' or 'abs'")

        if type(self.decay).__name__ not in ["int", 'float']:
            raise ValueError(
                "CMN(collaborative Memory Networks)'s decay {} not supported, should be 'int' or 'float'".format(
                    self.decay))

        if type(self.decay_sqrt).__name__ not in ['bool']:
            raise ValueError(
                "CMN(collaborative Memory Networks)'s decay_sqrt {} not supported, should be 'bool'".format(
                    self.decay_sqrt))

        if type(self.hops).__name__ not in ["int"]:
            raise ValueError(
                "CMN(collaborative Memory Networks)'s hops {} not supported, should be 'int'".format(
                    self.hops))

        if type(self.max_len).__name__ not in ["int"]:
            raise ValueError(
                "CMN(collaborative Memory Networks)'s max_len {} not supported, should be 'int'".format(
                    self.max_len))

        if type(self.neg_count).__name__ not in ["int"]:
            raise ValueError(
                "CMN(collaborative Memory Networks)'s neg_count {} not supported, should be 'int'".format(
                    self.neg_count))

        if type(self.l2_coef).__name__ not in ["int", "float"]:
            raise ValueError(
                "CMN(collaborative Memory Networks)'s l2_coef {} not supported, should be 'int' or 'float'".format(
                    self.l2_coef))

        return True

    def _parse_early_stop(self, param):
        """
           Examples:

               1. "early_stop": "diff"
               2. "early_stop": {
                       "early_stop": "diff",
                       "eps": 0.0001
                   }
        """
        default_eps = 0.0001
        if isinstance(param, str):
            return SimpleNamespace(converge_func=param, eps=default_eps)
        elif isinstance(param, dict):
            early_stop = param.get("early_stop", None)
            eps = param.get("eps", default_eps)
            if not early_stop:
                raise ValueError(f"early_stop config: {param} invalid")
            return SimpleNamespace(converge_func=early_stop, eps=eps)
        else:
            raise ValueError(f"invalid type for early_stop: {type(param)}")

    def _parse_optimizer(self, param):
        """
        Examples:

            1. "optimize": "SGD"
            2. "optimize": {
                    "optimizer": "SGD",
                    "learning_rate": 0.05
                }
        """
        kwargs = {}
        if isinstance(param, str):
            return SimpleNamespace(optimizer=param, kwargs=kwargs)
        elif isinstance(param, dict):
            optimizer = param.get("optimizer", kwargs)
            if not optimizer:
                raise ValueError(f"optimizer config: {param} invalid")
            kwargs = {k: v for k, v in param.items() if k != "optimizer"}
            return SimpleNamespace(optimizer=optimizer, kwargs=kwargs)
        else:
            raise ValueError(f"invalid type for optimize: {type(param)}")

    def _parse_metrics(self, param):
        """
        Examples:

            1. "metrics": "Accuracy"
            2. "metrics": ["Accuracy"]
        """
        if not param:
            return []
        elif isinstance(param, str):
            return [param]
        elif isinstance(param, list):
            return param
        else:
            raise ValueError(f"invalid metrics type: {type(param)}")


class HeteroCMNParam(CMNParam):
    """
    Parameters
    ----------
    re_encrypt_batches : int, default: 2
        Required when using encrypted version HomoLR. Since multiple batch updating coefficient may cause
        overflow error. The model need to be re-encrypt for every several batches. Please be careful when setting
        this parameter. Too large batches may cause training failure.

    aggregate_iters : int, default: 1
        Indicate how many iterations are aggregated once.

    """

    def __init__(self, penalty='L2',
                 tol=1e-5, alpha=1.0, optimizer='sgd',
                 batch_size=-1, learning_rate=0.01, init_param=CMNInitParam(),
                 max_iter=100, early_stop='diff',
                 predict_param=PredictParam(), cv_param=CrossValidationParam(),
                 decay=1, decay_sqrt=True,
                 aggregate_iters=1, validation_freqs=None,
                 hops=2, max_len=4, l2_coef=0.1, neg_count=10,
                 secure_aggregate: bool = True,
                 aggregate_every_n_epoch: int = 1,
                 loss: str = "mse",
                 metrics: str = "mse"):
        super(HeteroCMNParam, self).__init__(penalty=penalty, tol=tol, alpha=alpha, optimizer=optimizer,
                                             batch_size=batch_size,
                                             learning_rate=learning_rate,
                                             init_param=init_param, max_iter=max_iter,
                                             early_stop=early_stop,
                                             predict_param=predict_param,
                                             cv_param=cv_param,
                                             validation_freqs=validation_freqs,
                                             decay=decay,
                                             decay_sqrt=decay_sqrt,
                                             hops=hops,
                                             max_len=max_len,
                                             neg_count=neg_count,
                                             l2_coef=l2_coef,
                                             secure_aggregate=secure_aggregate,
                                             aggregate_every_n_epoch=aggregate_every_n_epoch,
                                             loss=loss,
                                             metrics=metrics)
        # TODO:修改下面的参数
        self.aggregate_iters = aggregate_iters

    def check(self):
        super().check()

        if not isinstance(self.aggregate_iters, int):
            raise ValueError(
                "CMN(collaborative Memory Networks)'s aggregate_iters {} not supported, should be int type".format(
                    self.aggregate_iters))

        return True

    def generate_pb(self):
        pb = cmn_model_meta_pb2.HeteroCMNParam()
        pb.secure_aggregate = self.secure_aggregate
        pb.aggregate_every_n_epoch = self.aggregate_every_n_epoch
        pb.batch_size = self.batch_size
        pb.max_iter = self.max_iter
        pb.early_stop.early_stop = self.early_stop.converge_func
        pb.early_stop.eps = self.early_stop.eps
        pb.hops = self.hops
        pb.max_len = self.max_len
        pb.l2_coef = self.l2_coef

        for metric in self.metrics:
            pb.metrics.append(metric)

        pb.optimizer.optimizer = self.optimizer.optimizer
        pb.optimizer.args = json.dumps(self.optimizer.kwargs)
        pb.loss = self.loss
        pb.embed_dim = self.init_param.embed_dim
        return pb

    def restore_from_pb(self, pb):
        self.secure_aggregate = pb.secure_aggregate
        self.aggregate_every_n_epoch = pb.aggregate_every_n_epoch
        self.batch_size = pb.batch_size
        self.max_iter = pb.max_iter
        self.early_stop = self._parse_early_stop(dict(early_stop=pb.early_stop.early_stop, eps=pb.early_stop.eps))
        self.metrics = list(pb.metrics)
        self.optimizer = self._parse_optimizer(dict(optimizer=pb.optimizer.optimizer, **json.loads(pb.optimizer.args)))
        self.loss = pb.loss
        self.max_len = pb.max_len
        self.hops = pb.hops
        self.l2_coef = pb.l2_coef
        return pb