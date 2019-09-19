#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import paddle.fluid as fluid
from parl import layers as L
from parl.core.fluid.layers.attr_holder import AttrHolder
from parl.core.fluid.layers.layer_wrappers import update_attr_name, LayerFunc

from utils import uniform_sample, arange_shuffle


def ensemble_fc(in_size,
                out_size,
                ensemble_size,
                param_attr=None,
                bias_attr=None,
                act=None,
                name=None):
    """
    Return a function that creates an ensemble FC layer.
    The input of an ensemble FC layer is in shape
    [batch-size, ensemble-size, in-size].
    """
    default_name = "ensemble_fc"
    param_attr = update_attr_name(name, default_name, param_attr, False)
    bias_attr = update_attr_name(name, default_name, bias_attr, True)

    class EnsembleFC(LayerFunc):
        def __init__(self):
            super(EnsembleFC, self).__init__(
                AttrHolder(param_attr=param_attr, bias_attr=bias_attr))
            self.weight = fluid.layers.create_parameter(
                [ensemble_size, in_size, out_size], dtype='float32',
                attr=param_attr)
            self.bias = fluid.layers.create_parameter(
                [ensemble_size, out_size], dtype='float32',
                attr=bias_attr)

        def __call__(self, input, ensemble_idxs, stop_gradient=False):
            if ensemble_idxs.shape[0] > ensemble_size:
                raise Exception("ensemble_idxs is larger than ensemble size")
            if isinstance(input, list):
                raise Exception("only accept single input tensor")
            if len(input.shape) != 3 or \
               input.shape[1] != ensemble_idxs.shape[0]:
                raise Exception("input tensor should have shape "
                                "[batch-size, ensemble-size, in-dim]")

            weight = fluid.layers.gather(self.weight, ensemble_idxs)
            bias = fluid.layers.gather(self.bias, ensemble_idxs)
            if stop_gradient:
                weight.stop_gradient = True
                bias.stop_gradient = True

            ensemble_results = []
            for input_var, w, b in zip(
                    fluid.layers.unstack(input, axis=1),
                    fluid.layers.unstack(weight, axis=0),
                    fluid.layers.unstack(bias, axis=0)):
                result = fluid.layers.elementwise_add(
                    fluid.layers.matmul(input_var, w), b)
                ensemble_results.append(result)

            return fluid.layers.stack(ensemble_results, axis=1)

    return EnsembleFC()


class EnsembleFeedForwardNet(object):
    """Custom feed-forward network layer with an ensemble."""
    def __init__(self, name, in_size, out_shape, layers=1, hidden_dim=32,
                 final_nonlinearity=None, ensemble_size=2,
                 train_sample_count=2, eval_sample_count=2):
        if train_sample_count > ensemble_size:
            raise Exception("train_sample_count cannot be larger than ensemble size")
        if eval_sample_count > ensemble_size:
            raise Exception("eval_sample_count cannot be larger than ensemble size")

        self.name = name
        self.in_size = in_size
        self.out_shape = out_shape
        self.out_size = np.prod(out_shape, dtype=int)
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.ensemble_size = ensemble_size
        self.train_sample_count = train_sample_count
        self.eval_sample_count = eval_sample_count
        self.final_nonlinearity = final_nonlinearity

        for i in range(layers):
            act, name = "relu", "efc%s" % (i+1)
            in_size = out_size = self.hidden_dim
            if i == 0:
                in_size = self.in_size
            if i == self.layers - 1:
                act = self.final_nonlinearity
                out_size = self.out_size
            fc = ensemble_fc(in_size, out_size, ensemble_size,
                             param_attr=fluid.initializer.Xavier(uniform=True),
                             bias_attr=fluid.initializer.Constant(value=0.0),
                             act=act, name=name)
            setattr(self, name, fc)

    def __call__(self, x, stop_gradient=False, is_eval=True,
                 ensemble_idxs=None, pre_expanded=None,
                 reduce_mode="none"):
        if pre_expanded is None:
            pre_expanded = ensemble_idxs is not None

        if ensemble_idxs is None:
            if is_eval:
                ensemble_sample_n = self.eval_sample_count
            else:
                ensemble_sample_n = self.train_sample_count

            ensemble_idxs = arange_shuffle(self.ensemble_size)
            ensemble_idxs = fluid.layers.slice(
                ensemble_idxs, [0], [0], [ensemble_sample_n])
        else:
            ensemble_sample_n = ensemble_idxs.shape[0]

        # When `pre_expanded` is True, `x` already has ensemble dim
        # otherwise, tile it at the ensemble axis
        if pre_expanded:
            h = fluid.layers.reshape(x, [-1, ensemble_sample_n, self.in_size])
        else:
            h = fluid.layers.expand(
                fluid.layers.reshape(x, [-1, 1, self.in_size]),
                [1, ensemble_sample_n, 1])

        for i in range(self.layers):
            fc = getattr(self, "efc%s" % (i+1))
            h = fc(h, ensemble_idxs, stop_gradient=stop_gradient)

        batch_size = h.shape[0]
        if reduce_mode == "none":
            pass
        elif reduce_mode == "random":
            h = uniform_sample(h, ensemble_sample_n, batch_size)
        elif reduce_mode == "mean":
            h = fluid.layers.reduce_mean(h, -2)
        else:
            raise Exception("use a valid reduce mode: none, random, or mean")

        # Reshape to match output shape
        if len(self.out_shape) > 0:
            out_shape = list(h.shape)[:-1] + list(self.out_shape)
        else:
            # out_shape is [], so squeeze last dim
            out_shape = list(h.shape)[:-1]
        h = fluid.layers.reshape(h, out_shape)

        return h


class FeedForwardNet(object):
    """Custom feed-forward network layer."""
    def __init__(self, name, in_size, out_shape, layers=1, hidden_dim=32,
                 final_nonlinearity=None):
        self.name = name
        self.in_size = in_size
        self.out_shape = out_shape
        self.out_size = np.prod(out_shape, dtype=int)
        self.layers = layers
        self.hidden_dim = hidden_dim
        self.final_nonlinearity = final_nonlinearity

        for i in range(layers):
            act, name = "relu", "fc%s" % (i+1)
            out_size = self.hidden_dim
            if i == self.layers - 1:
                act = self.final_nonlinearity
                out_size = self.out_size

            fc = L.fc(size=out_size, act=act)
            setattr(self, name, fc)

    def __call__(self, x, stop_gradient=False, **kwargs):
        original_shape = list(x.shape)
        h = fluid.layers.reshape(x, [-1, self.in_size])
        for i in range(self.layers):
            fc = getattr(self, "fc%s" % (i+1))
            h = fc(h)

        if stop_gradient:
            h.stop_gradient = True

        h = fluid.layers.reshape(h, original_shape[:-1] + self.out_shape)
        return h
