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


def create_tmp_var(name, dtype, shape):
    return fluid.default_main_program().current_block().create_var(
        name=name, dtype=dtype, shape=shape)


def uniform_sample(samples, sample_n, size):
    def perform_sample(x):
        x = np.array(x)
        indices = np.random.randint(0, sample_n, size=size)
        res = np.stack([x[i, j, ...] for i, j in zip(range(size), indices)])
        return res

    out = create_tmp_var("sample", samples.dtype,
                         [size] + list(samples.shape)[2:])
    fluid.layers.py_func(perform_sample, samples, out)
    return out


def arange_shuffle(n):
    def shuffle(n_):
        n_ = np.array(n_)
        x = np.arange(n_)
        np.random.shuffle(x)
        return x

    n_var = fluid.layers.fill_constant([], dtype='int32', value=n)
    out = create_tmp_var("shuffle", 'int32', [n])
    fluid.layers.py_func(shuffle, n_var, out)
    return out
