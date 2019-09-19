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

import os
import json
import numpy as np
from collections import defaultdict
import paddle.fluid as fluid
from parl import layers as L


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


class ConfigDict(dict):
    def __init__(self, loc=None, ghost=False):
        self._dict = defaultdict(lambda _: False)
        self.ghost = ghost
        if loc:
            with open(loc) as f:
                raw = json.load(f)
            if "inherits" in raw and raw["inherits"]:
                for dep_loc in raw["inherits"]:
                    self.update(ConfigDict(dep_loc))
            if "updates" in raw and raw["updates"]:
                self.update(raw["updates"], include_all=True)

    def __getitem__(self, key):
        return self._dict[key]

    def __setitem__(self, key, value):
        self._dict[key] = value

    def __str__(self):
        s = str(dict(self._dict))
        s = s.replace("'", "\"")
        s = s.replace("True", "true")
        s = s.replace("False", "false")
        pure_dict = json.loads(s)
        # pretty print
        return json.dumps(pure_dict, indent=2)

    def __repr__(self):
        return str(dict(self._dict))

    def __iter__(self):
        return self._dict.__iter__()

    def __bool__(self):
        return bool(self._dict)

    def __nonzero__(self):
        return bool(self._dict)

    def update(self, dictlike, include_all=False):
        for key in dictlike:
            value = dictlike[key]
            if isinstance(value, dict):
                if key[0] == "*":
                    # this means only override, do not set
                    key = key[1:]
                    ghost = True
                else:
                    ghost = False
                if not include_all and isinstance(value, ConfigDict) and \
                   key not in self._dict and value.ghost:
                    continue
                if key not in self._dict:
                    self._dict[key] = ConfigDict(ghost=ghost)
                self._dict[key].update(value)
            else:
                self._dict[key] = value


def create_directory(directory):
    dir_chunks = directory.split("/")
    for i in range(len(dir_chunks)):
        partial_dir = "/".join(dir_chunks[:i+1])
        try:
            os.makedirs(partial_dir)
        except OSError:
            pass
    return directory


def tile_ensemble_dim(var, size, axis=1):
    shape = list(var.shape)
    shape.insert(axis, 1)
    var_ = L.reshape(var, shape)
    expand_dims = [1 for _ in shape]
    expand_dims[axis] = size
    return L.expand(var_, expand_dims)
