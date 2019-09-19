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
import parl
from parl import layers as L
import paddle.fluid as fluid

from nn import EnsembleFeedForwardNet, FeedForwardNet
from utils import arange_shuffle, tile_ensemble_dim


class DynamicsModel(parl.Model):
    def __init__(self, name, env_config, learner_config):
        # self.name = self.model_id + "/" + name
        self.name = name
        self.obs_dim = np.prod(env_config["obs_dims"])
        self.action_dim = env_config["action_dim"]
        self.reward_scale = env_config["reward_scale"]
        self.discount = env_config["discount"]

        self.aux_hidden_dim = learner_config["aux_hidden_dim"]
        self.transition_hidden_dim = learner_config["transition_hidden_dim"]
        self.bayesian_config = learner_config["bayesian"]

        if self.bayesian_config:
            es = self.bayesian_config["ensemble_size"]
            c1 = self.bayesian_config["transition"]["train_sample_count"]
            c2 = self.bayesian_config["transition"]["eval_sample_count"]

            # T(s, a)
            self.transition_predictor = EnsembleFeedForwardNet(
                "transition_predictor", self.obs_dim + self.action_dim,
                [self.obs_dim], layers=8, hidden_dim=self.transition_hidden_dim,
                ensemble_size=es, train_sample_count=c1, eval_sample_count=c2)

            # d(s, s', a)
            self.done_predictor = EnsembleFeedForwardNet(
                "done_predictor", self.obs_dim * 2 + self.action_dim, [],
                layers=4, hidden_dim=self.aux_hidden_dim, ensemble_size=es,
                train_sample_count=c1, eval_sample_count=c2)

            # r(s, s', a)
            self.reward_predictor = EnsembleFeedForwardNet(
                "reward_predictor", self.obs_dim * 2 + self.action_dim, [],
                layers=4, hidden_dim=self.aux_hidden_dim, ensemble_size=es,
                train_sample_count=c1, eval_sample_count=c2)

        else:
            self.transition_predictor = FeedForwardNet(
                "transition_predictor", self.obs_dim + self.action_dim,
                [self.obs_dim], layers=8, hidden_dim=self.transition_hidden_dim)

            self.done_predictor = FeedForwardNet(
                "done_predictor", self.obs_dim * 2 + self.action_dim, [],
                layers=4, hidden_dim=self.aux_hidden_dim)

            self.reward_predictor = FeedForwardNet(
                "reward_predictor", self.obs_dim * 2 + self.action_dim, [],
                layers=4, hidden_dim=self.aux_hidden_dim)

    def get_ensemble_idx_info(self):
        if self.bayesian_config:
            ensemble_size = self.transition_predictor.ensemble_size
            ensemble_idxs = arange_shuffle(ensemble_size)

            # M in STEVE paper, i.e. transition sample count
            M = self.transition_predictor.eval_sample_count
            # N in STEVE paper, i.e. reward sample count
            N = self.reward_predictor.eval_sample_count

            return ensemble_idxs, M, N
        else:
            return None, 1, 1

    def transition(self, obs, action, ensemble_idxs=None, pre_expanded=None):
        """API for rollout in value expansion."""
        info = L.concat([obs, action], axis=-1)
        next_obs_delta = self.transition_predictor(
            info, ensemble_idxs=ensemble_idxs, pre_expanded=pre_expanded,
            reduce_mode="none")

        # tile the thick one according to pre_expanded
        if pre_expanded is None:
            pre_expanded = ensemble_idxs is not None
        if not pre_expanded and ensemble_idxs is not None:
            # ensemble model and not pre expanded
            ensemble_size = next_obs_delta.shape[1]
            info = tile_ensemble_dim(info, ensemble_size)
            obs = tile_ensemble_dim(obs, ensemble_size)
            next_obs = L.elementwise_add(obs, next_obs_delta)
            next_info = L.concat([next_obs, info], -1)
        else:
            next_obs = L.elementwise_add(obs, next_obs_delta)
            next_info = L.concat([next_obs, info], -1)

        done = L.sigmoid(self.done_predictor(
            next_info, ensemble_idxs=ensemble_idxs, pre_expanded=True,
            reduce_mode="none"))

        return next_obs, done

    def get_rewards(self, obs, action, next_obs, ensemble_idxs=None):
        """API for rollout in value expansion."""
        next_info = L.concat([next_obs, obs, action], -1)
        reward = self.reward_predictor(next_info, reduce_mode="none",
                                       ensemble_idxs=ensemble_idxs)
        return reward
