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

import paddle.fluid as fluid
from parl import layers as L
from parl.core.fluid.algorithm import Algorithm


class DynamicsModelLearner(Algorithm):
    def __init__(self, dynamics_model):
        """Supervised learning to get dynamics model.

        Args:
            dynamics_model (parl.Model): an subclass of `parl.Model`, with
                functions `transition_predictor`, `done_predictor`, and
                `reward_predictor` to construct different subnetworks.
        """
        self.model = dynamics_model

    def learn(self, obs, next_obs, actions, rewards, dones, learning_rate=3e-4,
              regularization_coeff=1e-4):
        """
        Args:
            obs: A float32 tensor of shape [B, obs_dim].
            next_obs: A float32 tensor of shape [B, obs_dim].
                Note that each observation is corresponds to the previous
                observation in `obs` respectively.
            actions: A float32 tensor of shape [B, action_dim].
                Note that each action corresponds to each observation in `obs`.
            rewards: A float32 tensor of shape ([B])
            dones: A float32 tensor of shape ([B])
            learning_rate: A float number for Adam optimizer.
            regularization_coeff: A float number for L2 decay regularizer.
        """
        info = L.concat(obs, actions, axis=-1)
        pred_next_obs = self.model.transition_predictor(
            info, is_eval=False, reduce_mode="random")
        next_info = L.concat([next_obs, info], axis=-1)
        pred_dones = self.model.done_predictor(
            next_info, is_eval=False, reduce_mode="random")
        pred_rewards = self.model.reward_predictor(
            next_info, is_eval=False, reduce_mode="random")

        done_losses = L.sigmoid_cross_entropy_with_logits(pred_dones, dones)
        reward_losses = .5 * L.square(L.elementwise_sub(rewards, pred_rewards))
        next_obs_losses = .5 * L.reduce_sum(L.square(
            L.elementwise_sub(next_obs, pred_next_obs)), dim=-1)

        done_loss = L.reduce_mean(done_losses, dim=-1)
        reward_loss = L.reduce_mean(reward_losses, dim=-1)
        next_obs_loss = L.reduce_mean(next_obs_losses, dim=-1)

        total_loss = done_loss + reward_loss + next_obs_loss
        reg = fluid.regularizer.L2DecayRegularizer(
            regularization_coeff=regularization_coeff)
        model_optimizer = fluid.optimizer.AdamOptimizer(
            learning_rate=learning_rate, regularization=reg)
        model_optimizer.minimize(total_loss)

        return total_loss, done_loss, reward_loss, next_obs_loss
