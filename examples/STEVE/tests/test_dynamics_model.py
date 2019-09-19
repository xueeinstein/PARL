import pytest
import paddle.fluid as fluid

from dynamics_model import DynamicsModel


class TestDynamicsModel(object):
    def setup_class(self):
        place = fluid.CPUPlace()
        self.exe = fluid.Executor(place)

        self.env_config = {
            "obs_dims": [16],
            "action_dim": 2,
            "reward_scale": 1,
            "discount": 0.9
        }

        self.learner_config = {
            "aux_hidden_dim": 8,
            "transition_hidden_dim": 8,
            "bayesian": {}
        }

        self.bayesian_config = {
            "ensemble_size": 4,
            "transition": {
                "train_sample_count": 4,
                "eval_sample_count": 4
            }
        }

    def test_transition(self):
        bs = 4
        self.learner_config["bayesian"] = {}
        prog = fluid.Program()
        with fluid.program_guard(prog):
            env = DynamicsModel("env", self.env_config, self.learner_config)
            E, M, N = env.get_ensemble_idx_info()
            obs = fluid.layers.uniform_random(
                [bs, *self.env_config["obs_dims"]], seed=1)
            action = fluid.layers.uniform_random(
                [bs, self.env_config["action_dim"]], seed=2)
            next_obs, done = env.transition(obs, action, ensemble_idxs=E)
            self.exe.run(fluid.default_startup_program())

        next_obs_np, done_np = self.exe.run(program=prog,
                                            fetch_list=[next_obs, done])
        assert next_obs_np.shape == (bs, *self.env_config["obs_dims"])
        assert done_np.shape == (bs,)

    def test_ensemble_transition(self):
        bs = 4
        c = self.bayesian_config["ensemble_size"]
        self.learner_config["bayesian"] = self.bayesian_config
        prog = fluid.Program()
        with fluid.program_guard(prog):
            env = DynamicsModel("env", self.env_config, self.learner_config)
            E, M, N = env.get_ensemble_idx_info()
            obs = fluid.layers.uniform_random(
                [bs, *self.env_config["obs_dims"]], seed=1)
            action = fluid.layers.uniform_random(
                [bs, self.env_config["action_dim"]], seed=2)
            next_obs, done = env.transition(obs, action, ensemble_idxs=E,
                                            pre_expanded=False)
            self.exe.run(fluid.default_startup_program())

        next_obs_np, done_np = self.exe.run(program=prog,
                                            fetch_list=[next_obs, done])
        assert next_obs_np.shape == (bs, c, *self.env_config["obs_dims"])
        assert done_np.shape == (bs, c)

    def test_rollout_ensemble_transition(self):
        bs, horizon = 4, 3
        c = self.bayesian_config["ensemble_size"]
        self.learner_config["bayesian"] = self.bayesian_config
        prog = fluid.Program()
        with fluid.program_guard(prog):
            env = DynamicsModel("env", self.env_config, self.learner_config)
            obs = fluid.layers.uniform_random(
                [bs, c, *self.env_config["obs_dims"]], seed=1)
            obs_arr = fluid.layers.create_array("float32")
            done_arr = fluid.layers.create_array("float32")
            d = fluid.layers.fill_constant(shape=[1], dtype="int64",
                                           value=horizon)
            i = fluid.layers.fill_constant(shape=[1], dtype="int64", value=0)
            # unsqueeze first dim for later `tensor_array_to_tensor` op
            fluid.layers.array_write(
                fluid.layers.unsqueeze(obs, axes=[0]), i, array=obs_arr)
            fluid.layers.array_write(
                fluid.layers.zeros([1, bs, c], "float32"), i, array=done_arr)
            cond = fluid.layers.less_than(x=i, y=d)
            while_op = fluid.layers.While(cond=cond)
            with while_op.block():
                E, M, N = env.get_ensemble_idx_info()
                obs = fluid.layers.array_read(obs_arr, i)
                obs = fluid.layers.squeeze(obs, axes=[0])
                action = fluid.layers.uniform_random(
                    [bs, c, self.env_config["action_dim"]])
                next_obs, done = env.transition(obs, action, ensemble_idxs=E,
                                                pre_expanded=True)
                fluid.layers.increment(i, value=1, in_place=True)
                fluid.layers.array_write(
                    fluid.layers.unsqueeze(next_obs, axes=[0]), i, array=obs_arr)
                fluid.layers.array_write(
                    fluid.layers.unsqueeze(done, axes=[0]), i, array=done_arr)

                fluid.layers.less_than(x=i, y=d, cond=cond)

            obs_rollout, _ = fluid.layers.tensor_array_to_tensor(obs_arr, axis=0)
            done_rollout, _ = fluid.layers.tensor_array_to_tensor(done_arr, axis=0)
            self.exe.run(fluid.default_startup_program())

        obs_rollout_np, done_rollout_np = self.exe.run(
            program=prog, fetch_list=[obs_rollout, done_rollout])
        assert obs_rollout_np.shape == (horizon + 1, bs, c, *self.env_config["obs_dims"])
        assert done_rollout_np.shape == (horizon + 1, bs, c)

    def test_rewards(self):
        bs = 4
        self.learner_config["bayesian"] = {}
        prog = fluid.Program()
        with fluid.program_guard(prog):
            env = DynamicsModel("env", self.env_config, self.learner_config)
            obs = fluid.layers.uniform_random(
                [bs, *self.env_config["obs_dims"]], seed=1)
            next_obs = fluid.layers.uniform_random(
                [bs, *self.env_config["obs_dims"]], seed=2)
            action = fluid.layers.uniform_random(
                [bs, self.env_config["action_dim"]], seed=3)
            rewards = env.get_rewards(obs, action, next_obs)
            self.exe.run(fluid.default_startup_program())

        res = self.exe.run(program=prog, fetch_list=[rewards])
        rewards_np = res[0]
        assert rewards_np.shape == (bs,)

    def test_ensemble_rewards(self):
        bs = 4
        c = self.bayesian_config["ensemble_size"]
        self.learner_config["bayesian"] = self.bayesian_config
        prog = fluid.Program()
        with fluid.program_guard(prog):
            env = DynamicsModel("env", self.env_config, self.learner_config)
            obs = fluid.layers.uniform_random(
                [bs, c, *self.env_config["obs_dims"]], seed=1)
            next_obs = fluid.layers.uniform_random(
                [bs, c, *self.env_config["obs_dims"]], seed=2)
            action = fluid.layers.uniform_random(
                [bs, c, self.env_config["action_dim"]])
            E, M, N = env.get_ensemble_idx_info()
            rewards = env.get_rewards(obs, action, next_obs, ensemble_idxs=E)
            self.exe.run(fluid.default_startup_program())

        res = self.exe.run(program=prog, fetch_list=[rewards])
        rewards_np = res[0]
        assert rewards_np.shape == (bs, c)
