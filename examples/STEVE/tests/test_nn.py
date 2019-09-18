import paddle.fluid as fluid

import nn


class TestEnsembleFeedForwardNet(object):
    def setup_class(self):
        place = fluid.CPUPlace()
        self.exe = fluid.Executor(place)

    def test_single_layer(self):
        es, eval_count = 4, 3
        bs, in_size, out_shape = 4, 5, [10]
        prog = fluid.Program()
        with fluid.program_guard(prog):
            net = nn.EnsembleFeedForwardNet(
                "ensemble", in_size, out_shape, ensemble_size=es,
                eval_sample_count=eval_count)
            data = fluid.layers.uniform_random([bs, in_size], seed=1)
            out = net(data, is_eval=True)
            self.exe.run(fluid.default_startup_program())

        res = self.exe.run(program=prog, fetch_list=[out])
        assert res[0].shape == (bs, eval_count, *out_shape)

    def test_multi_layers(self):
        layers = 2
        es, eval_count = 4, 3
        bs, in_size, out_shape = 4, 5, [10]
        prog = fluid.Program()
        with fluid.program_guard(prog):
            net = nn.EnsembleFeedForwardNet(
                "ensemble", in_size, out_shape, layers=layers,
                ensemble_size=es, eval_sample_count=eval_count)
            data = fluid.layers.uniform_random([bs, in_size], seed=1)
            out = net(data, is_eval=True)
            self.exe.run(fluid.default_startup_program())

        res = self.exe.run(program=prog, fetch_list=[out])
        assert res[0].shape == (bs, eval_count, *out_shape)

    def test_reduce_random(self):
        bs, in_size, out_shape = 4, 5, [10]
        prog = fluid.Program()
        with fluid.program_guard(prog):
            net = nn.EnsembleFeedForwardNet("ensemble", in_size, out_shape)
            data = fluid.layers.uniform_random([bs, in_size], seed=1)
            out = net(data, reduce_mode="random")
            self.exe.run(fluid.default_startup_program())

        res = self.exe.run(program=prog, fetch_list=[out])
        assert res[0].shape == (bs, *out_shape)

    def test_reduce_mean(self):
        bs, in_size, out_shape = 4, 5, [10]
        prog = fluid.Program()
        with fluid.program_guard(prog):
            net = nn.EnsembleFeedForwardNet("ensemble", in_size, out_shape)
            data = fluid.layers.uniform_random([bs, in_size], seed=1)
            out = net(data, reduce_mode="mean")
            self.exe.run(fluid.default_startup_program())

        res = self.exe.run(program=prog, fetch_list=[out])
        assert res[0].shape == (bs, *out_shape)


class TestFeedForwardNet(object):
    def setup_class(self):
        place = fluid.CPUPlace()
        self.exe = fluid.Executor(place)

    def test_multi_layers(self):
        layers = 2
        bs, in_size, out_shape = 4, 5, [10]
        prog = fluid.Program()
        with fluid.program_guard(prog):
            net = nn.FeedForwardNet("mlp", in_size, out_shape,
                                    layers=layers)
            data = fluid.layers.uniform_random([bs, in_size], seed=1)
            out = net(data)
            self.exe.run(fluid.default_startup_program())

        res = self.exe.run(program=prog, fetch_list=[out])
        assert res[0].shape == (bs, *out_shape)
