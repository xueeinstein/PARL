import paddle.fluid as fluid

from utils import *


class TestRandomUtils(object):
    def setup_class(self):
        place = fluid.CPUPlace()
        self.exe = fluid.Executor(place)

    def test_uniform_sample(self):
        sample_n, size = 5, 10
        prog = fluid.Program()
        with fluid.program_guard(prog):
            samples = fluid.layers.uniform_random([size, sample_n, 2, 2])
            samples = uniform_sample(samples, sample_n, size)

        res = self.exe.run(program=prog, fetch_list=[samples])
        assert res[0].shape == (size, 2, 2)

    def test_arange_shuffle(self):
        n = 10
        prog = fluid.Program()
        with fluid.program_guard(prog):
            arange = arange_shuffle(n)

        res1 = self.exe.run(program=prog, fetch_list=[arange])
        res2 = self.exe.run(program=prog, fetch_list=[arange])
        assert (res1[0] != res2[0]).any()
