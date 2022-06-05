

class tensor_desc_conv:
    def __init__(self, n, c, h, w) -> None:
        self.n = n
        self.c = c
        self.h = h
        self.w = w

class tensor_desc_gemm:
    def __init__(self, g, k, m) -> None:
        self.g = g
        self.k = k
        self.m = m

class kernel_tile:
    def __init__(self, *arg) -> None:
        self.args = arg

    def gen_kernel_tile(self):
        return f'<{self.args}>'

class perf_desc:
    def __init__(self, elapsed_time, tflops, band_width, kernel_tile) -> None:
        self.elapsed_time = elapsed_time
        self.tflops = tflops
        self.band_width = band_width
        self.kernel_tile = kernel_tile


class perf_data_conv:
    def __init__(self, in_t, wei_t, out_t, perf_t) -> None:
        self.in_t = in_t
        self.out_t = out_t
        self.wei_t = wei_t
        self.perf_t = perf_t

        self.perf_pd = None

    def gen_perf_df(self):
        #data_index = ['n', 'hi', 'wi', 'k', 'c', 'y', 'x', 'ho', 'wo', 'time', 'tflops', 'bandwidth', 'kernel']
        in_t = self.in_t
        wei_t = self.wei_t
        out_t = self.out_t
        perf_t = self.perf_t
        perf_pd = [in_t.n, in_t.h, in_t.w, \
                   wei_t.n, wei_t.c, wei_t.h, wei_t.w, \
                   out_t.h, out_t.w, \
                   perf_t.elapsed_time, perf_t.tflops, perf_t.band_width,\
                   perf_t.kernel_tile.gen_kernel_tile()]
        self.perf_pd = perf_pd
        return perf_pd

class perf_data_gemm:
    def __init__(self, a_t, b_t, c_t, perf_t) -> None:
        self.a_t = a_t
        self.b_t = b_t
        self.c_t = c_t
        self.perf_t = perf_t

        self.perf_pd = None

    def gen_perf_df(self):
        #data_index = ['a_g', 'a_k', 'a_m', 'b_g', 'b_k', 'b_m', 'c_g', 'c_k', 'c_m', 'time', 'tflops', 'bandwidth', 'kernel']
        a_t = self.a_t
        b_t = self.b_t
        c_t = self.c_t
        perf_t = self.perf_t
        perf_pd = [a_t.g, a_t.k, a_t.m, \
                   b_t.g, b_t.k, b_t.m, \
                   c_t.g, c_t.k, c_t.m,  \
                   perf_t.elapsed_time, perf_t.tflops, perf_t.band_width,\
                   perf_t.kernel_tile.gen_kernel_tile()]
        self.perf_pd = perf_pd
        return perf_pd