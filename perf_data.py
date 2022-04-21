

class tensor_desc:
    def __init__(self, n, c, h, w) -> None:
        self.n = n
        self.c = c
        self.h = h
        self.w = w

class kernel_tile:
    def __init__(self, block_size, m_per_block, n_per_block, k0_per_block) -> None:
        self.block_size = block_size
        self.m_per_block = m_per_block
        self.n_per_block = n_per_block
        self.k0_per_block = k0_per_block

    def gen_kernel_tile(self):
        return f'<{self.block_size}, {self.m_per_block}, {self.n_per_block}, {self.k0_per_block}>'

class perf_desc:
    def __init__(self, elapsed_time, tflops, band_width, kernel_tile) -> None:
        self.elapsed_time = elapsed_time
        self.tflops = tflops
        self.band_width = band_width
        self.kernel_tile = kernel_tile


class perf_data:
    def __init__(self, in_t, out_t, wei_t, perf_t) -> None:
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
        