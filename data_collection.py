import pandas as pd
import perf_data
import log_file_extraction
import re
import argparse

class perf_data_pd:
    def __init__(self) -> None:
        self.perf_pd = None
        self.perf_list = None

    def extract_log_file(self, log_file):
        assert isinstance(log_file, log_file_extraction.log_file)
        txt = log_file.txt_lines
        tensor_list = []
        perf_list = []
        for each_line in txt:
            # find input dim
            if log_file.op_type == log_file_extraction.OP_CONV:
                input_l = re.search(r'in_n_c_hi_wi: dim \d+, lengths \{\d+, \d+, \d+, \d+\}', each_line)
                if input_l:
                    tensor_list = []
                    nchw_p = re.search(r'lengths \{\d+, \d+, \d+, \d+\}', input_l.group())
                    nchw = re.findall(r'\d+', nchw_p.group())
                    nchw_l = [int(length) for length in nchw]
                    input_d = perf_data.tensor_desc_conv(nchw_l[0], nchw_l[1], nchw_l[2], nchw_l[3])
                    tensor_list.append(input_d)

                # find weight dim
                weight_l = re.search(r'wei_k_c_y_x: dim \d+, lengths \{\d+, \d+, \d+, \d+\}', each_line)
                if weight_l:
                    nchw_p = re.search(r'lengths \{\d+, \d+, \d+, \d+\}', weight_l.group())
                    nchw = re.findall(r'\d+', nchw_p.group())
                    nchw_l = [int(length) for length in nchw]
                    weight_d = perf_data.tensor_desc_conv(nchw_l[0], nchw_l[1], nchw_l[2], nchw_l[3])
                    tensor_list.append(weight_d)

                # find output dim
                output_l = re.search(r'out_n_k_ho_wo: dim \d+, lengths \{\d+, \d+, \d+, \d+\}', each_line)
                if output_l:
                    nchw_p = re.search(r'lengths \{\d+, \d+, \d+, \d+\}', output_l.group())
                    nchw = re.findall(r'\d+', nchw_p.group())
                    nchw_l = [int(length) for length in nchw]
                    output_d = perf_data.tensor_desc_conv(nchw_l[0], nchw_l[1], nchw_l[2], nchw_l[3])
                    tensor_list.append(output_d)

            elif log_file.op_type == log_file_extraction.OP_GEMM:
                a_l = re.search(r'a_.*: dim \d+, lengths \{(\d+, )+\d+\}', each_line)
                if a_l:
                    tensor_list = []
                    a_gkm_p = re.search(r'lengths \{(\d+, )+\d+\}', a_l.group())
                    a_gkm = re.findall(r'\d+', a_gkm_p.group())
                    a_gkm_l = [int(length) for length in a_gkm]
                    if len(a_gkm_l) == 2:
                        a_gkm_l.insert(0, 1)
                    a_d = perf_data.tensor_desc_gemm(a_gkm_l[0], a_gkm_l[1], a_gkm_l[2])
                    tensor_list.append(a_d)

                b_l = re.search(r'b_.*: dim \d+, lengths \{(\d+, )+\d+\}', each_line)
                if b_l:
                    b_gkm_p = re.search(r'lengths \{(\d+, )+\d+\}', b_l.group())
                    b_gkm = re.findall(r'\d+', b_gkm_p.group())
                    b_gkm_l = [int(length) for length in b_gkm]
                    if len(b_gkm_l) == 2:
                        b_gkm_l.insert(0, 1)
                    b_d = perf_data.tensor_desc_gemm(b_gkm_l[0], b_gkm_l[1], b_gkm_l[2])
                    tensor_list.append(b_d)

                c_l = re.search(r'c_.*: dim \d+, lengths \{(\d+, )+\d+\}', each_line)
                if c_l:
                    c_gkm_p = re.search(r'lengths \{(\d+, )+\d+\}', c_l.group())
                    c_gkm = re.findall(r'\d+', c_gkm_p.group())
                    c_gkm_l = [int(length) for length in c_gkm]
                    if len(c_gkm_l) == 2:
                        c_gkm_l.insert(0, 1)
                    c_d = perf_data.tensor_desc_gemm(c_gkm_l[0], c_gkm_l[1], c_gkm_l[2])
                    tensor_list.append(c_d)

            # find best perf
            best_perf_l = re.search(r'Best Perf:', each_line)
            if best_perf_l:
                # elapsed time
                ms_p = re.search(r'\d+\.?\d*(?= ms,)', each_line)
                ms_l = ms_p.group().split(' ')
                ms = float(ms_l[0])

                # tflops
                tflops_p = re.search(r'\d+\.?\d*(?= TFlops,)', each_line)
                tflops_l = tflops_p.group().split(' ')
                tflops = float(tflops_l[0])

                # bandwidth
                bw_p = re.search(r'\d+\.?\d*(?= GB/s,)', each_line)
                bw_l = bw_p.group().split(' ')
                bw = float(bw_l[0])

                # kernel tile size
                ts_p = re.search(r'\<(\d+, )+\d+\>', each_line)
                if ts_p != None:
                    ts_l = re.findall(r'\d+', ts_p.group())
                else:
                    ts_l = [0]
                ts = [int(ts_l_e) for ts_l_e in ts_l]
                tile_size_t = perf_data.kernel_tile(ts)

                perf_data_t = perf_data.perf_desc(ms, tflops, bw, tile_size_t)

                if log_file.op_type == log_file_extraction.OP_CONV:
                    perf_t = perf_data.perf_data_conv(tensor_list[0], tensor_list[1], tensor_list[2], perf_data_t)
                elif log_file.op_type == log_file_extraction.OP_GEMM:
                    perf_t = perf_data.perf_data_gemm(tensor_list[0], tensor_list[1], tensor_list[2], perf_data_t)
                
                perf_list.append(perf_t)

        self.perf_list = perf_list

        return perf_list

    def gen_perf_pd(self, log_file):
        assert isinstance(log_file, log_file_extraction.log_file)
        perf_list = self.perf_list
        perf_array = []
        if log_file.op_type == log_file_extraction.OP_CONV:
            data_index = ['n', 'hi', 'wi', 'k', 'c', 'y', 'x', 'ho', 'wo', 'time', 'tflops', 'bandwidth', 'kernel']
        elif log_file.op_type == log_file_extraction.OP_GEMM:
            data_index = ['a_g', 'a_k', 'a_m', 'b_g', 'b_k', 'b_m', 'c_g', 'c_k', 'c_m', 'time', 'tflops', 'bandwidth', 'kernel']
        for perf_ele in perf_list:
            perf_df = perf_ele.gen_perf_df()
            perf_array.append(perf_df)

        log_idx = list(range(len(perf_array)))
        ds_read_bank_df = pd.DataFrame(perf_array, columns=data_index, index=log_idx)

        file_name = log_file.file_path
        xlsx_name = './' + file_name.split('.')[-2] + '.xlsx'

        with pd.ExcelWriter(xlsx_name) as writer:  
            ds_read_bank_df.to_excel(writer, sheet_name='ck_perf')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("log_file", help="your log file")
    parser.add_argument("op_type", help="op type:1-conv, 2-gemm")

    args = parser.parse_args()

    file_p = args.log_file
    op_type = int(args.op_type)
    log_f_t = log_file_extraction.log_file(file_p, 'ck', op_type)
    perf_pd_t = perf_data_pd()
    #import cProfile
    #cProfile.run('perf_data_pd()')
    perf_l = perf_pd_t.extract_log_file(log_f_t)
    perf_pd_t.gen_perf_pd(log_f_t)

    #print(f"ms={perf_l[0].perf_t.elapsed_time}, tflops={perf_l[0].perf_t.tflops}, bw={perf_l[0].perf_t.band_width}")
    