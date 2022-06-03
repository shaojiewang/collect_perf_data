import pandas as pd
import perf_data
import log_file_extraction
import re

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
            input_l = re.search(r'in_n_c_hi_wi: dim \d+, lengths \{\d+, \d+, \d+, \d+\}', each_line)
            if input_l:
                tensor_list = []
                nchw_p = re.search(r'lengths \{\d+, \d+, \d+, \d+\}', input_l.group())
                nchw = re.findall(r'\d+', nchw_p.group())
                nchw_l = [int(length) for length in nchw]
                input_d = perf_data.tensor_desc(nchw_l[0], nchw_l[1], nchw_l[2], nchw_l[3])
                tensor_list.append(input_d)

            # find weight dim
            weight_l = re.search(r'wei_k_c_y_x: dim \d+, lengths \{\d+, \d+, \d+, \d+\}', each_line)
            if weight_l:
                nchw_p = re.search(r'lengths \{\d+, \d+, \d+, \d+\}', weight_l.group())
                nchw = re.findall(r'\d+', nchw_p.group())
                nchw_l = [int(length) for length in nchw]
                weight_d = perf_data.tensor_desc(nchw_l[0], nchw_l[1], nchw_l[2], nchw_l[3])
                tensor_list.append(weight_d)

            # find output dim
            output_l = re.search(r'out_n_k_ho_wo: dim \d+, lengths \{\d+, \d+, \d+, \d+\}', each_line)
            if output_l:
                nchw_p = re.search(r'lengths \{\d+, \d+, \d+, \d+\}', output_l.group())
                nchw = re.findall(r'\d+', nchw_p.group())
                nchw_l = [int(length) for length in nchw]
                output_d = perf_data.tensor_desc(nchw_l[0], nchw_l[1], nchw_l[2], nchw_l[3])
                tensor_list.append(output_d)

            # find best perf
            best_perf_l = re.search(r'Best Perf:', each_line)
            if best_perf_l:
                # elapsed time
                ms_p = re.search(r'\d+\.?\d+(?= ms,)', each_line)
                ms_l = ms_p.group().split(' ')
                ms = float(ms_l[0])

                # tflops
                tflops_p = re.search(r'\d+\.?\d+(?= TFlops,)', each_line)
                tflops_l = tflops_p.group().split(' ')
                tflops = float(tflops_l[0])

                # bandwidth
                bw_p = re.search(r'\d+\.?\d+(?= GB/s,)', each_line)
                bw_l = bw_p.group().split(' ')
                bw = float(bw_l[0])

                # kernel tile size
                ts_p = re.search(r'\<\d+, \d+, \d+, \d+\>', each_line)
                ts_l = re.findall(r'\d+', ts_p.group())
                ts = [int(ts_l_e) for ts_l_e in ts_l]
                tile_size_t = perf_data.kernel_tile(ts[0], ts[1], ts[2], ts[3])

                perf_data_t = perf_data.perf_desc(ms, tflops, bw, tile_size_t)

                perf_t = perf_data.perf_data(tensor_list[0], tensor_list[1], tensor_list[2], perf_data_t)
                perf_list.append(perf_t)

        self.perf_list = perf_list

        return perf_list

    def gen_perf_pd(self):
        perf_list = self.perf_list
        perf_array = []
        data_index = ['n', 'hi', 'wi', 'k', 'c', 'y', 'x', 'ho', 'wo', 'time', 'tflops', 'bandwidth', 'kernel']
        for perf_ele in perf_list:
            perf_df = perf_ele.gen_perf_df()
            perf_array.append(perf_df)

        log_idx = list(range(len(perf_array)))
        ds_read_bank_df = pd.DataFrame(perf_array, columns=data_index, index=log_idx)

        with pd.ExcelWriter('ck_perf_data_wrw_resnet50_bs256_mi200.xlsx') as writer:  
            ds_read_bank_df.to_excel(writer, sheet_name='ck_perf')


if __name__ == "__main__":
    file_p = "./ck_wrw_resnet50_mi200_bs256.log"
    log_f_t = log_file_extraction.log_file(file_p)
    perf_pd_t = perf_data_pd()
    import cProfile
    cProfile.run('perf_data_pd()')
    perf_l = perf_pd_t.extract_log_file(log_f_t)
    perf_pd_t.gen_perf_pd()

    #print(f"ms={perf_l[0].perf_t.elapsed_time}, tflops={perf_l[0].perf_t.tflops}, bw={perf_l[0].perf_t.band_width}")
    