OP_CONV = 1
OP_GEMM = 2

class log_file:
    def __init__(self, file_path, file_type = "ck", op_type = OP_CONV) -> None:
        self.file_path = file_path
        self.file_type = file_type
        self.op_type = op_type
        with open(file_path) as f:
            self.txt_lines = f.readlines()
