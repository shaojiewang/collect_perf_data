

class log_file:
    def __init__(self, file_path, file_type = "ck") -> None:
        self.file_path = file_path
        self.file_type = file_type
        with open(file_path) as f:
            self.txt_lines = f.readlines()
