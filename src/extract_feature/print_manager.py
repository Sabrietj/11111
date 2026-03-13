# -*- coding: UTF-8 -*-
from time import time

class PrintManager:
    def __init__(self):
        self.index_of_folder = 1
        self.t0 = None
        self.t1 = None
        self.dash_line = "\n<<<---------------------------------------------------------------->>>"

    def set_finish_time(self):
        self.t1 = time()

    def welcome_header(self):
        print(self.dash_line)
        print("\n<<< Welcome to Network-Traffic-Multiview-Feature-Extraction project")

    def dataset_folder_header(self, path, size):
        print(
            f"\n{self.dash_line}\n"
            f"<<< Program will start to evaluate dataset folder:\n"
            f"{path}\n"
            f"<<< total number of folders: {size}\n"
            f"{self.dash_line}"
        )

    def single_folder_header(self, path_to_single):
        print(f"\n{self.dash_line}\n<<< dataset No.{self.index_of_folder} | {path_to_single}\n{self.dash_line}")
        self.t0 = time()
        self.index_of_folder += 1

    def success_single_folder_header(self):
        self.t1 = time()
        print(f"\n<<< approximate running time: {self.t1 - self.t0:.2f} sec.\n{self.dash_line}")

    def evaluate_creating_plot(self):
        print("\n<<< AnalyzeData.py: Creating dataset ...")

    def success_evaluate_data(self):
        print("\n<<< AnalyzeData.py: Process complete !!!")


__PrintManager__ = PrintManager()