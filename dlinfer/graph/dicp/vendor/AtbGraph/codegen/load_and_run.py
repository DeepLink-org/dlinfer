import atexit
import os

import acl
import numpy as np
import torch
from torch.profiler import record_function


class AtbModel:
    def __init__(self, model_path) -> None:
        # print("### in_load_and_run_model_path:", model_path)
        self.model = torch.classes.DICPModel.DICPModel(model_path)

    @record_function("load_and_run")
    def run(self, inputs, outputs, param):
        self.model.execute_out(inputs, outputs, param)


if __name__ == "__main__":
    pass
