import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nn.nn_model import nn_model


class nn_controller:
    def __init__(self, data_pass="../data/dataset.npz", epoch=60):
        self.data_pass = data_pass
        self.nn = nn_model(epochs=epoch)

    def mark_predict(self):
        self.nn.load_data()
        self.nn.define_mark_model()
        self.nn.compile_model()
        self.nn.fitting()

    def get_results(self):
        pass
