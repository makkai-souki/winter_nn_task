import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from nn.nn_model import nn_model


class nn_controller:
    def __init__(
        self, *, data_pass="./data/dataset.npz", epochs=20, input_size=(50, 50)
    ):
        self.data_pass = data_pass
        self.nn = nn_model(
            epochs=epochs,
            categories=[
                "1",
                "2",
                "3",
                "4",
                "5",
                "6",
                "7",
                "8",
                "9",
                "10",
                "j",
                "q",
                "k",
            ],
        )
        self.size = input_size

    def mark_predict(self):
        self.nn.load_data(data_pass=self.data_pass)
        self.nn.define_mark_model(size=self.size)
        self.nn.compile_model()
        self.nn.fitting()

    def num_predict(self):
        self.nn.load_data(data_pass=self.data_pass)
        print("loaded")
        self.nn.define_num_model(size=self.size)
        print("defined")
        self.nn.compile_model()
        print("compiled")
        self.nn.fitting()

    def get_results(self):
        self.nn.save_model()
        self.nn.get_accepoch_graph()
        self.nn.get_lossepoch_graph()
        self.nn.get_graph()
        self.nn.get_predict_map()
        self.nn.get_precision()
        self.nn.get_recall()
        self.nn.get_f_mean()
