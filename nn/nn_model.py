from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
import keras
from keras import layers, models
from keras import optimizers
from keras.utils import np_utils


class nn_model:
    def __init__(self, *, epochs=20, batch=64, categories=["s", "h", "k", "d"]):
        """ nnの基本関数群の初期化
        
        Parameters
        ----------
        epochs : int, optional
            エポック数, by default 20
        batch : int, optional
            バッチサイズ, by default 64
        """
        self.epochs = epochs
        self.batch = batch
        self.categories = categories
        self.nb_classes = len(self.categories)

    def load_data(self, *, data_pass="./data/dataset.npz"):
        self.categories = ["s", "h", "k", "d"]
        self.nb_classes = len(self.categories)
        f = np.load(data_pass)
        X_train, self.y_train = f["x_train"], f["y_train"]
        X_test, self.y_test = f["x_test"], f["y_test"]
        f.close()
        self.x_train = X_train.astype("float") / 255
        self.x_test = X_test.astype("float") / 255
        np.random.seed(1)
        np.random.shuffle(self.x_train)
        np.random.seed(1)
        np.random.shuffle(self.y_train)

    def define_num_model(self, *, size=(50, 50)):
        self.model = models.Sequential()
        self.model.add(layers.Flatten(input_shape=size))
        self.model.add(layers.Dense(512, activation="relu"))
        self.model.add(layers.Dropout(0.5))
        self.model.add(layers.Dense(13, activation="softmax"))

    def define_mark_model(self, *, size=(50, 50)):
        self.model = models.Sequential()
        self.model.add(layers.Flatten(input_shape=size))
        self.model.add(layers.Dense(512, activation="relu"))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(4, activation="softmax"))

    def compile_model(self):
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        print(self.model.summary())

    def fitting(self):
        self.fit = self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            validation_data=(self.x_test, self.y_test),
        )

    def get_score(self):
        self.score = self.model.evaluate(self.x_train, self.y_test)

    def summery_model(self):
        pass

    def graphing_history(self):
        pass
