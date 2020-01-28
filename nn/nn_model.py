from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import keras
from keras import layers, models
from keras import optimizers
from keras.utils import np_utils
from keras.utils import plot_model
import cv2


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
        print("shuffled")
        # for image in self.x_test:
            # if image[20][20] > 1:
                # print(image)
        return self

    def define_num_model(self, *, size=(50, 50)):
        self.model = models.Sequential()
        self.model.add(layers.Flatten(input_shape=size))
        self.model.add(layers.Dense(512, activation="relu"))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(13, activation="softmax"))
        return self

    def define_mark_model(self, *, size=(50, 50)):
        self.model = models.Sequential()
        self.model.add(layers.Flatten(input_shape=size))
        self.model.add(layers.Dense(512, activation="relu"))
        self.model.add(layers.Dropout(0.2))
        self.model.add(layers.Dense(4, activation="softmax"))
        return self

    def compile_model(self):
        self.model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        print(self.model.summary())
        return self

    def fitting(self):
        self.history = self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            validation_data=(self.x_test, self.y_test),
        )
        self.acc = self.history.history["acc"]
        self.val_acc = self.history.history["val_acc"]
        self.loss = self.history.history["loss"]
        self.val_loss = self.history.history["val_loss"]
        return self

    def get_score(self):
        self.score = self.model.evaluate(self.x_train, self.y_test)

    def save_model(self, name="tramp.hdf5"):
        self.model.save_weights("./result/" + name)

    def test_model(self):
        self.pred = self.model.predict(self.x_test)
        self.score = model.evaluate(X_test, y_test, verbose=1)

    def get_image(self, mode=True, number=10):
        """
        
        Parameters
        ----------
        mode : bool, optional
            Trueなら正解のデータを出力, by default True
        number : in, optional
            出力するデータの件数
        """
        for image in self.x_test:
            if image[20][20] > 1:
                print(image)
        categories = self.categories
        for i, v in enumerate(self.pre):
            pre_ans = v.argmax()
            ans = self.y_test[i]
            dat = self.x_test[i]
            if pre_ans == self.y_test[i]:
                continue
            fname = (
                # "./result/NG_photo/"
                "./result/NG_photo/miss_"
                + str(i)
                + "-"
                + categories[pre_ans]
                + "-ne-"
                + categories[ans]
                + ".png"
            )
            dat *= 256
            cv2.imwrite(fname, np.uint8(dat))

    def predict(self):
        self.pre = self.model.predict(self.x_test)

    def get_lossepoch_graph(self, *, save_file="./result/lossepoch.png"):
        epoches = np.array(range(self.epochs))
        fig = plt.figure()
        fig.add_subplot(111)
        plt.plot(epoches, self.val_loss, color=(1, 0, 0), label="validation")
        plt.plot(epoches, self.loss, color=(0, 1, 0), label="train")
        plt.xlabel("epoch")
        plt.xlim(0, self.epochs)
        plt.grid()
        plt.legend()
        plt.savefig(save_file)

    def get_accepoch_graph(self, *, save_file="./result/accepoch.png"):
        epoches = np.array(range(self.epochs))
        fig = plt.figure()
        fig.add_subplot(111)
        plt.plot(epoches, self.val_acc, color=(1, 0, 0), label="validation")
        plt.plot(epoches, self.acc, color=(0, 1, 0), label="train")
        plt.xlabel("epoch")
        plt.xlim(0, self.epochs)
        plt.grid()
        plt.legend()
        plt.savefig(save_file)

    def get_graph(self, *, save_file="./result/log.png"):
        epoches = np.array(range(self.epochs))
        fig = plt.figure()
        fig.add_subplot(111)
        plt.plot(epoches, self.val_acc, color=(1, 0, 0), label="accuracy")
        plt.plot(epoches, self.val_loss, color=(0, 1, 0), label="loss")
        plt.xlabel("epoch")
        plt.xlim(0, self.epochs)
        plt.grid()
        plt.legend()
        plt.savefig(save_file)

    def get_predict_map(self):
        self.predict()
        dim = len(self.categories)
        self.predict_map = np.zeros((dim, dim), dtype=int)
        for i, v in enumerate(self.pre):
            # self.predict_map
            pre_ans = v.argmax()
            ans = self.y_test[i]
            self.predict_map[pre_ans][ans] += 1
        print(self.predict_map)
    
    def get_precision(self):
        dim = len(self.categories)
        self.precision = np.zeros((dim))
        tp_fp = 0
        for i in range(dim):
            tp_fp = 0
            for j in range(dim):
                tp_fp += self.predict_map[i][j]
            self.precision[i] = self.predict_map[i][i] / tp_fp
        print(self.precision)

    def get_recall(self):
        dim = len(self.categories)
        self.recall = np.zeros((dim))
        tp_fn = 0
        for j in range(dim):
            tp_fn = 0
            for i in range(dim):
                tp_fn += self.predict_map[i][j]
            # print(tp_fn)
            self.recall[j] = self.predict_map[j][j] / tp_fn
        print(self.recall)
    
    def get_f_mean(self):
        dim = len(self.categories)
        self.f_mean_array = np.zeros((dim))
        for i in range(dim):
            self.f_mean_array[i] = (2 * self.precision[i] * self.recall[i])/(self.precision[i] + self.recall[i])
        print(self.f_mean_array)
        self.f_mean = self.f_mean_array.sum() / dim
        print(self.f_mean)
