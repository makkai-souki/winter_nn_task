import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import random
from keras.preprocessing.image import array_to_img, img_to_array, load_img


class format_model:
    def __init__(self, *, save_dir="./data/"):
        """
        
        Parameters
        ----------
        save_dir : str, optional
            保存先のディレクトリ, by default 'data'
        """
        self.save_dir = save_dir

    def make_frames(self, video_name, label, *, base_name="frame", ext="jpg"):
        """ 動画をフレームに分割する変更する関数
        
        Parameters
        ----------
        video_name : string
            入力する動画の名称
        label : str
            つけるラベル
        dirpass : str, optional
            ディレクトリのパス, by default 'frames'
        base_name : str, optional
            出力画像の名称, by default 'frame'

        Returns
        -------
        out_frames : list
            出力するフレームの配列. opencv形式3channelのnumpy配列のリスト
        """
        label_dir = self.save_dir + label
        os.makedirs(label_dir, exist_ok=True)
        cap = cv2.VideoCapture(video_name)
        if not cap.isOpened():
            return -1
        while True:
            ret, frame = cap.read()
            if ret:
                n = random.randint(0, 100000000000)
                cv2.imwrite(
                    "{}/{}_{}.{}".format(label_dir, base_name, str(n), ext), frame
                )
            else:
                return 0

    def grayscale(self, *, filematrix=None, filename=None, outname="self", mode=0):
        """ グレースケール化の関数
        
        Parameters
        ----------
        img : np.array
            入力する画像. opencv3チャンネルのnumpy配列
        filename : str
            入力するファイル名
        mode : int, optional
            グレースケールのモード
        outname : str, optional
            アウトプットするファイル名（selfはfilenameを同様）
        outmode : int, optional
            アウトプットの形式（0: ファイル化, 1: np.array）
        
        Returns
        -------
        gray_image
        """
        if filename is None and filematrix is None:
            return -1
        if filematrix is None:
            out = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
            if outname == "self":
                outname = self.save_dir + filename
            else:
                outname = self.save_dir + outname
        else:
            out = cv2.cvtColor(filematrix, cv2.COLOR_BGR2GRAY)
        if mode == 0:
            cv2.imwrite(outname, out)
            return 0
        else:
            return out

    def resize_frame(
        self, *, filematrix=None, filename=None, outname="self", size=(50, 50), mode=0
    ):
        """ 画像をリサイズする関数
        
        Parameters
        ----------
        filematrix : [type], optional
            [description], by default None
        filename : [type], optional
            [description], by default None
        outname : str, optional
            [description], by default "self"
        size : tuple, optional
            画像サイズ, by default (50, 50)
        """
        if filename is None and filematrix is None:
            return -1
        if filename is not None:
            filematrix = cv2.imread(filename)
            if outname == "self":
                outname = self.save_dir + filename
            else:
                outname = self.save_dir + outname
        out = cv2.resize(filematrix, dsize=size)
        if mode == 0:
            cv2.imwrite(outname, out)
            return 0
        else:
            return out

    def convert_markdata2npz(
        self, test_ratio, *, base_dir="", labels=["s", "h", "k", "d"]
    ):
        X_train = []
        Y_train = []
        X_test = []
        Y_test = []
        num = 0
        for label in labels:
            n = 0
            data = os.listdir(self.save_dir + base_dir + label)
            file_num = len(data)
            test = int(file_num * test_ratio)
            for filename in data:
                if ".jpg" in filename:
                    img = self.save_dir + label + "/" + filename
                    tmp_img_array = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
                    # print(tmp_img_array)
                else:
                    continue
                if n > test:
                    X_train.append(tmp_img_array)
                    Y_train.append(num)
                else:
                    X_test.append(tmp_img_array)
                    Y_test.append(num)
                n += 1
            num += 1
        np.savez(
            self.save_dir + "dataset",
            x_train=X_train,
            y_train=Y_train,
            x_test=X_test,
            y_test=Y_test,
        )

    def convert_zip(self, dir_name):
        """ フォルダを圧縮する
        
        Parameters
        ----------
        dir_name : str
            圧縮するフォルダ名
        """
        pass
