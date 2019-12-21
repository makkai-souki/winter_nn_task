import numpy as np
import cv2
import matplotlib.pyplot as plt
import os


class format_model:
    def __init__(self, *, save_dir="./data/"):
        """
        
        Parameters
        ----------
        save_dir : str, optional
            保存先のディレクトリ, by default 'data'
        """
        self.save_dir = save_dir
        print(os.path.exists(self.save_dir))

    def make_flames(self, video_name, label, *, dirpass="flames", base_name="flame"):
        """ 動画をフレームに分割する変更する関数
        
        Parameters
        ----------
        video_name : string
            入力する動画の名称
        label : int
            つけるラベル
        dirpass : str, optional
            ディレクトリのパス, by default 'flames'
        base_name : str, optional
            出力画像の名称, by default 'flame'

        Returns
        -------
        out_flames : list
            出力するフレームの配列. opencv形式3channelのnumpy配列のリスト
        """
        label_dir = self.save_dir + label
        print(os.path.exists(label_dir))

    def grayscale(self, filename, *, mode=0, outname="self", out_mode=0):
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
            アウトプットの形式（0: ファイル化, 1: no.array）
        
        Returns
        -------
        gray_image
        """
        if filematrix is None or filename is None:
            return -1
        if filename is not None:
            pass

    def resize_flame(
        self, *, filematrix=None, filename=None, outname="self", size=(50, 50)
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
        pass

    def convert_zip(self, dir_name):
        """ フォルダを圧縮する
        
        Parameters
        ----------
        dir_name : str
            圧縮するフォルダ名
        """
        pass
