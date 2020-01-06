import numpy as np
import cv2
from format.format_model import format_model
import os


class format_controller:
    def __init__(self, video_pass, label):
        """
        
        Parameters
        ----------
        video_pass : str
            データセットの元になる動画
        label : str
            データセットのラベル
        """
        self.fm = format_model()
        self.label = label
        self.video_pass = video_pass

    def make_datasets(self, input_size=(50, 50)):
        self.fm.make_frames(self.video_pass, self.label)
        img_list = os.listdir(self.fm.save_dir + self.label)
        for img in img_list:
            tmp = self.fm.grayscale(filename="./data/" + self.label + "/" + img, mode=1)
            self.fm.resize_frame(
                filematrix=tmp,
                outname="./data/" + self.label + "/" + img,
                size=input_size,
            )
