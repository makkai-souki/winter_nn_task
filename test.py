from format.format_model import format_model
from format.format_controller import format_controller
import numpy as np
import matplotlib.pyplot as plt


# fm = format_model()
# print(fm.make_frames("./video/IMG_0078.MOV", "test2/new"))
# print(
#     fm.resize_frame(
#         filename="./data/test2/new/frame_1.jpg", outname="resize.jpg", size=(100, 100)
#     )
# )
# print("hello world")

fc = format_controller("./video/IMG_0078.MOV", "test")
fc.make_datasets()
