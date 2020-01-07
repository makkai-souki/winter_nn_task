from format.format_controller import format_controller
import numpy as np
import matplotlib.pyplot as plt

videos = {
    "s1_1": "s",
    "s1_2": "s",
    "h1_2": "h",
    "h1_3": "h",
    "h2_1": "h",
    "k1_1": "k",
    "k1_2": "k",
    "k1_3": "k",
    "d1_1": "d",
    "d1_2": "d",
    #
    "s2_1": "s",
    "h1_1": "h",
    "k2_1": "k",
    "d2_1": "d",
}
print(videos.items())
for key, value in videos.items():
    print("./video/" + key + ".MOV")
    fc = format_controller("./video/" + key + ".MOV", value)
    fc.make_datasets()
fc.fm.convert_markdata2npz(0.2)
# fs = format_controller("./video/s1.MOV", "s")
# fs.make_datasets()
# fk = format_controller("./video/k1.MOV", "k")
# fk.make_datasets()
# fh = format_controller("./video/h1.MOV", "h")
# fh.make_datasets()
# fd = format_controller("./video/d1.MOV", "d")
# fd.make_datasets()
# fd.fm.convert_markdata2npz(0.2)
