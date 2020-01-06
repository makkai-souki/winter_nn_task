from format.format_controller import format_controller
import numpy as np
import matplotlib.pyplot as plt

videos = {
    "s11": "s",
    "s12": "s",
    "h11": "h",
    "h12": "h",
    "h13": "h",
    "k11": "k",
    "k12": "k",
    "k13": "k",
    "d11": "d",
    "d12": "d",
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
