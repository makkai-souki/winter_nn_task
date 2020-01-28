from format.format_controller import format_controller
import numpy as np
import matplotlib.pyplot as plt

videos = {
    "s1_1": "1",
    "s2_1": "2",
    "s3_1": "3",
    "s4_1": "4",
    "s5_1": "5",
    "s6_1": "6",
    "s7_1": "7",
    "s7_2": "7",
    "s8_1": "8",
    "s9_1": "9",
    "s10_1": "10",
    "sj_1": "j",
    "sq_1": "q",
    "sk_1": "k",
    "h1_1": "1",
    "h2_1": "2",
    "h3_1": "3",
    "h4_1": "4",
    "h5_1": "5",
    "h6_1": "6",
    "h7_1": "7",
    "h7_2": "7",
    "h8_1": "8",
    "h9_1": "9",
    "h10_1": "10",
    "hj_1": "j",
    "hq_1": "q",
    "hk_1": "k",
}
print(videos.items())
for key, value in videos.items():
    print("./video/" + key + ".MOV")
    fc = format_controller("./video/" + key + ".MOV", value)
    fc.make_datasets(input_size=(50, 50))
fc.fm.convert_markdata2npz(
    0.2, labels=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "j", "q", "k"]
)
# fs = format_controller("./video/s1.MOV", "s")
# fs.make_datasets()
# fk = format_controller("./video/k1.MOV", "k")
# fk.make_datasets()
# fh = format_controller("./video/h1.MOV", "h")
# fh.make_datasets()
# fd = format_controller("./video/d1.MOV", "d")
# fd.make_datasets()
# fd.fm.convert_markdata2npz(0.2)
