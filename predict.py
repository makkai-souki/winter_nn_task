import numpy as np
import matplotlib.pyplot as plt
from nn.nn_controller import nn_controller

nc = nn_controller(epochs=20, input_size=(200, 200))
nc.num_predict()
nc.get_results()

