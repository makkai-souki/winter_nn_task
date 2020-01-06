import numpy as np
import matplotlib.pyplot as plt
from nn.nn_controller import nn_controller

nc = nn_controller(epochs=100)
nc.mark_predict()
nc.get_results()
