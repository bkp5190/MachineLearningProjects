import random
import numpy as np
import matplotlib.pyplot as plt
from Problem5 import *

if __name__ == '__main__':
    # Put the code for the plots here, you can use different functions for each
    # part

    df = np.load('data.npy')
    y = df[:, 1]
    eta = np.array([0.05, 0.1, 0.1, 0.1])
    delta = np.array([0.1, 0.01, 0, 0])
    lambda1 = np.array([0.001, 0.001, 0.001, 0])
    num_iter = np.array([50, 50, 100, 100])
    w = np.zeros(len(df[0]))

    for x in range(0, len(df) -1):
        df[:, 1] = 1
    
    i = 0

    while i <= 4:
        new_w, history_fw = bgd_12(data, y, eta[i], delta[i], lambda1[i], num_iter[i])
        plt.xlabel('Iteration Number')
        plt.ylabel('History of Objective Function')
        plt.plot(range(len(history_fw)), history_fw)
        i += 1

    i = 0
    eta = np.array([1, 1, 1, 1])
    delta = np.array([0.1, 0.01, 0, 0])
    lambda1 = np.array([0.05, 0.01, 0, 0])
    num_iter = np.array([800, 800, 40, 800])
    w = np.zeros(len(df[0]))
    while i <= 4:
        new_w, history_fw = sgd_12(data, y, eta[i], delta[i], lambda1[i], num_iter[i])
        plt.xlabel('Iteration Number')
        plt.ylabel('History of Objective Function')
        pl.yscale('log')
        plt.plot(range(len(history_fw)), history_fw)
        i += 1