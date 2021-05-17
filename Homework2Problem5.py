import math
import random
import numpy as np


def bgd_l2(data, y, w, eta, delta, lam, num_iter):
    new_w = []
    history_fw = []
    new_w = w

    for i in range(num_iter):
        w_g = 0
        length = len(y)
        history = 0
        reg = 0
        
        for j in range(0, length):
            if y[j] >= (np.dot(np.transpose(new_w), data[j]) + delta):
                w_g += -(2/length) * data[j] * (y[j] - np.dot(np.transpose(new_w), data[j]) - delta) + (2 * lam * new_w)
            elif y[j] <= (np.dot(np.transpose(new_w), data[j]) - delta):
                w_g += -(2/length) * data[j] * (y[j] - np.dot(np.transpose(new_w), data[j]) + delta) + (2 * lam * new_w)
            else:
                w_g = (2 * lam * new_w)

        for j in range(0, length):
            if y[j] >= (np.dot(np.transpose(new_w), data[j]) + delta):
                history += ((y[j] - np.dot(np.transpose(new_w), data[j]) - delta)**2)
            elif y[j] <= (np.dot(np.transpose(new_w), data[j]) - delta):
                history += ((y[j] - np.dot(np.transpose(new_w), data[j]) + delta)**2)
            
        new_w -= eta * w_g
        for j in range(0, len(data[j])):
            reg += w[j]**2
        reg *= lam
        
        history *= (1 / length)
        history += reg
        history_fw.append(history)

    return new_w, history_fw

def sgd_l2(data, y, w, eta, delta, lam, num_iter, i=-1):
    new_w = []
    history_fw = []
    new_w = w

    if i == -1:
        end = num_iter
    else:
        end = 2
        
    for i in range(1, end):
        w_g = 0
        length = len(y)
        history = 0
        reg = 0
        randomIndex = random.randint(0, length-1)
        
        if y[randomIndex] >= (np.dot(np.transpose(new_w), data[randomIndex]) + delta):
            w_g += -(2/length) * data[randomIndex] * (y[randomIndex] - np.dot(np.transpose(new_w), data[randomIndex]) - delta) + (2 * lam * new_w)
        elif y[randomIndex] <= (np.dot(np.transpose(new_w), data[randomIndex]) - delta):
            w_g += -(2/length) * data[randomIndex] * (y[randomIndex] - np.dot(np.transpose(new_w), data[randomIndex]) + delta) + (2 * lam * new_w)
        else:
            w_g = (2 * lam * new_w)

        for j in range(0, length):
            if y[j] >= (np.dot(np.transpose(new_w), data[j]) + delta):
                history += ((y[j] - np.dot(np.transpose(new_w), data[j]) - delta)**2)
            elif y[j] <= (np.dot(np.transpose(new_w), data[j]) - delta):
                history += ((y[j] - np.dot(np.transpose(new_w), data[j]) + delta)**2)
        new_w -= ((eta / i**(1/2)) * w_g)
        
        for j in range(0, len(data[j])):
            reg += w[j]**2
        reg *= lam
        
        history *= (1 / length)
        history += reg
        history_fw.append(history)

    return new_w, history_fw
