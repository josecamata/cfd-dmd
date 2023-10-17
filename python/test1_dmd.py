
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dmd_class import DMD

def generate_data(xsize, tsize, noise=5e-2):
    x = np.arange(xsize)
    y = np.sin(0.02 * x) + noise * np.random.randn(np.prod(x.shape))
    y.flatten()
    # print("y shape" + str(y.shape))
    # print(y)
    ts_length = tsize
    data = np.array([
            y[start : start + ts_length] for start in range(0, y.shape[0] - ts_length)
        ])

    return x, y, data.T

def process_data_to_array(data):
    datat = data.T

    xsize = np.arange(datat.shape[0])
    tsize = np.arange(datat.shape[1])
    array = []
    for i in range(0, data.shape[1]):
        array.append(data[0,i])

    for i in range(1, data.shape[0]):
        array.append(data[i,tsize-1])
    print(array)
    return np.array(array)

if __name__ == "__main__":

    x_length = 1000
    ts_length = 200
    x,y,data = generate_data(x_length, ts_length)

    print(data.shape)
    print(data)
    # y = process_data_to_array(data)

    figsize = (24, 10)


    dmd = DMD()
    dmd.fit(data)


    fname = "synthetic-data.png"
    plt.clf()
    plt.figure(figsize=figsize)
    plt.plot(y, "r", label="raw data")

    t = 0
    pred_x = np.arange(t, t + ts_length)
    pred_y = dmd.predict_future(t)
    plt.plot(pred_x, pred_y, "b", label="fitted data")

    t = y.shape[0]
    pred_x = np.arange(t, t + ts_length)
    pred_y = dmd.predict_future(t)
    plt.plot(pred_x, pred_y, "c", label="future prediction")
    plt.legend()
    plt.savefig(fname)