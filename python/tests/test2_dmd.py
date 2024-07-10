
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm

from dmd_class import DMD

np.set_printoptions(formatter={'float': lambda x: f"{x:15.4e}"})

def sech(x):
    return 1/np.cosh(x)

if __name__ == "__main__":

    xsize = 400
    tsize = 200
    xi = np.linspace ( -10 ,10 ,xsize) 
    t  = np.linspace (0 ,4*np.pi ,tsize)
    dt = t[1] - t[0] 
    Xgrid ,T = np.meshgrid (xi ,t)

    f1 = sech(Xgrid+3)*(1* np.exp(1j*2.3*T))
    f2 = sech(Xgrid)*np.tanh (Xgrid)*(2* np.exp(1j *2.8*T))
    F = f1 + f2

    #np.disp(F.real)
    X = F.T

    X1 = X[:,:-1]
    X2 = X[:, 1:]

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(221, projection='3d')
    surf = ax.plot_surface(Xgrid, T, F.real, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('f')

    dmd = DMD()
    dmd.fit(X,dt=dt)
    Fdmd = dmd.predict(t)
   
    error = np.linalg.norm(Fdmd.T-F)
    print("Error: " + str(error))


    ax = fig.add_subplot(222, projection='3d')
    surf = ax.plot_surface(Xgrid, T, Fdmd.T.real, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    ax.set_xlabel('x')
    ax.set_ylabel('t')
    ax.set_zlabel('fdmd')

    plt.show()


