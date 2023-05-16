import numpy as np
import scipy.io 

import matplotlib.pyplot as plt
# from dmd_class import DMD

from pydmd import DMD


file_path = '../DATA/FLUIDS/CYLINDER_ALL.mat'
mat = scipy.io.loadmat(file_path)
print(mat.keys())
print(mat['UALL'].shape)
X = mat['UALL']


# n = 100
# m = 500
# # Fit the DMD object to the data
# X = np.zeros((n, m))


# for i in range(n):
#     for j in range(m):
#         X[i, j] = j + i
 

dmd = DMD(svd_rank=2)
dmd.fit(X)
N,T = X.shape
  
# print(dmd.A.shape)

Xp = X[:,0]
for t in range(T+2):
    Xn = dmd.predict(Xp)
    Xp = Xn

print(Xn.T.real)


