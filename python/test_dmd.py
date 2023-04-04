import numpy as np
import matplotlib.pyplot as plt
from dmd_class import DMD

# Create a DMD object
dmd = DMD(r=2)

n = 100
m = 500
# Fit the DMD object to the data
X = np.zeros((n, m))


for i in range(n):
    for j in range(m):
        X[i, j] = j + i


dmd = DMD()
dmd.fit(X)
N,T = X.shape
  
print(dmd.A.shape)

Xp = X[:,0]
for t in range(T+10):
    Xn = dmd.A @ Xp.real
    Xp = Xn

print(Xn.T.real)


