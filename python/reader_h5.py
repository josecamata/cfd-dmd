import h5py
import numpy as np
from pydmd import DMD
import dmd

path_h5 = '/home/breno/cfd-dmd/python/cylinder.h5'
file_h5 = h5py.File(path_h5, 'r')

extracted_column = []

for key in file_h5['/Function/u']:
    dataset = file_h5['/Function/u/' + key]
    column = dataset[:, 0]
    extracted_column.append(column)

data_matrix = np.array(extracted_column).T  # Transpõe para obter as snapshots como colunas

r = 20
delta_t = 0.0025

predictions = dmd.DMD4cast(data_matrix, r, delta_t)
print(predictions)

# X = np.array(extracted_column).T
# # print(X)

# dmd = DMD(svd_rank=2)
# dmd.fit(X)

# N, T = X.shape

# num_predictions = 2
# Xn = X[:, 0]

# for t in range(T + num_predictions):
#     Xn = dmd.predict(Xn)

# print(Xn.T.real)