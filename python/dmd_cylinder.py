import h5py
import numpy as np
import matplotlib.pyplot as plt
from pydmd import DMD

path_h5 = '/home/breno/cfd-dmd/python/cylinder.h5'
file_h5 = h5py.File(path_h5, 'r')

extracted_column = []

for key in file_h5['/Function/u']:
    dataset = file_h5['/Function/u/' + key]
    column = dataset[:, 0]
    extracted_column.append(column)

X = np.array(extracted_column)
print(X.shape)

dmd = DMD()
dmd.fit(X)

total_time_points = X.shape[0]
time_interval = 0.0025
t_values = np.arange(0, total_time_points * time_interval, time_interval)
predicted_data = dmd.predict(t_values)

print("Matriz de Previsões:")
print(predicted_data)
X_data = np.array(predicted_data)
print(X_data.shape)

# frobenius_errors = []

# for i in range(X_data.shape[0]):
#     original_data = X[:, i]
#     predicted_data_at_time = predicted_data[:, i]
#     error = np.linalg.norm(original_data - predicted_data_at_time, 'fro') / np.linalg.norm(original_data)
#     frobenius_errors.append(error)