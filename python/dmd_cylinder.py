import h5py
import numpy as np
import matplotlib.pyplot as plt
from dmd_class import DMD

path_h5 = '/home/breno/cfd-dmd/python/cylinder.h5'
file_h5 = h5py.File(path_h5, 'r')

extracted_column = []
total_time_points = 0

for key in file_h5['/Function/u']:
    dataset = file_h5['/Function/u/' + key]
    column = dataset[:, 0]
    extracted_column.append(column)
    total_time_points += 1

    if total_time_points == 50:
        break

X = np.array(extracted_column).T
print(X.shape)

dmd = DMD()
dmd.fit(X)

# total_time_points = X.shape[1]
time_interval = 0.0025
t_values = np.arange(0, total_time_points * time_interval, time_interval)
predicted_data = dmd.predict(t_values)

X_data = np.array(predicted_data)
print(X_data.shape)

frobenius_errors = []

for i in range(X_data.shape[1]):
    original_data = X[:, i]
    predicted_data_at_time = predicted_data[:, i]
    error = np.linalg.norm(original_data - predicted_data_at_time)
    frobenius_errors.append(error)

plt.plot(np.arange(0, total_time_points * time_interval, time_interval), frobenius_errors, label='Erro de Frobenius')
plt.xlabel('Tempo')
plt.ylabel('Erro de Frobenius')
plt.title('Erro de Frobenius entre imagens originais e previstas')
plt.legend()
plt.show()