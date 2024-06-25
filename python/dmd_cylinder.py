import h5py
import numpy as np
import matplotlib.pyplot as plt
from dmd_class import DMD

path_h5_predicted = '/home/breno/cfd-dmd/cylinder_h5/cylinder.h5'
file_h5_predicted = h5py.File(path_h5_predicted, 'r+')

extracted_column = []
total_time_points = 0

for key in file_h5_predicted['/Function/u']:
    dataset = file_h5_predicted['/Function/u/' + key]
    column = dataset[:, 0]
    extracted_column.append(column)
    total_time_points += 1

    # if total_time_points == 1000:
    #     break

X = np.array(extracted_column).T
shape = dataset.shape
# print(X.shape)

dmd = DMD()
dmd.fit(X, dt=0.0025, thresh=1.0e-3)

time_interval = 0.0025
t_values = np.arange(0, total_time_points * time_interval, time_interval)
predicted_data = dmd.predict(t_values)

# Sobrescrever os dados preditos no arquivo HDF5
for i, key in enumerate(file_h5_predicted['/Function/u']):
    dataset = file_h5_predicted['/Function/u/' + key]
    dataset[:, 0] = predicted_data[:, i]

file_h5_predicted.close()

# Calcular os erros de Frobenius entre os dados originais e previstos
frobenius_errors = []
for i in range(predicted_data.shape[1]):
    original_data = X[:, i]
    predicted_data_at_time = predicted_data[:, i]
    error = np.linalg.norm(original_data - predicted_data_at_time) / np.linalg.norm(original_data)
    frobenius_errors.append(error)

# Plotar os erros de Frobenius
plt.plot(np.arange(0, total_time_points * time_interval, time_interval), frobenius_errors, label='Erro de Frobenius')
plt.xlabel('Tempo')
plt.ylabel('Erro de Frobenius')
plt.title('Erro de Frobenius entre imagens originais e previstas')
plt.legend()
plt.show()