import h5py
import numpy as np

path_h5 = '/home/breno/cfd-dmd/python/cylinder.h5'
file_h5 = h5py.File(path_h5, 'r')

extracted_column = []

for key in file_h5['/Function/u']:
    dataset = file_h5['/Function/u/' + key]
    column = dataset[:, 0]
    extracted_column.append(column)

size = np.shape(extracted_column)
print("Dimensões do vetor criado:", size)
