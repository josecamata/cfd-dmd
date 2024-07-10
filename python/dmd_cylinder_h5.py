import h5py
import numpy as np
import matplotlib.pyplot as plt
from dmd_class import DMD

ROOT_DIR = '/home/camata/git/cfd-dmd'

path_h5  = ROOT_DIR + '/DATA/cylinder_xdmf/cylinder.h5'
path_out = ROOT_DIR + '/OUTPUT/cylinder.h5'
file_h5 = h5py.File(path_h5, 'r')

extracted_column = []
total_time_points = 0
keys = []
time_interval = 0.0025
time = []
t    = 0.0

for key in file_h5['/Function/u']:
    t += time_interval
    dataset = file_h5['/Function/u/' + key]
    column = dataset[:, 0]
    if(total_time_points > 0):
        extracted_column.append(column)
        keys.append(key)
        time.append(t)
    total_time_points += 1
    if total_time_points == 51:
        break

# close file
file_h5.close()



X = np.array(extracted_column).T
print(X.shape)

dmd = DMD()
dmd.fit(X)

# total_time_points = X.shape[1]
# time_interval  = 0.0025
# t_values       = np.arange(0, total_time_points * time_interval, time_interval)
t_values = np.array(time)
xDMD = dmd.predict(t_values)

file_h5 = h5py.File(path_out, 'r+')

errors_mse = []
errors_inf = []
for i in range(xDMD.shape[1]):
    original_data          = X[:, i]
    predicted_data_at_time = xDMD[:, i]
    diff = original_data - predicted_data_at_time
    # compute mse
    mse = np.sum(diff**2) / diff.size
    # print(f'MSE at time {t_values[i]}: {mse}')
    error_diff  = np.linalg.norm(diff,np.inf)
    error_tmp   = np.linalg.norm(original_data,np.inf)
    error = error_diff
    print(f'Infty Norm Error at time {t_values[i]}: {error}')
   
    errors_mse.append(mse)
    errors_inf.append(error)
    # update h5 dataset with predicted data
    dataset = file_h5['/Function/u/' + keys[i]]
    dataset[:, 0] = predicted_data_at_time


# close file

plt.plot(t_values, errors_mse, label='MSE')
plt.xlabel('Time')
plt.ylabel('MSE')
plt.legend()
plt.show()

plt.plot(t_values, errors_inf, label='Infty Norm')
plt.xlabel('Time')
plt.ylabel('Infty Norm Error')
plt.legend()
plt.show()