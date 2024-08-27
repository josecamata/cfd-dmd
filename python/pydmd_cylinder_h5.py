import h5py
import numpy as np
import matplotlib.pyplot as plt
from pydmd import DMD
from pydmd import RDMD



ROOT_DIR = '/home/camata/git/cfd-dmd'

path_h5  = ROOT_DIR + '/DATA/cylinder_xdmf/cylinder.h5'
file_h5 = h5py.File(path_h5, 'r')


t    = 0.0

xy = file_h5['/Mesh/mesh/geometry']

n_points = xy.shape[0]
# get number of datasets
n_times = len(file_h5['/Function/u'])

print('Number of points:', n_points)
print('Number of datasets:', n_times)

u    = np.zeros((n_points, n_times))
time = np.zeros(n_times)

j = 0
for key in file_h5['/Function/u']:
    # replace '_' by '.'
    t_str = key.replace('_', '.')
    # convert to float
    t = float(t_str)
    dataset = file_h5['/Function/u/' + key]
    u[:,j] = dataset[:, 0]
    time[j] = t
    j+=1


time_interval = time[1] - time[0]
# close file


INTERVALO_INICIAL   = 1
INTERVALO_FINAL     = 250
N_SNAPSHOTS         = INTERVALO_FINAL - INTERVALO_INICIAL
PREDICT_INTERVAL_START = INTERVALO_FINAL
PREDICT_LEN            = 10
PREDICT_INTERVAL_END   = PREDICT_INTERVAL_START + PREDICT_LEN

X = u[:,INTERVALO_INICIAL:INTERVALO_FINAL]

print('Matriz de snapshots preenchida')
print(' Shape:', X.shape)

dmd = DMD(svd_rank=0)
dmd.fit(X)


# total_time_points = X.shape[1]
# time_interval  = 0.0025
# t_values       = np.arange(0, total_time_points * time_interval, time_interval)
t_values = time[INTERVALO_INICIAL:INTERVALO_FINAL]
print("predicted from time ", t_values[0], " to ", t_values[-1])
xDMD = dmd.reconstructed_data.real


print('DMD finalizado')
print(' Shape:', xDMD.shape)

errors_mse = []
errors_inf = []
for i in range(X.shape[1]):
    #print(f'Processing time {t_values[i]}')
    original_data          = X[:, i]
    predicted_data_at_time = xDMD[:, i]
    diff = original_data - predicted_data_at_time
    # compute mse
    mse = np.sum(diff**2) / diff.size
    # print(f'MSE at time {t_values[i]}: {mse}')
    error_diff  = np.linalg.norm(diff,np.inf)
    error_tmp   = np.linalg.norm(original_data,np.inf)
    error = error_diff
    #print(f'Infty Norm Error at time {t_values[i]}: {error}')
    errors_mse.append(mse)
    errors_inf.append(error)


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

t = xDMD.shape[1] - 1

# Plot the velocity distribution of the flow field:
x_true  = xy[:, 0]
y_true  = xy[:, 1]
u_true  = X[:, t]
u_pred  = xDMD[:, t]
fig, ax = plt.subplots(2, 1)
cntr0   = ax[0].tricontourf(x_true, y_true, u_pred, levels=80, cmap="rainbow")
cb0     = plt.colorbar(cntr0, ax=ax[0])
cntr1   = ax[1].tricontourf(x_true, y_true, u_true, levels=80, cmap="rainbow")
cb1     = plt.colorbar(cntr1, ax=ax[1])
ax[0].set_title("u-DMD " + "(t=" + str(t_values[t]) + ")", fontsize=9.5)
ax[0].axis("scaled")
ax[0].set_xlabel("X", fontsize=7.5, family="Arial")
ax[0].set_ylabel("Y", fontsize=7.5, family="Arial")
ax[1].set_title("u-Reference solution " + "(t=" + str(t_values[t]) + ")", fontsize=9.5)
ax[1].axis("scaled")
ax[1].set_xlabel("X", fontsize=7.5, family="Arial")
ax[1].set_ylabel("Y", fontsize=7.5, family="Arial")
fig.tight_layout()
plt.show()

file_h5.close()

