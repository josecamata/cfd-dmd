import numpy as np
from PIL import Image
from dmd_class import DMD
import matplotlib.pyplot as plt

ROOT_DIR              = '/home/camata/git/cfd-dmd/'
INPUT_DIR              = ROOT_DIR + 'DATA/cylinder_cropped_bw'
OUTPUT_DIR             = ROOT_DIR + 'OUTPUT/cylinder'
INTERVALO_INICIAL      = 1500
INTERVALO_FINAL        = 2000
N_SNAPSHOTS            = INTERVALO_FINAL - INTERVALO_INICIAL
PREDICT_INTERVAL_START = INTERVALO_FINAL
PREDICT_LEN            = 10
PREDICT_INTERVAL_END   = PREDICT_INTERVAL_START + PREDICT_LEN

im = Image.open(INPUT_DIR + f'/cilindro.{str(INTERVALO_INICIAL).zfill(4)}.png')

arr = np.array(im)
shape = arr.shape
size = arr.size

X = np.zeros((size, N_SNAPSHOTS))
j = 0
for i in range(INTERVALO_INICIAL, INTERVALO_FINAL):
    im = Image.open(INPUT_DIR + f'/cilindro.{str(i).zfill(4)}.png')
    arr = np.array(im).reshape((size, 1))
    X[:, j] = arr[:, 0]
    j += 1

dmd = DMD()
dmd.fit(X,svd_rank=0.1, dt=0.0025)
t = np.arange(INTERVALO_INICIAL*0.0025, 0.0025*INTERVALO_FINAL, 0.0025)

xDMD = dmd.predict(t)

mse_errors   = []
infty_errors = []

j = 0
for i in range(INTERVALO_INICIAL, INTERVALO_FINAL):
    arr_pred = xDMD[:, j].real.reshape(shape)
    arr_pred = arr_pred.astype(np.uint8)

    img_pred = Image.fromarray(arr_pred)
    img_pred.save(OUTPUT_DIR + f'/imagem_predita_{str(i).zfill(4)}.png')

    # compute mse
    diff = X[:, j] - xDMD[:, j]
    mse = np.sum(diff**2) / diff.size
    mse_errors.append(mse)
    # compute infinity norm
    infty = np.linalg.norm(diff, np.inf)
    infty_errors.append(infty)
    j += 1

# plot log y scale  
plt.plot(t, mse_errors, label='MSE')
plt.xlabel('Time')
plt.ylabel('MSE')
plt.legend()
plt.show()

plt.plot(t, infty_errors, label='Infinity Norm')
plt.xlabel('Time')
plt.ylabel('Infinity Norm')
plt.legend()
plt.show()

if(PREDICT_LEN > 0  ):
    # generate predicted images
    t_predict = np.arange(PREDICT_INTERVAL_START*0.0025, 0.0025*PREDICT_INTERVAL_END, 0.0025)
    xDMD = dmd.predict(t_predict)
    for (i, t) in enumerate(t_predict):
        arr_pred = xDMD[:, i].real.reshape(shape)
        arr_pred = arr_pred.astype(np.uint8)
        img_pred = Image.fromarray(arr_pred)
        img_pred.save(OUTPUT_DIR + f'/imagem_predita_{str(i+PREDICT_INTERVAL_START).zfill(4)}.png')

