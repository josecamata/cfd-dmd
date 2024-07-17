import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from dmd_class import DMD

ROOT_DIR            = '/home/camata/git/cfd-dmd'
INPUT_DIR            = ROOT_DIR+'/DATA/sediment_cropped_bw'
OUTPUT_DIR_PREDICTED = ROOT_DIR+'/OUTPUT/sediment'
INTERVALO_INICIAL   = 3500
INTERVALO_FINAL     = 4000
N_SNAPSHOTS         = INTERVALO_FINAL - INTERVALO_INICIAL
PREDICT_INTERVAL_START = INTERVALO_FINAL
PREDICT_LEN            = 10
PREDICT_INTERVAL_END   = PREDICT_INTERVAL_START + PREDICT_LEN

im = Image.open(INPUT_DIR + f'/RenderView1_{INTERVALO_INICIAL}.png')

arr   = np.array(im)
shape = arr.shape
size  = arr.size

# Matriz de snapshots

X     = np.zeros((size, N_SNAPSHOTS))

# Preenchendo a matriz de snapshots
j = 0
for i in range(INTERVALO_INICIAL, INTERVALO_FINAL):
    im = Image.open(INPUT_DIR + f'/RenderView1_{i}.png')
    arr = np.array(im).reshape((size,1))
    X[:, j] = arr[:, 0]
    j += 1
 
print('Matriz de snapshots preenchida')
print(' Shape:', X.shape)
time_step =  1
# DMD
dmd = DMD()
dmd.fit(X,svd_rank=0, dt=time_step)
t = np.arange(INTERVALO_INICIAL*time_step, time_step*INTERVALO_FINAL, time_step)

xDMD = dmd.predict(t)

print('DMD finalizado')
print(' Shape:', xDMD.shape)

# Salvando as imagens previstas e
# calculando o erro de Frobenius

relative_errors = []

j = 0
for i in range(INTERVALO_INICIAL, INTERVALO_FINAL):
    arr_pred = xDMD[:, j].real.reshape(shape)
    arr_pred = arr_pred.astype(np.uint8)

    img_pred = Image.fromarray(arr_pred)
    img_pred.save(OUTPUT_DIR_PREDICTED + f'/imagem_predita_{str(i).zfill(5)}.png')

    #error = np.linalg.norm(X[:,j] - xDMD[:,j]) / np.linalg.norm(X[:,j])
    # compute mse
    diff = X[:, j] - xDMD[:, j]
    mse = np.sum(diff**2) / diff.size
    relative_errors.append(mse)
    j += 1

# plot log y scale  
plt.plot(t, relative_errors, label='MSE')
plt.xlabel('Time')
plt.ylabel('MSE')
plt.legend()
plt.show()

# compute prediction for next 10 time steps

if(PREDICT_LEN > 0  ):
    t_future = np.arange(PREDICT_INTERVAL_START*time_step, time_step*PREDICT_INTERVAL_END, time_step)
    xPredict = dmd.predict(t_future)

    for(i, t) in enumerate(t_future):
        arr_pred = xPredict[:, i].real.reshape(shape)
        arr_pred = arr_pred.astype(np.uint8)
        img_pred = Image.fromarray(arr_pred)
        img_pred.save(OUTPUT_DIR_PREDICTED + f'/imagem_predita_{str(i+PREDICT_INTERVAL_START).zfill(5)}.png')
