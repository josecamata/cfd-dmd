import numpy as np
from PIL import Image
from dmd_class import DMD

INPUT_DIR = '/home/breno/cfd-dmd/pngs_bw'
INPUT_DIR_PREDICTED = '/home/breno/cfd-dmd/predicted_imgs_1000'
OUTPUT_PREDICTED = "/home/breno/cfd-dmd/predicted_imgs"
INTERVALO_INICIAL = 900
INTERVALO_FINAL = 1001
N_SNAPSHOTS = INTERVALO_FINAL - INTERVALO_INICIAL
PREDICT_INTERVAL_START = 1000
PREDICT_INTERVAL_END = 1011

im = Image.open(INPUT_DIR + f'/RenderView1_{INTERVALO_INICIAL}.png')

arr = np.array(im)
shape = arr.shape
size = arr.size

X = np.zeros((size, N_SNAPSHOTS))

for i in range(INTERVALO_INICIAL, INTERVALO_FINAL):
    im = Image.open(INPUT_DIR + f'/RenderView1_{i}.png')
    arr = np.array(im).reshape((size, 1))
    X[:, i - INTERVALO_INICIAL] = arr[:, 0]

dmd = DMD()
dmd.fit(X)
t = np.arange(0, N_SNAPSHOTS, 1)
xDMD = dmd.predict(t)

for i in range(INTERVALO_FINAL, PREDICT_INTERVAL_END):
    arr_pred = xDMD[:, i - INTERVALO_FINAL].real.reshape(shape)
    arr_pred = arr_pred.astype(np.uint8)

    img_pred = Image.fromarray(arr_pred)
    img_pred.save(OUTPUT_PREDICTED + f'/imagem_predita_{i}.png')
