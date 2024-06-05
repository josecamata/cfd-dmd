import numpy as np
from PIL import Image
from dmd_class import DMD

INPUT_DIR = '/home/breno/cfd-dmd/cilindro_tratado'
OUTPUT_PREDICTED = "/home/breno/cfd-dmd/cilindro_predicted"
INTERVALO_INICIAL = 1500
INTERVALO_FINAL = 1601
N_SNAPSHOTS = INTERVALO_FINAL - INTERVALO_INICIAL
PREDICT_INTERVAL_START = 1600
PREDICT_INTERVAL_END = 1611

im = Image.open(INPUT_DIR + f'/cilindro.{str(INTERVALO_INICIAL).zfill(4)}.png')

arr = np.array(im)
shape = arr.shape
size = arr.size

X = np.zeros((size, N_SNAPSHOTS))

for i in range(INTERVALO_INICIAL, INTERVALO_FINAL):
    im = Image.open(INPUT_DIR + f'/cilindro.{str(i).zfill(4)}.png')
    arr = np.array(im).reshape((size, 1))
    X[:, i - INTERVALO_INICIAL] = arr[:, 0]

dmd = DMD()
dmd.fit(X)
# t = np.arange(0, N_SNAPSHOTS, 1)

# TODO
t = np.arange(INTERVALO_INICIAL, INTERVALO_FINAL + 10, 1)

xDMD = dmd.predict(t)

for i in range(PREDICT_INTERVAL_START, PREDICT_INTERVAL_END):
    arr_pred = xDMD[:, i - PREDICT_INTERVAL_START].real.reshape(shape)
    #normalize
    # arr_pred = (arr_pred - np.min(arr_pred)) / (np.max(arr_pred) - np.min(arr_pred)) * 255
    # arr_pred = arr_pred/np.norm(arr_pred)
    arr_pred = arr_pred.astype(np.uint8)

    img_pred = Image.fromarray(arr_pred)
    img_pred.save(OUTPUT_PREDICTED + f'/cilindro_predito_{str(i).zfill(4)}.png')