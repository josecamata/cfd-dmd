import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from dmd_class import DMD

INPUT_DIR = '/home/breno/cfd-dmd/pngs_bw'
INPUT_DIR_PREDICTED = '/home/breno/cfd-dmd/predicted_imgs_1000'
INTERVALO_INICIAL = 3900
INTERVALO_FINAL = 4000
N_SNAPSHOTS = INTERVALO_FINAL - INTERVALO_INICIAL

im = Image.open(INPUT_DIR + f'/RenderView1_{INTERVALO_INICIAL}.png')

arr   = np.array(im)
shape = arr.shape
size  = arr.size

X     = np.zeros((size, N_SNAPSHOTS))

for i in range(INTERVALO_INICIAL, INTERVALO_FINAL):
    im = Image.open(INPUT_DIR + f'/RenderView1_{i}.png')
    arr = np.array(im).reshape((size,1))
    X[:, i - INTERVALO_INICIAL] = arr[:, 0]
 
dmd = DMD()
dmd.fit(X)
t = np.arange(0, N_SNAPSHOTS, 1)
xDMD = dmd.predict(t)


# Xpred = np.zeros((N, T))
# Xpred[:, 0] = X[:, 0]

# xDMD = dmd.predict(0)

# for t in range(1, T):
#     Xpred[:, t] = dmd.predict(t)

for i in range(INTERVALO_INICIAL, INTERVALO_FINAL):
    arr_pred = xDMD[:, i - INTERVALO_INICIAL].real.reshape(shape)
    arr_pred = arr_pred.astype(np.uint8)

    img_pred = Image.fromarray(arr_pred)
    img_pred.save(INPUT_DIR_PREDICTED + f'/imagem_predita_{i}.png')

original_images = []

for i in range(INTERVALO_INICIAL, INTERVALO_FINAL):
    im = Image.open(INPUT_DIR + f'/RenderView1_{i}.png')
    arr = np.array(im)
    original_images.append(arr)

predicted_images = []

for i in range(INTERVALO_INICIAL, INTERVALO_FINAL):
    im = Image.open(INPUT_DIR_PREDICTED + f'/imagem_predita_{i}.png')
    arr_pred = np.array(im)
    predicted_images.append(arr_pred)

frobenius_errors = []

for i in range(INTERVALO_INICIAL, INTERVALO_FINAL):
    error = np.linalg.norm(original_images[i - INTERVALO_INICIAL] - predicted_images[i - INTERVALO_INICIAL], 'fro') / np.linalg.norm(original_images[i - INTERVALO_INICIAL])
    frobenius_errors.append(error)

plt.plot(range(INTERVALO_INICIAL, INTERVALO_FINAL), frobenius_errors, label='Erro de Frobenius')
plt.xlabel('Tempo')
plt.ylabel('Erro de Frobenius')
plt.title('Erro de Frobenius entre imagens originais e previstas')
plt.legend()
plt.show()

# dmd.plot_modes_2D(figsize=(12,5))


# Xpred[:,0] = X[:,0]
# for t in range(1,T):
#     Xpred[:,t] = dmd.predict(Xpred[:,t-1])


# # print(Xp)
# arr2 = Xpred[:,T-1].reshape(shape)
# #onvert float arrary to int array
# arr2 = arr2.astype(np.uint8)

# print(arr2.shape)
# print(arr2)

# # # make a PIL image
# img2 = Image.fromarray(arr2)
# img2.save('imagem2.png')

# Xpred[:, 0] = X[:, 0]
# for t in range(1, T):
#     Xpred[:, t] = dmd.predict(Xpred[:, t-1])

# for i in range(0, 100):
#     arr_pred = Xpred[:, i].reshape(shape)
#     arr_pred = arr_pred.astype(np.uint8)

#     img_pred = Image.fromarray(arr_pred)
#     img_pred.save('/home/breno/cfd-dmd/predicted_imgs' + f'/imagem_predita_{i}.png')