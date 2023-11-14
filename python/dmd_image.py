import numpy as np
from PIL import Image
import scipy.io 
import matplotlib.pyplot as plt
from dmd_class import DMD
# from pydmd import DMD

INPUT_DIR = '/home/breno/cfd-dmd/pngs_bw'
INPUT_DIR_PREDICTED = '/home/breno/cfd-dmd/predicted_imgs_1000'
N_SNAPSHOTS = 1000

# im = Image.open(INPUT_DIR + f'/RenderView1_0.png')

# arr   = np.array(im)
# shape = arr.shape
# size  = arr.size

# print(shape)

# X     = np.zeros((size, N_SNAPSHOTS))

# for i in range(0, N_SNAPSHOTS, 1):
#     im = Image.open(INPUT_DIR + f'/RenderView1_{i}.png')
#     arr = np.array(im).reshape((size,1))
#     X[:,i] = arr[:,0]
 
# dmd = DMD()
# dmd.fit(X)
# t = np.arange(0, N_SNAPSHOTS, 1)
# xDMD = dmd.predict(t)


# # Xpred = np.zeros((N, T))
# # Xpred[:, 0] = X[:, 0]

# # xDMD = dmd.predict(0)

# # for t in range(1, T):
# #     Xpred[:, t] = dmd.predict(t)

# for i in range(0, N_SNAPSHOTS):
#     arr_pred = xDMD[:, i].real.reshape(shape)
#     arr_pred = arr_pred.astype(np.uint8)

#     img_pred = Image.fromarray(arr_pred)
#     img_pred.save('/home/breno/cfd-dmd/predicted_imgs_1000' + f'/imagem_predita_{i}.png')

# Calculo do erro de Frobenius
original_images = []
for i in range(N_SNAPSHOTS):
    im = Image.open(INPUT_DIR + f'/RenderView1_{i}.png')
    arr = np.array(im)
    original_images.append(arr)

# Calcular imagens previstas pela DMD
predicted_images = []
for i in range(N_SNAPSHOTS):
    im = Image.open(INPUT_DIR_PREDICTED + f'/imagem_predita_{i}.png')
    arr_pred = np.array(im)
    predicted_images.append(arr_pred)

# Calcular o erro de Frobenius
frobenius_errors = []
for i in range(N_SNAPSHOTS):
    error = np.linalg.norm(original_images[i] - predicted_images[i], 'fro')
    frobenius_errors.append(error)

# Plotar o erro de Frobenius ao longo do tempo
plt.plot(range(N_SNAPSHOTS), frobenius_errors, label='Erro de Frobenius')
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