
import numpy as np
from PIL import Image
import scipy.io 

import matplotlib.pyplot as plt
#from dmd_class import DMD

from pydmd import DMD
INPUT_DIR = '/home/camata/git/cfd-dmd/DATA/pngs_bw/'

im= Image.open(INPUT_DIR + f'RenderView1_0.png')
arr = np.array(im)
shape = arr.shape
print(shape)
size = arr.size

X = np.zeros((size,100))

for i in range(0,100,1):
    # print("i: %d\n" % i)
    im= Image.open(INPUT_DIR + f'RenderView1_{i}.png')
    # Convert the image into an array

    arr = np.array(im).reshape((size,1))
    # print(arr)
    X[:,i] = arr[:,0]


 
dmd = DMD(svd_rank=2)
dmd.fit(X)
N,T = X.shape

Xp = X[:,0]
for t in range(T+2):
    Xn = dmd.predict(Xp)
    Xp = Xn



# print(Xp)
arr2 = Xp.reshape(shape)
#onvert float arrary to int array
arr2 = arr2.astype(np.uint8)

print(arr2.shape)
print(arr2)

# # make a PIL image
img2 = Image.fromarray(arr2)
img2.save('imagem2.png')
