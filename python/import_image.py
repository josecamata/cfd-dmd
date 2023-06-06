import numpy as np
from PIL import Image
import scipy.io as scipy
import matplotlib.pyplot as plt


INPUT_DIR = '/home/camata/git/cfd-dmd/DATA/pngs_bw/'



im= Image.open(INPUT_DIR + 'RenderView1_0.png')

# Convert the image into an array
arr = np.array(im)
shape = arr.shape
print(shape)
size = arr.size
print(size)


# print(img)
arr_col = np.array(im).reshape((size,1))
print(arr_col.shape)
# record the original shape
# shape = arr.shape



arr2 = arr_col.reshape(shape)
# # make a PIL image
img2 = Image.fromarray(arr2)
img2.save('imagem2.png')
