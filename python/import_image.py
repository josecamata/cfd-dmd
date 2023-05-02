import numpy as np
#from PIL import Image
import scipy.io as scipy
import matplotlib.pyplot as plt

im = plt.imread('imagem.png')
print(im.shape)


#img = Image.open('imagem.png').convert('RGBA')
# print(img)
# arr = np.array(img)

# record the original shape
# shape = arr.shape

# make a 1-dimensional view of arr
flat_arr = im.ravel()
print(flat_arr.shape)

# convert it to a matrix
# vector = np.matrix(flat_arr)

# do something to the vector
# vector[:,::10] = 128

# reform a numpy array of the original shape
# arr2 = np.asarray(vector).reshape(shape)

# make a PIL image
# img2 = Image.fromarray(arr2, 'RGBA')
# img2.show()

