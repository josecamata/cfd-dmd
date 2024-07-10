import os
import numpy as np
import cv2


ROOT_DIR    = '/home/camata/git/cfd-dmd'
INPUT_DATA  = ROOT_DIR + 'DATA/cylinder_original_img'
OUTPUT_DATA = ROOT_DIR + 'DATA/cylinder_cropped_bw'


#obter numeros de arquivos de um diretório

files = os.listdir(INPUT_DATA)
n_files = len(files)

# image position to crop
# position 45 x 92
# size 912 x 170

# fazer o crop e salvar as imagens em escala de cinza usando cv2
for i in range(n_files):
    print (f'Processing file {files[i]} ...')
    img = cv2.imread(INPUT_DATA + f'/'+ files[i], cv2.IMREAD_GRAYSCALE)
    imgCropped = img[92:92+170, 45:45+912]
    #gray_image = cv2.cvtColor(imgCropped, cv2.COLOR_BGR2GRAY )
    cv2.imwrite(OUTPUT_DATA + f'/'+ files[i], imgCropped)
