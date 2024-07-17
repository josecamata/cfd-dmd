import os
import numpy as np
import cv2


ROOT_DATA   = '/home/camata/git/cfd-dmd/DATA'
INPUT_DATA  = ROOT_DATA + '/sediment_original_img'
OUTPUT_DATA = ROOT_DATA + '/sediment_cropped_bw'


#obter numeros de arquivos de um diretório

files = os.listdir(INPUT_DATA)
n_files = len(files)

img = cv2.imread(INPUT_DATA + f'/RenderView1_0.png')
imgCropped = img[316:477, 16:1669]
gray_image = cv2.cvtColor(imgCropped, cv2.COLOR_BGR2GRAY )
cv2.imwrite(OUTPUT_DATA + f'/RenderView1_0.png', gray_image)


# fazer o crop e salvar as imagens em escala de cinza usando cv2
file_id = 4
for i in range(1,n_files):
    img = cv2.imread(INPUT_DATA + f'/RenderView1_{file_id}.png')
    imgCropped = img[316:477, 16:1669]
    gray_image = cv2.cvtColor(imgCropped, cv2.COLOR_BGR2GRAY )
    cv2.imwrite(OUTPUT_DATA + f'/RenderView1_{i}.png', gray_image)
    file_id += 5

