import numpy as np
from PIL import Image

ROOT_DIR             = '/home/breno/cfd-dmd-new/cfd-dmd'
INPUT_DIR            = ROOT_DIR + '/DATA/sediment_cropped_bw'
OUTPUT_DIR_PREDICTED = ROOT_DIR + '/OUTPUT/sediment'
IMAGE = 2000

def psnr(img, ref_img):
    mse = np.mean((np.array(ref_img) - np.array(img)) ** 2)
    if mse == 0:
        return float('inf')
    range_max = 255.0
    psnr = 20 * np.log10(range_max / np.sqrt(mse))
    return psnr

img_predicted = Image.open(OUTPUT_DIR_PREDICTED + f'/imagem_predita_{str(IMAGE).zfill(5)}.png')
img_original = Image.open(INPUT_DIR + f'/RenderView1_{IMAGE}.png')

psnr_value = psnr(img_predicted, img_original)
print("PSNR:", psnr_value)
