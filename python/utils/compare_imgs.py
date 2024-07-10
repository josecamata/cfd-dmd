from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import math

def calculate_psnr(original_image, generated_image):
    img_original = Image.open(original_image)
    img_generated = Image.open(generated_image)

    if img_original.size != img_generated.size:
        raise ValueError("As imagens têm dimensões diferentes.")

    arr_original = np.array(img_original, dtype=float)
    arr_generated = np.array(img_generated, dtype=float)

    diff = arr_original - arr_generated
    squared_diff = np.square(diff)

    mse = np.mean(squared_diff)
    max_pixel = 255

    if mse == 0:
        psnr = math.inf
    else:
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))

    return psnr

original_images = []
generated_images = []
psnr_values = []

for i in range(0, 100):
    original_images += [f"/home/breno/cfd-dmd/pngs_bw/RenderView1_{i}.png"]
    generated_images += [f"/home/breno/cfd-dmd/predicted_imgs/imagem_predita_{i}.png"]

for i in range(len(original_images)):
    original_image = original_images[i]
    generated_image = generated_images[i]

    psnr = calculate_psnr(original_image, generated_image)
    psnr_values.append(psnr)

image_numbers = range(1, len(original_images) + 1)
plt.plot(image_numbers, psnr_values)
plt.xlabel('Número da Imagem')
plt.ylabel('PSNR (dB)')
plt.title('PSNR em função da imagem')
plt.show()