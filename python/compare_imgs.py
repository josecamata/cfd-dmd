import cv2
import numpy as np
import math

def calculate_psnr(original_image, generated_image):
    # Carrega as imagens como matrizes
    img_original = cv2.imread(original_image, cv2.IMREAD_UNCHANGED)
    img_generated = cv2.imread(generated_image, cv2.IMREAD_UNCHANGED)

    # Verifica se as imagens têm as mesmas dimensões
    if img_original.shape != img_generated.shape:
        raise ValueError("As imagens têm dimensões diferentes.")

    # Calcula a diferença ao quadrado entre as imagens
    diff = np.subtract(img_original.astype(float), img_generated.astype(float))
    squared_diff = np.square(diff)

    # Calcula o MSE
    mse = np.mean(squared_diff)

    # Calcula o valor máximo possível para um pixel (255 para escala de cinza de 8 bits)
    max_pixel = 255

    # Calcula o PSNR
    if mse == 0:
        psnr = math.inf

    else:
        psnr = 20 * math.log10(max_pixel / math.sqrt(mse))

    return psnr

# Caminho para as 100 imagens originais e geradas
original_images = []
generated_images = []

for i in range(0, 100):
    original_images += [f"/home/breno/cfd-dmd/pngs_bw/RenderView1_{i}.png"]
    generated_images += [f"/home/breno/cfd-dmd/predicted_imgs/imagem_predita_{i}.png"]

# Loop para calcular o PSNR para cada par de imagens
for i in range(len(original_images)):
    original_image = original_images[i]
    generated_image = generated_images[i]

    psnr = calculate_psnr(original_image, generated_image)
    print(f"PSNR para as imagens {original_image} e {generated_image}: {psnr} dB")
