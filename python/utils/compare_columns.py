import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

EXTRACT_COLUMN_PERCENTAGE = (1/8)

INPUT_ORIGIN = "/home/breno/cfd-dmd/pngs_bw"
INPUT_PREDICTED = "/home/breno/cfd-dmd/predicted_imgs_1000"

def load_image(nome_arquivo):
    imagem = Image.open(nome_arquivo)
    return np.array(imagem)

img = 1000

original_image = load_image(INPUT_ORIGIN + f"/RenderView1_{img}.png")
predicted_image = load_image(INPUT_PREDICTED + f"/imagem_predita_{img}.png")

NUM_COLUMNS = original_image.shape[1]

extracted_column_index = int(NUM_COLUMNS * EXTRACT_COLUMN_PERCENTAGE)

origin_column = original_image[:, extracted_column_index]
predicted_column = predicted_image[:, extracted_column_index]

x = np.arange(len(origin_column))

plt.plot(x, origin_column, label='Imagem original')
plt.plot(x, predicted_column, label='Imagem prevista')
plt.xlabel('Posição na coluna')
plt.ylabel('Valor do pixel')
plt.title('Comparação de uma coluna em duas imagens: original e prevista')
plt.legend()
plt.show()