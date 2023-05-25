from PIL import Image

# Open and convert img 1
img = Image.open('../pngs/RenderView1_0.png').convert('L')
imgCropped = img.crop(box=(16,316,1669,477))
imgCropped.save('../pngs_bw/RenderView1_0.png')

for i in range(4, 20000, 5):
    img = Image.open(f'../pngs/RenderView1_{i}.png').convert('L')
    imgCropped = img.crop(box=(16,316,1669,477))
    imgCropped.save(f'../pngs_bw/RenderView1_{i}.png')