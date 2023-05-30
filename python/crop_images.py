from PIL import Image

thresh = 50
fn = lambda x : 0 if x > thresh else 255

# Open and convert img 1
img = Image.open('../DATA/pngs/RenderView1_0.png')
imgCropped = img.crop(box=(16,316,1669,477))
imgCropped = imgCropped.convert('L')
imgCropped.save('../DATA/pngs_bw/RenderView1_0.png')

for i in range(4, 20000, 5):
    img = Image.open(f'../DATA/pngs/RenderView1_{i}.png')
    imgCropped = img.crop(box=(16,316,1669,477))
    imgCropped = imgCropped.convert('L')
    imgCropped.save(f'../DATA/pngs_bw/RenderView1_{i}.png')