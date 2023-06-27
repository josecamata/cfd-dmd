from PIL import Image

thresh = 50
fn = lambda x : 0 if x > thresh else 255

# Open and convert img 1
img = Image.open('/home/breno/cfd-dmd/pngs/RenderView1_0.png')
imgCropped = img.crop(box=(16,316,1669,477))
imgCropped = imgCropped.convert('L')
imgCropped.save('/home/breno/cfd-dmd/pngs_bw1/RenderView1_0.png')

j = 1
for i in range(4, 20000, 5):
    img = Image.open(f'/home/breno/cfd-dmd/pngs/RenderView1_{i}.png')
    imgCropped = img.crop(box=(16,316,1669,477))
    imgCropped = imgCropped.convert('L')
    imgCropped.save(f'/home/breno/cfd-dmd/pngs_bw1/RenderView1_{j}.png')
    j += 1