from PIL import Image


thresh = 50
fn = lambda x : 0 if x > thresh else 255

# Open and convert img 1
img = Image.open('/home/camata/git/cfd-dmd/DATA/pngs/RenderView1_0.png')
imgCropped = img.crop(box=(16,316,1669,477))
imgCropped.save('/home/camata/git/cfd-dmd/DATA/pngs_bw2/RenderView1_0.png')

j = 1
for i in range(4, 20000, 5):
    img = Image.open(f'/home/camata/git/cfd-dmd/DATA/pngs/RenderView1_{i}.png')
    imgCropped = img.crop(box=(16,316,1669,477))
    imgCropped.save(f'/home/camata/git/cfd-dmd/DATA/pngs_bw1/RenderView1_{j}.png')
    j += 1