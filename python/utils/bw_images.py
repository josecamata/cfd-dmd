import cv2


for i in range(0, 4000):
    img = cv2.imread(f'/home/camata/git/cfd-dmd/DATA/pngs_cropped/RenderView1_{i}.png')
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY )
    cv2.imwrite(f'/home/camata/git/cfd-dmd/DATA/pngs_bw/RenderView1_{i}.png', gray_image)
