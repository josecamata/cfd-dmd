import gdown
url = 'https://drive.google.com/uc?id=1_nUqi0e-GAHQDK33or8x_eS4-e_A0ipO'
output = 'DATA.tar.xz'
gdown.download(url, output, quiet=False)