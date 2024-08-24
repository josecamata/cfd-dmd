# delete the old files
rm -rf DATA
rm -f OUTPUT/cylinder/*.png
rm -f OUTPUT/cylinder/*.h5
rm -f OUTPUT/cylinder/*.xdmf
rm -f OUTPUT/sediment/*.png

tar -xf DATA.tar.xz 

# copy h5 and xdmf files to the OUTPUT folder
cp DATA/cylinder_xdmf/cylinder.h5   OUTPUT/cylinder/
cp DATA/cylinder_xdmf/cylinder.xdmf OUTPUT/cylinder/

