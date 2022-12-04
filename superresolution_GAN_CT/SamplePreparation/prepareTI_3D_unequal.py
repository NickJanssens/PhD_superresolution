#Prepare TI volumes

#Import module
import os
from Processing.createTIvolumes_3D_unequal import write_training_images

"""
USER INPUT
"""
edgelength_xy = 256
edgelength_z = 64
stride_xy = 180
stride_z = 32
imgfiles_orig = r'E:\UNet\AM23\THREEDIMENSIONAL\TESTING\MaximalSize\New\LR_slices'
imgfiles_segm = r'E:\UNet\AM23\THREEDIMENSIONAL\TESTING\MaximalSize\New\HR_slices'

main_path = os.path.split(imgfiles_orig)[0]
outputdirs = [os.path.join(main_path, 'LR_volumes_test'),
              os.path.join(main_path, 'HR_volumes_test')]

for opdir in outputdirs:
    if not os.path.isdir(opdir):
        os.mkdir(opdir)

outputnames = ['original', 'segmented']

"""
Write training volumes
"""
write_training_images(edgelength_xy, edgelength_z, stride_xy, stride_z, imgfiles_orig, imgfiles_segm,
                          outputdirs, outputnames)
