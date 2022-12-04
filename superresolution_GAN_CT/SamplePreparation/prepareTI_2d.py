# Prepare TI volumes

# Import module
from Processing.createTIslices import write_training_images

"""
USER INPUT - folders in presentdir should be called "HR_slices" and "LR_slice"
"""
stride = 128
edgelength = 128
traintest_ratio = 0.8
slice_freq = 20
presentdir = r'C:\Users\u0105452\PhD\NeuralNetworks\UNet\XRE\test_ct_to_ct' #'F:/MicroCT/Nick/AM23_multiscale/Training_unet/COMPARING'
"""
Write training volumes
"""
write_training_images(stride, edgelength, presentdir, traintest_ratio, slice_freq)
