import os
import cv2
import numpy as np
from Processing.Models_test_3D import generator_unet_real

input_path = r'F:\UNet\AM23\THREEDIMENSIONAL\TESTING\New\HR_volumes'
input_LR_path = r'F:\UNet\AM23\THREEDIMENSIONAL\TESTING\New\LR_volumes'
result_path = r'F:\UNet\AM23\THREEDIMENSIONAL\TESTING\New\generated_volumes'
mean_values_input = []
mean_values_result = []
for vol_path, res_path in zip(os.listdir(input_path), os.listdir(result_path)):
    volume = np.load(os.path.join(input_path, vol_path))
    volume_res = np.load(os.path.join(result_path, res_path))
    mean_values_input.append(volume.mean())
    mean_values_result.append(volume_res.mean())

mean_values_input = np.asarray(mean_values_input)
mean_values_result = np.asarray(mean_values_result)

delta_mn = np.abs(mean_values_result - mean_values_input)
sorted_id = np.argsort(-delta_mn)


vol_inp = np.load(os.path.join(input_LR_path, 'original_02245.npy'))
vol_res = np.load(os.path.join(result_path, 'result_02245.npy'))




generator_path = r'F:\UNet\AM23\THREEDIMENSIONAL\output_gan_3D\weights\81\generator_1000_1.h5'

input_slice_path_lr = r'F:\UNet\AM23\THREEDIMENSIONAL\TESTING\New\LR_slices'
lr_slice_files = os.listdir(input_slice_path_lr)
vol = np.zeros((1200,1200,64))
for idx in np.arange(64):
    vol[:,:,idx] = cv2.imread(os.path.join(input_slice_path_lr, lr_slice_files[idx]),0)

vol = (vol-127.5)/127.5


testvol = vol[0:256, 0:256, 0:64]
testvol = np.expand_dims(testvol, axis=0)
testvol = np.expand_dims(testvol, axis=4)
unet = generator_unet_real((256, 256, 64, 1))
unet.load_weights(generator_path)
ttt = unet.predict(testvol)

