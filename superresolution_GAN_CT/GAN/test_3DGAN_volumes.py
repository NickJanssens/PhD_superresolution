import os
import tqdm
import cv2
import numpy as np
from Processing.Models_test_3D import generator_unet_real

generator_path = r'E:\UNet\AM23\THREEDIMENSIONAL\output_gan_3D\weights\81\generator_1000_1.h5'
volumes_input_path = r'E:\UNet\AM23\THREEDIMENSIONAL\TESTING\MaximalSize\AlreadySeen\LR_volumes_test'
dim_xy = 256
dim_z = 64
stride_xy = 180
stride_z = 32
dim_xy_final = 1200
n_channels = 1
input_shape = (dim_xy, dim_xy, dim_z, n_channels)

main_path = os.path.split(volumes_input_path)[0]
output_path = os.path.join(main_path, 'generated_volumes')
if not os.path.exists(output_path):
    os.mkdir(output_path)

output_path_slices = os.path.join(main_path, 'generated_slices')
if not os.path.exists(output_path_slices):
    os.mkdir(output_path_slices)

dim_z_final = np.int(dim_z + stride_z * (
        len(os.listdir(volumes_input_path)) / ((np.floor((dim_xy_final - dim_xy) / stride_xy) + 1) ** 2) - 1))
output_shape = (dim_xy_final, dim_xy_final, dim_z_final)


def simulate(generator, input_path, output_path):
    for idx, volume_path in tqdm.tqdm(enumerate(os.listdir(input_path))):
        volume = np.load(os.path.join(volumes_input_path, volume_path))
        volume = np.expand_dims(volume, axis=0)
        volume = np.expand_dims(volume, axis=4)
        hr_volume = generator.predict(volume)
        hr_volume = hr_volume[0, :, :, :, 0]
        np.save(os.path.join(output_path, 'result_%05d.npy' % idx), hr_volume)


def create_output_slices_unequal(input_path_volumes, output_path_slices, volume_dim,  edgelength_xy, edgelength_z, stride_xy, stride_z):
    div_volume = np.zeros(volume_dim)
    result_volume = np.zeros(volume_dim)
    add_ones = np.ones((edgelength_xy, edgelength_xy, edgelength_z))
    result_volumes_list = os.listdir(input_path_volumes)

    id = 0
    for i in range(0, volume_dim[0], stride_xy):
        for j in range(0, volume_dim[1], stride_xy):
            for k in range(0, volume_dim[2], stride_z):
                endi, endj, endk = i + edgelength_xy, j + edgelength_xy, k + edgelength_z

                if endi <= volume_dim[0] and endj <= volume_dim[1] and endk <= volume_dim[2]:
                    result = np.load(os.path.join(input_path_volumes, result_volumes_list[id]))
                    result_volume[i:endi, j:endj, k:endk] = result_volume[i:endi, j:endj, k:endk] + result
                    div_volume[i:endi, j:endj, k:endk] = div_volume[i:endi, j:endj, k:endk] + add_ones
                    id += 1
    result_volume = np.divide(result_volume, div_volume)

    x = int(np.floor((volume_dim[0] - edgelength_xy) / stride_xy) * stride_xy + edgelength_xy)
    y = int(np.floor((volume_dim[1] - edgelength_xy) / stride_xy) * stride_xy + edgelength_xy)
    z = int(np.floor((volume_dim[2] - edgelength_z) / stride_z) * stride_z + edgelength_z)
    result_volume = result_volume[0:x, 0:y, 0:z]

    for idx in np.arange(volume_dim[2]):
        result_slice = result_volume[:, :, idx]
        result_slice_new = np.zeros((result_slice.shape[0], result_slice.shape[1]))
        result_slice_new[result_slice > 0] = 1
        # result_slice_new[result_slice <= 0.6] = 0.5
        result_slice_new[result_slice <= 0] = 0
        result_slice_new = result_slice_new * 255
        # result_slice = ((result_slice > 0).astype('uint8'))*255
        cv2.imwrite(os.path.join(output_path_slices, 'slice_%05d.tif' % idx), result_slice_new)


generator = generator_unet_real(input_shape)
generator.load_weights(generator_path)

simulate(generator, volumes_input_path, output_path)
create_output_slices_unequal(output_path, output_path_slices, output_shape, dim_xy, dim_z, stride_xy, stride_z)
