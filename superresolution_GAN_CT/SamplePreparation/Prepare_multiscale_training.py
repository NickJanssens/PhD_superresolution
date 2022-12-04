import os
import cv2
import numpy as np
from skimage.measure import compare_ssim as ssim
from Processing.helper_functions import updating_slicenumber, matching_procedure
from random import randint

'''
Manual input
'''
HR_directory = 'D:/MicroCT/Nick/AM23_multiscale/LBM_segmentations/AM23_04micro'
LR_directory = 'D:/MicroCT/Nick/AM23_multiscale/LBM_segmentations/AM23_12micro'
HR_directory_original = 'D:/MicroCT/Nick/AM23_multiscale/AM23_4micro_alufilt/slices'
LR_directory_original = 'D:/MicroCT/Nick/AM23_multiscale/AM23_12micro/slices'
volume_dimensions = 1000
length_volume = 1200
search_dimensions = 1000
expected_factor = 3
op_path = 'D:/MicroCT/Nick/AM23_unet/TwoD_trainingsets/Factor3_04vs12'

n_slices = len(os.listdir(HR_directory))
HR_slice_number_begin = (np.round(n_slices / 2) - (length_volume / 2)).astype(int)
# Get last slice number
HR_slice_number_end = HR_slice_number_begin + length_volume - 1
op_path_HR = os.path.join(op_path, 'AM23_04micro_training_volume')
op_path_LR = os.path.join(op_path, 'AM23_12micro_training_volume')
if not os.path.exists(op_path_HR):
    os.mkdir(op_path_HR)
if not os.path.exists(op_path_LR):
    os.mkdir(op_path_LR)

'''
Pre processing
'''

# Open HR images to which LR images will be compared
img_HR_begin = cv2.imread(os.path.join(HR_directory, 'slice_%05d.tif' % HR_slice_number_begin), 0)
img_HR_end = cv2.imread(os.path.join(HR_directory, 'slice_%05d.tif' % HR_slice_number_end), 0)

# Open LR test image to determine dimensions of these images
test_LR = cv2.imread(os.path.join(LR_directory, 'slice_%05d.tif' % 1), 0)

# Reshape HR images to LR size - this will speed up SSIM steps
img_HR_begin_resize = cv2.resize(img_HR_begin, (test_LR.shape[1], test_LR.shape[0]))
img_HR_end_resize = cv2.resize(img_HR_end, (test_LR.shape[1], test_LR.shape[0]))

'''
Find matching LR slices 
'''
# Initiate empty structsim arrays
structsim_begin = []
structsim_end = []

# Get lists of image files over which will be iterated
LR_images_list = os.listdir(LR_directory)
HR_images_list = os.listdir(HR_directory)

for img_path_LR in LR_images_list:
    img_LR_path = os.path.join(LR_directory, img_path_LR)
    img_LR = cv2.imread(img_LR_path, 0)

    structsim_begin.append(ssim(img_LR, img_HR_begin_resize))
    structsim_end.append(ssim(img_LR, img_HR_end_resize))

maxidx_ssim_begin = np.argmax(structsim_begin)
maxidx_ssim_end = np.argmax(structsim_end)

HR_slice_number_begin_update = updating_slicenumber(HR_directory, LR_directory, HR_slice_number_begin,
                                                    LR_images_list, HR_images_list, maxidx_ssim_begin)
HR_slice_number_end_update = updating_slicenumber(HR_directory, LR_directory, HR_slice_number_end,
                                                  LR_images_list, HR_images_list, maxidx_ssim_end)

# Get update factor
updated_factor = (HR_slice_number_end_update - HR_slice_number_begin_update + 1) / \
                 (maxidx_ssim_end - maxidx_ssim_begin + 1)

'''
Cut volume
'''
# Load volume
HR_dimensions = (img_HR_begin.shape[0], img_HR_begin.shape[1],
                 (HR_slice_number_end_update - HR_slice_number_begin_update + 1))
LR_dimensions = (test_LR.shape[0], test_LR.shape[1],
                 (maxidx_ssim_end - maxidx_ssim_begin + 1))
HR_volume = np.zeros(HR_dimensions)
LR_volume = np.zeros(LR_dimensions)
HR_images_list_update = HR_images_list[HR_slice_number_begin_update: HR_slice_number_end_update + 1]
LR_images_list_update = LR_images_list[maxidx_ssim_begin: maxidx_ssim_end + 1]

for idx, HR_file in enumerate(HR_images_list_update):
    HR_volume[:, :, idx] = cv2.imread(os.path.join(HR_directory, HR_file), 0)

for idx, LR_file in enumerate(LR_images_list_update):
    LR_volume[:, :, idx] = cv2.imread(os.path.join(LR_directory, LR_file), 0)

# Determine starting coordinates

central_coord = np.array([round(img_HR_begin.shape[0] / 2), round(img_HR_begin.shape[1] / 2)])
half_size = round(search_dimensions / 2)
minimal_begin_coord = central_coord - half_size + 1
maximal_end_coord = central_coord + half_size
maximal_begin_coord = maximal_end_coord - volume_dimensions + 1

begin_coord = np.array([randint(minimal_begin_coord[0], maximal_begin_coord[0]),
                        randint(minimal_begin_coord[1], maximal_begin_coord[1])])
end_coord = np.array([begin_coord[0] + volume_dimensions, begin_coord[1] + volume_dimensions])

row1_1, row2_1, col_1, row1_HR_1, row2_HR_1, col_HR_1, row1_2, row2_2, col_2, \
row1_HR_2, row2_HR_2, col_HR_2, row_3, col1_3, col2_3, row_HR_3, col1_HR_3, col2_HR_3, \
row_4, col1_4, col2_4, row_HR_4, col1_HR_4, col2_HR_4 = matching_procedure(HR_volume,
                                                                           LR_volume,
                                                                           begin_coord,
                                                                           end_coord,
                                                                           updated_factor)

updated_begin_coord_LR = np.round(np.array([np.mean([row1_1, row1_2, row_3]),
                                            np.mean([col1_3, col1_4, col_1])])).astype(int)
updated_end_coord_LR = np.round(np.array([np.mean([row2_1, row2_2, row_4]),
                                          np.mean([col2_3, col2_4, col_2])])).astype(int)
updated_begin_coord_HR = np.round(np.array([np.mean([row1_HR_1, row1_HR_2, row_HR_3]),
                                            np.mean([col1_HR_3, col1_HR_4, col_HR_1])])).astype(int)
updated_end_coord_HR = np.round(np.array([np.mean([row2_HR_1, row2_HR_2, row_HR_4]),
                                          np.mean([col2_HR_3, col2_HR_4, col_HR_2])])).astype(int)

# Actually cut volumes
for slice_number in range(HR_volume.shape[2]):
    slice_HR = HR_volume[:, :, slice_number].copy()
    slice_cut = slice_HR[updated_begin_coord_HR[0]:updated_end_coord_HR[0],
                updated_begin_coord_HR[1]:updated_end_coord_HR[1]]
    slice_cut = slice_cut > 129
    slice_cut.dtype = 'uint8'
    cv2.imwrite(os.path.join(op_path_HR, 'slice_%05d.tif' % slice_number), slice_cut * 255)
for slice_number in range(LR_volume.shape[2]):
    slice_LR = LR_volume[:, :, slice_number]
    slice_cut = slice_LR[updated_begin_coord_LR[0]:updated_end_coord_LR[0],
                updated_begin_coord_LR[1]:updated_end_coord_LR[1]]
    slice_cut = slice_cut > 129
    slice_cut.dtype = 'uint8'
    cv2.imwrite(os.path.join(op_path_LR, 'slice_%05d.tif' % slice_number), slice_cut * 255)

factor_xy = (updated_end_coord_HR-updated_begin_coord_HR) / (updated_end_coord_LR-updated_begin_coord_LR)
factor_row = factor_xy[0]
factor_col = factor_xy[1]
mean_factor = np.mean([factor_xy, updated_factor])

with open(os.path.join(op_path, 'log.txt'), 'w') as f:
    f.write('begin_coord_LR')
    f.write('\n')
    np.savetxt(f, updated_begin_coord_LR, fmt = '%d', delimiter=',')
    f.write('begin_coord_HR')
    f.write('\n')
    np.savetxt(f, updated_begin_coord_HR, fmt = '%d', delimiter=',')
    f.write('end_coord_LR')
    f.write('\n')
    np.savetxt(f, updated_end_coord_LR, fmt = '%d', delimiter=',')
    f.write('end_coord_HR')
    f.write('\n')
    np.savetxt(f, updated_end_coord_HR, fmt = '%d', delimiter=',')
    f.write('Factor_z')
    f.write('\n%0.5f' % updated_factor)
    f.write('\nFactors_x(row)')
    f.write('\n%0.5f' % factor_row)
    f.write('\nFactors_y(col)')
    f.write('\n%0.5f' % factor_col)


