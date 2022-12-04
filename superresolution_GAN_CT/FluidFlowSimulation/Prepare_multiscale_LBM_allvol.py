import os
import re
import cv2
import fnmatch
import numpy as np
from skimage.measure import compare_ssim as ssim
from Processing.helper_functions import updating_slicenumber, matching_procedure
from random import randint

'''
Manual input
'''
input_path = 'D:/PhD/Multiscale_datasets/LBM_segmentations'
HR_slice_number_begin = 200
volume_dimensions = 500
search_dimensions = 1200
expected_factor = 3
op_path = 'D:/PhD/Multiscale_datasets/LBM_segmentations'

all_directories = fnmatch.filter(os.listdir(input_path), '*_*micro')
all_resolutions = []
for path in all_directories:
    s = [int(s) for s in re.findall(r'\d+', path)]
    all_resolutions.append(s[-1])

highest_resolution = min(all_resolutions)
idx_highest_resolution = np.argmin(all_resolutions)
name_highest_resolution = all_directories[idx_highest_resolution]
lower_resolutions = all_resolutions.copy()
lower_resolutions.remove(highest_resolution)
lower_res_directories = all_directories.copy()
del lower_res_directories[idx_highest_resolution]

HR_output_path = os.path.join(input_path, '%s_LBM_volume' % name_highest_resolution)
if not os.path.exists(HR_output_path):
    os.mkdir(HR_output_path)

'''
Pre processing
'''
# Get last slice number
HR_slice_number_end = HR_slice_number_begin + volume_dimensions
# Open HR images to which LR images will be compared
HR_directory = os.path.join(input_path, name_highest_resolution)
img_HR_begin = cv2.imread(os.path.join(HR_directory, 'slice_%05d.tif' % HR_slice_number_begin), 0)
img_HR_end = cv2.imread(os.path.join(HR_directory, 'slice_%05d.tif' % HR_slice_number_end), 0)

# Determine starting and ending coordinates of high resolution volume. Should be the same for all resolutions...
# otherwise, no comparison is possible.
central_coord = np.array([round(img_HR_begin.shape[0] / 2), round(img_HR_begin.shape[1] / 2)])
half_size = round(search_dimensions / 2)
minimal_begin_coord = central_coord - half_size + 1
maximal_end_coord = central_coord + half_size
maximal_begin_coord = maximal_end_coord - volume_dimensions + 1
begin_coord = np.array([randint(minimal_begin_coord[0], maximal_begin_coord[0]),
                        randint(minimal_begin_coord[1], maximal_begin_coord[1])])
end_coord = np.array([begin_coord[0] + volume_dimensions, begin_coord[1] + volume_dimensions])

final_end_coord_LR = []
final_begin_coord_LR = []
final_end_coord_HR = []
final_begin_coord_HR = []
factors = []

for res, LR_path in zip(lower_resolutions, lower_res_directories):
    LR_output_path = os.path.join(input_path, '%s_LBM_volume' % LR_path)
    HR_matching_LR_op_path = os.path.join(input_path, '%s_LBM_volume_matching_%s' %
                                          (name_highest_resolution, LR_path))
    LR_directory = os.path.join(input_path, LR_path)

    if not os.path.exists(LR_output_path):
        os.mkdir(LR_output_path)
    if not os.path.exists(HR_matching_LR_op_path):
        os.mkdir(HR_matching_LR_op_path)

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
    factors.append(updated_factor)
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

    final_begin_coord_LR.append(updated_begin_coord_LR)
    final_begin_coord_HR.append(updated_begin_coord_HR)
    final_end_coord_LR.append(updated_end_coord_LR)
    final_end_coord_HR.append(updated_end_coord_HR)

    # Actually cut volumes
    for slice_number in range(HR_volume.shape[2]):
        slice_HR = HR_volume[:, :, slice_number].copy()
        slice_cut = slice_HR[updated_begin_coord_HR[0]:updated_end_coord_HR[0],
                    updated_begin_coord_HR[1]:updated_end_coord_HR[1]]
        slice_cut = slice_cut > 129
        slice_cut.dtype = 'uint8'
        cv2.imwrite(os.path.join(HR_matching_LR_op_path, 'slice_%05d.tif' % slice_number), slice_cut * 255)
    for slice_number in range(LR_volume.shape[2]):
        slice_LR = LR_volume[:, :, slice_number]
        slice_cut = slice_LR[updated_begin_coord_LR[0]:updated_end_coord_LR[0],
                    updated_begin_coord_LR[1]:updated_end_coord_LR[1]]
        slice_cut = slice_cut > 129
        slice_cut.dtype = 'uint8'
        cv2.imwrite(os.path.join(LR_output_path, 'slice_%05d.tif' % slice_number), slice_cut * 255)

final_begin_coord_LR_copy = final_begin_coord_LR.copy()
final_begin_coord_HR_copy = final_begin_coord_HR.copy()
final_end_coord_LR_copy = final_end_coord_LR.copy()
final_end_coord_HR_copy = final_end_coord_HR.copy()
final_begin_coord_LR = np.round(np.mean(np.asarray(final_begin_coord_LR), axis=0)).astype(int)
final_begin_coord_HR = np.round(np.mean(np.asarray(final_begin_coord_HR), axis=0)).astype(int)
final_end_coord_LR = np.round(np.mean(np.asarray(final_end_coord_LR), axis=0)).astype(int)
final_end_coord_HR = np.round(np.mean(np.asarray(final_end_coord_HR), axis=0)).astype(int)

for slice_number in range(HR_volume.shape[2]):
    slice_HR = HR_volume[:, :, slice_number].copy()
    slice_cut = slice_HR[final_begin_coord_HR[0]:final_end_coord_HR[0],
                final_begin_coord_HR[1]:final_end_coord_HR[1]]
    slice_cut = slice_cut > 129
    slice_cut.dtype = 'uint8'
    cv2.imwrite(os.path.join(HR_output_path, 'slice_%05d.tif' % slice_number), slice_cut * 255)

factors_xy = []
for (begin_LR, end_LR, begin_HR, end_HR) in zip(final_begin_coord_LR_copy, final_end_coord_LR_copy,
                                                final_begin_coord_HR_copy, final_end_coord_HR_copy):
    factors_xy.append(np.mean((end_HR - begin_HR) / (end_LR - begin_LR)))

'''
Write log file
'''
with open(os.path.join(input_path, 'log.txt'), 'w') as f:
    f.write('final_begin_coord_LR')
    f.write('\n')
    np.savetxt(f, final_begin_coord_LR, fmt = '%d', delimiter=',')
    f.write('final_begin_coord_HR')
    f.write('\n')
    np.savetxt(f, final_begin_coord_HR, fmt = '%d', delimiter=',')
    f.write('final_end_coord_LR')
    f.write('\n')
    np.savetxt(f, final_end_coord_LR, fmt = '%d', delimiter=',')
    f.write('final_end_coord_HR')
    f.write('\n')
    np.savetxt(f, final_end_coord_HR, fmt = '%d', delimiter=',')
    f.write('Factors_z')
    f.write('\n')
    np.savetxt(f, factors, fmt = '%0.5f', delimiter=',')
    f.write('Factors_xy')
    f.write('\n')
    np.savetxt(f, factors_xy, fmt='%0.5f', delimiter=',')

