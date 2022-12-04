import os
import cv2
import re
import numpy as np
from skimage.measure import compare_ssim as ssim


def updating_slicenumber(HR_directory, LR_directory, HR_slice_number, LR_images_list, HR_images_list, maxidx_ssim):
    img_LR_match = cv2.imread(os.path.join(LR_directory, LR_images_list[maxidx_ssim]), 0)

    LR_segm_idcs = np.where(img_LR_match > 0)
    x = LR_segm_idcs[0]
    y = LR_segm_idcs[1]
    img_LR_crop = img_LR_match[x.min() - 1:x.max() + 1, y.min() - 1:y.max() + 1]

    # img_LR_match = cv2.resize(img_LR_match, newdim)
    HR_images_sublist = HR_images_list[HR_slice_number - 5: HR_slice_number + 5]
    slice_numbers = np.arange(HR_slice_number - 5, HR_slice_number + 5)

    structsim_step2 = []
    for img_path_HR in HR_images_sublist:
        img_HR_path = os.path.join(HR_directory, img_path_HR)
        img_HR = cv2.imread(img_HR_path, 0)
        # img_HR_resize = cv2.resize(img_HR, (img_LR_match.shape[1], img_LR_match.shape[0]))

        HR_segm_idcs = np.where(img_HR > 0)
        x = HR_segm_idcs[0]
        y = HR_segm_idcs[1]
        img_HR_crop = img_HR[x.min() - 1:x.max() + 1, y.min() - 1:y.max() + 1]

        img_HR_crop_resize = cv2.resize(img_HR_crop, (img_LR_crop.shape[1], img_LR_crop.shape[0]))
        structsim_step2.append(ssim(img_LR_crop, img_HR_crop_resize))

    HR_slice_number_update = slice_numbers[np.argmax(structsim_step2)]

    return HR_slice_number_update


def move_around_till_match_row(row1, row2, col, HR_slice, LR_volume, row1_HR, row2_HR, col_HR, HR_volume):
    range = 10
    row1_range = np.arange(row1 - range, row1 + range)
    row2_range = np.arange(row2 - range, row2 + range)
    col_range = np.arange(col - range, col + range)

    LR_slice_test = LR_volume[row1:row2, col, :]
    HR_slice = cv2.resize(HR_slice, (LR_slice_test.shape[1], LR_slice_test.shape[0]))

    ssim_table = np.zeros((len(row1_range), len(col_range)))

    for idx_r, (r1, r2) in enumerate(zip(row1_range, row2_range)):
        for idx_c, c in enumerate(col_range):
            LR_slice = LR_volume[r1:r2, c, :]
            ssim_table[idx_r, idx_c] = ssim(LR_slice, HR_slice)

    highest_ssim = np.where(ssim_table == np.max(ssim_table))
    row1_update = row1_range[highest_ssim[0]][0]
    row2_update = row2_range[highest_ssim[0]][0]
    col_update = col_range[highest_ssim[1]][0]

    LR_match = LR_volume[row1_update:row2_update, col_update, :]

    # Do the reverse move
    row1_range = np.arange(row1_HR - range, row1_HR + range)
    row2_range = np.arange(row2_HR - range, row2_HR + range)
    col_range = np.arange(col_HR - range, col_HR + range)

    ssim_table2 = np.zeros((len(row1_range), len(col_range)))

    for idx_r, (r1, r2) in enumerate(zip(row1_range, row2_range)):
        for idx_c, c in enumerate(col_range):
            HR_slice = HR_volume[r1:r2, c, :]
            HR_slice = cv2.resize(HR_slice, (LR_match.shape[1], LR_match.shape[0]))
            ssim_table2[idx_r, idx_c] = ssim(HR_slice, LR_match)

    highest_ssim2 = np.where(ssim_table2 == np.max(ssim_table2))

    row1_HR_update = row1_range[highest_ssim2[0]][0]
    row2_HR_update = row2_range[highest_ssim2[0]][0]
    col_HR_update = col_range[highest_ssim2[1]][0]

    return row1_update, row2_update, col_update, row1_HR_update, row2_HR_update, col_HR_update


def move_around_till_match_col(row, col1, col2, HR_slice, LR_volume, row_HR, col1_HR, col2_HR, HR_volume):
    range = 10
    row_range = np.arange(row - range, row + range)
    col1_range = np.arange(col1 - range, col1 + range)
    col2_range = np.arange(col2 - range, col2 + range)


    LR_slice_test = LR_volume[row, col1:col2, :]
    HR_slice = cv2.resize(HR_slice, (LR_slice_test.shape[1], LR_slice_test.shape[0]))

    ssim_table = np.zeros((len(row_range), len(col1_range)))


    for idx_r, r in enumerate(row_range):
        for idx_c, (c1, c2) in enumerate(zip(col1_range, col2_range)):
            LR_slice = LR_volume[r, c1:c2, :]
            ssim_table[idx_r, idx_c] = ssim(LR_slice, HR_slice)

    highest_ssim = np.where(ssim_table == np.max(ssim_table))
    row_update = row_range[highest_ssim[0]][0]
    col1_update = col1_range[highest_ssim[1]][0]
    col2_update = col2_range[highest_ssim[1]][0]

    LR_match = LR_volume[row_update, col1_update:col2_update, :]

    # Do the reverse move
    row_range = np.arange(row_HR - range, row_HR + range)
    col1_range = np.arange(col1_HR - range, col1_HR + range)
    col2_range = np.arange(col2_HR - range, col2_HR + range)

    ssim_table2 = np.zeros((len(row_range), len(col1_range)))

    for idx_r, r in enumerate(row_range):
        for idx_c, (c1, c2) in enumerate(zip(col1_range, col2_range)):
            HR_slice = HR_volume[r, c1:c2, :]
            HR_slice = cv2.resize(HR_slice, (LR_match.shape[1], LR_match.shape[0]))
            ssim_table2[idx_r, idx_c] = ssim(HR_slice, LR_match)

    highest_ssim2 = np.where(ssim_table2 == np.max(ssim_table2))

    row_HR_update = row_range[highest_ssim2[0]][0]
    col1_HR_update = col1_range[highest_ssim2[0]][0]
    col2_HR_update = col2_range[highest_ssim2[1]][0]

    return row_update, col1_update, col2_update, row_HR_update, col1_HR_update, col2_HR_update


def matching_procedure(HR_volume, LR_volume, HR_begin_coordinates, HR_end_coordinates, updated_factor):
    LR_begin_coordinates = np.round(HR_begin_coordinates / updated_factor).astype(int)
    LR_end_coordinates = np.round(HR_end_coordinates / updated_factor).astype(int)

    slice_1 = HR_volume[HR_begin_coordinates[0]:HR_end_coordinates[0], HR_begin_coordinates[1], :]
    slice_2 = HR_volume[HR_begin_coordinates[0]:HR_end_coordinates[0], HR_end_coordinates[1], :]
    slice_3 = HR_volume[HR_begin_coordinates[0], HR_begin_coordinates[1]:HR_end_coordinates[1], :]
    slice_4 = HR_volume[HR_end_coordinates[0], HR_begin_coordinates[1]:HR_end_coordinates[1], :]

    row1_1, row2_1, col_1, row1_HR_1, row2_HR_1, col_HR_1 = \
        move_around_till_match_row(LR_begin_coordinates[0], LR_end_coordinates[0], LR_begin_coordinates[1],
                                   slice_1, LR_volume, HR_begin_coordinates[0], HR_end_coordinates[0],
                                   HR_begin_coordinates[1], HR_volume)
    row1_2, row2_2, col_2, row1_HR_2, row2_HR_2, col_HR_2 = \
        move_around_till_match_row(LR_begin_coordinates[0], LR_end_coordinates[0], LR_end_coordinates[1],
                                   slice_2, LR_volume, HR_begin_coordinates[0], HR_end_coordinates[0],
                                   HR_end_coordinates[1], HR_volume)

    row_3, col1_3, col2_3, row_HR_3, col1_HR_3, col2_HR_3 = \
        move_around_till_match_col(LR_begin_coordinates[0], LR_begin_coordinates[1], LR_end_coordinates[1],
                                   slice_3, LR_volume, HR_begin_coordinates[0], HR_begin_coordinates[1],
                                   HR_end_coordinates[1], HR_volume)
    row_4, col1_4, col2_4, row_HR_4, col1_HR_4, col2_HR_4 = \
        move_around_till_match_col(LR_end_coordinates[0], LR_begin_coordinates[1], LR_end_coordinates[1],
                                   slice_4, LR_volume, HR_end_coordinates[0], HR_begin_coordinates[1],
                                   HR_end_coordinates[1], HR_volume)

    # try to bring all updated values together into a nice cube...
    return row1_1, row2_1, col_1, row1_HR_1, row2_HR_1, col_HR_1,  row1_2, row2_2, col_2, \
           row1_HR_2, row2_HR_2, col_HR_2, row_3, col1_3, col2_3, row_HR_3, col1_HR_3, col2_HR_3, \
           row_4, col1_4, col2_4, row_HR_4, col1_HR_4, col2_HR_4


def find_begin_and_end_slice_segm(HR_directory, LR_directory, HR_directory_orig, LR_directory_orig):
    HR_orig_list = [i for i in os.listdir(HR_directory_orig) if i.endswith('.tif')]
    LR_orig_list = [i for i in os.listdir(LR_directory_orig) if i.endswith('.tif')]
    HR_segm_list = os.listdir(HR_directory)
    LR_segm_list = os.listdir(LR_directory)

    first_orig_HR = int((re.findall('\d+', HR_orig_list[0]))[-1])
    first_orig_LR = int((re.findall('\d+', LR_orig_list[0]))[-1])
    last_orig_HR = int((re.findall('\d+', HR_orig_list[-1]))[-1])
    last_orig_LR = int((re.findall('\d+', LR_orig_list[-1]))[-1])
    first_segm_HR = int((re.findall('\d+', HR_segm_list[0]))[-1])
    first_segm_LR = int((re.findall('\d+', LR_segm_list[0]))[-1])
    last_segm_HR = int((re.findall('\d+', HR_segm_list[-1]))[-1])
    last_segm_LR = int((re.findall('\d+', LR_segm_list[-1]))[-1])




def random_slice_add(fraction_random):

