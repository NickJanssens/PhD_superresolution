import os
import cv2
import random
import numpy as np
from shutil import copyfile


def prepocessing_trainingset(input_path_HR, input_path_LR, input_path, random_val):
    output_selected_LR = os.path.join(input_path, 'LR_slices_selected')
    output_not_selected_LR = os.path.join(input_path, 'LR_slices_not_selected')
    output_selected_HR = os.path.join(input_path, 'HR_slices_selected')
    output_not_selected_HR = os.path.join(input_path, 'HR_slices_not_selected')
    if not os.path.exists(output_selected_LR):
        os.mkdir(output_selected_LR)
    if not os.path.exists(output_not_selected_LR):
        os.mkdir(output_not_selected_LR)
    if not os.path.exists(output_selected_HR):
        os.mkdir(output_selected_HR)
    if not os.path.exists(output_not_selected_HR):
        os.mkdir(output_not_selected_HR)

    image_list_LR = os.listdir(input_path_LR)
    image_list_HR = os.listdir(input_path_HR)

    test_image = cv2.imread(os.path.join(input_path_HR, image_list_HR[0]), 0)
    total_pixels = test_image.shape[0] * test_image.shape[1]
    variances = []
    means = []
    porosity = []
    for (slice_path_HR, slice_path_LR) in zip(image_list_HR, image_list_LR):
        img = cv2.imread(os.path.join(input_path_HR, slice_path_HR), 0)
        # Segment image to boolean
        img = img > 5
        # Determine porosity, based on pore vs total pixels
        n_white_pixels = np.sum(img)
        poro = 1 - (n_white_pixels / total_pixels)
        porosity.append(poro)

        # variances.append(ndimage.variance(img))
        # means.append(ndimage.mean(img))
        if poro > 0.001:
            copyfile(os.path.join(input_path_LR, slice_path_LR),
                     os.path.join(output_selected_LR, slice_path_LR))
            copyfile(os.path.join(input_path_HR, slice_path_HR),
                     os.path.join(output_selected_HR, slice_path_HR))
        else:
            random_float = random.random()
            if random_float < random_val:
                copyfile(os.path.join(input_path_LR, slice_path_LR),
                         os.path.join(output_selected_LR, slice_path_LR))
                copyfile(os.path.join(input_path_HR, slice_path_HR),
                         os.path.join(output_selected_HR, slice_path_HR))
            else:
                copyfile(os.path.join(input_path_LR, slice_path_LR),
                         os.path.join(output_not_selected_LR, slice_path_LR))
                copyfile(os.path.join(input_path_HR, slice_path_HR),
                         os.path.join(output_not_selected_HR, slice_path_HR))

    # mn_var = np.mean(variances)
    # std_var = np.std(variances)
    # cut_off = mn_var-std_var
    #
    # for var, slice_path in zip(variances, image_list):
    #     if var > cut_off:
    #         copyfile(os.path.join(input_directory, slice_path),
    #                  os.path.join(output_selected, slice_path))
    #     else:
    #         copyfile(os.path.join(input_directory, slice_path),
    #                  os.path.join(output_not_selected, slice_path))


def test_suitable_trainingImage(HR_image, random_val):
    total_pixels = HR_image.shape[0] * HR_image.shape[1]
    HR_image = HR_image > 0.1
    # Determine porosity, based on pore vs total pixels
    n_white_pixels = np.sum(HR_image)
    poro = 1 - (n_white_pixels / total_pixels)

    # variances.append(ndimage.variance(img))
    # means.append(ndimage.mean(img))
    if poro > 0.001:
        return True
    else:
        random_float = random.random()
        if random_float < random_val:
            return True
        else:
            return False
