# import packages
import os
import sys
import numpy as np
import matplotlib.image as img
import matplotlib.pyplot as plt
import scipy.misc as sc
from pathlib import Path
from Processing.PreprocessingTrainingSet import prepocessing_trainingset
from Processing.PreprocessingTrainingSet import test_suitable_trainingImage


def create_and_write_slices(stride, edgelength, presentdir, LR_path, HR_path, LR_files, HR_files):
    count = 0
    presentdir_LR = os.path.join(presentdir, 'LR_slices')
    presentdir_HR = os.path.join(presentdir, 'HR_slices')
    for LRfile, HRfile in zip(LR_files, HR_files):
        LR_slice = img.imread(os.path.join(presentdir_LR, LRfile))
        HR_slice = img.imread(os.path.join(presentdir_HR, HRfile))

        assert (LR_slice.shape == HR_slice.shape)

        for i in range(0, LR_slice.shape[0], stride):
            for j in range(0, LR_slice.shape[1], stride):
                subset_LR = LR_slice[i:i + edgelength, j:j + edgelength]
                subset_HR = HR_slice[i:i + edgelength, j:j + edgelength]
                subset_LR = subset_LR / 255
                subset_HR = subset_HR / 65535
                #subset_HR[subset_HR > 0.6] = 1
                #subset_HR[subset_HR <= 0.6] = 0

                if subset_LR.shape == (edgelength, edgelength):
                    im_LR = sc.toimage(subset_LR, cmin=0, cmax=1)
                    im_LR.save(os.path.join(LR_path, "LR_%05d.tif" % (count)))
                    im_HR = sc.toimage(subset_HR, cmin=0, cmax=1)
                    im_HR.save(os.path.join(HR_path, "HR_%05d.tif" % (count)))
                    count += 1



def create_and_write_slices_all(stride, edgelength, LR_path, HR_path, present_path, LR_files, HR_files):
    count = 0
    random_val = 0.2
    index_accept = 0
    index_no_accept = 0

    output_selected_LR = os.path.join(present_path, 'LR_slices_selected')
    output_not_selected_LR = os.path.join(present_path, 'LR_slices_not_selected')
    output_selected_HR = os.path.join(present_path, 'HR_slices_selected')
    output_not_selected_HR = os.path.join(present_path, 'HR_slices_not_selected')
    if not os.path.exists(output_selected_LR):
        os.mkdir(output_selected_LR)
    if not os.path.exists(output_not_selected_LR):
        os.mkdir(output_not_selected_LR)
    if not os.path.exists(output_selected_HR):
        os.mkdir(output_selected_HR)
    if not os.path.exists(output_not_selected_HR):
        os.mkdir(output_not_selected_HR)


    for LRfile, HRfile in zip(LR_files, HR_files):
        LR_slice = img.imread(os.path.join(LR_path, LRfile))
        HR_slice = img.imread(os.path.join(HR_path, HRfile))

        assert (LR_slice.shape == HR_slice.shape)

        for i in range(0, LR_slice.shape[0], stride):
            for j in range(0, LR_slice.shape[1], stride):
                subset_LR = LR_slice[i:i + edgelength, j:j + edgelength]
                subset_HR = HR_slice[i:i + edgelength, j:j + edgelength]
                subset_LR = subset_LR / 255
                subset_HR = subset_HR / 255
                test_HR = subset_HR.copy()
                subset_HR[subset_HR > 0.6] = 1
                subset_HR[subset_HR <= 0.6] = 0

                suitable_TI = test_suitable_trainingImage(test_HR, random_val)

                if subset_LR.shape == (edgelength, edgelength):
                    if suitable_TI:
                        im_LR = sc.toimage(subset_LR, cmin=0, cmax=1)
                        im_LR.save(os.path.join(output_selected_LR, "LR_%05d.tif" % (index_accept)))
                        im_HR = sc.toimage(subset_HR, cmin=0, cmax=1)
                        im_HR.save(os.path.join(output_selected_HR, "HR_%05d.tif" % (index_accept)))
                        index_accept += 1
                    else:
                        im_LR = sc.toimage(subset_LR, cmin=0, cmax=1)
                        im_LR.save(os.path.join(output_not_selected_LR, "LR_%05d.tif" % (index_no_accept)))
                        im_HR = sc.toimage(subset_HR, cmin=0, cmax=1)
                        im_HR.save(os.path.join(output_not_selected_HR, "HR_%05d.tif" % (index_no_accept)))
                        index_no_accept += 1


# def rescaled_vol(vol_orig, downscale_factor):
#    #Rescale volume
#    shape_vol = vol_orig.shape
#    shape_resize = tuple(sz//downscale_factor for sz in shape_vol)
#    vol_resized = resize(vol_orig, 1/downscale_factor, anti_aliasing = False)
#    vol_rescaled = imresize(vol_resized, shape_vol, interp = 'nearest')
#
#    return vol_rescaled


def write_training_images(stride, edgelength, presentdir, traintest_ratio, slice_freq):
    #   Get list of folder content - tif files
    LR_path = os.path.join(presentdir, 'LR_slices')
    HR_path = os.path.join(presentdir, 'HR_slices')
    imagefiles_LR = os.listdir(LR_path)
    imagefiles_HR = os.listdir(HR_path)

    #prepocessing_trainingset(LR_path, HR_path, presentdir, 0.2)

    #   Separate into training and testing folders, make these directories IF they do not exist
    LRtrain_path = os.path.join(presentdir, 'LR_train')
    HRtrain_path = os.path.join(presentdir, 'HR_train')
    LRtest_path = os.path.join(presentdir, 'LR_test')
    HRtest_path = os.path.join(presentdir, 'HR_test')

    if not os.path.exists(LRtrain_path):
        os.makedirs(LRtrain_path)
        os.makedirs(os.path.join(LRtrain_path, 'LR_train'))
    if not os.path.exists(HRtrain_path):
        os.makedirs(HRtrain_path)
        os.makedirs(os.path.join(HRtrain_path, 'HR_train'))
    if not os.path.exists(LRtest_path):
        os.makedirs(LRtest_path)
        os.makedirs(os.path.join(LRtest_path, 'LR_test'))
    if not os.path.exists(HRtest_path):
        os.makedirs(HRtest_path)
        os.makedirs(os.path.join(HRtest_path, 'HR_test'))

    LRtrain_path = os.path.join(LRtrain_path, 'LR_train')
    HRtrain_path = os.path.join(HRtrain_path, 'HR_train')
    LRtest_path = os.path.join(LRtest_path, 'LR_test')
    HRtest_path = os.path.join(HRtest_path, 'HR_test')

    #   Create subslices and write them to folders - do separately for training and testing
    nLR_slices = len(imagefiles_LR)
    nHR_slices = len(imagefiles_HR)
    assert (nLR_slices == nHR_slices)

    ntraining_slices = int(round(nHR_slices * traintest_ratio))
    ntesting_slice = nHR_slices - ntraining_slices

    LRtrain_files = [imagefiles_LR[i] for i in np.arange(0, ntraining_slices, slice_freq)]
    LRtest_files = [imagefiles_LR[i] for i in np.arange(ntraining_slices, nLR_slices, slice_freq)]
    HRtrain_files = [imagefiles_HR[i] for i in np.arange(0, ntraining_slices, slice_freq)]
    HRtest_files = [imagefiles_HR[i] for i in np.arange(ntraining_slices, nLR_slices, slice_freq)]

    create_and_write_slices(stride, edgelength, presentdir, LRtrain_path, HRtrain_path, LRtrain_files, HRtrain_files)
    create_and_write_slices(stride, edgelength, presentdir, LRtest_path, HRtest_path, LRtest_files, HRtest_files)
