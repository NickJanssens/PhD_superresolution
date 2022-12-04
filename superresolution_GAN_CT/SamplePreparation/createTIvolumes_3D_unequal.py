# import packages
import os
import sys
import numpy as np
import matplotlib.image as img
from pathlib import Path


# import h5py
# from scipy.misc import imresize
# from skimage.transform import rescale, resize, downscale_local_mean


def write_volume(subset, outputdir, outputname, count, edgelength_xy, edgelength_z):
    data_folder = Path(outputdir)

    if subset.shape == (edgelength_xy, edgelength_xy, edgelength_z):
        file_to_open = data_folder / ("%s_%05d.npy" % (outputname, count))  # (outputname+"_"+str(count)+".npy")
        np.save(file_to_open, subset)
        #        f = h5py.File(outputdir+"/"+outputname+"_"+str(count)+".hdf5", "w")
        #        f.create_dataset('data', data=subset, dtype="i8", compression="gzip")
        #        f.close()
        return 1
    else:
        return 0


def create_and_write_volumes(edgelength_xy, edgelength_z, stride_xy, stride_z, vol_orig, vol_segm,
                             output_dirs, output_names):
    assert (sys.getsizeof(vol_orig) ==
            sys.getsizeof(vol_segm)), 'One or more of the volumes have unequal sizes'
    count = 0
    for i in range(0, vol_orig.shape[0], stride_xy):
        for j in range(0, vol_orig.shape[1], stride_xy):
            for k in range(0, vol_orig.shape[2], stride_z):
                subset_orig = ((vol_orig[i:i + edgelength_xy, j:j + edgelength_xy, k:k + edgelength_z]) - 127.5) / 127.5
                subset_segm = vol_segm[i:i + edgelength_xy, j:j + edgelength_xy, k:k + edgelength_z]
                subset_segm = subset_segm / 255
                subset_segm[subset_segm > 0.6] = 1
                subset_segm[subset_segm <= 0.6] = -1
                t1 = write_volume(subset_orig, output_dirs[0], output_names[0], count, edgelength_xy, edgelength_z)
                # t2 = write_volume(vol_rescaled, output_dirs[1], output_names[1],count)
                t3 = write_volume(subset_segm, output_dirs[1], output_names[1], count, edgelength_xy, edgelength_z)
                if sum([t1 + t3]) == 2:
                    count += 1


def generate_volume(imgfiles):
    # Import data
    filesindir = next(os.walk(imgfiles))[2]
    nfiles = len(filesindir)
    testimage = img.imread(os.path.join(imgfiles, filesindir[0]))
    vol = np.zeros((testimage.shape[0], testimage.shape[1], nfiles))

    for n, f in enumerate(filesindir):
        image = img.imread(os.path.join(imgfiles, f))
        vol[:, :, n] = image

    return vol


# def rescaled_vol(vol_orig, downscale_factor):
#    #Rescale volume
#    shape_vol = vol_orig.shape
#    shape_resize = tuple(sz//downscale_factor for sz in shape_vol)
#    vol_resized = resize(vol_orig, 1/downscale_factor, anti_aliasing = False)
#    vol_rescaled = imresize(vol_resized, shape_vol, interp = 'nearest')
#
#    return vol_rescaled


def write_training_images(edgelength_xy, edgelength_z, stride_xy, stride_z, imgfiles_orig, imgfiles_segm,
                          outputdirs, outputnames):
    vol_orig = generate_volume(imgfiles_orig)
    vol_segm = generate_volume(imgfiles_segm)
    # vol_rescaled = rescaled_vol(vol_orig, downscale_factor)
    create_and_write_volumes(edgelength_xy, edgelength_z, stride_xy, stride_z, vol_orig,
                             vol_segm, outputdirs, outputnames)
