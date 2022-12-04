import os
import shutil
import numpy as np
import openpnm as op

dimension_volume_xy = 1000
dimension_volume_z = 160
newdimension = 100
resolution = 4.0
n_volumes = 100

datafile_name = {'LR_original.raw', 'HR_original.raw', 'HR_simulation.raw'}
pn_directory = r'C:\Users\u0105452\PhD\PNM_extraction\pnextract-master'
path_exe = r'.\bin\pnextract.exe'
substrings = ('link1.dat', 'link2.dat', 'node1.dat', 'node2.dat', 'VElems.mhd', 'VElems.tif')

output_folder = os.path.join(pn_directory, 'subvolumes_inputfiles')
if not os.path.isdir(output_folder):
    os.mkdir(output_folder)
output_folders = []
output_folders_pnm = []
for df in datafile_name:
    new_op_folder = os.path.join(output_folder, df.split('.')[0])
    output_folders.append(new_op_folder)
    if not os.path.isdir(new_op_folder):
        os.mkdir(new_op_folder)
    output_folder_pnm = os.path.join(new_op_folder, 'pnm')
    if not os.path.isdir(output_folder_pnm):
        os.mkdir(output_folder_pnm)
    output_folders_pnm.append(output_folder_pnm)

def createRandomVolume(dimension_volume_xy, dimension_volume_z, newdimension, resolution, datafile_name,
                       index, startx, starty, startz):
    stopx, stopy, stopz = startx + newdimension, starty + newdimension, startz + newdimension

    with open(r'C:\Users\u0105452\PhD\PNM_extraction\pnextract-master\doc\Image.mhd', 'r') as file:
        filedata = file.readlines()

    filedata[4] = filedata[4].replace('1000', '%d' % dimension_volume_z, )
    filedata[4] = filedata[4].replace('1000', '%d' % dimension_volume_xy, 2)
    filedata[5] = filedata[5].replace('1.6', '%0.1f' % resolution)
    filedata[8] = filedata[8].replace('Image.raw.gz', '%s' % datafile_name)
    filedata[15] = filedata[15].replace('0', '%d' % startx, 1)
    filedata[15] = filedata[15].replace('0', '%d' % starty, 1)
    filedata[15] = filedata[15].replace('0', '%d' % startz, 1)
    filedata[15] = filedata[15].replace('300', '%d' % stopx, 1)
    filedata[15] = filedata[15].replace('300', '%d' % stopy, 1)
    filedata[15] = filedata[15].replace('300', '%d' % stopz, 1)

    file_output = r'C:\Users\u0105452\PhD\PNM_extraction\pnextract-master\doc\volume_input_%05d.mhd' % index
    file = open(file_output, 'w', newline='')
    for text in filedata:
        file.write(text)
    file.close()

    return file_output





for index in range(n_volumes):
    startx = np.random.randint(0, dimension_volume_xy - newdimension)
    starty = np.random.randint(0, dimension_volume_xy - newdimension)
    startz = np.random.randint(0, dimension_volume_z - newdimension)

    for df_name, op_folder, op_folder_pnm in zip(datafile_name, output_folders, output_folders_pnm):

        file_output = createRandomVolume(dimension_volume_xy, dimension_volume_z, newdimension, resolution, df_name,
                                         index, startx, starty, startz)
        os.system('cd %s & %s %s' % (pn_directory, path_exe, file_output))
        # Move files generated
        files_in_pndir = os.listdir(pn_directory)
        for ss in substrings:
            filename = [fn for fn in files_in_pndir if ss in fn]
            shutil.move(os.path.join(pn_directory, filename[0]), os.path.join(op_folder, 'volume_%05d_%s' % (index, ss)))
        prefix = 'volume_%05d' % (index)
        project = op.io.Statoil.load(path=op_folder, prefix=prefix)
        pn = project.network
        pn.name = 'index'
        project.export_data(filename=os.path.join(op_folder_pnm, prefix))








