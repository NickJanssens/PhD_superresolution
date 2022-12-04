import os
import openpnm as op
import numpy as np
import scipy as sp
from pathlib import Path


path = Path(r'C:\Users\u0105452\PhD\PNM_extraction\pnextract-master/')
project = op.io.Statoil.load(path=path, prefix='LR_slices_segm_png')
pn = project.network
pn.name = 'LR'
project.export_data(filename=r'C:\Users\u0105452\PhD\PNM_extraction\pnextract-master\LR_slices_segm_png')
