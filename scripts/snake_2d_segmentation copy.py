

import numpy as np
import napari
import morphsnakes
import tifffile
from matplotlib.backends.backend_qt5agg import FigureCanvas
import matplotlib.pyplot as plt
import scipy.ndimage as ndi

path_to_data = '/home/jvanaret/data/data_paper_tapenade/comparison_12views_globlocnorm'

input_image = tifffile.imread(f'{path_to_data}/Hoechst_FoxA2_Oct4_Bra_78h_big_1.tif')
roi_data = input_image[27,225-15:225+15,296-15:296+15]

viewer = napari.Viewer()
image_layer = viewer.add_image(roi_data)


def iter_callback(u):
    global viewer
    viewer.add_image(u)



center_pos = (roi_data.shape[0] // 2, roi_data.shape[1] // 2)

init_ls = morphsnakes.circle_level_set(
            roi_data.shape,
            center_pos,
            3
        )
snake = morphsnakes.morphological_chan_vese(
            roi_data,
            iterations=10,
            init_level_set=init_ls,
            smoothing=1,
            lambda1=0.1,
            lambda2=10,
            iter_callback=iter_callback
        )
    

napari.run()