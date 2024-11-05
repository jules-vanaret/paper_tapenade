import napari
import numpy as np
import tifffile
from skimage.morphology import binary_dilation


path_to_data = '/home/jvanaret/data/data_paper_tapenade/morphology/processed/all_quantities_midplane'


labels = tifffile.imread(
    f'{path_to_data}/labels/ag1_norm_labels.tif'
)
labels_mask = labels.astype(bool)
mask = tifffile.imread(
    f'{path_to_data}/mask/ag1_mask.tif'
)
mask_contour = binary_dilation(mask) & ~mask


S, A = np.meshgrid(
    np.arange(labels.shape[1]),
    np.arange(labels.shape[0]),
)

S = np.where(mask, S, 0)
S = 0.1+(S/np.max(S))**1.3
S = np.where(mask, S, 0)
A = np.where(mask, A, 0)
A = 0.1+(A/np.max(A))**1.3
A = np.where(mask, A, 0)

epsilon = np.random.uniform(0, 0.2, labels.shape)
I = labels_mask * S * A + epsilon
I = np.clip(I, 0, 1)


viewer = napari.Viewer()

mask_contour = np.where(mask_contour, True, np.nan)

viewer.add_image(I, colormap='gray_r')
viewer.add_image(mask_contour, blending='translucent', opacity=0.25, colormap='gray_r')
viewer.add_image(labels_mask, colormap='gray_r')
viewer.add_image(mask_contour, blending='translucent', opacity=0.25, colormap='gray_r')
viewer.add_image(S, colormap='gray_r')
viewer.add_image(mask_contour, blending='translucent', opacity=0.25, colormap='gray_r')
viewer.add_image(A, colormap='gray_r')
viewer.add_image(mask_contour, blending='translucent', opacity=0.25, colormap='gray_r')
viewer.add_image(epsilon, colormap='gray_r', contrast_limits=[np.min(I), np.max(I)])
viewer.add_image(mask_contour, blending='translucent', visible=False, colormap='gray_r')

viewer.grid.enabled = True
viewer.grid.shape = (1, -1)
viewer.grid.stride = -2
napari.run()









