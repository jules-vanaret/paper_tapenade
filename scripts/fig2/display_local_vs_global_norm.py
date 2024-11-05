import napari
import tifffile
import numpy as np
from tapenade.preprocessing import crop_array_using_mask


path_to_data = '/home/jvanaret/data/data_paper_tapenade/1_vs_2_views'

mask = tifffile.imread(f'{path_to_data}/g2_dapi_fused_mask.tif')
data = tifffile.imread(f'{path_to_data}/g2_dapi_fused.tif')
print(data.shape)
percs = np.percentile(data[mask], [1, 99])
data = np.clip(
    (data-percs[0])/(percs[1]-percs[0]),
    0, 1
)
data = np.transpose(data, (1, 0, 2))

data_equalized = tifffile.imread(f'{path_to_data}/g2_dapi_fused_equalized.tif')
print(data_equalized.shape)
data_equalized = np.transpose(data_equalized, (1, 0, 2))

mask = np.transpose(mask, (1, 0, 2))

print(mask.shape)
_, data = crop_array_using_mask(image=data, mask=mask.copy())
print(mask.shape)
_, data_equalized = crop_array_using_mask(image=data_equalized, mask=mask)

print(data.shape)
print(data_equalized.shape)

labels_global = tifffile.imread(f'{path_to_data}/g2_dapi_fused_global_labels.tif')
# labels_global = np.transpose(labels_global, (1, 0, 2))
print(labels_global.shape)
_, labels_global = crop_array_using_mask(labels=labels_global, mask=mask)

labels_equalized = tifffile.imread(f'{path_to_data}/g2_dapi_fused_equalized_labels.tif')
labels_equalized = np.transpose(labels_equalized, (1, 0, 2))
print(labels_equalized.shape)
_, labels_equalized = crop_array_using_mask(labels=labels_equalized, mask=mask)

print(labels_global.shape)
print(labels_equalized.shape)


viewer = napari.Viewer()
viewer.add_image(data, name='original', colormap='gray_r')
viewer.add_image(data_equalized, name='equalized', colormap='gray_r')
viewer.add_labels(labels_global, name='labels_global', opacity=1)
viewer.add_labels(labels_equalized, name='labels_equalized', opacity=1)

viewer.grid.enabled = True
viewer.grid.shape = (2, 2)

viewer.grid.stride=-1
napari.run()