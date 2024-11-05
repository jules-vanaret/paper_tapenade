import numpy as np
from tapenade.analysis.spatial_correlation import SpatialCorrelationPlotter
from tapenade.preprocessing import crop_array_using_mask
import tifffile
import matplotlib.pyplot as plt
import napari
from tapenade.preprocessing import masked_gaussian_smooth_dense_two_arrays_gpu


path_to_data = '/home/jvanaret/data/data_paper_tapenade/data_fig_correlation_fig_napari/processed/'

mask = tifffile.imread(
    f'{path_to_data}/mask.tif'
)

labels = tifffile.imread(
    f'{path_to_data}/labels.tif'
)
dapi = tifffile.imread(
    f'{path_to_data}/dapi isotropized equalized.tif'
)

bra = tifffile.imread(
    f'{path_to_data}/normalized/bra isotropized normalized.tif'
)

ecad = tifffile.imread(
    f'{path_to_data}/normalized/ecad isotropized normalized.tif'
)

mask_nuclei = labels.astype(bool)
mask_membranes = np.where(mask, ~mask_nuclei, False)




bra_smoothed, ecad_smoothed = masked_gaussian_smooth_dense_two_arrays_gpu(
    datas=[bra, ecad],
    sigmas=10,
    mask=mask,
    masks_for_volume=[mask_membranes, mask_nuclei],
)



### Napari
if True:
    viewer = napari.Viewer()

    bra_napari = np.where(mask, bra, np.nan)
    ecad_napari = np.where(mask, ecad, np.nan)

    viewer.add_image(dapi, name='dapi', colormap='gray', opacity=1)

    viewer.add_image(bra_napari, name='bra', colormap='red', opacity=1)
    viewer.add_image(ecad_napari, name='ecad', colormap='green', opacity=1)

    viewer.add_image(bra_smoothed, name='bra_smoothed', colormap='red', opacity=1)
    viewer.add_image(ecad_smoothed, name='ecad_smoothed', colormap='green', opacity=1)

    viewer.add_image(mask_nuclei, name='mask_nuclei', colormap='gray', opacity=0.5)
    viewer.add_image(mask_membranes, name='mask_membranes', colormap='gray', opacity=0.5)

    viewer.grid.enabled = True
    # viewer.grid.shape = (2, 3)
    viewer.grid.shape = (1,2)

    napari.run()



plotter = SpatialCorrelationPlotter(
            quantity_X=bra_smoothed,
            quantity_Y=ecad_smoothed,
            mask=mask,
            labels=labels,
        )

fig, ax = plotter.get_heatmap_figure(
    bins=[40,40],
    show_individual_cells=True,
    label_X='Bra signal (A.U)',
    label_Y='Ecad signal (A.U)',
    show_linear_fit=False,
    display_quadrants=False
)




# def plot(X, Y, mask ,labels,
#          lX, lY):
#     # fig, axes = plt.subplots(2,2)
#     # fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 4.2))
#     fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(9, 12))
#     plotter = SpatialCorrelationPlotter(
#             quantity_X=X,
#             quantity_Y=Y,
#             mask=mask,
#             labels=labels,
#         )

#     fig, ax = plotter.get_heatmap_figure(
#         bins=[40,40],
#         label_X=lX,
#         label_Y=lY,
#         extent_X=[0, 2000],
#         extent_Y=[0, 2000],
#         # fig_ax_tuple=(fig, axes[0, 0]),
#         fig_ax_tuple=(fig, axes[1,0]),
#         show_linear_fit=True,
#         display_quadrants=False
#     )

#     ax.ticklabel_format(scilimits=(-5, 8))
#     # ax.set_xticks([0,500,1000,1500,2000])
#     # ax.set_yticks([0,500,1000,1500,2000])
#     # ax.legend(loc=4)

#     third_dim = int(mask.shape[0]/3)

#     for i in range(3):
#         slice_ = slice(i*third_dim, (i+1)*third_dim)

#         plotter = SpatialCorrelationPlotter(
#             quantity_X=X[slice_],
#             quantity_Y=Y[slice_],
#             mask=mask[slice_],
#             labels=labels[slice_],
#         )

#         print((int(i>0), 1+int(i%2)))

#         fig, ax = plotter.get_heatmap_figure(
#             bins=[40,40],
#             label_X=lX,
#             label_Y=lY,
#             extent_X=[0, 2000],
#             extent_Y=[0, 2000],
#             # fig_ax_tuple=(fig, axes[int(i>0), 1-int(i%2)]),
#             fig_ax_tuple=(fig, axes[i, 1]),
#             show_linear_fit=True,
#             display_quadrants=False
#         )
#         add_median(ax)
#         add_identity(ax)
#         ax.ticklabel_format(scilimits=(-5, 8))
#         ax.set_xticks([0,500,1000,1500,2000])
#         ax.set_yticks([0,500,1000,1500,2000])
#         ax.legend(loc=4)

#     fig.tight_layout()

# ### FIRST FIGURE: No normalization, full img + 3 depths

# plot(endogen, reporter, mask, labels, 'Endogen signal (A.U)', 'Reporter signal (A.U)')



plt.show()