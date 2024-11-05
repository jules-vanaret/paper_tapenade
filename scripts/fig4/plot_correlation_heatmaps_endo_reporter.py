import numpy as np
from tapenade.analysis.spatial_correlation import SpatialCorrelationPlotter
from tapenade.preprocessing import crop_array_using_mask
import tifffile
import matplotlib.pyplot as plt
import napari


path_to_data = '/home/jvanaret/data/data_paper_tapenade/correlation_staining_endogen'

mask = tifffile.imread(
    f'{path_to_data}/mask.tif'
)

labels = tifffile.imread(
    f'{path_to_data}/labels.tif'
)
dapi = tifffile.imread(
    f'{path_to_data}/dapi_before_norm.tif'
)
dapi = crop_array_using_mask(array=dapi, mask=mask.copy())
dapi_norm = tifffile.imread(
    f'{path_to_data}/dapi_iso_normalized_12.tif'
)
dapi_norm = crop_array_using_mask(array=dapi_norm, mask=mask.copy())
endogen = tifffile.imread(
    f'{path_to_data}/endo_iso.tif'
)
endogen = crop_array_using_mask(array=endogen, mask=mask.copy())
reporter = tifffile.imread(
    f'{path_to_data}/reporter_iso.tif'
)
reporter = crop_array_using_mask(array=reporter, mask=mask.copy())
endogen_norm = tifffile.imread(
    f'{path_to_data}/endo_iso_normalized_12.tif'
)
endogen_norm = crop_array_using_mask(array=endogen_norm, mask=mask.copy())
reporter_norm = tifffile.imread(
    f'{path_to_data}/reporter_iso_normalized_12.tif'
)
# mask, reporter_norm, labels = crop_array_using_mask(
#     image=reporter_norm, 
#     mask=mask.copy(), 
#     labels=labels
# )
reporter_norm = crop_array_using_mask(mask=mask, array=reporter_norm)
labels = crop_array_using_mask(mask=mask, array=labels)
mask = crop_array_using_mask(mask=mask, array=mask)






### Napari
if True:
    viewer = napari.Viewer()

    min_max_dapi = (
        min(np.percentile(dapi[mask],3), np.percentile(dapi_norm[mask],3)), 
        max(np.percentile(dapi[mask],97), np.percentile(dapi_norm[mask],97))
    )
    min_max_endogen = (
        min(np.percentile(endogen[mask],3), np.percentile(endogen_norm[mask],3)), 
        max(np.percentile(endogen[mask],97), np.percentile(endogen_norm[mask],97))
    )
    min_max_reporter = (
        min(np.percentile(reporter[mask],3), np.percentile(reporter_norm[mask],3)), 
        max(np.percentile(reporter[mask],97), np.percentile(reporter_norm[mask],97))
    )

    dapi_napari = np.where(mask, dapi, np.nan)
    endogen_napari = np.where(mask, endogen, np.nan)
    reporter_napari = np.where(mask, reporter, np.nan)
    dapi_norm_napari = np.where(mask, dapi_norm, np.nan)
    endogen_norm_napari = np.where(mask, endogen_norm, np.nan)
    reporter_norm_napari = np.where(mask, reporter_norm, np.nan)

    viewer.add_image(dapi_napari, name='dapi', contrast_limits=min_max_dapi, colormap='gray_r')
    viewer.add_image(dapi_norm_napari, name='dapi_norm', contrast_limits=min_max_dapi, colormap='gray_r')
    viewer.add_image(endogen_napari, name='endogen', contrast_limits=min_max_endogen, colormap='gray_r')
    viewer.add_image(endogen_norm_napari, name='endogen_norm', contrast_limits=min_max_endogen, colormap='gray_r')
    viewer.add_image(reporter_napari, name='reporter', contrast_limits=min_max_reporter, colormap='gray_r')
    viewer.add_image(reporter_norm_napari, name='reporter_norm', contrast_limits=min_max_reporter, colormap='gray_r')

    # viewer.add_image(mask, name='mask', colormap='gray', opacity=1)

    for l in viewer.layers:
        l.data = np.transpose(l.data, (1, 0, 2))

    viewer.grid.enabled = True
    # viewer.grid.shape = (2, 3)
    viewer.grid.shape = (3,2)
    viewer.grid.stride = -1

    napari.run()



#! PUT EVERY SCATTER ON THE SAME EXTENT

def add_median(ax):
    medX = np.median(ax.collections[0].get_offsets()[:,0].data)
    medY = np.median(ax.collections[0].get_offsets()[:,1].data)

    ax.axvline(medX, color='black', lw=3, alpha=0.5)
    ax.axhline(medY, color='black', lw=3, alpha=0.5)

def add_identity(ax):
    ax.plot([0, 2000], [0, 2000], '--', c='#648FFF', lw=3, label='Y = X')

def plot(X, Y, mask ,labels,
         lX, lY):
    # fig, axes = plt.subplots(2,2)
    # fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 4.2))
    fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(9, 12))
    plotter = SpatialCorrelationPlotter(
            quantity_X=X,
            quantity_Y=Y,
            mask=mask,
            labels=labels,
        )

    fig, ax = plotter.get_heatmap_figure(
        bins=[40,40],
        label_X=lX,
        label_Y=lY,
        extent_X=[0, 2000],
        extent_Y=[0, 2000],
        # fig_ax_tuple=(fig, axes[0, 0]),
        fig_ax_tuple=(fig, axes[1,0]),
        show_linear_fit=True,
        display_quadrants=False
    )

    add_median(ax)
    add_identity(ax)
    ax.ticklabel_format(scilimits=(-5, 8))
    ax.set_xticks([0,500,1000,1500,2000])
    ax.set_yticks([0,500,1000,1500,2000])
    ax.legend(loc=4)

    third_dim = int(mask.shape[0]/3)

    for i in range(3):
        slice_ = slice(i*third_dim, (i+1)*third_dim)

        plotter = SpatialCorrelationPlotter(
            quantity_X=X[slice_],
            quantity_Y=Y[slice_],
            mask=mask[slice_],
            labels=labels[slice_],
        )

        print((int(i>0), 1+int(i%2)))

        fig, ax = plotter.get_heatmap_figure(
            bins=[40,40],
            label_X=lX,
            label_Y=lY,
            extent_X=[0, 2000],
            extent_Y=[0, 2000],
            # fig_ax_tuple=(fig, axes[int(i>0), 1-int(i%2)]),
            fig_ax_tuple=(fig, axes[i, 1]),
            show_linear_fit=True,
            display_quadrants=False
        )
        add_median(ax)
        add_identity(ax)
        ax.ticklabel_format(scilimits=(-5, 8))
        ax.set_xticks([0,500,1000,1500,2000])
        ax.set_yticks([0,500,1000,1500,2000])
        ax.legend(loc=4)

    fig.tight_layout()

### FIRST FIGURE: No normalization, full img + 3 depths

plot(endogen, reporter, mask, labels, 'Endogen signal (A.U)', 'Reporter signal (A.U)')


# ### SECOND FIGURE: Normalization, full img + 3 depths

plot(endogen_norm, reporter_norm, mask, labels, 'Endogen signal normalized (A.U)', 'Reporter signal normalized (A.U)')

plt.show()