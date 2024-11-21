import numpy as np
from tapenade.analysis.spatial_correlation import SpatialCorrelationPlotter
from tapenade.preprocessing import crop_array_using_mask
from tapenade.preprocessing._intensity_normalization import _normalize_intensity
import tifffile
import matplotlib.pyplot as plt
import napari
from pathlib import Path

folder = ...

mask = tifffile.imread(Path(folder)/"S7_endogenous_vs_immunostained/mask.tif")

labels = tifffile.imread(Path(folder)/"S7_endogenous_vs_immunostained/labels.tif")
hoechst = tifffile.imread(Path(folder)/"S7_endogenous_vs_immunostained/hoechst_before_norm.tif")

endogen = tifffile.imread(Path(folder)/"S7_endogenous_vs_immunostained/endo_iso.tif")
immunostained = tifffile.imread(Path(folder)/"S7_endogenous_vs_immunostained/immunostained_iso.tif")

hoechst_norm = _normalize_intensity(
    array=hoechst,
    ref_array=hoechst,
    mask=mask,
    labels=labels,
    image_wavelength=405,
    sigma=14,
)
endogen_norm = _normalize_intensity(
    array=endogen,
    ref_array=endogen,
    mask=mask,
    labels=labels,
    image_wavelength=488,
    sigma=14,
)
immunostained_norm = _normalize_intensity(
    array=immunostained,
    ref_array=immunostained,
    mask=mask,
    labels=labels,
    image_wavelength=555,
    sigma=14,
)
# mask, immunostained_norm, labels = crop_array_using_mask(
#     image=immunostained_norm,
#     mask=mask.copy(),
#     labels=labels
# )
hoechst = crop_array_using_mask(array=hoechst, mask=mask.copy())
hoechst_norm = crop_array_using_mask(array=hoechst_norm, mask=mask.copy())
endogen = crop_array_using_mask(array=endogen, mask=mask.copy())
endogen_norm = crop_array_using_mask(array=endogen_norm, mask=mask.copy())
immunostained = crop_array_using_mask(array=immunostained, mask=mask.copy())
immunostained_norm = crop_array_using_mask(mask=mask, array=immunostained_norm)
labels = crop_array_using_mask(mask=mask, array=labels)
mask = crop_array_using_mask(mask=mask, array=mask)

### Napari
if True:
    viewer = napari.Viewer()

    min_max_hoechst = (
        min(np.percentile(hoechst[mask], 3), np.percentile(hoechst_norm[mask], 3)),
        max(np.percentile(hoechst[mask], 97), np.percentile(hoechst_norm[mask], 97)),
    )
    min_max_endogen = (
        min(np.percentile(endogen[mask], 3), np.percentile(endogen_norm[mask], 3)),
        max(np.percentile(endogen[mask], 97), np.percentile(endogen_norm[mask], 97)),
    )
    min_max_immunostained = (
        min(
            np.percentile(immunostained[mask], 3),
            np.percentile(immunostained_norm[mask], 3),
        ),
        max(
            np.percentile(immunostained[mask], 97),
            np.percentile(immunostained_norm[mask], 97),
        ),
    )

    hoechst_napari = np.where(mask, hoechst, np.nan)
    endogen_napari = np.where(mask, endogen, np.nan)
    immunostained_napari = np.where(mask, immunostained, np.nan)
    hoechst_norm_napari = np.where(mask, hoechst_norm, np.nan)
    endogen_norm_napari = np.where(mask, endogen_norm, np.nan)
    immunostained_norm_napari = np.where(mask, immunostained_norm, np.nan)

    viewer.add_image(
        hoechst_napari,
        name="hoechst",
        contrast_limits=min_max_hoechst,
        colormap="gray_r",
    )
    viewer.add_image(
        hoechst_norm_napari,
        name="hoechst_norm",
        contrast_limits=min_max_hoechst,
        colormap="gray_r",
    )
    viewer.add_image(
        endogen_napari,
        name="endogen",
        contrast_limits=min_max_endogen,
        colormap="gray_r",
    )
    viewer.add_image(
        endogen_norm_napari,
        name="endogen_norm",
        contrast_limits=min_max_endogen,
        colormap="gray_r",
    )
    viewer.add_image(
        immunostained_napari,
        name="immunostained",
        contrast_limits=min_max_immunostained,
        colormap="gray_r",
    )
    viewer.add_image(
        immunostained_norm_napari,
        name="immunostained_norm",
        contrast_limits=min_max_immunostained,
        colormap="gray_r",
    )

    # viewer.add_image(mask, name='mask', colormap='gray', opacity=1)

    for l in viewer.layers:
        l.data = np.transpose(l.data, (1, 0, 2))

    viewer.grid.enabled = True
    # viewer.grid.shape = (2, 3)
    viewer.grid.shape = (3, 2)
    viewer.grid.stride = -1

    napari.run()


#! PUT EVERY SCATTER ON THE SAME EXTENT


def add_median(ax):
    medX = np.median(ax.collections[0].get_offsets()[:, 0].data)
    medY = np.median(ax.collections[0].get_offsets()[:, 1].data)

    ax.axvline(medX, color="black", lw=3, alpha=0.5)
    ax.axhline(medY, color="black", lw=3, alpha=0.5)


def add_identity(ax):
    ax.plot([0, 2000], [0, 2000], "--", c="#648FFF", lw=3, label="Y = X")


def plot(X, Y, mask, labels, lX, lY):
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
        bins=[40, 40],
        label_X=lX,
        label_Y=lY,
        extent_X=[0, 2000],
        extent_Y=[0, 2000],
        # fig_ax_tuple=(fig, axes[0, 0]),
        fig_ax_tuple=(fig, axes[1, 0]),
        show_linear_fit=True,
        display_quadrants=False,
    )

    add_median(ax)
    add_identity(ax)
    ax.ticklabel_format(scilimits=(-5, 8))
    ax.set_xticks([0, 500, 1000, 1500, 2000])
    ax.set_yticks([0, 500, 1000, 1500, 2000])
    ax.legend(loc=4)

    third_dim = int(mask.shape[0] / 3)

    for i in range(3):
        slice_ = slice(i * third_dim, (i + 1) * third_dim)

        plotter = SpatialCorrelationPlotter(
            quantity_X=X[slice_],
            quantity_Y=Y[slice_],
            mask=mask[slice_],
            labels=labels[slice_],
        )

        print((int(i > 0), 1 + int(i % 2)))

        fig, ax = plotter.get_heatmap_figure(
            bins=[40, 40],
            label_X=lX,
            label_Y=lY,
            extent_X=[0, 2000],
            extent_Y=[0, 2000],
            # fig_ax_tuple=(fig, axes[int(i>0), 1-int(i%2)]),
            fig_ax_tuple=(fig, axes[i, 1]),
            show_linear_fit=True,
            display_quadrants=False,
        )
        add_median(ax)
        add_identity(ax)
        ax.ticklabel_format(scilimits=(-5, 8))
        ax.set_xticks([0, 500, 1000, 1500, 2000])
        ax.set_yticks([0, 500, 1000, 1500, 2000])
        ax.legend(loc=4)

    fig.tight_layout()


### FIRST FIGURE: No normalization, full img + 3 depths

plot(
    endogen,
    immunostained,
    mask,
    labels,
    "Endogenous signal (A.U)",
    "Immunostained signal (A.U)",
)
plt.savefig(Path(folder)/"S7_b_c.png")


# ### SECOND FIGURE: Normalization, full img + 3 depths

plot(
    endogen_norm,
    immunostained_norm,
    mask,
    labels,
    "Endogenous signal normalized (A.U)",
    "Immunostained signal \n normalized (A.U)",
)
plt.savefig(Path(folder)/"S7_d.png")
plt.show()
