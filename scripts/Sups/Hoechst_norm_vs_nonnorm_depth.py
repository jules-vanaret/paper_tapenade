# Plot intensity as a function of depth for Hoechst, non normalized and normalized by Hoechst intensity.

import tifffile
import napari
import matplotlib.pyplot as plt
import numpy as np
from tapenade.preprocessing._preprocessing import (
    change_array_pixelsize,
    normalize_intensity,
)
from tapenade.preprocessing._smoothing import _masked_smooth_gaussian


def save_fig(data, path_fig, cmap, vmin, vmax):
    """
    Save an image as a figure

    data : 2D array
    cmap : colormap
    vmin : min value of the colormap
    vmax : max value of the colormap
    """
    fig = plt.figure()
    fig = plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    fig = plt.xticks([])
    fig = plt.yticks([])
    fig = plt.savefig(rf"{path_fig}", dpi=500)
    return fig


folder = ...
name = "2"
im = tifffile.imread(rf"{folder}\data\{name}.tif")
mask = (tifffile.imread(rf"{folder}\masks\{name}.tif")).astype(bool)
seg = tifffile.imread(rf"{folder}\segmentation\{name}.tif")
hoechst = im[:, 0, :, :]

sigma = 20
scale = (1, 0.6, 0.6)

hoechst_iso = change_array_pixelsize(array=hoechst, input_pixelsize=scale)
mask_iso = change_array_pixelsize(array=mask, input_pixelsize=scale, order=0)
seg_iso = change_array_pixelsize(array=seg, input_pixelsize=scale, order=0)

hoechst_norm = normalize_intensity(
    image=hoechst_iso,
    ref_image=hoechst_iso,
    mask=mask_iso.astype(bool),
    labels=seg_iso,
    sigma=sigma,
    image_wavelength=405,
)

hoechst_nan = np.where(mask_iso == 1, hoechst_iso, np.nan).astype(float)

Int_hoechst_non_norm = []
Int_hoechst_norm = []
for z in range(len(mask)):
    Int_hoechst_non_norm.append(np.nanmedian(hoechst_nan[z, :, :]))
    Int_hoechst_norm.append(np.nanmedian(hoechst_norm[z, :, :]))
Int_hoechst_non_norm = [i for i in Int_hoechst_non_norm if np.isnan(i) == False]
Int_hoechst_norm = [i for i in Int_hoechst_norm if np.isnan(i) == False]

fig, ax = plt.subplots(1, figsize=(10, 7))

ax.plot(Int_hoechst_norm, label="normalized", color="#009988", linewidth=4)
ax.plot(
    Int_hoechst_non_norm,
    label="not normalized",
    color="#009988",
    linewidth=4,
    linestyle="dashed",
)

ax.set_xlabel("Depth (µm)", fontsize=30)
ax.set_ylabel("Median intensity in \n Hoechst (A.U)", fontsize=30)
ax.tick_params(axis="y", labelsize=30)
ax.tick_params(axis="x", labelsize=30)
ax.legend(fontsize=25)
plt.legend()
plt.show()
fig.savefig(rf"{folder_fig}\zprofile_{name}.svg")
