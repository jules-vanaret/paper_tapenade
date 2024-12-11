# plots the intensity profile of Hoechst in depth for different sigmas of the gaussian filter used for normalization
import tifffile
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from tapenade.preprocessing._smoothing import _masked_smooth_gaussian
from tapenade.preprocessing._preprocessing import (
    change_array_pixelsize,
    normalize_intensity,
)
from tqdm import tqdm


def save_fig(data, path, cmap, vmin, vmax):
    fig = plt.figure()
    fig = plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    fig = plt.xticks([])
    fig = plt.yticks([])
    fig = plt.savefig(Path(path), dpi=500)
    return fig


folder = Path(__file__).parents[3] / 'data'

fig, ax = plt.subplots(1, figsize=(10, 7))

colors = ["#33BBEE", "#009988", "#EE7733", "#CC3311", "#EE3377"]
scale = (1, 0.6, 0.6)

im = tifffile.imread(Path(folder) / "2k_Hoechst_FoxA2_Oct4_Bra_78h/big/data/1.tif")
mask = (tifffile.imread(Path(folder) / "2k_Hoechst_FoxA2_Oct4_Bra_78h/big/masks/1_mask.tif")).astype(bool)
seg = tifffile.imread(Path(folder) / "2k_Hoechst_FoxA2_Oct4_Bra_78h/big/segmentation/1_seg.tif")
hoechst = im[:, 0, :, :]
hoechst_iso = change_array_pixelsize(array=hoechst, input_pixelsize=scale)
mask_iso = change_array_pixelsize(array=mask, input_pixelsize=scale, order=0)
seg_iso = change_array_pixelsize(array=seg, input_pixelsize=scale, order=0)

hoechst_nan = np.where(mask_iso == 1, hoechst_iso, np.nan).astype(float)
Int_hoechst_non_norm = []
for z in range(len(mask)):
    Int_hoechst_non_norm.append(np.nanmedian(hoechst_nan[z, :, :]))

i = 0  # index for color change
for sigma in tqdm([10, 20, 40]):
    hoechst_norm = normalize_intensity(
        image=hoechst_iso,
        ref_image=hoechst_iso,
        mask=mask_iso.astype(bool),
        labels=seg_iso,
        sigma=sigma,
        image_wavelength=405,
    )
    hoechst_norm[mask_iso == 0] = np.nan
    hoechst_smooth = _masked_smooth_gaussian(
        hoechst_iso, sigmas=sigma, mask_for_volume=seg_iso, mask=mask_iso
    )
    hoechst_nan = np.where(mask_iso == 1, hoechst_smooth, np.nan).astype(float)
    # save_fig(
    #     hoechst_nan[120],
    #     Path(folder) / f"S8e_smoothed_hoechst_sigma_{sigma}.svg",
    #     "viridis",
    #     0,
    #     120,
    # )

    Int_hoechst_norm = []
    for z in range(len(hoechst_iso)):
        Int_hoechst_norm.append(np.nanmedian(hoechst_norm[z, :, :]))
    ax.plot(
        Int_hoechst_norm,
        label="σ=" + str(sigma) + "µm",
        color=colors[i],
        linewidth=4,
    )
    i += 1

ax.plot(
    Int_hoechst_non_norm, label="Raw data, unnormalized", color=colors[i], linewidth=4
)
ax.set_xlabel("Depth (µm)", fontsize=30)
ax.set_ylabel("Intensity in \nHoechst (A.U)", fontsize=30)
ax.set_yticks([0, 50, 100, 150])
ax.set_xticks([0, 100, 200, 300])
ax.tick_params(axis="y", labelsize=30)
ax.tick_params(axis="x", labelsize=30)
ax.legend(fontsize=15)
fig.tight_layout()
# fig.savefig(Path(folder) / "S8e_plot.svg")
# plt.legend()
plt.show()
