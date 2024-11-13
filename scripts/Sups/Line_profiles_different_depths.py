# plot intensity profile in a XY plane for Hoechst and T-Bra, normalized. we look a different planes to check (for dapi) if there is no decrease in z.

import tifffile
import napari
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage as ndi
from tapenade.preprocessing._preprocessing import (
    change_array_pixelsize,
    normalize_intensity,
)
from tapenade.preprocessing._smoothing import _masked_smooth_gaussian


def save_fig(data, folder, cmap, vmin, vmax):
    fig = plt.figure()
    fig = plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    fig = plt.xticks([])
    fig = plt.yticks([])
    fig = plt.savefig(rf"{folder}", dpi=500)
    return fig


def plot_profile(im2D, mask):
    ymid = im2D.shape[0] // 2
    midline_region = slice(ymid - 30, ymid + 30)  # 60 µm around the midline
    masked_image = np.where(
        mask[midline_region] == 1, im2D[midline_region], np.nan
    )  # image which value is im2D in the mask, and nan outside
    line = np.nanmedian(masked_image, axis=0)  # proj intensity along the y axis
    return line


folder = ...

# colormaps by #https://personal.sron.nl/~pault/
colors_bra = ["#FEE391", "#FE9A29", "#CC4C02", "#993404", "#662506"]
colors_hoechst = ["#B5DDD8", "#81C4E7", "#9398D2", "#906388", "#684957"]

scale = (1, 0.6, 0.6)
sigma_plot = 30
sigma_norm = 11
dz = 5  # half thickness of each subvolume around the slices
visualize_napari = False
im = tifffile.imread(rf"{folder}\data\1.tif")
mask = (tifffile.imread(rf"{folder}\masks\1.tif")).astype(bool)
seg = tifffile.imread(rf"{folder}\segmentation\1.tif")
hoechst = im[:, 0, :, :]
bra = im[:, 3, :, :]

hoechst_iso = change_array_pixelsize(array=hoechst, input_pixelsize=scale)
bra_iso = change_array_pixelsize(array=bra, input_pixelsize=scale)
seg_iso = change_array_pixelsize(array=seg, input_pixelsize=scale, order=0)
mask_iso = change_array_pixelsize(array=mask, input_pixelsize=scale, order=0)
bra_norm = normalize_intensity(
    image=bra_iso,
    ref_image=hoechst_iso,
    mask=mask_iso,
    labels=seg_iso,
    sigma=sigma_norm,
    image_wavelength=555,
)
hoechst_norm = normalize_intensity(
    image=hoechst_iso,
    ref_image=hoechst_iso,
    mask=mask_iso,
    labels=seg_iso,
    sigma=sigma_norm,
    image_wavelength=405,
)

hoechst_smooth = _masked_smooth_gaussian(
    hoechst_iso, sigmas=sigma_norm, mask_for_volume=seg_iso, mask=mask_iso
)

if visualize_napari:
    viewer = napari.Viewer()
    viewer.add_image(bra_norm, name="bra_norm", colormap="inferno")
    viewer.add_image(hoechst_norm, name="hoechst_norm", colormap="inferno")
    viewer.add_image(hoechst_iso, name="hoechst", colormap="inferno")
    viewer.add_image(bra_iso, name="bra", colormap="inferno")
    viewer.add_image(hoechst_smooth, colormap="turbo")
    napari.run()

fig, ax = plt.subplots(2, 2, figsize=(15, 9))

for ind_z, z in enumerate([50, 100, 150, 200]):
    # averaging few slices around z
    bra_norm_2D = np.mean(bra_norm[z - dz : z + dz], axis=0)
    hoechst_norm_2D = np.mean(hoechst_norm[z - dz : z + dz], axis=0)
    mask_2D = np.mean(mask_iso[z - dz : z + dz], axis=0).astype(bool)
    bra_2D = np.mean(bra_iso[z - dz : z + dz], axis=0)
    hoechst_2D = np.mean(hoechst_iso[z - dz : z + dz], axis=0)
    bra_norm_2D[np.isnan(bra_norm_2D)] = 0
    hoechst_norm_2D[np.isnan(hoechst_norm_2D)] = 0

    # save_fig(bra_2D,rf'{folder}\bra_2D.svg','gray_r',0,300)
    # save_fig(hoechst_2D,rf'{folder}\hoechst_2D.svg','gray_r',0,300)
    # save_fig(bra_norm_2D,rf'{folder}\bra_norm_2D.svg','gray_r',0,500)
    # save_fig(hoechst_norm_2D,rf'{folder}\hoechst_norm_2D.svg','gray_r',0,500)

    bra_norm_gauss = _masked_smooth_gaussian(
        array=bra_norm_2D, mask=mask_2D, sigmas=sigma_plot
    )
    hoechst_norm_gauss = _masked_smooth_gaussian(
        array=hoechst_norm_2D, mask=mask_2D, sigmas=sigma_plot
    )
    bra_gauss = _masked_smooth_gaussian(array=bra_2D, mask=mask_2D, sigmas=sigma_plot)
    hoechst_gauss = _masked_smooth_gaussian(
        array=hoechst_2D, mask=mask_2D, sigmas=sigma_plot
    )

    line_bra_norm = plot_profile(bra_norm_gauss, mask=mask_2D)
    line_hoechst_norm = plot_profile(hoechst_norm_gauss, mask=mask_2D)
    line_bra = plot_profile(bra_gauss, mask=mask_2D)
    line_hoechst = plot_profile(hoechst_gauss, mask=mask_2D)
    y_coords = [i for i in range(len(line_hoechst))]

    ax[0, 0].plot(
        y_coords,
        line_hoechst,
        linewidth=4,
        label="z=" + str(z) + " µm",
        color=colors_hoechst[ind_z],
    )
    ax[0, 1].plot(
        y_coords,
        line_hoechst_norm,
        linewidth=4,
        label="z=" + str(z) + " µm",
        color=colors_hoechst[ind_z],
    )
    ax[1, 0].plot(
        y_coords,
        line_bra,
        linewidth=4,
        label="z=" + str(z) + " µm",
        color=colors_bra[ind_z],
    )
    ax[1, 1].plot(
        y_coords,
        line_bra_norm,
        linewidth=4,
        label="z=" + str(z) + " µm",
        color=colors_bra[ind_z],
    )


def shape_axis(ax, ymax):
    ax.set_xlabel("Distance along the line (µm)", fontsize=30)
    ax.set_ylabel("Avg. intensity in \nHoechst (A.U)", fontsize=30)
    ax.set_xticks([0, 100, 200, 300])
    ax.tick_params(axis="x", labelsize=30)
    ax.tick_params(axis="y", labelsize=30)
    ax.set_ylim(0, ymax)
    ax.legend()


ymax_hoechst = 1100
ymax_tbra = 900
shape_axis(ax[0, 0], ymax=ymax_hoechst)
shape_axis(ax[0, 1], ymax=ymax_hoechst)
shape_axis(ax[1, 0], ymax=ymax_tbra)
shape_axis(ax[1, 1], ymax=ymax_tbra)

ax[0, 0].legend()
ax[1, 0].legend()

fig.savefig(rf"{folder}\line_profile.svg")
plt.show()
