# Plots line profile of the intensity of Hoechst and Tbra in the midplane of the num_sample, and their normalized versions.

from pathlib import Path
import tifffile
import matplotlib.pyplot as plt
import numpy as np
from tapenade.preprocessing import (
    masked_gaussian_smoothing,
    normalize_intensity,
)


def save_fig(data, path, cmap, vmin, vmax):
    fig = plt.figure()
    fig = plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    fig = plt.xticks([])
    fig = plt.yticks([])
    fig = plt.savefig(Path(path), dpi=500)
    return fig


def plot_profile(im2D, mask):
    ymid = im2D.shape[0] // 2
    midline_region = slice(ymid - 30, ymid + 30)  # 60 µm around the midline
    masked_image = np.where(
        mask[midline_region] == 1, im2D[midline_region], np.nan
    )  # image which value is im2D in the mask, and nan outside
    line = np.nanmedian(masked_image, axis=0)  # proj intensity along the y axis
    return line, len(line)


# colormaps by https://personal.sron.nl/~pault/
light_green = "#44BB99"
dark_green = "#225522"
light_blue = "#77AADD"
dark_blue = "#222255"
light_red = "#EE8866"
dark_red = "#663333"

sigma_norm = 11.8  # size of normalization kernel, found with optimization in the function normalize_intensity
z = 150  # middle slice of the gastruloid
half_thickness = 30  # max proj on 60µm subvolume around the midplane
sigma_plot = 5  # size of the smoothing kernel for the profile plot
scale = (1, 0.6, 0.6)
folder = Path(__file__).parents[2] / 'data'
# num_sample = 1  # plot 5b
num_sample = 6 #plot 5a
im = tifffile.imread(
    Path(folder) / f"5a_Dapi_Ecad_bra_sox2_725h_re/data/{num_sample}.tif"
)
mask = (
    tifffile.imread(
        Path(folder) / f"5a_Dapi_Ecad_bra_sox2_725h_re/masks/{num_sample}_mask.tif"
    )
).astype(bool)
seg = tifffile.imread(
    Path(folder) / f"5a_Dapi_Ecad_bra_sox2_725h_re/segmentation/{num_sample}_seg.tif"
)
Hoechst = im[:, 0, :, :]
Tbra = im[:, 2, :, :]

Tbra_norm = normalize_intensity(
    image=Tbra, ref_image=Hoechst, mask=mask, labels=seg, sigma=sigma_norm
)
Hoechst_norm = normalize_intensity(
    image=Hoechst, ref_image=Hoechst, mask=mask, labels=seg, sigma=sigma_norm
)
Hoechst_smooth = masked_gaussian_smoothing(
    image=Hoechst, mask=mask, mask_for_volume=seg, sigmas=sigma_norm
)

Tbra_norm_2D = np.mean(Tbra_norm[z - half_thickness : z + half_thickness], axis=0)
Hoechst_norm_2D = np.mean(Hoechst_norm[z - half_thickness : z + half_thickness], axis=0)
mask_2D = np.mean(mask[z - half_thickness : z + half_thickness], axis=0).astype(bool)
Tbra_2D = np.mean(Tbra[z - half_thickness : z + half_thickness], axis=0)
Hoechst_2D = np.mean(Hoechst[z - half_thickness : z + half_thickness], axis=0)
Tbra_norm_2D[np.isnan(Tbra_norm_2D)] = 0
Hoechst_norm_2D[np.isnan(Hoechst_norm_2D)] = 0

if False:
    save_fig(
        Tbra_2D,
        Path(folder) / "5a_Dapi_Ecad_bra_sox2_725h_re/Tbra_2D.svg",
        "gray_r",
        0,
        150,
    )
    save_fig(
        Hoechst_2D,
        Path(folder) / "5a_Dapi_Ecad_bra_sox2_725h_re/Hoechst_2D.svg",
        "gray_r",
        0,
        150,
    )
    save_fig(
        Tbra_norm_2D,
        Path(folder) / "5a_Dapi_Ecad_bra_sox2_725h_re/Tbra_norm_2D.svg",
        "gray_r",
        0,
        500,
    )
    save_fig(
        Hoechst_norm_2D,
        Path(folder) / "5a_Dapi_Ecad_bra_sox2_725h_re/Hoechst_norm_2D.svg",
        "gray_r",
        0,
        500,
    )
    save_fig(
        Hoechst_smooth[z],
        Path(folder) / "5a_Dapi_Ecad_bra_sox2_725h_re/Hoechst_smooth.svg",
        "turbo",
        0,
        200,
    )

fig, ax = plt.subplots(2, figsize=(14, 18))
# signal is smoothed to extract large scale variations
Tbra_norm_gauss = masked_gaussian_smoothing(
    image=Tbra_norm_2D, mask=mask_2D, sigmas=sigma_plot
)
Hoechst_norm_gauss = masked_gaussian_smoothing(
    image=Hoechst_norm_2D, mask=mask_2D, sigmas=sigma_plot
)
Tbra_gauss = masked_gaussian_smoothing(image=Tbra_2D, mask=mask_2D, sigmas=sigma_plot)
Hoechst_gauss = masked_gaussian_smoothing(
    image=Hoechst_2D, mask=mask_2D, sigmas=sigma_plot
)

line_Tbra_norm, length = plot_profile(Tbra_norm_gauss, mask=mask_2D)
line_Hoechst_norm, length = plot_profile(Hoechst_norm_gauss, mask=mask_2D)
line_Tbra, _ = plot_profile(Tbra_gauss, mask=mask_2D)
line_Hoechst, _ = plot_profile(Hoechst_gauss, mask=mask_2D)
y_coords = [i for i in range(len(line_Hoechst))]

ax[0].plot(y_coords, line_Hoechst, linewidth=4, label="Hoechst", color=light_blue)
ax0_twin = ax[0].twinx()
ax0_twin.plot(
    y_coords, line_Hoechst_norm, linewidth=4, label="Hoechst norm.", color=dark_blue
)

ax[1].plot(y_coords, line_Tbra, linewidth=4, label="T-Bra", color=light_red)
ax1_twin = ax[1].twinx()
ax1_twin.plot(
    y_coords, line_Tbra_norm, linewidth=4, label="T-Bra norm.", color=dark_red
)

ax[0].set_xlabel("Distance along the line (µm)", fontsize=30)
ax[1].set_xlabel("Distance along the line (µm)", fontsize=30)
#
ax[0].set_ylabel("Avg. intensity in \nHoechst (A.U)", fontsize=30, color=light_blue)
ax0_twin.set_ylabel(
    "Avg. intensity in \nHoechst  norm.(A.U)", fontsize=30, color=dark_blue
)
ax[1].set_ylabel("Avg. intensity in \nT-Bra (A.U)", fontsize=30, color=light_red)
ax1_twin.set_ylabel("Avg. intensity in \nT-Bra norm.(A.U)", fontsize=30, color=dark_red)


ax[0].set_yticks([0, 40, 80])
ax0_twin.set_yticks([0, 125, 250])
ax[1].set_yticks([0, 40, 80])
ax1_twin.set_yticks([0, 100, 200])

ax[0].set_xticks([0, 100, 200, 300])
ax[1].set_xticks([0, 100, 200, 300])

ax[0].tick_params(axis="x", labelsize=30)
ax[1].tick_params(axis="x", labelsize=30)

ax[0].tick_params(axis="y", labelsize=30, colors=light_blue)
ax0_twin.tick_params(axis="y", labelsize=30, colors=dark_blue)
ax[1].tick_params(axis="y", labelsize=30, colors=light_red)
ax1_twin.tick_params(axis="y", labelsize=30, colors=dark_red)

lines_1, labels_1 = ax[0].get_legend_handles_labels()
lines_1_1, labels_1_1 = ax0_twin.get_legend_handles_labels()
ax[0].legend(lines_1 + lines_1_1, labels_1 + labels_1_1, fontsize=25)

lines_0, labels_0 = ax[1].get_legend_handles_labels()
lines_0_1, labels_0_1 = ax1_twin.get_legend_handles_labels()
ax[1].legend(lines_0 + lines_0_1, labels_0 + labels_0_1, fontsize=25)

fig.tight_layout()
# fig.savefig(Path(folder) / "5a_plot.svg")
plt.show()
