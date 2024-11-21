# This script is used to generate the 2D maps of cell density, division density and proliferation for the wholemount samples.
# Uses a segmentation of all nuclei and a segmentation of all dividing cells.
# save as tif the 3D maps, plot on napari and save 2D screenshots of the midplane

import numpy as np
import napari
from skimage.measure import regionprops
import tifffile
import matplotlib.pyplot as plt
from tqdm import tqdm
from tapenade.preprocessing import normalize_intensity
from tapenade.preprocessing._smoothing import _masked_smooth_gaussian
from pathlib import Path


def im_centroids(labels):
    labels_ppties = regionprops(labels)
    centroids_coord = np.array([prop.centroid for prop in labels_ppties]).T
    coords_in_array = [coord.astype(int).tolist() for coord in centroids_coord]
    im_centroids = np.zeros_like(div)
    im_centroids[coords_in_array[0], coords_in_array[1], coords_in_array[2]] = 1
    return im_centroids


def save_fig(data, folder, cmap, vmin, vmax):
    fig = plt.figure()
    fig = plt.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax)
    fig = plt.xticks([])
    fig = plt.yticks([])
    fig = plt.savefig(rf"{folder}", dpi=500)
    return fig


folder = ...
sigma = 35
scale = (1, 1, 1)

visualize_napari = True
save_2D_maps = True
z = 85  # if save_2D_maps is True, the z plane to look at
#S6b : Path(folder) / f"4b_S6b_proliferation/10b/1_data.tif"
mask = tifffile.imread(Path(folder) / f"4b_S6b_proliferation/4_mask.tif")
image = tifffile.imread(Path(folder) / f"4b_S6b_proliferation/4_data.tif")
seg = tifffile.imread(Path(folder) / f"4b_S6b_proliferation/4_seg.tif").astype(int)
div = tifffile.imread(Path(folder) / f"4b_S6b_proliferation/4_div.tif").astype(int)
hoechst = image[:, 0, :, :]
hoechst[mask == 0] = 0
ph3 = image[:, 1, :, :]
hoechst_norm = normalize_intensity(
    image=hoechst,
    ref_image=hoechst,
    mask=mask.astype(bool),
    labels=seg,
    sigma=sigma,
)
ph3_norm = normalize_intensity(
    image=ph3, ref_image=hoechst, mask=mask.astype(bool), labels=seg, sigma=sigma
)

# if one does not have a segmentation of the sparse signal to plot the map with, one can do it below using thresholding :
# signal = image[:,2,:,:]
# signal_norm = normalize_intensity(image=signal,ref_image=hoechst,mask=mask.astype(bool),labels=seg,sigma=sigma)
# ppties = regionprops(seg, intensity_image=signal_norm)
# cellular_signal= [prop.intensity_mean for prop in ppties]
# thresh_signal = threshold_otsu(np.asarray([i for i in cellular_signal if np.isnan(i)==False]))
# seg_signal=np.zeros_like(seg)
# for prop in ppties:
#     if prop.mean_intensity>thresh_signal:
#         seg_signal[seg==prop.label]=prop.label
# tifffile.imwrite(rf'{folder}/4_seg_signal.tif',seg_signal.astype('uint16'))

im_centroids_seg = im_centroids(seg)
im_centroids_div = im_centroids(div)
cell_density_map = _masked_smooth_gaussian(
    im_centroids_seg, sigmas=sigma, mask=mask
) * (10 * 10 * 10)
division_density_map = _masked_smooth_gaussian(
    im_centroids_div, sigmas=sigma, mask=mask
) * (10 * 10 * 10)
proliferation = division_density_map / cell_density_map

division_density_map[mask == 0] = np.nan
cell_density_map[mask == 0] = np.nan

tifffile.imwrite(
    Path(folder) / f"4b_10b_proliferation/4_proliferation.tif", proliferation
)
tifffile.imwrite(
    Path(folder) / f"4b_10b_proliferation/4_celldens.tif", cell_density_map
)
tifffile.imwrite(
    Path(folder) / f"4b_10b_proliferation/4_divdens.tif", division_density_map
)

if visualize_napari:
    viewer = napari.Viewer()
    viewer.add_image(hoechst_norm, name="4_hoechst")
    viewer.add_image(ph3_norm, name="4_ph3")
    viewer.add_labels(div, name="4_div")
    viewer.add_labels(seg, name="4_seg")
    viewer.add_image(
        cell_density_map,
        name="4_celldensity",
        colormap="inferno",
    )
    viewer.add_image(
        division_density_map,
        name="4_divisiondensity",
        colormap="inferno",
    )
    viewer.add_image(
        proliferation,
        name="4_proliferation",
        colormap="inferno",
    )
    viewer.grid.enabled = True
    napari.run()

if save_2D_maps:

    im = cell_density_map[z, :, :]
    save_fig(
        im,
        Path(folder) / f"4b_10b_proliferation/4_z{z}_celldens.svg",
        "inferno",
        np.nanpercentile(im, 1),
        np.nanpercentile(im, 99),
    )
    print(
        "cell density maps in [",
        np.nanpercentile(im, 1),
        ",",
        np.nanpercentile(im, 99),
        "]",
    )

    im = division_density_map[z, :, :]
    save_fig(
        im,
        Path(folder) / f"4b_10b_proliferation/4_z{z}_divdens.svg",
        "inferno",
        np.nanpercentile(im, 1),
        np.nanpercentile(im, 99),
    )
    print(
        "division density maps in [",
        np.nanpercentile(im, 1),
        ",",
        np.nanpercentile(im, 99),
        "]",
    )
    im = proliferation[z, :, :]
    save_fig(
        im,
        Path(folder) / f"4b_10b_proliferation/4_z{z}_prol.svg",
        "inferno",
        np.nanpercentile(im, 1),
        np.nanpercentile(im, 99),
    )
    print(
        "proliferation maps in [",
        np.nanpercentile(im, 1),
        ",",
        np.nanpercentile(im, 99),
        "]",
    )
    im = hoechst_norm[z, :, :]
    save_fig(
        im,
        Path(folder) / f"4b_10b_proliferation/4_z{z}_hoechst.svg",
        "gray_r",
        np.nanpercentile(im, 1),
        np.nanpercentile(im, 99),
    )
    im = ph3_norm[z, :, :]
    save_fig(
        im,
        Path(folder) / f"4b_10b_proliferation/4_z{z}_ph3.svg",
        "gray_r",
        np.nanpercentile(im, 1),
        np.nanpercentile(im, 99),
    )
