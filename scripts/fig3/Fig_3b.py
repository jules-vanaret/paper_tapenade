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
list_samples = []  # nums
sigma = 35
scale = (1, 1, 1)

visualize_napari = True
save_2D_maps = True

for num in tqdm(list_samples):
    mask = tifffile.imread(rf"{folder}\masks\{num}.tif")
    image = tifffile.imread(rf"{folder}\data\{num}.tif")
    seg = tifffile.imread(rf"{folder}\segmentation\{num}_seg.tif").astype(int)
    div = tifffile.imread(rf"{folder}\div\{num}_div.tif").astype(int)
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
    # tifffile.imwrite(rf'{folder}\{num}_seg_signal.tif',seg_signal.astype('uint16'))

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

    tifffile.imwrite(rf"{folder}\{num}_proliferation.tif", proliferation)
    tifffile.imwrite(rf"{folder}\{num}_celldens.tif", cell_density_map)
    tifffile.imwrite(rf"{folder}\{num}_divdens.tif", division_density_map)

    if visualize_napari:
        viewer = napari.Viewer()
        viewer.add_image(hoechst_norm, name=str(num) + "_hoechst")
        viewer.add_image(ph3_norm, name=str(num) + "_ph3")
        viewer.add_labels(div, name=str(num) + "_div")
        viewer.add_labels(seg, name=str(num) + "_seg")
        viewer.add_image(
            cell_density_map,
            name=str(num) + "_celldensity",
            colormap="inferno",
            contrast_limits=[0.08, 0.8],
        )
        viewer.add_image(
            division_density_map,
            name=str(num) + "_divisiondensity",
            colormap="inferno",
            contrast_limits=[0.08, 0.8],
        )
        viewer.add_image(
            proliferation,
            name=str(num) + "_proliferation",
            colormap="inferno",
            contrast_limits=[0.08, 0.8],
        )
        napari.run()

    if save_2D_maps:
        z = 85
        im = cell_density_map[z, :, :]
        save_fig(
            im,
            rf"{folder}\{num}_z{z}_celldens.svg",
            "inferno",
            np.nanpercentile(im, 1),
            np.nanpercentile(im, 99),
        )
        im = division_density_map[z, :, :]
        save_fig(
            im,
            rf"{folder}\{num}_z{z}_divdens.svg",
            "inferno",
            np.nanpercentile(im, 1),
            np.nanpercentile(im, 99),
        )
        im = proliferation[z, :, :]
        save_fig(
            im,
            rf"{folder}\{num}_z{z}_prol.svg",
            "inferno",
            np.nanpercentile(im, 1),
            np.nanpercentile(im, 99),
        )
        im = hoechst_norm[z, :, :]
        save_fig(
            im,
            rf"{folder}\{num}_z{z}_hoechst.svg",
            "gray_r",
            np.nanpercentile(im, 1),
            np.nanpercentile(im, 99),
        )
        im = ph3_norm[z, :, :]
        save_fig(
            im,
            rf"{folder}\{num}_z{z}_ph3.svg",
            "gray_r",
            np.nanpercentile(im, 1),
            np.nanpercentile(im, 99),
        )
