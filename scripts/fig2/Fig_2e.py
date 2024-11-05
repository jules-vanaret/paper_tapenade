# This script generates a plot of the density of segmented cells in the organoid as a function of the depth

import tifffile
from skimage.measure import regionprops
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tapenade.preprocessing._preprocessing import change_array_pixelsize

folder = ...
path_fig = ...
samples = []
folder_data = rf"{folder}\hoechst"
folder_mask = rf"{folder}\masks"
folder_seg = rf"{folder}\seg"
paths = sorted(glob(rf"{folder_data}\*.tif"))

scale = (0.65, 0.65, 0.65)
samples = [...]

values = np.linspace(0, 1, len(samples) + 1)
cmap = ["teal", "navy", "mediumvioletred", "firebrick"]
index_cmap = 0


fig = plt.figure(figsize=(14, 10))
for index, name in enumerate(samples):
    list_depths = []
    list_densities = []
    mask = tifffile.imread(
        rf"{folder_mask}\{name}_mask.tif"
    )  # mask can also be just the segmentation binarized. we use it to measure the sample total volume
    seg = tifffile.imread(rf"{folder_seg}\{name}_seg.tif")
    mask_iso = change_array_pixelsize(
        array=mask, input_pixelsize=scale, output_pixelsize=(1, 1, 1), order=0
    )
    seg_iso = change_array_pixelsize(
        array=seg, input_pixelsize=scale, output_pixelsize=(1, 1, 1), order=0
    )

    rg = regionprops(mask_iso.astype(int))
    for prop in rg:
        (zmin, ymin, xmin, zmax, ymax, xmax) = prop.bbox
    depth = zmax - zmin

    for z in range(0, len(seg_iso), 3):  # one every 3 plans to make it a bit faster
        nb_cells = len(np.unique(seg_iso[z, :, :]))
        surface = np.sum(mask_iso[z, :, :])
        if surface < 500:  # prevent nonsense density values due to very small plans
            density = 0
        else:
            density = nb_cells / surface  # cells/um2
            density = density * (100 * 100)  # cells/(100um2)
        list_densities.append(density)
        list_depths.append(
            z / depth
        )  # normalized depth : relative position of the considered plan in the z axis
    plt.plot(
        list_depths,
        list_densities,
        label="o=" + str(depth) + "µm",
        color=cmap[index],
        linewidth=5,
    )


plt.legend(fontsize=20)
plt.xlabel("normalized depth", fontsize=30)
plt.ylabel("Cells/(100µm)2", fontsize=30)
plt.xticks([0.25, 0.5, 0.75, 1], fontsize=30)
plt.yticks([0, 40, 80, 120], fontsize=30)
plt.show()
fig.savefig(rf"{folder}\figure.svg")
