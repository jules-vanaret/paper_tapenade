import tifffile
import numpy as np
import matplotlib.pyplot as plt

folder = ...
oneview = tifffile.imread(rf"{folder}\1_view.tif")
twoviews = tifffile.imread(rf"{folder}\2_views.tif")
mask_oneview = tifffile.imread(rf"{folder}\1_view_mask.tif")
mask_twoviews = tifffile.imread(rf"{folder}\1_views_mask.tif")


def list_intensities(im, mask):
    int_im = []
    for z in range(0, len(im)):
        im2D = (im[z, :, :]).astype(float)
        im2D[mask[z, :, :] == 0] = np.nan
        int_im.append(np.nanmean(im2D))
    return int_im


int_oneview = list_intensities(oneview, mask_oneview)
list_depths_1side = np.arange(0, len(oneview))
int_twoviews = list_intensities(twoviews, mask_twoviews)
list_depths_2sides = np.arange(0, len(twoviews))


fig, ax = plt.subplots(1, figsize=(5, 4))
plt.plot(
    list_depths_2sides,
    int_twoviews,
    color="darkturquoise",
    label="two views \n+ reconstruction",
    linewidth=5,
)
plt.plot(list_depths_1side, int_oneview, color="orchid", label="one view", linewidth=5)
plt.xlabel("depth(µm)", fontsize=20)
plt.ylabel("Avg. intensity in \n each plane (A.U)", fontsize=20)
plt.title("Averaged intensity for one\n vs two views imaging", fontsize=30)
plt.legend(loc="lower left", prop={"size": 20})
plt.xticks([0, 100, 200, 300], fontsize=20)
plt.yticks([0, 100, 200], fontsize=20)
plt.savefig(rf"{folder}\plot.svg")
