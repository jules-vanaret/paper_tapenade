import tifffile
import numpy as np
import matplotlib.pyplot as plt
import napari
from pathlib import Path

folder = Path(__file__).parents[2] / 'data'
oneview_bottom = tifffile.imread(Path(folder) / "2f_1vs2views_imaging/6_bottom_iso.tif")
oneview_top = tifffile.imread(Path(folder) / "2f_1vs2views_imaging/6_top_iso.tif")
twoviews = tifffile.imread(Path(folder) / "2f_1vs2views_imaging/6_reg_iso.tif")
mask_oneview_bottom = tifffile.imread(
    Path(folder) / "2f_1vs2views_imaging/6_bottom_iso_mask.tif"
)
mask_twoviews = tifffile.imread(
    Path(folder) / "2f_1vs2views_imaging/6_reg_iso_mask.tif"
)


def list_intensities(im, mask):
    int_im = []
    for z in range(0, len(im)):
        im2D = (im[z, :, :]).astype(float)
        im2D[mask[z, :, :] == 0] = np.nan
        int_im.append(np.nanmean(im2D))
    return int_im


int_oneview = list_intensities(oneview_bottom, mask_oneview_bottom)
list_depths_1side = np.arange(0, len(oneview_bottom))
int_twoviews = list_intensities(twoviews, mask_twoviews)
list_depths_2sides = np.arange(0, len(twoviews))


fig, ax = plt.subplots(1, figsize=(10, 8))
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
plt.tight_layout()
# plt.savefig(Path(folder) / "2f_plot.svg")
plt.show()


viewer = napari.Viewer()
viewer.add_image(oneview_bottom, colormap="gray_r")
viewer.add_image(oneview_top, colormap="gray_r")
viewer.add_image(twoviews, colormap="gray_r")
for l in viewer.layers:
    l.data = np.transpose(l.data, (1, 0, 2))
viewer.grid.enabled = True
napari.run()
