import numpy as np
import tifffile
# import napari
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from skimage.measure import regionprops

# cle.select_device(0)

def density_from_labels(labels):
    props = regionprops(labels)

    densities = np.zeros(labels.shape[0])

    # positions = [int(np.round(prop.centroid[0])) for prop in props]
    # # _, densities = np.unique(positions, return_counts=True)

    # for position in positions:
    #     densities[position] += 1
    for prop in props:
        densities[int(np.round(prop.centroid[0]))] += 1

    return densities




path_to_data = '/data1/data_paper_tapenade/dapi_stacks_high_magn/processed'

media = ['water', 'optiprep', 'gold', 'glycerol']

all_slices = [
    (slice(0,None), slice(260, 660), slice(230, 630)),
    (slice(0,None), slice(310, 710), slice(258, 658)),
    (slice(0,None), slice(374, 774), slice(248, 648)),
    (slice(0,None), slice(382, 702), slice(222, 622)),

]

fig, axes = plt.subplots(2, 2)

axes[0,0].set_xlabel('depth (voxels)')
axes[0,0].set_ylabel('median intensity')

axes[1,0].set_xlabel('depth (voxels)')
axes[1,0].set_ylabel('median intensity')

axes[0,1].set_xlabel('depth (voxels)')
axes[0,1].set_ylabel('labels density')

axes[1,1].set_xlabel('depth (voxels)')
axes[1,1].set_ylabel('labels density')

for i,(medium, slices) in enumerate(zip(media, all_slices)):
    image = tifffile.imread(f'{path_to_data}/{medium}.tif')[slices]
    image_norm = tifffile.imread(f'{path_to_data}/{medium}_norm.tif')[slices]

    labels = tifffile.imread(f'{path_to_data}/{medium}_labels.tif')[slices]
    labels_norm = tifffile.imread(f'{path_to_data}/{medium}_norm_labels.tif')[slices]

    mask = tifffile.imread(f'{path_to_data}/{medium}_mask.tif')[slices]

    intensities = [np.median(image_z[mask_z]) for image_z, mask_z in zip(image, mask)]
    intensities_norm = [np.median(image_z[mask_z]) for image_z, mask_z in zip(image_norm, mask)]

    densities = density_from_labels(labels)
    densities = [d/np.sum(m) for d, m in zip(densities, mask)]
    densities_norm = density_from_labels(labels_norm)
    densities_norm = [d/np.sum(m) for d, m in zip(densities_norm, mask)]

    axes[0,0].plot(uniform_filter(intensities, size=3,), label=medium)
    axes[1,0].plot(uniform_filter(intensities_norm, size=3), linestyle='--')

    axes[0,1].plot(uniform_filter(densities, size=10))
    axes[1,1].plot(uniform_filter(densities_norm, size=10), linestyle='--')


axes[0,0].legend()
fig.tight_layout()
plt.show()