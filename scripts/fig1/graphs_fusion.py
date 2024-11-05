import numpy as np
import tifffile
# import napari
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter
from skimage.measure import regionprops
import os
import pandas as pd


def density_from_labels(labels):
    props = regionprops(labels)

    densities = np.zeros(labels.shape[0])
    for prop in props:
        densities[int(np.round(prop.centroid[0]))] += 1

    return densities




path_to_data = '/data1/data_paper_tapenade/1_vs_2_views/processed'
path_to_csv_one = f'{path_to_data}/results_one.csv'
path_to_csv_two = f'{path_to_data}/results_two.csv'


if not os.path.exists(path_to_csv_one):
    
    one_view = tifffile.imread(f'{path_to_data}/one_cropped.tif')
    one_view_mask = tifffile.imread(f'{path_to_data}/one_mask_cropped.tif')
    one_view_norm = tifffile.imread(f'{path_to_data}/one_cropped_norm.tif')
    one_view_labels = tifffile.imread(f'{path_to_data}/one_cropped_labels.tif')
    one_view_labels_norm = tifffile.imread(f'{path_to_data}/one_cropped_norm_labels.tif')

    two_views = tifffile.imread(f'{path_to_data}/two_cropped.tif')
    two_views_mask = tifffile.imread(f'{path_to_data}/two_mask_cropped.tif')
    two_views_norm = tifffile.imread(f'{path_to_data}/two_cropped_norm.tif')
    two_views_labels = tifffile.imread(f'{path_to_data}/two_cropped_labels.tif')
    two_views_labels_norm = tifffile.imread(f'{path_to_data}/two_cropped_norm_labels.tif')


    #intensity profiles
    intensity_one = [np.median(one_view_z[one_view_mask_z]) for one_view_z, one_view_mask_z in zip(one_view, one_view_mask)]
    intensity_one_norm = [np.median(one_view_norm_z[one_view_mask_z]) for one_view_norm_z, one_view_mask_z in zip(one_view_norm, one_view_mask)]
    intensity_two = [np.median(two_views_z[two_views_mask_z]) for two_views_z, two_views_mask_z in zip(two_views, two_views_mask)]
    intensity_two_norm = [np.median(two_views_norm_z[two_views_mask_z]) for two_views_norm_z, two_views_mask_z in zip(two_views_norm, two_views_mask)]

    # number of labels profiles
    nlabels_one = density_from_labels(one_view_labels)
    nlabels_one_norm = density_from_labels(one_view_labels_norm)
    nlabels_two = density_from_labels(two_views_labels)
    nlabels_two_norm = density_from_labels(two_views_labels_norm)

    # number of labels divided by area profiles
    density_one = [d/np.sum(m) for d, m in zip(nlabels_one, two_views_mask)]
    density_one_norm = [d/np.sum(m) for d, m in zip(nlabels_one_norm, two_views_mask)]
    density_two = [d/np.sum(m) for d, m in zip(nlabels_two, two_views_mask)]
    density_two_norm = [d/np.sum(m) for d, m in zip(nlabels_two_norm, two_views_mask)]

    df_one = pd.DataFrame({
        'intensity_one': intensity_one,
        'intensity_one_norm': intensity_one_norm,
        'nlabels_one': nlabels_one,
        'nlabels_one_norm': nlabels_one_norm,
        'density_one': density_one,
        'density_one_norm': density_one_norm,
    })

    df_one.to_csv(path_to_csv_one)

    df_two = pd.DataFrame({
        'intensity_two': intensity_two,
        'intensity_two_norm': intensity_two_norm,
        'nlabels_two': nlabels_two,
        'nlabels_two_norm': nlabels_two_norm,
        'density_two': density_two,
        'density_two_norm': density_two_norm,
    })

    df_two.to_csv(path_to_csv_two)

else:
    df_one = pd.read_csv(path_to_csv_one)
    df_two = pd.read_csv(path_to_csv_two)

    intensity_one = df_one['intensity_one'].values
    intensity_one_norm = df_one['intensity_one_norm'].values
    nlabels_one = df_one['nlabels_one'].values
    nlabels_one_norm = df_one['nlabels_one_norm'].values
    density_one = df_one['density_one'].values
    density_one_norm = df_one['density_one_norm'].values

    intensity_two = df_two['intensity_two'].values
    intensity_two_norm = df_two['intensity_two_norm'].values
    nlabels_two = df_two['nlabels_two'].values
    nlabels_two_norm = df_two['nlabels_two_norm'].values
    density_two = df_two['density_two'].values
    density_two_norm = df_two['density_two_norm'].values


fig, axes = plt.subplots(2, 3, figsize=(10, 7))

ax1 = axes[0, 0]
ax2 = axes[1, 0]

ax1.plot(uniform_filter(intensity_one, size=3), label='one', c='royalblue')
ax2.plot(uniform_filter(intensity_one_norm, size=3), label='one norm', c='royalblue', linestyle='--')
ax1.plot(uniform_filter(intensity_two, size=3), label='two', c='orange')
ax2.plot(uniform_filter(intensity_two_norm, size=3), label='two norm', c='orange', linestyle='--')

ax1.legend()
ax1.set_xlabel('depth')
ax1.set_ylabel('median intensity')
ax2.legend()
ax2.set_xlabel('depth')
ax2.set_ylabel('median intensity')




ax1 = axes[0, 1]
ax2 = axes[1, 1]

ax1.plot(uniform_filter(nlabels_one, size=10), label='one', c='royalblue')
ax2.plot(uniform_filter(nlabels_one_norm, size=10), label='one norm', c='royalblue', linestyle='--')
ax1.plot(uniform_filter(nlabels_two, size=10), label='two', c='orange')
ax2.plot(uniform_filter(nlabels_two_norm, size=10), label='two norm', c='orange', linestyle='--')

ax1.legend()
ax1.set_xlabel('depth')
ax1.set_ylabel('number of labels')
ax2.legend()
ax2.set_xlabel('depth')
ax2.set_ylabel('number of labels')



ax1 = axes[0, 2]
ax2 = axes[1, 2]

ax1.plot(uniform_filter(density_one, size=10), label='one', c='royalblue')
ax2.plot(uniform_filter(density_one_norm, size=10), label='one norm', c='royalblue', linestyle='--')
ax1.plot(uniform_filter(density_two, size=10), label='two', c='orange')
ax2.plot(uniform_filter(density_two_norm, size=10), label='two norm', c='orange', linestyle='--')

ax1.legend()
ax1.set_xlabel('depth')
ax1.set_ylabel('density normalized by mask area')
ax2.legend()
ax2.set_xlabel('depth')
ax2.set_ylabel('density normalized by mask area')

plt.tight_layout()

plt.savefig(f'{path_to_data}/results.svg')
plt.savefig(f'{path_to_data}/results.png')
plt.show()