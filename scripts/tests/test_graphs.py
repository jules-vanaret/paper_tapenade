import numpy as np
import tifffile

# import napari
import matplotlib.pyplot as plt

from stardist.models import StarDist3D
from tqdm import tqdm
from tapenade.preprocessing import local_image_equalization
from tapenade.preprocessing import change_arrays_pixelsize


path_to_data = "/data1/data_paper_tapenade/1_vs_2_views"

one_view = tifffile.imread(f"{path_to_data}/iso.tif")
one_view_mask = tifffile.imread(f"{path_to_data}/iso_mask.tif")
one_view_norm = local_image_equalization(
    one_view, box_size=10, perc_low=1, perc_high=99, mask=one_view_mask
)


# intensities = [np.median(one_view_z[mask_z]) for one_view_z, mask_z in zip(one_view, one_view_mask)]
# intensities_norm = [np.median(one_view_norm_z[mask_z]) for one_view_norm_z, mask_z in zip(one_view_norm, one_view_mask)]

two_views = tifffile.imread(f"{path_to_data}/g2_dapi_fused.tif")
two_views_mask = tifffile.imread(f"{path_to_data}/g2_dapi_fused_mask.tif")
two_views_norm = local_image_equalization(
    two_views, box_size=10, perc_low=1, perc_high=99, mask=two_views_mask
)

intensities2 = [
    np.median(two_views_z[mask_z])
    for two_views_z, mask_z in zip(two_views, two_views_mask)
]
# intensities_norm2 = [np.median(two_views_norm_z[mask_z]) for two_views_norm_z, mask_z in zip(two_views_norm, two_views_mask)]


def predict_lennedist(array, zoom_factors, normalize=False):

    is_temporal = array.ndim == 4

    if normalize:
        if is_temporal:
            perc_low, perc_high = np.percentile(array, (1, 99), axis=(1, 2, 3))

            array = (array - perc_low[:, None, None, None]) / (perc_high - perc_low)[
                :, None, None, None
            ]
            array = np.clip(array, 0, 1)
        else:
            perc_low, perc_high = np.percentile(array, (1, 99))

            array = (array - perc_low) / (perc_high - perc_low)
            array = np.clip(array, 0, 1)

    # isotropize to reach target object size
    array = change_arrays_pixelsize(image=array, zoom_factors=zoom_factors, order=1)
    print(array.min(), array.max())

    model = StarDist3D(
        None, name="lennedist_3d_grid222_rays64", basedir="/data1/lennedist_data/models"
    )
    model.config.use_gpu = True

    if is_temporal:
        labels = np.zeros(array.shape, dtype=np.uint16)

        for index_t, im in enumerate(tqdm(array)):
            labs, _ = model.predict_instances(im, n_tiles=model._guess_n_tiles(im))
            labels[index_t] = labs

    else:
        labels, _ = model.predict_instances(array, n_tiles=model._guess_n_tiles(array))

    # stretch by a factor of 2 in all dims to account for binning, plus
    # initial zoom_factors
    second_zoom_factors = [1 / zf for zf in zoom_factors]
    labels = change_arrays_pixelsize(labels, zoom_factors=second_zoom_factors, order=0)

    return labels


# labels = predict_lennedist(np.random.rand(300,600,600), (1,1,1), normalize=False)

# labels = predict_lennedist(one_view, (1/0.621, 1/0.621, 1/0.621), normalize=True)
labels_norm = predict_lennedist(
    one_view_norm, (1 / 0.621, 1 / 0.621, 1 / 0.621), normalize=False
)
tifffile.imwrite(f"{path_to_data}/iso_norm_labels.tif", labels_norm)

# tifffile.imwrite(f'{path_to_data}/iso_labels.tif', labels)

# labels2 = predict_lennedist(two_views, (1/0.621, 1/0.621, 1/0.621), normalize=True)
labels_norm2 = predict_lennedist(
    two_views_norm, (1 / 0.621, 1 / 0.621, 1 / 0.621), normalize=False
)

# tifffile.imwrite(f'{path_to_data}/g2_dapi_fused_labels.tif', labels2)
tifffile.imwrite(f"{path_to_data}/g2_dapi_fused_norm_labels.tif", labels_norm2)

# plt.figure()
# plt.plot(intensities)
# plt.plot(intensities2)
# plt.xlabel('Depth')
# plt.ylabel('Median intensity')

# plt.figure()
# plt.plot(intensities_norm)
# plt.plot(intensities_norm2)
# plt.xlabel('Depth')
# plt.ylabel('Median intensity')

# plt.show()


# viewer = napari.Viewer()
# viewer.add_image(one_view)
# viewer.add_image(one_view_norm)
# # viewer.add_image(one_view_mask, opacity=0.5, colormap='red')
# viewer.add_image(two_views)
# viewer.add_image(two_views_norm)

# napari.run()
