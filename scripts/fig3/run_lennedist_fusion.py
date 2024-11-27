import numpy as np
import tifffile

# import napari
import matplotlib.pyplot as plt

from stardist.models import StarDist3D
from tqdm import tqdm
from tapenade.preprocessing import local_contrast_enhancement
from tapenade.preprocessing import change_array_pixelsize


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
    array = change_array_pixelsize(image=array, zoom_factors=zoom_factors, order=1)
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
    labels = change_array_pixelsize(labels, zoom_factors=second_zoom_factors, order=0)

    return labels


path_to_data = "/data1/data_paper_tapenade/1_vs_2_views/processed"


for prefix in ["one", "two"]:
    image = tifffile.imread(f"{path_to_data}/{prefix}_cropped.tif")
    labels = predict_lennedist(image, (1 / 0.621, 1 / 0.621, 1 / 0.621), normalize=True)

    image_norm = tifffile.imread(f"{path_to_data}/{prefix}_cropped_norm.tif")
    labels_norm = predict_lennedist(
        image_norm, (1 / 0.621, 1 / 0.621, 1 / 0.621), normalize=False
    )

    tifffile.imwrite(f"{path_to_data}/{prefix}_cropped_labels.tif", labels)
    tifffile.imwrite(f"{path_to_data}/{prefix}_cropped_norm_labels.tif", labels_norm)
